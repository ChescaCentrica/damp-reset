"""Ventilation moisture model - well-mixed single-zone room.

Answers the question: "if the window is open for X minutes and outdoor
absolute humidity is C_out, how much moisture remains in the room?" using
the analytic solution to the well-mixed-zone moisture balance.

The governing ODE and its analytic solution are documented in full on
``predict_final_absolute_humidity``. This module docstring focuses on the
modelling assumptions and what each one costs in real-world accuracy.

MODELLING ASSUMPTIONS AND THEIR REAL-WORLD IMPLICATIONS
=======================================================

1. Perfectly mixed room air.
   What it means: at every instant, the moisture concentration is uniform
   throughout the room volume. Air entering through the window immediately
   equilibrates with the whole room rather than forming a plume.
   Real-world implication: real rooms show stratification (warm humid air
   near the ceiling, cold dry air near a cracked window), local jets from
   openings, and dead corners with slower turnover. The single-zone model
   over-estimates how quickly the *average* indoor sensor reading catches
   up with outdoor air, and it cannot represent within-room gradients that
   condensation risk actually depends on (a cold external wall may reach
   dew point long before the middle of the room does).

2. Air-change rate (ACH) is constant while the window state does not change.
   What it means: throughout a single window-open (or window-closed)
   interval, the exchange rate has one value. Changing the window state is
   the only thing that changes ACH.
   Real-world implication: real ACH varies with wind speed and direction,
   indoor/outdoor temperature difference (stack effect), open door state,
   whether HVAC is running, and how far the window is cracked. Predictions
   over intervals when weather or occupant behaviour changes will drift.

3. Outdoor absolute humidity remains constant during a ventilation event.
   What it means: C_out is a scalar, not a time series, for the duration
   of a single predict_moisture call.
   Real-world implication: fine for the ~5-30 minute events this POC is
   built around; degrades over multi-hour horizons where outdoor humidity
   shifts with weather. Chaining short predictions with fresh outdoor
   readings between them is the intended workaround.

4. No internal moisture generation.
   What it means: no source term in the ODE for occupant respiration,
   cooking, showering, drying laundry, indoor plants, aquaria, pets.
   Real-world implication: real rooms usually have SOME source most of
   the time. Two adults sleeping add ~40-80 g/h of water vapour; a
   simmering pot easily adds 500+ g/h; drying a load of laundry indoors
   can add 1-3 kg over a day. The model UNDERESTIMATES how much
   ventilation is needed to hit any given target because it ignores the
   inflow that has to be compensated for.

5. No moisture buffering from walls, furniture, textiles or other fabrics.
   What it means: the only reservoir modelled is the room air. Plaster,
   wood, gypsum, textiles, and bedding are treated as inert.
   Real-world implication: these hygroscopic materials hold ORDERS OF
   MAGNITUDE more water than the air (typically 10-100x by mass in a
   normally furnished residential room), buffer moisture in and out on
   timescales of hours to days, and can dominate the moisture balance
   after the fast initial air-exchange transient. A short ventilation
   event may dry the AIR far more effectively than it dries the ROOM,
   because the fabric slowly re-wets the air over subsequent hours.
   Damp-risk conclusions drawn from air-only predictions are therefore
   optimistic in the long run.

6. No condensation or evaporation at surfaces.
   What it means: no phase change is modelled anywhere in the room.
   Real-world implication: cold surfaces (single-glazed windows, thermal
   bridges, external walls) below the local dew point continuously
   REMOVE water from the air by condensation while ventilation is
   removing it by exchange; warm damp surfaces (recent shower splashes,
   drying floor tiles) continuously ADD water by evaporation. Neither is
   in the model. Predictions closest to a saturating surface should be
   trusted least.

7. No inter-room airflow.
   What it means: the model treats one room in isolation. Doors are
   effectively either shut or represented only via a lumped ACH.
   Real-world implication: real dwellings behave as networks of rooms
   connected by doors and vents, driven by pressure differences,
   temperature differences, and mechanical ventilation. Moisture
   generated in a bathroom typically spreads to bedrooms overnight
   through door gaps; a single-room model cannot represent that at all.

8. Temperature changes during ventilation are not modelled.
   What it means: no thermal model runs alongside the moisture model.
   Absolute humidity in g/m^3 is the conserved quantity here.
   Real-world implication: opening a window in winter cools the room,
   which shrinks the air volume slightly, changes the RH reading at any
   given absolute humidity, and can open up condensation risk on
   newly-cold surfaces. Absolute humidity in g/m^3 is only strictly
   conserved under isothermal ventilation, so predictions become less
   trustworthy the larger the indoor/outdoor temperature difference is
   and the longer the event runs.

OUT OF SCOPE FOR THIS MODULE
============================
    - Thermal / energy model (heat loss when the window is open).
    - Moisture buffering, condensation, or evaporation at surfaces.
    - Internal moisture generation (occupants, cooking, laundry).
    - Multi-zone / inter-room airflow.
    - Sensor integration, ACH estimation from measurements, optimiser,
      dashboard, control logic, mould-risk modelling, weather forecasts.
"""

from dataclasses import dataclass
from math import exp, isfinite

from psychrometrics import AirState

MINUTES_PER_HOUR: float = 60.0


@dataclass(frozen=True)
class Room:
    """Immutable description of a room and its ventilation properties.

    Fields:
        volume_m3: room volume in cubic metres. Must be strictly positive.
        indoor_temperature_c: current indoor air temperature in degrees
            Celsius. Not range-checked here; downstream psychrometric calls
            enforce their own residential-range contract.
        indoor_relative_humidity_pct: current indoor relative humidity on a
            0-100 scale. Must lie in [0, 100].
        ach_closed: air changes per hour with windows and doors shut, i.e.
            the building's baseline infiltration rate. Must be non-negative;
            0 is allowed (a perfectly sealed envelope, useful as a limit
            case in tests).
        ach_window_open: air changes per hour while the ventilation window
            is open. Must be non-negative. Not asserted to exceed
            ``ach_closed`` - that is a modelling assumption for the caller,
            not a data-integrity check.

    Validation is performed in ``__post_init__`` so an invalid Room cannot
    be constructed at all. All fields must additionally be finite; NaN or
    infinite inputs are rejected.
    """

    volume_m3: float
    indoor_temperature_c: float
    indoor_relative_humidity_pct: float
    ach_closed: float
    ach_window_open: float

    def __post_init__(self) -> None:
        """Validate every field; raise ValueError with a targeted message on failure."""
        for name, value in (
            ("volume_m3", self.volume_m3),
            ("indoor_temperature_c", self.indoor_temperature_c),
            ("indoor_relative_humidity_pct", self.indoor_relative_humidity_pct),
            ("ach_closed", self.ach_closed),
            ("ach_window_open", self.ach_window_open),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if self.volume_m3 <= 0.0:
            raise ValueError(
                f"volume_m3 must be strictly positive, got {self.volume_m3}"
            )
        if not 0.0 <= self.indoor_relative_humidity_pct <= 100.0:
            raise ValueError(
                f"indoor_relative_humidity_pct={self.indoor_relative_humidity_pct} "
                f"is outside the valid range [0, 100]"
            )
        if self.ach_closed < 0.0:
            raise ValueError(
                f"ach_closed must be non-negative, got {self.ach_closed}"
            )
        if self.ach_window_open < 0.0:
            raise ValueError(
                f"ach_window_open must be non-negative, got {self.ach_window_open}"
            )


def predict_final_absolute_humidity(
    indoor_ah_g_m3: float,
    outdoor_ah_g_m3: float,
    ach: float,
    duration_minutes: float,
) -> float:
    """Indoor absolute humidity after ``duration_minutes`` of steady ventilation.

    Solves the well-mixed single-zone moisture balance with no internal
    moisture generation:

        dC/dt = n * (C_out - C)                       [g/m^3 per hour]

    where C is indoor absolute humidity in g/m^3, C_out is outdoor absolute
    humidity in g/m^3, n is the air-change rate in hours^-1 (ACH), and t is
    time in hours. The analytic solution starting from C(0) = C_0 is:

        C(t) = C_out + (C_0 - C_out) * exp(-n * t)

    Interpretation:
        - As t grows, the exponential decays to 0 and C -> C_out: the room
          asymptotically equilibrates with outdoor air.
        - The time constant is tau = 1 / n hours; after one tau the
          indoor-outdoor gap has shrunk by a factor of e (~= 63 % of the
          way there).
        - If n = 0 (no exchange), exp(0) = 1 and C(t) = C_0 for any t.
        - If t = 0, exp(0) = 1 and C(0) = C_0 as required.

    Both degenerate cases fall out of the formula naturally; no special
    branches are used.

    Modelling assumptions are documented at module level; the most
    important ones for this function are perfect mixing, constant ACH
    across the event, constant outdoor AH, and no moisture sources.

    Args:
        indoor_ah_g_m3: initial indoor absolute humidity, g/m^3. Must be
            non-negative and finite.
        outdoor_ah_g_m3: outdoor absolute humidity, g/m^3, assumed constant
            for the duration of the event. Must be non-negative and finite.
        ach: air-change rate for the event, in hours^-1. Must be
            non-negative and finite.
        duration_minutes: length of the event, in minutes. Must be
            non-negative and finite. Converted to hours internally so the
            exponent stays dimensionless.

    Returns:
        Indoor absolute humidity at the end of the event, in g/m^3.

    Raises:
        ValueError: if any argument is negative, NaN, or infinite.
    """
    for name, value in (
        ("indoor_ah_g_m3", indoor_ah_g_m3),
        ("outdoor_ah_g_m3", outdoor_ah_g_m3),
        ("ach", ach),
        ("duration_minutes", duration_minutes),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    duration_hours = duration_minutes / MINUTES_PER_HOUR
    return outdoor_ah_g_m3 + (indoor_ah_g_m3 - outdoor_ah_g_m3) * exp(
        -ach * duration_hours
    )


def predict_final_absolute_humidity_with_source(
    indoor_ah_g_m3: float,
    outdoor_ah_g_m3: float,
    ach: float,
    duration_minutes: float,
    moisture_generation_g_per_hour: float,
    room_volume_m3: float,
) -> float:
    """Indoor absolute humidity after ventilation WITH an internal moisture source.

    Extends the ventilation-only equation this module already owns
    from

        dC/dt = n * (C_out - C)                       [g/m^3 per hour]

    to

        dC/dt = n * (C_out - C) + G / V              [g/m^3 per hour]

    where the additional term is a constant moisture-generation rate
    G [g/hour] distributed uniformly across the room volume V [m^3].
    G/V has units of (g/hour)/m^3 = (g/m^3) per hour, dimensionally
    matching the ventilation term.

    Analytic solution (piecewise-constant G over the event):

        Let C_eq = C_out + G / (n * V).
        C(t) = C_eq + (C_0 - C_eq) * exp(-n * t)

    ``C_eq`` is the steady-state indoor AH the room would reach if
    the source and the ventilation could balance indefinitely: the
    ventilation term drives C toward C_out at rate n, while the
    source raises C at G/V per hour, and the two balance when C
    exceeds C_out by exactly G/(n*V).

    Degenerate cases:
        - G = 0: reduces exactly to the ventilation-only equation
          (predict_final_absolute_humidity). Verified as a test.
        - n = 0 (no ventilation, source active): dC/dt = G/V,
          integrating to C(t) = C_0 + G * t / V. Handled by a
          separate branch because the analytic form above divides
          by n.
        - G = 0 and n = 0: the room is inert, C(t) = C_0. Falls out
          of either branch.

    Units contract (units are enforced by parameter names):
        indoor_ah_g_m3            g/m^3           room air, initial
        outdoor_ah_g_m3           g/m^3           outdoor air, constant
        ach                       hours^-1        air-change rate
        duration_minutes          minutes         event length
        moisture_generation_g_per_hour  g/hour    room-wide source rate
        room_volume_m3            m^3             room volume, > 0

    Args:
        indoor_ah_g_m3: initial indoor absolute humidity.
        outdoor_ah_g_m3: outdoor absolute humidity, constant across
            the event.
        ach: air-change rate for the event, in hours^-1. Non-negative.
        duration_minutes: length of the event, in minutes.
            Non-negative. Converted to hours internally so the ODE
            variable stays dimensionless.
        moisture_generation_g_per_hour: rate at which occupants /
            activities add water vapour to the room air, in grams
            per hour. Must be non-negative and finite. This is the
            SUM of every simultaneous source; scheduling of
            individual sources is handled by the caller (see
            ``moisture_sources.MoistureSourceSchedule``).
        room_volume_m3: room volume in cubic metres. Must be
            strictly positive because the source-per-unit-volume
            term ``G / V`` diverges as V approaches zero.

    Returns:
        Indoor absolute humidity at the end of the event, in g/m^3.

    Raises:
        ValueError: on any invalid argument.
    """
    for name, value in (
        ("indoor_ah_g_m3", indoor_ah_g_m3),
        ("outdoor_ah_g_m3", outdoor_ah_g_m3),
        ("ach", ach),
        ("duration_minutes", duration_minutes),
        (
            "moisture_generation_g_per_hour",
            moisture_generation_g_per_hour,
        ),
        ("room_volume_m3", room_volume_m3),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if room_volume_m3 <= 0.0:
        raise ValueError(
            f"room_volume_m3 must be strictly positive, got {room_volume_m3}"
        )

    duration_hours = duration_minutes / MINUTES_PER_HOUR
    # Special-case n = 0: no ventilation. The homogeneous solution
    # collapses (C_eq diverges) so integrate the source term
    # directly. C(t) = C_0 + (G/V) * t.
    if ach == 0.0:
        return (
            indoor_ah_g_m3
            + moisture_generation_g_per_hour * duration_hours / room_volume_m3
        )
    equilibrium_ah_g_m3 = (
        outdoor_ah_g_m3
        + moisture_generation_g_per_hour / (ach * room_volume_m3)
    )
    return equilibrium_ah_g_m3 + (
        indoor_ah_g_m3 - equilibrium_ah_g_m3
    ) * exp(-ach * duration_hours)


@dataclass(frozen=True)
class MoisturePrediction:
    """Bundled result of a well-mixed ventilation moisture prediction.

    Sign conventions:
        - ``absolute_humidity_change_g_m3`` = final - initial. Positive when
          ventilation ADDED water to the room (outdoor was wetter), negative
          when it REMOVED water.
        - ``absolute_humidity_reduction_g_m3`` = initial - final. Positive
          when ventilation DRIED the room (outdoor was drier), negative when
          it wet it. This is simply -change; both are kept because the
          downstream ventilation logic reads more naturally with the sign
          convention matching "did we dry?".
        - ``percentage_reduction`` = 100 * reduction / initial, using the
          INITIAL indoor absolute humidity as the denominator. So:
              +30 %  -> we removed 30 % of the water that was in the room
                0 %  -> no net change
              -50 %  -> we ADDED water equal to 50 % of what we started with
        - ``water_removed_g`` = (initial_ah - final_ah) * room_volume
          [g/m^3 * m^3 = g]. Positive = water removed (drying), zero = no
          change, negative = water added (ventilation moistened the room).
          The sign is preserved deliberately, no absolute value taken; the
          direction is useful downstream.

    Important scope note on ``water_removed_g``:
        This is the water removed from (or added to) the AIR VOLUME
        represented by the well-mixed room model. It is NOT the total
        moisture removed from walls, furniture, soft furnishings, plaster,
        or bedding - those reservoirs hold orders of magnitude more water
        than the air and are out of scope for this module.

    Why there is no ``final_relative_humidity`` (or final dew point / final
    humidity ratio from RH) field:
        Relative humidity is a function of BOTH water content and air
        temperature (RH = P_v / P_sat(T)). Opening a window changes both
        - the room dries out AND cools down. Computing a "final RH" from
        the model's final absolute humidity together with the room's
        ORIGINAL indoor temperature would silently assume the room stayed
        at that temperature, which is physically wrong and would produce
        a misleadingly high RH (the same water content at a lower T looks
        more humid, but the room did not stay at the original T).
        A valid final RH can only be computed once a thermal model
        predicts the room's final temperature. Until that model exists,
        the moisture layer deliberately stops at absolute humidity, which
        is a well-defined function of water content alone in the
        isothermal approximation this module operates under.

    Edge case: if ``initial_absolute_humidity_g_m3`` is exactly 0.0 the
    denominator is undefined; ``percentage_reduction`` is set to NaN. The
    absolute-value fields still describe what happened correctly.
    """

    initial_absolute_humidity_g_m3: float
    outdoor_absolute_humidity_g_m3: float
    final_absolute_humidity_g_m3: float
    absolute_humidity_change_g_m3: float
    absolute_humidity_reduction_g_m3: float
    percentage_reduction: float
    water_removed_g: float
    room_volume_m3: float
    ach: float
    duration_minutes: float


def predict_moisture(
    indoor_ah_g_m3: float,
    outdoor_ah_g_m3: float,
    ach: float,
    duration_minutes: float,
    room_volume_m3: float,
) -> MoisturePrediction:
    """Predict the room moisture after a ventilation event and report the details.

    Delegates the physics to ``predict_final_absolute_humidity`` (the
    analytic well-mixed-zone solution) so the equation lives in exactly one
    place. This wrapper adds the framing quantities a caller usually wants:
    change and reduction in g/m^3, a percentage relative to the initial
    room state, and the mass of water gained or lost from the room air.

    Args:
        indoor_ah_g_m3: initial indoor absolute humidity, g/m^3.
        outdoor_ah_g_m3: outdoor absolute humidity, g/m^3.
        ach: air-change rate while the window is open, hours^-1.
        duration_minutes: length of the ventilation event, minutes.
        room_volume_m3: room volume in cubic metres. Must be positive and
            finite. Used only to convert the change in absolute humidity
            into a mass of water via ``water = reduction_g_m3 *
            room_volume_m3``.

    Returns:
        A ``MoisturePrediction`` with initial / outdoor / final absolute
        humidities, the signed change and reduction in g/m^3, the
        percentage reduction (using the initial indoor value as the
        denominator; NaN if the initial value is exactly 0), the signed
        mass of water removed from the room AIR in grams, and the
        ACH / duration / volume inputs echoed back for audit.

    Raises:
        ValueError: if ``room_volume_m3`` is not finite or is non-positive;
            all other input validation propagates from
            ``predict_final_absolute_humidity``.
    """
    if not isfinite(room_volume_m3):
        raise ValueError(
            f"room_volume_m3 must be finite, got {room_volume_m3!r}"
        )
    if room_volume_m3 <= 0.0:
        raise ValueError(
            f"room_volume_m3 must be strictly positive, got {room_volume_m3}"
        )
    final_ah = predict_final_absolute_humidity(
        indoor_ah_g_m3=indoor_ah_g_m3,
        outdoor_ah_g_m3=outdoor_ah_g_m3,
        ach=ach,
        duration_minutes=duration_minutes,
    )
    change = final_ah - indoor_ah_g_m3
    reduction = indoor_ah_g_m3 - final_ah
    if indoor_ah_g_m3 == 0.0:
        percentage_reduction = float("nan")
    else:
        percentage_reduction = 100.0 * reduction / indoor_ah_g_m3
    water_removed_g = reduction * room_volume_m3
    return MoisturePrediction(
        initial_absolute_humidity_g_m3=indoor_ah_g_m3,
        outdoor_absolute_humidity_g_m3=outdoor_ah_g_m3,
        final_absolute_humidity_g_m3=final_ah,
        absolute_humidity_change_g_m3=change,
        absolute_humidity_reduction_g_m3=reduction,
        percentage_reduction=percentage_reduction,
        water_removed_g=water_removed_g,
        room_volume_m3=room_volume_m3,
        ach=ach,
        duration_minutes=duration_minutes,
    )


def predict_room_moisture(
    room: Room,
    outdoor: AirState,
    duration_minutes: float,
    window_open: bool = True,
) -> MoisturePrediction:
    """High-level ventilation prediction from a ``Room`` and outdoor ``AirState``.

    Convenience wrapper that:
        1. Reads indoor temperature and RH from ``room`` and asks the
           psychrometric module for the corresponding indoor absolute
           humidity (via ``AirState.absolute_humidity`` - the module-level
           ideal-gas equation, not reimplemented here).
        2. Reads outdoor absolute humidity from ``outdoor`` in the same way.
        3. Selects the correct air-change rate from the room -
           ``room.ach_window_open`` when ``window_open`` is True,
           ``room.ach_closed`` when False.
        4. Delegates the physics to ``predict_moisture``, which delegates
           to ``predict_final_absolute_humidity`` (the single source of the
           analytic solution).

    Args:
        room: the room whose indoor state and ventilation properties will
            drive the prediction.
        outdoor: the outdoor air state driving the ventilation exchange.
            Assumed constant for the duration of the event.
        duration_minutes: length of the ventilation event, in minutes.
            Passed through to ``predict_moisture``; must be non-negative
            and finite.
        window_open: if True, the exchange rate is ``room.ach_window_open``;
            if False, ``room.ach_closed``. Defaults to True since the model
            exists to answer window-open scenarios.

    Returns:
        A ``MoisturePrediction`` describing the room after the event.

    Notes:
        Input validation on the psychrometric side (temperature and RH
        ranges, finiteness) is performed by ``AirState``'s property
        accessors and the psychrometric standalone functions. Validation
        on ``duration_minutes`` and the ACH value happens in
        ``predict_final_absolute_humidity``.
    """
    indoor_state = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    )
    ach = room.ach_window_open if window_open else room.ach_closed
    return predict_moisture(
        indoor_ah_g_m3=indoor_state.absolute_humidity,
        outdoor_ah_g_m3=outdoor.absolute_humidity,
        ach=ach,
        duration_minutes=duration_minutes,
        room_volume_m3=room.volume_m3,
    )
