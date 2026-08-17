"""Ventilation thermal model - heat loss during window-open events.

Purpose
=======
Estimates the thermal penalty caused by exchanging warm indoor air with
colder outdoor air during a ventilation event: how much heat leaves the
room via the exchanged air, and (in a later slice) how much the room
cools as a result. This layer exists so the moisture benefit computed by
``moisture.py`` can eventually be weighed against its temperature cost;
that trade-off will be modelled by a future component, not here.

This slice contains only the scaffolding needed for the eventual
calculation: the modelling assumptions, the named SI constants that
appear in the heat-loss expressions, and a couple of unit-conversion
helpers. The actual temperature-drop expression is deliberately NOT in
this slice - the assumptions and unit contract are reviewed and locked
down here first.

Modelling assumptions
=====================
1. Room air is perfectly mixed.
   Real-world implication: the same caveat as ``moisture.py`` -
   stratification, jets, and dead corners are ignored. The single lumped
   indoor temperature will over-estimate how quickly a room-averaged
   sensor catches up with outdoor conditions.

2. Air-change rate is constant across a ventilation event.
   Real-world implication: real ACH varies with wind, stack effect, door
   state, and window opening area. Predictions over intervals when those
   change will drift.

3. Outdoor temperature is constant across a ventilation event.
   Real-world implication: fine for the short (~5-30 minute) events this
   POC targets; degrades over multi-hour horizons.

4. No active heating during the ventilation event.
   Real-world implication: in practice most residential heating systems
   keep running while a window is open; the model in this slice
   over-estimates how far the room cools because it ignores the
   compensating heat input. That gap is intentional - the goal here is
   to isolate the raw ventilation heat loss, so a later slice can bolt
   on a heating-power term with its own set of assumptions.

5. Solar gains are ignored.
   Real-world implication: on a sunny winter day, direct-beam solar gain
   through a window can offset a large fraction of ventilation heat
   loss; a rainy day removes that offset. Neither is in the model.

6. Occupant and equipment heat gains are ignored.
   Real-world implication: a seated adult produces ~100 W of sensible
   heat, a laptop 20-60 W, a running oven several kW. Predictions will
   over-estimate the cooling in occupied or equipment-loaded rooms.

7. Lumped effective thermal capacitance.
   The room and its immediately-coupled contents are represented by a
   single scalar C_eff [J/K] that includes the sensible heat capacity of
   the room air plus whatever fraction of walls, floor, furniture, and
   soft furnishings responds to short-timescale ventilation events. This
   is intentionally a modelling knob, not a first-principles quantity;
   its correct value for a specific room has to be identified against
   real measurements later.
   Real-world implication: a single scalar cannot represent the
   different time constants of the immediate air, the fast surface
   layers, and the slow deep-mass response. Short events (~minutes) are
   dominated by air + surface; long events (~hours) start pulling on
   deeper mass. One C_eff cannot fit both regimes.

8. Other heat-loss paths are ignored (conduction through walls, radiant
   loss to cold surfaces, infiltration when the window is closed).
   Real-world implication: this is deliberate for the first slice - it
   isolates ventilation heat loss so its magnitude and time constant can
   be reasoned about on their own. A realistic energy-balance model will
   need conduction terms too, and those will not be built in this
   module.

Out of scope for this module
============================
    - The temperature-drop calculation itself (deferred to the next
      slice, once the assumptions above are agreed).
    - Multi-zone airflow, wall conduction, radiation, moisture-thermal
      coupling.
    - Occupant / equipment heat gains, solar gains, active heating.
    - Weather forecasts, sensors, dashboards, optimiser, control logic.

SI unit contract
================
    Temperature       degrees Celsius (_c) for room conditions;
                      kelvin (_k) or kelvin-difference (_k) where an
                      absolute scale is required. Function signatures
                      encode which.
    Volume            cubic metres (_m3)
    Mass              kilograms (_kg)
    Time              seconds (_s) internally; minutes and hours only in
                      caller-facing helpers.
    Energy            joules (_j) internally; kilowatt-hours (_kwh) only
                      in caller-facing helpers.
    Power             watts (_w), i.e. J/s.

Every named constant below states its units in its name; every helper
does the same. There are no bare-number physical constants in this
module.
"""

from dataclasses import dataclass
from math import exp, isfinite

AIR_DENSITY_KG_PER_M3: float = 1.204
"""Density of dry air at 20 degC and 101325 Pa.

Derived from the ideal-gas law: rho = P * M_dry_air / (R * T_K)
                                    = 101325 * 0.028966 / (8.31446 * 293.15)
                                    ~= 1.2041 kg/m^3.
This is a residential-conditions reference value; real indoor density
varies by a few percent with temperature and altitude. A later slice can
promote it to rho(T) if that error matters for the eventual thermal
calculation, but for typical residential events keeping it constant is
consistent with the isothermal-approximation stance already taken by
``moisture.py``.
"""

AIR_SPECIFIC_HEAT_J_PER_KG_K: float = 1005.0
"""Specific heat capacity of air at constant pressure, near room temperature.

Standard reference value (ASHRAE Handbook of Fundamentals, ch. 1) is
~1005 J/(kg.K) at 20 degC and ~1006 at 25 degC. Variation across the
residential-ventilation range is under 1 percent, so a constant is fine
for this POC.
"""

SECONDS_PER_HOUR: float = 3600.0
"""Exact time-unit conversion (60 min * 60 s)."""

JOULES_PER_KWH: float = 3_600_000.0
"""Exact energy-unit conversion.

1 kWh = 1000 W * 3600 s = 3.6e6 J. Named so that energy results computed
internally in joules can be presented in the more human-readable kWh
without an implicit factor sitting in caller code.
"""


def joules_to_kwh(energy_j: float) -> float:
    """Convert an energy in joules to kilowatt-hours."""
    return energy_j / JOULES_PER_KWH


def kwh_to_joules(energy_kwh: float) -> float:
    """Convert an energy in kilowatt-hours to joules."""
    return energy_kwh * JOULES_PER_KWH


def airflow_rate_from_ach(
    room_volume_m3: float,
    ach: float,
) -> float:
    """Volumetric airflow rate through a room in cubic metres per second.

    Converts an air-change rate (per hour) and a room volume into the
    equivalent volumetric flow rate an SI-based thermal or moisture step
    can consume:

        V_dot = (ACH * V) / 3600      [m^3 / s]

    Derivation:
        ACH counts complete room-air replacements per HOUR. Multiplying
        by the room volume V [m^3] gives the total volume of air passing
        through in one hour: [1/h] * [m^3] = [m^3/h]. The heat-loss and
        moisture-balance expressions this module will produce are
        integrated with time in SECONDS (energy in joules and power in
        watts are both defined per second in SI), so the m^3/h figure has
        to be divided by SECONDS_PER_HOUR = 3600 to reach m^3/s. The
        3600 is not a fudge factor - it is the unit conversion between
        hours (the unit ACH is quoted in) and seconds (the unit
        downstream thermal maths uses).

    Args:
        room_volume_m3: room volume in cubic metres. Must be finite and
            strictly positive.
        ach: air-change rate in hours^-1. Must be finite and
            non-negative; ACH = 0 is allowed (the physical limit case of
            a sealed room, and the returned flow rate is exactly zero).

    Returns:
        Volumetric airflow rate in cubic metres per second.

    Raises:
        ValueError: if ``room_volume_m3`` is not finite or is not
            strictly positive, or if ``ach`` is not finite or is negative.
    """
    if not isfinite(room_volume_m3):
        raise ValueError(f"room_volume_m3 must be finite, got {room_volume_m3!r}")
    if room_volume_m3 <= 0.0:
        raise ValueError(
            f"room_volume_m3 must be strictly positive, got {room_volume_m3}"
        )
    if not isfinite(ach):
        raise ValueError(f"ach must be finite, got {ach!r}")
    if ach < 0.0:
        raise ValueError(f"ach must be non-negative, got {ach}")
    return ach * room_volume_m3 / SECONDS_PER_HOUR


def ventilation_heat_loss_coefficient(
    room_volume_m3: float,
    ach: float,
    air_density_kg_m3: float = AIR_DENSITY_KG_PER_M3,
    air_specific_heat_j_kg_k: float = AIR_SPECIFIC_HEAT_J_PER_KG_K,
) -> float:
    """Ventilation heat-loss coefficient in watts per kelvin of indoor-outdoor gap.

    Physical form:

        H_vent = rho * c_p * V_dot                             [W / K]

    where ``V_dot`` is the volumetric airflow rate ``(ACH * V) / 3600``
    in m^3/s. Reuses ``airflow_rate_from_ach`` so the units chain
    (ACH per hour -> m^3/s) lives in exactly one place.

    Interpretation:
        H_vent is a RATE per unit temperature difference. Combined with an
        indoor-outdoor temperature gap dT [K] it gives the instantaneous
        ventilation heat-loss POWER in watts:

            Q_dot [W] = H_vent [W/K] * dT [K]

        Multiplying by an event duration t [s] gives the total ventilation
        heat lost as energy in joules:

            Q [J] = H_vent [W/K] * dT [K] * t [s]

        This function does NOT integrate; it returns the coefficient
        only. Callers combine it with dT (and eventually a time
        integral, once the temperature-drop step exists) to compute
        power or energy.

    Sanity check against the common rule of thumb:
        Building-services texts often quote H_vent ~= 0.33 * ACH * V or
        0.34 * ACH * V, with the coefficient absorbing rho, c_p, and the
        /3600 hour-to-second conversion. With this module's reference
        constants:
            rho * c_p / 3600 = 1.204 * 1005 / 3600 ~= 0.336
        which sits between the two rounded rules-of-thumb, so both
        agree with this function to within ~2 %. The 0.33 form is not
        the primary implementation - it is a spot-check target. Tests
        below verify agreement to within a few percent.

    Assumptions inherited from the module docstring, plus these two:
        - Constant air density rho. Real rho varies with temperature and
          pressure; using 1.204 kg/m^3 (dry air at 20 degC, 101325 Pa)
          is a residential-conditions reference. For a room at 5 degC
          rho is closer to 1.27 kg/m^3, a ~5 % error the caller absorbs.
          A later slice can promote rho to a function of T if needed.
        - Constant specific heat capacity c_p at constant pressure. Real
          c_p varies by under 1 % across the residential range and with
          humidity, so a constant is a fine approximation here.

    Args:
        room_volume_m3: room volume in cubic metres. Strictly positive.
        ach: air-change rate in hours^-1. Non-negative; ACH = 0 returns
            H_vent = 0 (a closed, sealed room loses no heat via
            ventilation).
        air_density_kg_m3: air density in kg/m^3. Defaults to
            ``AIR_DENSITY_KG_PER_M3`` (dry air at 20 degC). Must be
            finite and non-negative; callers may pass a temperature-
            corrected value once such a helper exists.
        air_specific_heat_j_kg_k: specific heat capacity of air at
            constant pressure in J/(kg.K). Defaults to
            ``AIR_SPECIFIC_HEAT_J_PER_KG_K``. Must be finite and
            non-negative.

    Returns:
        Ventilation heat-loss coefficient in watts per kelvin.

    Raises:
        ValueError: on any invalid argument. Volume / ACH validation is
            delegated to ``airflow_rate_from_ach``; density and specific
            heat are validated here (finite, non-negative).
    """
    if not isfinite(air_density_kg_m3):
        raise ValueError(
            f"air_density_kg_m3 must be finite, got {air_density_kg_m3!r}"
        )
    if air_density_kg_m3 < 0.0:
        raise ValueError(
            f"air_density_kg_m3 must be non-negative, got {air_density_kg_m3}"
        )
    if not isfinite(air_specific_heat_j_kg_k):
        raise ValueError(
            f"air_specific_heat_j_kg_k must be finite, "
            f"got {air_specific_heat_j_kg_k!r}"
        )
    if air_specific_heat_j_kg_k < 0.0:
        raise ValueError(
            f"air_specific_heat_j_kg_k must be non-negative, "
            f"got {air_specific_heat_j_kg_k}"
        )
    volumetric_flow_m3_per_s = airflow_rate_from_ach(room_volume_m3, ach)
    return air_density_kg_m3 * air_specific_heat_j_kg_k * volumetric_flow_m3_per_s


def ventilation_heat_loss_power(
    indoor_temperature_c: float,
    outdoor_temperature_c: float,
    room_volume_m3: float,
    ach: float,
) -> float:
    """Instantaneous sensible-heat power leaving the room through ventilation.

    Physical form:

        Q_dot = H_vent * (T_in - T_out)
              = rho * c_p * V_dot * (T_in - T_out)               [W]

    where ``V_dot`` is the volumetric airflow rate ``ACH * V / 3600``
    in m^3/s. Reuses ``ventilation_heat_loss_coefficient`` so the
    physical formula (rho * c_p * V_dot) lives in exactly one place;
    this function only adds the temperature difference.

    Sign convention (explicit, no clamping):
        * positive ``Q_dot`` -> indoor is WARMER than outdoor, so
          ventilation removes sensible heat from the room ("heat is
          leaving the room");
        * zero -> indoor and outdoor temperatures are equal, no
          temperature-driven ventilation heat exchange;
        * negative ``Q_dot`` -> outdoor is WARMER than indoor, so
          ventilation adds sensible heat to the room.

    The negative branch is preserved deliberately - a warm draught into
    a cooler room is a real physical effect this POC will want to
    reason about (e.g. summer conditions where opening a window at
    midday would heat the interior). Clamping negatives to zero would
    silently hide that case and be misleading to downstream code.

    Temperature units: the temperature DIFFERENCE ``T_in - T_out`` is
    the same in degrees Celsius and kelvin (both scales share a unit
    size), so this function takes both inputs in degrees Celsius while
    the underlying coefficient is quoted in W/K. No conversion is
    required.

    Args:
        indoor_temperature_c: indoor air temperature in degrees Celsius.
            No residential-range clamp here; the caller is expected to
            supply a real room temperature. NaN / infinite values are
            rejected.
        outdoor_temperature_c: outdoor air temperature in degrees
            Celsius. Same finiteness contract as indoor.
        room_volume_m3: room volume in cubic metres. Strictly positive.
            Validation propagates from ``ventilation_heat_loss_coefficient``.
        ach: air-change rate in hours^-1. Non-negative; ACH = 0 returns
            exactly 0 W regardless of the temperature difference (a
            sealed room has no ventilation heat exchange).

    Returns:
        Sensible-heat power in watts, signed per the convention above.

    Raises:
        ValueError: if any temperature is NaN or infinite; volume / ACH
            validation propagates from
            ``ventilation_heat_loss_coefficient``.
    """
    if not isfinite(indoor_temperature_c):
        raise ValueError(
            f"indoor_temperature_c must be finite, got {indoor_temperature_c!r}"
        )
    if not isfinite(outdoor_temperature_c):
        raise ValueError(
            f"outdoor_temperature_c must be finite, got {outdoor_temperature_c!r}"
        )
    h_vent_w_per_k = ventilation_heat_loss_coefficient(room_volume_m3, ach)
    return h_vent_w_per_k * (indoor_temperature_c - outdoor_temperature_c)


def ventilation_energy_loss_constant_temperature(
    indoor_temperature_c: float,
    outdoor_temperature_c: float,
    room_volume_m3: float,
    ach: float,
    duration_minutes: float,
) -> float:
    """First-order energy loss estimate assuming CONSTANT indoor temperature.

    ================================================================
    CONSTANT-INDOOR-TEMPERATURE APPROXIMATION - NOT THE FINAL MODEL.
    ================================================================
    Treats ``ventilation_heat_loss_power(...)`` as if it stayed at its
    initial value for the whole event and multiplies by duration:

        E = P_loss * t                                      [J]

    In reality the room COOLS while the window is open, which shrinks
    ``T_in - T_out`` and therefore shrinks ``P_loss``; the true energy
    lost is smaller than this constant-temperature product. The gap
    grows with event length. Use this function for order-of-magnitude
    ventilation-cost estimates only. Once the dynamic thermal model
    exists, do NOT quietly swap this in as a stand-in - always
    distinguish the constant-T estimate from the dynamic result at the
    call site.

    Sign convention (inherited from ``ventilation_heat_loss_power``):
        * positive kWh -> heat LEFT the room (indoor warmer than outdoor);
        * zero -> no temperature-driven exchange during the event;
        * negative kWh -> outdoor was warmer; ventilation ADDED heat.
    No abs, no clamp.

    Args:
        indoor_temperature_c: indoor air temperature in degrees Celsius.
            Treated as constant across the event by construction.
        outdoor_temperature_c: outdoor air temperature in degrees Celsius,
            also constant across the event (same assumption as the
            moisture layer).
        room_volume_m3: room volume in cubic metres. Strictly positive.
        ach: air-change rate in hours^-1. Non-negative; ACH = 0 returns
            exactly 0 kWh regardless of duration or temperature gap.
        duration_minutes: length of the event in minutes. Must be finite
            and non-negative; 0 returns exactly 0 kWh.

    Returns:
        Signed ventilation energy in kilowatt-hours.

    Raises:
        ValueError: on any invalid argument. Temperature / volume / ACH
            validation propagates from ``ventilation_heat_loss_power``;
            duration is validated here (finite, non-negative).
    """
    if not isfinite(duration_minutes):
        raise ValueError(
            f"duration_minutes must be finite, got {duration_minutes!r}"
        )
    if duration_minutes < 0.0:
        raise ValueError(
            f"duration_minutes must be non-negative, got {duration_minutes}"
        )
    power_w = ventilation_heat_loss_power(
        indoor_temperature_c=indoor_temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        room_volume_m3=room_volume_m3,
        ach=ach,
    )
    duration_seconds = duration_minutes * 60.0
    energy_joules = power_w * duration_seconds
    return joules_to_kwh(energy_joules)


# --- Effective thermal capacitance -----------------------------------------
#
# Why not just air volume * air density * specific heat?
#
# For a 40 m^3 room the sensible heat capacity of the ROOM AIR ALONE is:
#     C_air = rho * c_p * V = 1.204 * 1005 * 40 ~= 48 400 J/K
# With a ventilation heat-loss coefficient of ~67 W/K (5 ACH, 40 m^3), the
# implied thermal time constant if only air mattered would be:
#     tau_air_only = C_air / H_vent ~= 48 400 / 67 ~= 720 s ~= 12 minutes
# In practice residential rooms exhibit short-term thermal time constants
# more like 30 to 90 minutes: opening a window in a real room cools it
# noticeably slower than a "just the air" model predicts, because heat
# also flows out of walls, floor, ceiling, furniture, soft furnishings,
# and internal partitions - all of which store far more sensible heat
# than the ~48 kJ/K held in the air. Using the air-only capacity would
# therefore predict UNREALISTICALLY RAPID indoor temperature drops
# during ventilation.
#
# The right way to represent this in a lumped POC model is a single
# effective thermal capacitance C_eff [J/K] that bundles the fast-
# responding fabric of the room (surface layers of walls and floor,
# furniture, textiles) together with the air. Its correct value is
# building-specific and should ultimately be identified from measured
# temperature response to a controlled ventilation event; we CANNOT set
# one universally correct default. This POC exposes it as a
# configurable model input on ``ThermalProperties`` and refuses to
# guess.

ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K: float = 500_000.0
"""Illustrative effective thermal capacitance for a small residential room.

Roughly ten times the air-only sensible heat capacity of a 40 m^3 room.
This is a placeholder for demos and worked examples ONLY - it is not
calibrated against any specific building, and any downstream cooling
prediction using this value should be labelled illustrative.

The single-scalar lumped-C model itself is an approximation: real rooms
have several thermal masses with different time constants (fast air +
surface, slow deep mass). One C_eff cannot fit both short and long
event horizons. For this POC, treat it as a knob the caller sets per
room, then eventually identify from measured temperature response.
"""


@dataclass(frozen=True)
class ThermalProperties:
    """Configurable thermal parameters for a residential room.

    Kept separate from ``moisture.Room`` because the moisture layer does
    not yet consume any thermal quantity; the two dataclasses can later
    compose when a joint moisture-thermal step exists, without either
    growing a field it does not use.

    Field:
        effective_thermal_capacity_j_per_k: lumped effective heat
            capacity of the room and its thermally-coupled contents in
            joules per kelvin. Bundles the sensible heat capacity of the
            room air PLUS the fast-responding fraction of walls, floor,
            ceiling, internal partitions, furniture, soft furnishings,
            and any other thermal mass that participates in a short
            (minutes-to-hours) ventilation event. Must be strictly
            positive and finite.

    Why this is a configurable model input, not a computed value:
        See the module comment above. Air-only capacity under-estimates
        the real C by roughly an order of magnitude for a typical
        furnished residential room, which would produce unrealistically
        rapid predicted temperature drops. A calibrated value depends
        on the specific room's construction, contents, and how deep
        into the fabric the event pulls; there is no universally
        correct default.

    Future work (documented, not implemented):
        The correct C_eff for a specific room can be INFERRED from a
        measured temperature response - open a window at a known ACH
        for a known duration, log the indoor temperature time series,
        and fit the exponential decay constant tau against the known
        H_vent (which we already compute). Then C_eff = H_vent * tau.
        This POC does not do that identification step; it accepts
        C_eff as an input.
    """

    effective_thermal_capacity_j_per_k: float

    def __post_init__(self) -> None:
        """Validate the effective thermal capacity."""
        if not isfinite(self.effective_thermal_capacity_j_per_k):
            raise ValueError(
                "effective_thermal_capacity_j_per_k must be finite, "
                f"got {self.effective_thermal_capacity_j_per_k!r}"
            )
        if self.effective_thermal_capacity_j_per_k <= 0.0:
            raise ValueError(
                "effective_thermal_capacity_j_per_k must be strictly "
                f"positive, got {self.effective_thermal_capacity_j_per_k}"
            )


def predict_indoor_temperature(
    initial_indoor_temperature_c: float,
    outdoor_temperature_c: float,
    room_volume_m3: float,
    ach: float,
    effective_thermal_capacity_j_per_k: float,
    duration_minutes: float,
) -> float:
    """Indoor temperature after ``duration_minutes`` of steady ventilation.

    Solves the single-zone energy balance under the assumption that
    ventilation is the ONLY heat-transfer mechanism (no wall conduction,
    no radiation, no active heating, no solar / occupant / equipment
    gains):

        C_eff * dT/dt = H_vent * (T_out - T)

    which rearranges to the same first-order-decay form as the
    moisture ODE:

        dT/dt = (H_vent / C_eff) * (T_out - T)                [K/s]

    The analytic solution, starting from T(0) = T_0, is:

        T(t) = T_out + (T_0 - T_out) * exp(-H_vent / C_eff * t)

    with t in seconds. The thermal time constant is

        tau = C_eff / H_vent                                   [s]

    Interpretation:
        - As t grows, the exponential decays to 0 and T -> T_out: the
          room asymptotically equilibrates with outdoor air.
        - After one tau the indoor-outdoor gap has shrunk by a factor of
          e (~63 % of the way there).
        - ACH = 0 -> H_vent = 0 -> exp(0) = 1 -> T(t) = T_0 for any t.
        - duration = 0 -> exp(0) = 1 -> T(0) = T_0.

    Both degenerate cases fall out of the formula, no branching.

    Symmetric to ``moisture.predict_final_absolute_humidity``: same
    first-order-decay shape, same "asymptote to the outdoor value"
    behaviour, same time-constant argument. The moisture rate is n =
    ACH in hours^-1; the thermal rate is H_vent / C_eff in s^-1. Both
    models still assume ventilation is the only mechanism - a real
    house does more than just ventilate, and joint predictions will
    need at least conduction and radiation terms to be trustworthy.
    Out of scope for this slice.

    Args:
        initial_indoor_temperature_c: room temperature at t = 0, in
            degrees Celsius. Finite; no residential-range clamp.
        outdoor_temperature_c: outdoor temperature, assumed constant
            across the event, in degrees Celsius. Finite.
        room_volume_m3: room volume in cubic metres. Strictly positive.
            Passed through to ``ventilation_heat_loss_coefficient``.
        ach: air-change rate in hours^-1. Non-negative; ACH = 0 returns
            ``initial_indoor_temperature_c`` unchanged.
        effective_thermal_capacity_j_per_k: lumped effective heat
            capacity of the room and its coupled contents in J/K.
            Strictly positive; the model's rate H_vent / C_eff diverges
            as C_eff -> 0. Not a computed quantity - see
            ``ThermalProperties`` for why this must be a caller-set
            input.
        duration_minutes: length of the ventilation event in minutes.
            Non-negative and finite; converted to seconds internally so
            H_vent / C_eff * t stays dimensionless.

    Returns:
        Indoor air temperature at the end of the event, in degrees
        Celsius.

    Raises:
        ValueError: on any invalid argument. Volume and ACH validation
            propagate from ``ventilation_heat_loss_coefficient``; the
            temperatures, effective capacity, and duration are validated
            here.
    """
    if not isfinite(initial_indoor_temperature_c):
        raise ValueError(
            "initial_indoor_temperature_c must be finite, "
            f"got {initial_indoor_temperature_c!r}"
        )
    if not isfinite(outdoor_temperature_c):
        raise ValueError(
            f"outdoor_temperature_c must be finite, got {outdoor_temperature_c!r}"
        )
    if not isfinite(effective_thermal_capacity_j_per_k):
        raise ValueError(
            "effective_thermal_capacity_j_per_k must be finite, "
            f"got {effective_thermal_capacity_j_per_k!r}"
        )
    if effective_thermal_capacity_j_per_k <= 0.0:
        raise ValueError(
            "effective_thermal_capacity_j_per_k must be strictly positive, "
            f"got {effective_thermal_capacity_j_per_k}"
        )
    if not isfinite(duration_minutes):
        raise ValueError(
            f"duration_minutes must be finite, got {duration_minutes!r}"
        )
    if duration_minutes < 0.0:
        raise ValueError(
            f"duration_minutes must be non-negative, got {duration_minutes}"
        )

    h_vent_w_per_k = ventilation_heat_loss_coefficient(room_volume_m3, ach)
    duration_seconds = duration_minutes * 60.0
    decay_factor = exp(
        -h_vent_w_per_k / effective_thermal_capacity_j_per_k * duration_seconds
    )
    return outdoor_temperature_c + (
        initial_indoor_temperature_c - outdoor_temperature_c
    ) * decay_factor


@dataclass(frozen=True)
class ThermalPrediction:
    """Bundled result of a well-mixed ventilation thermal prediction.

    Sign conventions - two distinct conventions live on this class,
    both preserved deliberately (no absolute values taken anywhere):

        1. State-delta convention (used by ``temperature_change_c``):
           final MINUS initial. A cooling event runs T downward, so a
           cooling event produces a NEGATIVE ``temperature_change_c``.

        2. Heat-leaving-the-room convention (used by
           ``initial_heat_loss_power_w``, ``energy_removed_j`` /
           ``_kwh``, and ``energy_removed_constant_temperature_j`` /
           ``_kwh``): positive = heat leaves the room, negative =
           ventilation adds heat. A cooling event moves heat out of
           the room, so a cooling event produces POSITIVE heat-loss
           power and POSITIVE energy removed.

    So a winter ventilation event yields a NEGATIVE
    ``temperature_change_c`` and a POSITIVE ``energy_removed_j``
    simultaneously - that is the intended asymmetry between the two
    conventions, not a bug. They are algebraically consistent:
    ``energy_removed_j = -C_eff * temperature_change_c``.

    Per-field notes:
        - ``initial_heat_loss_power_w`` is the power at t = 0; because
          the driving gap shrinks during the event, the true
          instantaneous power falls in magnitude across the duration.
          Only the initial value is reported here.
        - ``energy_removed_j`` (dynamic) is the physically consistent
          time-integrated energy under the simplified single-source
          model, given by ``C_eff * (initial_temperature_c -
          final_temperature_c)``. Prefer this over the constant-T
          approximation whenever a temperature trajectory is being
          reasoned about.

    Fields:
        initial_temperature_c: room temperature at the start of the
            event, degrees Celsius.
        outdoor_temperature_c: outdoor temperature during the event
            (assumed constant), degrees Celsius.
        final_temperature_c: predicted room temperature at the end of
            the event, degrees Celsius.
        temperature_change_c: signed change from initial to final,
            kelvin (= degrees Celsius for a difference).
        ventilation_heat_loss_coefficient_w_per_k: H_vent for the
            event, watts per kelvin. Echoed on the result so the
            time-constant tau = C_eff / H_vent can be recomputed by a
            caller without re-invoking the physics.
        initial_heat_loss_power_w: signed sensible-heat power at t = 0,
            watts.
        energy_removed_j: DYNAMIC energy removed from the lumped room
            mass during the event, in joules. Computed as
            ``C_eff * (T_0 - T_f)``. Because ventilation is the only
            heat-transfer mechanism in this model, energy conservation
            makes this identically equal to the time-integral
            ``integral_0^t H_vent * (T(tau) - T_out) d(tau)``. This is
            the physically consistent energy number for the dynamic
            model and should be preferred over the constant-T
            approximation whenever a temperature trajectory is being
            reasoned about.
        energy_removed_kwh: same quantity as ``energy_removed_j``,
            reported in kilowatt-hours for human-readable comparisons.
        energy_removed_constant_temperature_j: FIRST-ORDER APPROXIMATION
            of the energy loss, computed as ``P_loss(t=0) * duration``
            with T held fixed at ``initial_temperature_c`` for the whole
            event. Over-estimates the true energy for a cooling event
            because it ignores the shrinking driving gap. Retained for
            comparison against the dynamic result at the call site; do
            not silently substitute one for the other.
        energy_removed_constant_temperature_kwh: same, in kWh.
        duration_minutes: length of the event, minutes.
        ach: air-change rate used for the event, hours^-1.

    Important scope note on the energy fields:
        ``C_eff`` is a LUMPED effective heat capacity that bundles the
        room air with the fast-responding fraction of walls, floor,
        ceiling, furniture, and soft furnishings. Computing
        ``energy_removed_j = C_eff * (T_0 - T_f)`` assumes that this
        entire lumped mass remains thermally coupled to the room air
        for the whole ventilation event - i.e. that whatever fabric is
        represented by ``C_eff`` is close enough to the air that it
        actually cools (or warms) with it, rather than sitting at a
        different temperature. That is a POC approximation. In reality
        different bits of fabric respond on different timescales, and
        their coupling to the well-mixed air varies with distance,
        surface area, and material. A layered thermal model would
        replace this single ``C_eff`` with several coupled masses and
        their exchange coefficients; that is out of scope for this
        module.
    """

    initial_temperature_c: float
    outdoor_temperature_c: float
    final_temperature_c: float
    temperature_change_c: float
    ventilation_heat_loss_coefficient_w_per_k: float
    initial_heat_loss_power_w: float
    energy_removed_j: float
    energy_removed_kwh: float
    energy_removed_constant_temperature_j: float
    energy_removed_constant_temperature_kwh: float
    duration_minutes: float
    ach: float


def predict_thermal_response(
    initial_indoor_temperature_c: float,
    outdoor_temperature_c: float,
    room_volume_m3: float,
    ach: float,
    effective_thermal_capacity_j_per_k: float,
    duration_minutes: float,
) -> ThermalPrediction:
    """Predict the room's thermal response to a ventilation event.

    Convenience wrapper that reuses the existing physics helpers - no
    equations are duplicated here:
        * ``ventilation_heat_loss_coefficient`` for H_vent [W/K];
        * ``ventilation_heat_loss_power`` for the initial (t=0) power;
        * ``predict_indoor_temperature`` for the final temperature via
          the analytic first-order-decay solution;
        * ``ventilation_energy_loss_constant_temperature`` for the
          first-order energy-loss approximation reported alongside the
          dynamic result.

    Dynamic energy removed:
        Because ventilation is the only heat-transfer mechanism in
        this model, energy conservation on the lumped room mass gives:

            energy_removed_j = C_eff * (T_0 - T_f)

        which is identically equal to the time-integral of the
        instantaneous heat-loss power ``H_vent * (T(t) - T_out)`` over
        the event. Positive for a cooling event (T_0 > T_f), negative
        for a warming event, zero for no gradient. See the note on
        ``ThermalPrediction`` about the lumped-C coupling assumption.

    Args:
        initial_indoor_temperature_c: room temperature at t = 0, in
            degrees Celsius.
        outdoor_temperature_c: outdoor temperature during the event, in
            degrees Celsius. Assumed constant.
        room_volume_m3: room volume in cubic metres. Strictly positive.
        ach: air-change rate in hours^-1. Non-negative.
        effective_thermal_capacity_j_per_k: lumped effective heat
            capacity of the room and its coupled contents in J/K.
            Strictly positive.
        duration_minutes: length of the event in minutes. Non-negative.

    Returns:
        A ``ThermalPrediction`` bundling the initial / outdoor / final
        temperatures, the signed change, the ventilation heat-loss
        coefficient, the initial (t=0) heat-loss power, both the
        dynamic and constant-T energy estimates in J and kWh, and the
        ACH / duration inputs echoed for audit.

    Raises:
        ValueError: propagates from the underlying helpers.
    """
    h_vent_w_per_k = ventilation_heat_loss_coefficient(room_volume_m3, ach)
    initial_power_w = ventilation_heat_loss_power(
        indoor_temperature_c=initial_indoor_temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        room_volume_m3=room_volume_m3,
        ach=ach,
    )
    final_temperature_c = predict_indoor_temperature(
        initial_indoor_temperature_c=initial_indoor_temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        room_volume_m3=room_volume_m3,
        ach=ach,
        effective_thermal_capacity_j_per_k=effective_thermal_capacity_j_per_k,
        duration_minutes=duration_minutes,
    )
    # Dynamic energy (integrated over the event): under the simplified
    # model where ventilation is the only heat-transfer path, energy
    # conservation on the lumped mass gives C_eff*(T_0 - T_f) exactly.
    energy_removed_j = effective_thermal_capacity_j_per_k * (
        initial_indoor_temperature_c - final_temperature_c
    )
    energy_removed_kwh = joules_to_kwh(energy_removed_j)
    # Constant-T energy (first-order approximation): P_loss(t=0) * duration,
    # with T held fixed at the initial value for the whole event. Computed
    # in joules directly and converted once to kWh so there is no J -> kWh
    # -> J round-trip through ventilation_energy_loss_constant_temperature
    # (which returns kWh). That standalone helper stays callable and tested
    # for callers who want the kWh number without a ThermalPrediction.
    duration_seconds = duration_minutes * 60.0
    energy_removed_constant_temperature_j = initial_power_w * duration_seconds
    energy_removed_constant_temperature_kwh = joules_to_kwh(
        energy_removed_constant_temperature_j
    )
    return ThermalPrediction(
        initial_temperature_c=initial_indoor_temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        final_temperature_c=final_temperature_c,
        temperature_change_c=final_temperature_c - initial_indoor_temperature_c,
        ventilation_heat_loss_coefficient_w_per_k=h_vent_w_per_k,
        initial_heat_loss_power_w=initial_power_w,
        energy_removed_j=energy_removed_j,
        energy_removed_kwh=energy_removed_kwh,
        energy_removed_constant_temperature_j=energy_removed_constant_temperature_j,
        energy_removed_constant_temperature_kwh=energy_removed_constant_temperature_kwh,
        duration_minutes=duration_minutes,
        ach=ach,
    )
