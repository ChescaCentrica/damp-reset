"""Combined ventilation prediction: moisture + thermal + inverse psychrometrics.

Single entry point that composes the three existing layers:

    1. ``moisture.predict_room_moisture`` -> final indoor absolute humidity
    2. ``thermal.predict_thermal_response`` -> final indoor temperature
    3. ``psychrometrics.relative_humidity_from_absolute_humidity`` uses
       the PREDICTED final temperature (not the initial one) together
       with the predicted final AH to recover a final relative humidity.

No physics is implemented here. Every equation - the moisture ODE, the
thermal ODE, the ideal-gas absolute-humidity inversion, and the Magnus
saturation curve - already lives in exactly one place in the codebase,
and this module only assembles their results.

The point of step 3 is explicit: a valid final RH must be derived from
the final temperature the thermal model predicts, NOT from the room's
initial temperature. Opening a window changes both moisture content and
indoor temperature; RH depends on both, and reporting an RH computed
against the initial temperature would be misleadingly high in a cooling
scenario (the same water content in cooler air reads more humid). See
the moisture module docstring for the design invariant.
"""

from dataclasses import dataclass

from moisture import MoisturePrediction, Room, predict_room_moisture
from psychrometrics import AirState, relative_humidity_from_absolute_humidity
from thermal import ThermalPrediction, ThermalProperties, predict_thermal_response


@dataclass(frozen=True)
class VentilationPrediction:
    """Bundled moisture + thermal result of a ventilation event.

    Composes a ``MoisturePrediction`` and a ``ThermalPrediction`` for
    the same window-open event and, crucially, provides a final RH
    computed from the PREDICTED final temperature and the PREDICTED
    final absolute humidity - not from the room's original temperature.

    Fields:
        moisture: full ``MoisturePrediction`` bundle - see
            ``moisture.MoisturePrediction`` for the sign conventions
            on absolute-humidity change, reduction, percentage, and
            water mass removed.
        thermal: full ``ThermalPrediction`` bundle - see
            ``thermal.ThermalPrediction`` for the sign conventions on
            temperature change, initial heat-loss power, and the two
            energy estimates (dynamic and constant-T).
        final_relative_humidity_pct: relative humidity at the end of
            the event, in percent (0-100 for physically consistent
            inputs; may exceed 100 if the predicted final AH is above
            saturation at the predicted final temperature, indicating
            a supersaturated state - see
            ``relative_humidity_from_absolute_humidity`` for the
            documented behaviour).
        duration_minutes: length of the event, echoed for audit.
        window_open: which ACH mode was used, echoed for audit.
    """

    moisture: MoisturePrediction
    thermal: ThermalPrediction
    final_relative_humidity_pct: float
    duration_minutes: float
    window_open: bool


def predict_ventilation(
    room: Room,
    thermal_properties: ThermalProperties,
    outdoor: AirState,
    duration_minutes: float,
    window_open: bool = True,
) -> VentilationPrediction:
    """Predict indoor state after a ventilation event, in both moisture and thermal terms.

    Convenience wrapper that composes:
        * ``predict_room_moisture`` for the moisture trajectory;
        * ``predict_thermal_response`` for the temperature trajectory;
        * ``relative_humidity_from_absolute_humidity`` to derive the
          final RH from the PREDICTED final temperature and the
          PREDICTED final absolute humidity.

    Args:
        room: ``moisture.Room`` describing the room dimensions,
            initial indoor state, and both ACH values.
        thermal_properties: ``thermal.ThermalProperties`` supplying
            the lumped effective heat capacity of the room and its
            thermally-coupled fabric.
        outdoor: outdoor air state driving the ventilation exchange.
            Assumed constant across the event.
        duration_minutes: length of the event, in minutes.
        window_open: if True (default), the moisture and thermal
            models both use ``room.ach_window_open``; if False, both
            use ``room.ach_closed``. Using the same ACH for both is
            correct: the physical air-exchange rate is a single number
            per window state.

    Returns:
        A ``VentilationPrediction`` bundling both physics predictions
        plus the derived final RH.

    Raises:
        ValueError: propagates from the underlying helpers.
    """
    ach = room.ach_window_open if window_open else room.ach_closed

    moisture_result = predict_room_moisture(
        room=room,
        outdoor=outdoor,
        duration_minutes=duration_minutes,
        window_open=window_open,
    )
    thermal_result = predict_thermal_response(
        initial_indoor_temperature_c=room.indoor_temperature_c,
        outdoor_temperature_c=outdoor.temperature_c,
        room_volume_m3=room.volume_m3,
        ach=ach,
        effective_thermal_capacity_j_per_k=(
            thermal_properties.effective_thermal_capacity_j_per_k
        ),
        duration_minutes=duration_minutes,
    )
    final_relative_humidity_pct = relative_humidity_from_absolute_humidity(
        temperature_c=thermal_result.final_temperature_c,
        absolute_humidity_g_m3=moisture_result.final_absolute_humidity_g_m3,
    )
    return VentilationPrediction(
        moisture=moisture_result,
        thermal=thermal_result,
        final_relative_humidity_pct=final_relative_humidity_pct,
        duration_minutes=duration_minutes,
        window_open=window_open,
    )


@dataclass(frozen=True)
class VentilationSimulationResult:
    """Flat scalar summary of one ventilation event.

    Ten fields, one per quantity the top-level POC simulation is
    expected to report. Each is derived from the corresponding field on
    a ``VentilationPrediction`` via the existing physics helpers - no
    new maths is done in this bundle.

    Sign conventions inherit from the underlying layers:
        - ``water_removed_g``: positive = water left the room air.
        - ``temperature_drop_c``: positive = room cooled (initial - final).
        - ``ventilation_energy_removed_kwh``: positive = heat left the
          room (from ``ThermalPrediction.energy_removed_kwh``, the
          dynamic estimate C_eff * (T_0 - T_f)).

    RH values are quoted at the correct temperature: initial RH is
    stated at the initial indoor temperature (the caller's input),
    final RH is derived from the PREDICTED final temperature and
    PREDICTED final absolute humidity.
    """

    initial_absolute_humidity_g_m3: float
    final_absolute_humidity_g_m3: float
    water_removed_g: float
    initial_relative_humidity_pct: float
    final_relative_humidity_pct: float
    initial_temperature_c: float
    final_temperature_c: float
    temperature_drop_c: float
    ventilation_heat_loss_coefficient_w_per_k: float
    ventilation_energy_removed_kwh: float


def simulate_ventilation_event(
    room_volume_m3: float,
    initial_indoor_temperature_c: float,
    initial_indoor_relative_humidity_pct: float,
    outdoor_temperature_c: float,
    outdoor_relative_humidity_pct: float,
    ach: float,
    effective_thermal_capacity_j_per_k: float,
    duration_minutes: float,
) -> VentilationSimulationResult:
    """Top-level POC simulation of a single ventilation event.

    Thin facade over ``predict_ventilation``: accepts raw scalars,
    builds the internal ``Room`` / ``AirState`` / ``ThermalProperties``
    value objects, runs the combined moisture / thermal / RH
    prediction, and returns a flat ``VentilationSimulationResult``
    with the ten quantities the POC-level summary reports.

    All physics still lives in the underlying single-source helpers
    (moisture / thermal / psychrometric modules). This function does
    not introduce any new equations.

    Args:
        room_volume_m3: room volume in cubic metres. Strictly positive.
        initial_indoor_temperature_c: room air temperature at the
            start of the event, degrees Celsius.
        initial_indoor_relative_humidity_pct: room air relative
            humidity at the start of the event, 0-100 %.
        outdoor_temperature_c: outdoor air temperature during the
            event, degrees Celsius. Assumed constant.
        outdoor_relative_humidity_pct: outdoor air relative humidity
            during the event, 0-100 %. Assumed constant.
        ach: air-change rate for the event, in hours^-1. Non-negative.
            One ACH value; the facade treats the event as "the window
            is at this exchange rate for this duration". If a caller
            needs both a closed and an open state, they can call twice.
        effective_thermal_capacity_j_per_k: lumped effective heat
            capacity of the room and its coupled contents in J/K.
            Strictly positive. Not a computed quantity - see
            ``ThermalProperties`` for why this must be a caller-set
            input.
        duration_minutes: length of the event in minutes. Non-negative.

    Returns:
        A ``VentilationSimulationResult`` with the ten flat scalar
        fields the top-level POC reports.

    Raises:
        ValueError: propagates from the underlying helpers.
    """
    # Use `ach` as both closed and open on the internal Room, then run
    # window_open=True so the moisture and thermal layers both see the
    # requested ACH. The composed predict_ventilation stays unchanged;
    # the facade just plugs into it.
    room = Room(
        volume_m3=room_volume_m3,
        indoor_temperature_c=initial_indoor_temperature_c,
        indoor_relative_humidity_pct=initial_indoor_relative_humidity_pct,
        ach_closed=ach,
        ach_window_open=ach,
    )
    outdoor = AirState(
        temperature_c=outdoor_temperature_c,
        relative_humidity_percent=outdoor_relative_humidity_pct,
    )
    thermal_properties = ThermalProperties(
        effective_thermal_capacity_j_per_k=effective_thermal_capacity_j_per_k
    )
    prediction = predict_ventilation(
        room=room,
        thermal_properties=thermal_properties,
        outdoor=outdoor,
        duration_minutes=duration_minutes,
        window_open=True,
    )
    return VentilationSimulationResult(
        initial_absolute_humidity_g_m3=(
            prediction.moisture.initial_absolute_humidity_g_m3
        ),
        final_absolute_humidity_g_m3=(
            prediction.moisture.final_absolute_humidity_g_m3
        ),
        water_removed_g=prediction.moisture.water_removed_g,
        initial_relative_humidity_pct=initial_indoor_relative_humidity_pct,
        final_relative_humidity_pct=prediction.final_relative_humidity_pct,
        initial_temperature_c=prediction.thermal.initial_temperature_c,
        final_temperature_c=prediction.thermal.final_temperature_c,
        temperature_drop_c=(
            prediction.thermal.initial_temperature_c
            - prediction.thermal.final_temperature_c
        ),
        ventilation_heat_loss_coefficient_w_per_k=(
            prediction.thermal.ventilation_heat_loss_coefficient_w_per_k
        ),
        ventilation_energy_removed_kwh=prediction.thermal.energy_removed_kwh,
    )
