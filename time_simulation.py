"""Time-domain simulation of a room over multiple hours.

Composes the existing single-event simulator over many small steps
to produce a trajectory of indoor state through a schedule of
moisture-generating events and window-open events. No new physics:
every step calls ``ventilation.simulate_ventilation_event`` for the
ventilation + thermal update, then applies a moisture-source
contribution ``ΔC_source = G · Δt / V`` between simulator calls
(operator-split).

Public shape:
    RoomTrajectory
    VentilationEvent
    simulate_room_period(...)

The trajectory is a fixed-time-step time series of indoor
temperature, indoor absolute humidity, indoor RH (derived from the
step-end T and AH via the existing psychrometric inverse), outdoor
T / AH at each step, plus the current window-open state and the
active moisture-generation rate.

Explicitly NOT in this slice:
    mould modelling (surface RH / integrated risk), heating system,
    weather forecasts, calibration, control decisions. Each of
    those is in the architecture proposal as a separate module.
"""

from dataclasses import dataclass
from math import isfinite
from typing import List, Sequence, Tuple

from moisture import (
    MINUTES_PER_HOUR,
    Room,
    predict_final_absolute_humidity,
)
from moisture_sources import (
    MoistureSourceSchedule,
    moisture_generation_rate_g_per_hour_at,
)
from psychrometrics import (
    AirState,
    relative_humidity_from_absolute_humidity,
)
from thermal import ThermalProperties, predict_indoor_temperature
from weather_forecast import WeatherForecast


@dataclass(frozen=True)
class VentilationEvent:
    """A window-open interval in the schedule.

    Fields:
        start_time_hours: interval start, in hours since the
            schedule's origin. Non-negative.
        end_time_hours: interval end, strictly greater than
            start_time_hours.

    Half-open convention: ``start_time_hours <= t < end_time_hours``.
    The room's ``ach_window_open`` applies inside the interval;
    ``ach_closed`` applies outside every interval.
    """

    start_time_hours: float
    end_time_hours: float

    def __post_init__(self) -> None:
        for name, value in (
            ("start_time_hours", self.start_time_hours),
            ("end_time_hours", self.end_time_hours),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if self.end_time_hours <= self.start_time_hours:
            raise ValueError(
                f"end_time_hours ({self.end_time_hours}) must be strictly "
                f"greater than start_time_hours ({self.start_time_hours})"
            )


@dataclass(frozen=True)
class RoomTrajectory:
    """Fixed-time-step time series of indoor and outdoor state.

    Every field is a tuple of the same length; index i refers to
    the same simulation time as index i of every other field. The
    time-series origin is t = 0.0 hours.

    Fields:
        times_hours: monotone increasing simulation times.
        indoor_temperature_c: indoor air temperature at each time.
        indoor_absolute_humidity_g_m3: indoor AH at each time.
        indoor_relative_humidity_pct: indoor RH at each time,
            derived from the step-end T and AH via the psychrometric
            inverse (matches the "predicted final T + AH" invariant
            the rest of the pipeline respects).
        outdoor_temperature_c: outdoor T at the START of each step
            (piecewise-constant across the step).
        outdoor_absolute_humidity_g_m3: outdoor AH at each step,
            derived from the outdoor state that was applied.
        window_open: True if the step ran with the window open.
        moisture_generation_g_per_hour: source rate at the START of
            each step (piecewise-constant across the step).
    """

    times_hours: Tuple[float, ...]
    indoor_temperature_c: Tuple[float, ...]
    indoor_absolute_humidity_g_m3: Tuple[float, ...]
    indoor_relative_humidity_pct: Tuple[float, ...]
    outdoor_temperature_c: Tuple[float, ...]
    outdoor_absolute_humidity_g_m3: Tuple[float, ...]
    window_open: Tuple[bool, ...]
    moisture_generation_g_per_hour: Tuple[float, ...]


def _is_window_open_at(
    events: Sequence[VentilationEvent], time_hours: float
) -> bool:
    """True if any event's half-open interval contains ``time_hours``."""
    for event in events:
        if event.start_time_hours <= time_hours < event.end_time_hours:
            return True
    return False


def simulate_room_period(
    room: Room,
    thermal_properties: ThermalProperties,
    outdoor: AirState,
    moisture_schedule: MoistureSourceSchedule,
    ventilation_events: Sequence[VentilationEvent],
    duration_hours: float,
    timestep_minutes: float,
) -> RoomTrajectory:
    """Simulate the room over ``duration_hours`` at fixed time steps.

    At every step:
        1. Look up the moisture-generation rate at the step start.
        2. Look up whether a window is open at the step start
           (piecewise-constant across the step).
        3. Advance the moisture ODE by calling
           ``predict_final_absolute_humidity`` with the current ACH.
        4. Add the moisture-source contribution ``ΔC = G·Δt/V``
           by operator-split (source applied AFTER the ventilation
           step, matching the standard operator-split scheme).
        5. Advance the thermal ODE by calling
           ``predict_indoor_temperature`` with the current ACH.
        6. Compute step-end indoor RH from step-end T and AH via
           the psychrometric inverse.
        7. Record everything and step forward.

    Outdoor conditions are constant across the run in this slice; a
    future ``weather.py`` module can supply a time-varying outdoor
    state without changing this function's signature (only how the
    outdoor AH per step is looked up).

    Args:
        room: room description. ``room.ach_window_open`` is used
            during windows; ``room.ach_closed`` is used outside.
        thermal_properties: lumped effective heat capacity used
            by the thermal ODE.
        outdoor: outdoor air state, held constant across the run.
        moisture_schedule: moisture-source schedule (background +
            events). Consulted once per step.
        ventilation_events: window-open intervals. May be empty.
            Overlapping events collapse to a single open interval;
            the exact ACH does not sum across overlapping events.
        duration_hours: total run length, in hours. Must be
            non-negative and finite. Zero produces a single-point
            trajectory containing the initial state.
        timestep_minutes: step size, in minutes. Must be strictly
            positive and finite. Small enough for operator-split
            error to be tolerable (rule of thumb: at least ten
            steps per shortest time constant). 1-5 min is the POC
            default.

    Returns:
        A ``RoomTrajectory`` with one entry per step boundary.

    Raises:
        ValueError: on any invalid input; validation errors from
            the underlying simulator propagate.
    """
    if not isfinite(duration_hours):
        raise ValueError(
            f"duration_hours must be finite, got {duration_hours!r}"
        )
    if duration_hours < 0.0:
        raise ValueError(
            f"duration_hours must be non-negative, got {duration_hours}"
        )
    if not isfinite(timestep_minutes):
        raise ValueError(
            f"timestep_minutes must be finite, got {timestep_minutes!r}"
        )
    if timestep_minutes <= 0.0:
        raise ValueError(
            f"timestep_minutes must be strictly positive, got {timestep_minutes}"
        )

    step_hours = timestep_minutes / MINUTES_PER_HOUR
    outdoor_ah_g_m3 = outdoor.absolute_humidity

    # Number of steps: floor(duration / step). We record the state
    # at t = 0 as the first sample, then after each step.
    if duration_hours == 0.0:
        step_count = 0
    else:
        step_count = int(duration_hours / step_hours + 1e-9)

    # Initial state.
    initial_air_state = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    )
    indoor_ah = initial_air_state.absolute_humidity
    indoor_t = room.indoor_temperature_c

    times: List[float] = [0.0]
    indoor_temperatures: List[float] = [indoor_t]
    indoor_ahs: List[float] = [indoor_ah]
    indoor_rhs: List[float] = [room.indoor_relative_humidity_pct]
    outdoor_ts: List[float] = [outdoor.temperature_c]
    outdoor_ahs: List[float] = [outdoor_ah_g_m3]
    window_states: List[bool] = [
        _is_window_open_at(ventilation_events, 0.0)
    ]
    generation_rates: List[float] = [
        moisture_generation_rate_g_per_hour_at(moisture_schedule, 0.0)
    ]

    for step_index in range(step_count):
        time_now_hours = step_index * step_hours

        # Piecewise-constant lookups at the START of this step.
        window_is_open = _is_window_open_at(
            ventilation_events, time_now_hours
        )
        ach = (
            room.ach_window_open if window_is_open else room.ach_closed
        )
        generation_g_per_hour = moisture_generation_rate_g_per_hour_at(
            moisture_schedule, time_now_hours
        )

        # Ventilation-only moisture update via the existing simulator.
        next_indoor_ah = predict_final_absolute_humidity(
            indoor_ah_g_m3=indoor_ah,
            outdoor_ah_g_m3=outdoor_ah_g_m3,
            ach=ach,
            duration_minutes=timestep_minutes,
        )
        # Operator-split: add the source contribution over the step.
        next_indoor_ah = next_indoor_ah + (
            generation_g_per_hour * step_hours / room.volume_m3
        )

        # Thermal update via the existing simulator.
        next_indoor_t = predict_indoor_temperature(
            initial_indoor_temperature_c=indoor_t,
            outdoor_temperature_c=outdoor.temperature_c,
            room_volume_m3=room.volume_m3,
            ach=ach,
            effective_thermal_capacity_j_per_k=(
                thermal_properties.effective_thermal_capacity_j_per_k
            ),
            duration_minutes=timestep_minutes,
        )

        # Step-end RH from the step-end T and AH via the
        # psychrometric inverse.
        next_indoor_rh = relative_humidity_from_absolute_humidity(
            temperature_c=next_indoor_t,
            absolute_humidity_g_m3=next_indoor_ah,
        )

        # Record.
        step_end_hours = (step_index + 1) * step_hours
        times.append(step_end_hours)
        indoor_temperatures.append(next_indoor_t)
        indoor_ahs.append(next_indoor_ah)
        indoor_rhs.append(next_indoor_rh)
        outdoor_ts.append(outdoor.temperature_c)
        outdoor_ahs.append(outdoor_ah_g_m3)
        window_states.append(window_is_open)
        generation_rates.append(generation_g_per_hour)

        # Advance state.
        indoor_ah = next_indoor_ah
        indoor_t = next_indoor_t

    return RoomTrajectory(
        times_hours=tuple(times),
        indoor_temperature_c=tuple(indoor_temperatures),
        indoor_absolute_humidity_g_m3=tuple(indoor_ahs),
        indoor_relative_humidity_pct=tuple(indoor_rhs),
        outdoor_temperature_c=tuple(outdoor_ts),
        outdoor_absolute_humidity_g_m3=tuple(outdoor_ahs),
        window_open=tuple(window_states),
        moisture_generation_g_per_hour=tuple(generation_rates),
    )


def simulate_room_period_with_forecast(
    room: Room,
    thermal_properties: ThermalProperties,
    forecast: WeatherForecast,
    moisture_schedule: MoistureSourceSchedule,
    ventilation_events: Sequence[VentilationEvent],
    duration_hours: float,
    timestep_minutes: float,
) -> RoomTrajectory:
    """Simulate a room over a horizon with time-varying outdoor conditions.

    Structurally identical to ``simulate_room_period`` except the
    outdoor state is looked up on the ``WeatherForecast`` at the
    START of every step (piecewise-constant across the step),
    matching the same convention the schedule and window lookups
    already use. No new physics: outdoor T and outdoor AH per step
    feed into the same simulator calls that the constant-outdoor
    routine drives.

    Args:
        room, thermal_properties, moisture_schedule,
        ventilation_events, duration_hours, timestep_minutes: same as
            ``simulate_room_period``.
        forecast: outdoor forecast. Query times outside its
            [first_ts, last_ts] range hold the nearest endpoint (see
            ``weather_forecast`` module docstring).

    Returns:
        A ``RoomTrajectory`` with one entry per step boundary.

    Raises:
        ValueError: on any invalid input; validation errors from
            the underlying simulator propagate.
    """
    if not isfinite(duration_hours):
        raise ValueError(
            f"duration_hours must be finite, got {duration_hours!r}"
        )
    if duration_hours < 0.0:
        raise ValueError(
            f"duration_hours must be non-negative, got {duration_hours}"
        )
    if not isfinite(timestep_minutes):
        raise ValueError(
            f"timestep_minutes must be finite, got {timestep_minutes!r}"
        )
    if timestep_minutes <= 0.0:
        raise ValueError(
            f"timestep_minutes must be strictly positive, got {timestep_minutes}"
        )

    step_hours = timestep_minutes / MINUTES_PER_HOUR
    if duration_hours == 0.0:
        step_count = 0
    else:
        step_count = int(duration_hours / step_hours + 1e-9)

    initial_air_state = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    )
    indoor_ah = initial_air_state.absolute_humidity
    indoor_t = room.indoor_temperature_c

    outdoor_at_zero = forecast.sample_at(0.0)
    times: List[float] = [0.0]
    indoor_temperatures: List[float] = [indoor_t]
    indoor_ahs: List[float] = [indoor_ah]
    indoor_rhs: List[float] = [room.indoor_relative_humidity_pct]
    outdoor_ts: List[float] = [outdoor_at_zero.temperature_c]
    outdoor_ahs: List[float] = [outdoor_at_zero.absolute_humidity]
    window_states: List[bool] = [_is_window_open_at(ventilation_events, 0.0)]
    generation_rates: List[float] = [
        moisture_generation_rate_g_per_hour_at(moisture_schedule, 0.0)
    ]

    for step_index in range(step_count):
        time_now_hours = step_index * step_hours

        window_is_open = _is_window_open_at(
            ventilation_events, time_now_hours
        )
        ach = (
            room.ach_window_open if window_is_open else room.ach_closed
        )
        generation_g_per_hour = moisture_generation_rate_g_per_hour_at(
            moisture_schedule, time_now_hours
        )
        outdoor_now = forecast.sample_at(time_now_hours)
        outdoor_ah_now = outdoor_now.absolute_humidity

        next_indoor_ah = predict_final_absolute_humidity(
            indoor_ah_g_m3=indoor_ah,
            outdoor_ah_g_m3=outdoor_ah_now,
            ach=ach,
            duration_minutes=timestep_minutes,
        )
        next_indoor_ah = next_indoor_ah + (
            generation_g_per_hour * step_hours / room.volume_m3
        )

        next_indoor_t = predict_indoor_temperature(
            initial_indoor_temperature_c=indoor_t,
            outdoor_temperature_c=outdoor_now.temperature_c,
            room_volume_m3=room.volume_m3,
            ach=ach,
            effective_thermal_capacity_j_per_k=(
                thermal_properties.effective_thermal_capacity_j_per_k
            ),
            duration_minutes=timestep_minutes,
        )

        next_indoor_rh = relative_humidity_from_absolute_humidity(
            temperature_c=next_indoor_t,
            absolute_humidity_g_m3=next_indoor_ah,
        )

        step_end_hours = (step_index + 1) * step_hours
        times.append(step_end_hours)
        indoor_temperatures.append(next_indoor_t)
        indoor_ahs.append(next_indoor_ah)
        indoor_rhs.append(next_indoor_rh)
        outdoor_ts.append(outdoor_now.temperature_c)
        outdoor_ahs.append(outdoor_ah_now)
        window_states.append(window_is_open)
        generation_rates.append(generation_g_per_hour)

        indoor_ah = next_indoor_ah
        indoor_t = next_indoor_t

    return RoomTrajectory(
        times_hours=tuple(times),
        indoor_temperature_c=tuple(indoor_temperatures),
        indoor_absolute_humidity_g_m3=tuple(indoor_ahs),
        indoor_relative_humidity_pct=tuple(indoor_rhs),
        outdoor_temperature_c=tuple(outdoor_ts),
        outdoor_absolute_humidity_g_m3=tuple(outdoor_ahs),
        window_open=tuple(window_states),
        moisture_generation_g_per_hour=tuple(generation_rates),
    )
