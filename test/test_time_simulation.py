"""Tests for the time-domain simulation module.

Covers:
    - Trajectory shape (all arrays same length, monotone times).
    - Regression against the single-event simulator: source-free,
      always-closed run reproduces the existing physics at every
      step to floating-point precision (composition invariant).
    - Ventilation events change the ACH used during their interval.
    - Moisture source raises indoor AH; balance with ventilation.
    - Zero-duration produces a single-sample trajectory.
    - Validation errors bubble up.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room, predict_final_absolute_humidity
from moisture_sources import (
    MoistureSourceEvent,
    MoistureSourceSchedule,
)
from psychrometrics import AirState
from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    ThermalProperties,
    predict_indoor_temperature,
)
from time_simulation import (
    RoomTrajectory,
    VentilationEvent,
    simulate_room_period,
)


def _default_room() -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )


def _default_thermal_properties() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _default_outdoor() -> AirState:
    return AirState(temperature_c=5.0, relative_humidity_percent=85.0)


# --- Trajectory shape ------------------------------------------------------


def test_trajectory_arrays_all_same_length() -> None:
    """Every field on RoomTrajectory has the same length."""
    trajectory = simulate_room_period(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=2.0,
        timestep_minutes=5.0,
    )
    n = len(trajectory.times_hours)
    for field in (
        trajectory.indoor_temperature_c,
        trajectory.indoor_absolute_humidity_g_m3,
        trajectory.indoor_relative_humidity_pct,
        trajectory.outdoor_temperature_c,
        trajectory.outdoor_absolute_humidity_g_m3,
        trajectory.window_open,
        trajectory.moisture_generation_g_per_hour,
    ):
        assert len(field) == n


def test_trajectory_times_are_monotone_increasing() -> None:
    """Times run forward from the origin at the caller's timestep."""
    trajectory = simulate_room_period(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=1.0,
        timestep_minutes=5.0,
    )
    for earlier, later in zip(
        trajectory.times_hours, trajectory.times_hours[1:]
    ):
        assert later > earlier
    # Number of steps + 1 samples.
    assert len(trajectory.times_hours) == 13  # 12 five-min steps + initial


def test_trajectory_first_sample_matches_initial_room_state() -> None:
    """t = 0 sample is the room's initial state, not a stepped result."""
    trajectory = simulate_room_period(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=0.5,
        timestep_minutes=5.0,
    )
    assert trajectory.times_hours[0] == 0.0
    assert trajectory.indoor_temperature_c[0] == 20.0
    assert trajectory.indoor_relative_humidity_pct[0] == 70.0
    initial_ah = AirState(20.0, 70.0).absolute_humidity
    assert trajectory.indoor_absolute_humidity_g_m3[0] == initial_ah


def test_zero_duration_returns_single_sample() -> None:
    """duration_hours = 0 -> just the initial state, one sample."""
    trajectory = simulate_room_period(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=0.0,
        timestep_minutes=5.0,
    )
    assert len(trajectory.times_hours) == 1
    assert trajectory.times_hours == (0.0,)


# --- Regression: single-event simulator as a special case ------------------


def test_source_free_closed_room_reproduces_single_event_ah() -> None:
    """Source-free, always-closed run matches predict_final_absolute_humidity
    at every step to floating-point precision.

    This is the load-bearing composition-invariant test: the time
    simulation must call the existing simulator faithfully. Any
    drift here means the operator-split composition is wrong.
    """
    room = _default_room()
    thermal_props = _default_thermal_properties()
    outdoor = _default_outdoor()
    duration_hours = 1.0
    step_minutes = 5.0

    trajectory = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=duration_hours,
        timestep_minutes=step_minutes,
    )

    # Reproduce the AH at each step by calling the single-event
    # simulator directly with the CUMULATIVE duration and closed
    # ACH.
    initial_ah = AirState(20.0, 70.0).absolute_humidity
    for i, t_hours in enumerate(trajectory.times_hours):
        elapsed_minutes = t_hours * 60.0
        expected_ah = predict_final_absolute_humidity(
            indoor_ah_g_m3=initial_ah,
            outdoor_ah_g_m3=outdoor.absolute_humidity,
            ach=room.ach_closed,
            duration_minutes=elapsed_minutes,
        )
        # Step-wise composition using the analytic exponential is
        # mathematically identical to a single call at the total
        # duration (exp is a monoid); expect FP-precision equality.
        assert trajectory.indoor_absolute_humidity_g_m3[i] == pytest.approx(
            expected_ah, rel=1e-12, abs=1e-12
        )


def test_source_free_closed_room_reproduces_single_event_temperature() -> None:
    """Twin test for the thermal step: source-free, always-closed
    trajectory matches predict_indoor_temperature at every step."""
    room = _default_room()
    thermal_props = _default_thermal_properties()
    outdoor = _default_outdoor()
    trajectory = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=1.0,
        timestep_minutes=5.0,
    )
    for i, t_hours in enumerate(trajectory.times_hours):
        elapsed_minutes = t_hours * 60.0
        expected_t = predict_indoor_temperature(
            initial_indoor_temperature_c=room.indoor_temperature_c,
            outdoor_temperature_c=outdoor.temperature_c,
            room_volume_m3=room.volume_m3,
            ach=room.ach_closed,
            effective_thermal_capacity_j_per_k=(
                thermal_props.effective_thermal_capacity_j_per_k
            ),
            duration_minutes=elapsed_minutes,
        )
        assert trajectory.indoor_temperature_c[i] == pytest.approx(
            expected_t, rel=1e-12, abs=1e-12
        )


# --- Ventilation events ----------------------------------------------------


def test_ventilation_event_changes_ach_during_its_interval() -> None:
    """Inside a ventilation event, the trajectory drops AH faster
    than an equivalent closed run.

    Runs two 1-hour simulations of an identical room. In the first,
    the window is closed the whole hour. In the second, the window
    is open for the middle 15 minutes. The final AH must be lower
    in the open-window case.
    """
    room = _default_room()
    thermal_props = _default_thermal_properties()
    outdoor = _default_outdoor()

    closed = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        duration_hours=1.0,
        timestep_minutes=1.0,
    )
    with_window = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(
            VentilationEvent(
                start_time_hours=0.25, end_time_hours=0.5
            ),
        ),
        duration_hours=1.0,
        timestep_minutes=1.0,
    )
    # Same initial state, so index 0 matches.
    assert with_window.indoor_absolute_humidity_g_m3[0] == pytest.approx(
        closed.indoor_absolute_humidity_g_m3[0], rel=1e-12
    )
    # But by the end the open-window case is drier (lower AH).
    assert (
        with_window.indoor_absolute_humidity_g_m3[-1]
        < closed.indoor_absolute_humidity_g_m3[-1]
    )


def test_ventilation_events_control_window_open_flag() -> None:
    """The window_open field flips inside intervals and back outside."""
    trajectory = simulate_room_period(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(
            VentilationEvent(
                start_time_hours=0.5, end_time_hours=1.0
            ),
        ),
        duration_hours=1.5,
        timestep_minutes=5.0,
    )
    # Sample the flag at three known times. Times are sample times,
    # not step-start times, so we probe by index.
    time_to_index = {
        round(t, 6): i for i, t in enumerate(trajectory.times_hours)
    }
    # Before the event: closed.
    assert trajectory.window_open[time_to_index[0.25]] is False
    # Inside the event: open.
    assert trajectory.window_open[time_to_index[0.75]] is True
    # After the event: closed again.
    assert trajectory.window_open[time_to_index[1.25]] is False


# --- Moisture sources ------------------------------------------------------


def test_moisture_source_raises_indoor_ah() -> None:
    """A background moisture source with a closed room raises indoor AH.

    Constant background rate + no ventilation event + no drying
    gradient (indoor and outdoor AH equal at t = 0). The source is
    the only mechanism that can change indoor AH, so it must rise
    over time.
    """
    # Construct a scenario where indoor AH equals outdoor AH at
    # t = 0 so the ventilation drift alone doesn't change AH.
    # Indoor 20 C at 50 %RH -> ~8.65 g/m^3; back-solve outdoor RH
    # at 20 C for the same AH.
    indoor = AirState(20.0, 50.0).absolute_humidity
    outdoor = AirState(20.0, 50.0)  # same conditions
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=50.0,
        ach_closed=0.0,
        ach_window_open=5.0,
    )
    trajectory = simulate_room_period(
        room=room,
        thermal_properties=_default_thermal_properties(),
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(
            constant_background_rate_g_per_hour=100.0
        ),
        ventilation_events=(),
        duration_hours=1.0,
        timestep_minutes=5.0,
    )
    # Indoor AH must rise monotonically (source > 0, ACH = 0).
    values = trajectory.indoor_absolute_humidity_g_m3
    assert values[0] == pytest.approx(indoor, rel=1e-12)
    for earlier, later in zip(values, values[1:]):
        assert later > earlier
    # And by the end the total rise is ~ G*t/V = 100*1/40 = 2.5 g/m^3.
    total_rise = values[-1] - values[0]
    assert total_rise == pytest.approx(2.5, rel=1e-6)


def test_ventilation_can_balance_moisture_generation() -> None:
    """With ventilation and a source, indoor AH tends to a shifted equilibrium.

    ACH_open = 5, V = 40, G = 100 g/h, outdoor AH = 5 g/m^3.
    C_eq = C_out + G/(n*V) = 5 + 100/(5*40) = 5.5 g/m^3.
    Starting from indoor AH = 5.5, indoor AH stays at 5.5 with
    window continuously open and steady source.
    """
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)
    # Outdoor AH here is ~5.77 not 5.0; use it directly.
    outdoor_ah = outdoor.absolute_humidity
    # Set indoor RH to give the equilibrium AH.
    # C_eq = outdoor_ah + G/(n*V). Aim for continuous open window
    # over the whole run.
    ach = 5.0
    volume = 40.0
    G = 100.0
    equilibrium_ah = outdoor_ah + G / (ach * volume)
    # Compute the indoor RH that gives this AH at 20 C, using the
    # psychrometric inverse via AirState.
    from psychrometrics import relative_humidity_from_absolute_humidity
    indoor_rh = relative_humidity_from_absolute_humidity(
        temperature_c=20.0,
        absolute_humidity_g_m3=equilibrium_ah,
    )
    room = Room(
        volume_m3=volume,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=indoor_rh,
        ach_closed=0.5,
        ach_window_open=ach,
    )
    trajectory = simulate_room_period(
        room=room,
        thermal_properties=_default_thermal_properties(),
        outdoor=outdoor,
        moisture_schedule=MoistureSourceSchedule(
            constant_background_rate_g_per_hour=G
        ),
        # Window open the entire run.
        ventilation_events=(
            VentilationEvent(start_time_hours=0.0, end_time_hours=2.0),
        ),
        duration_hours=1.0,
        timestep_minutes=5.0,
    )
    # Indoor AH stays close to the equilibrium at every step. Small
    # deviation is OK due to operator-split error (source applied
    # after ventilation step introduces small drift).
    for ah in trajectory.indoor_absolute_humidity_g_m3:
        assert ah == pytest.approx(equilibrium_ah, rel=0.02)


# --- Validation -----------------------------------------------------------


def test_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_hours"):
        simulate_room_period(
            room=_default_room(),
            thermal_properties=_default_thermal_properties(),
            outdoor=_default_outdoor(),
            moisture_schedule=MoistureSourceSchedule(),
            ventilation_events=(),
            duration_hours=-1.0,
            timestep_minutes=5.0,
        )


def test_rejects_zero_or_negative_timestep() -> None:
    with pytest.raises(ValueError, match="timestep_minutes"):
        simulate_room_period(
            room=_default_room(),
            thermal_properties=_default_thermal_properties(),
            outdoor=_default_outdoor(),
            moisture_schedule=MoistureSourceSchedule(),
            ventilation_events=(),
            duration_hours=1.0,
            timestep_minutes=0.0,
        )


def test_rejects_non_finite_arguments() -> None:
    """NaN / inf on duration or timestep is rejected."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duration_hours"):
            simulate_room_period(
                room=_default_room(),
                thermal_properties=_default_thermal_properties(),
                outdoor=_default_outdoor(),
                moisture_schedule=MoistureSourceSchedule(),
                ventilation_events=(),
                duration_hours=bad,
                timestep_minutes=5.0,
            )
        with pytest.raises(ValueError, match="timestep_minutes"):
            simulate_room_period(
                room=_default_room(),
                thermal_properties=_default_thermal_properties(),
                outdoor=_default_outdoor(),
                moisture_schedule=MoistureSourceSchedule(),
                ventilation_events=(),
                duration_hours=1.0,
                timestep_minutes=bad,
            )


def test_ventilation_event_rejects_zero_width_interval() -> None:
    """VentilationEvent validation catches bad intervals."""
    with pytest.raises(ValueError, match="end_time_hours"):
        VentilationEvent(start_time_hours=0.5, end_time_hours=0.5)
    with pytest.raises(ValueError, match="end_time_hours"):
        VentilationEvent(start_time_hours=1.0, end_time_hours=0.5)
