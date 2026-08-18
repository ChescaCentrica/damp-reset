"""Regression tests: the optimiser's result equals an independent re-simulation.

After the "heating-aware optimiser" refactor,
``optimise_scheduled_action_under_risk_limit`` evaluates every
candidate action using ``simulate_room_period_with_heating`` and
``evaluate_moisture_risk`` on the resulting trajectory. This test
module proves that the result the OPTIMISER returned for the
winning candidate is EXACTLY what a caller reproduces by:

    1. calling ``simulate_room_period_with_heating`` on the
       selected action, then
    2. calling ``evaluate_moisture_risk`` on the resulting
       trajectory.

The historical bug this catches: the previous optimiser planned
against a no-heating trajectory but the demo re-simulated with the
heating model, so the two risk scores disagreed. If someone ever
adds a stand-in in the optimiser again, these tests fail.

Every check uses ``rel=1e-9`` - exact float agreement.
"""

import sys
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from heating import NoHeating, ThermostaticHeating
from moisture import Room
from moisture_sources import MoistureSourceEvent, MoistureSourceSchedule
from mould_risk import RiskConfig, evaluate_moisture_risk
from psychrometrics import AirState
from surface_risk import SurfaceDescriptor
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from time_simulation import (
    VentilationEvent,
    simulate_room_period_with_heating,
)
from weather_forecast import ForecastPoint, WeatherForecast

from optimiser import (
    ScheduledAction,
    VentilationConstraints,
    optimise_scheduled_action_under_risk_limit,
)


def _room(rh: float = 71.0) -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.2,
        indoor_relative_humidity_pct=rh,
        ach_closed=0.3,
        ach_window_open=5.0,
    )


def _thermal() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _surface() -> SurfaceDescriptor:
    return SurfaceDescriptor(
        label="cold external wall corner (kitchen)",
        surface_temperature_factor=0.72,
    )


def _schedule(background: float = 80.0) -> MoistureSourceSchedule:
    return MoistureSourceSchedule(
        constant_background_rate_g_per_hour=background,
    )


def _forecast() -> WeatherForecast:
    return WeatherForecast(
        points=(
            ForecastPoint(0.0, -1.0, 70.0),
            ForecastPoint(0.5, 4.0, 65.0),
            ForecastPoint(2.0, 8.0, 60.0),
            ForecastPoint(4.0, 10.0, 60.0),
            ForecastPoint(6.0, 12.0, 60.0),
        )
    )


def _candidates() -> tuple:
    actions = [ScheduledAction(0.0, 0.0)]
    starts_hours = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
    durations_min = (5.0, 10.0, 15.0, 20.0, 30.0)
    for start in starts_hours:
        for duration in durations_min:
            if start + duration / 60.0 <= 6.0:
                actions.append(ScheduledAction(start, duration))
    return tuple(actions)


def _resimulate(action: ScheduledAction, heating_model, room=None):
    """Independently re-run the winning action with the same heating model."""
    if action.is_do_nothing:
        events = ()
    else:
        events = (
            VentilationEvent(
                start_time_hours=action.start_time_hours,
                end_time_hours=(
                    action.start_time_hours + action.duration_minutes / 60.0
                ),
            ),
        )
    return simulate_room_period_with_heating(
        room=room or _room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        ventilation_events=events,
        heating_model=heating_model,
        duration_hours=6.0,
        timestep_minutes=5.0,
    )


# --- No-heating regime (default) -----------------------------------------


def test_no_heating_selected_risk_equals_resimulation() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    resim = _resimulate(result.selected_action, NoHeating())
    resim_risk = evaluate_moisture_risk(
        trajectory=resim.trajectory, surface=_surface(), config=RiskConfig()
    )
    assert (
        result.selected_risk.cumulative_risk_score
        == pytest.approx(resim_risk.cumulative_risk_score, rel=1e-9)
    )
    assert (
        result.selected_risk.maximum_surface_rh_percent
        == pytest.approx(resim_risk.maximum_surface_rh_percent, rel=1e-9)
    )
    assert (
        result.selected_risk.time_in_condensation_hours
        == pytest.approx(resim_risk.time_in_condensation_hours, rel=1e-9)
    )


def test_no_heating_baseline_risk_equals_resimulation() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    resim = _resimulate(ScheduledAction(0.0, 0.0), NoHeating())
    resim_risk = evaluate_moisture_risk(
        trajectory=resim.trajectory, surface=_surface(), config=RiskConfig()
    )
    assert (
        result.baseline_risk.cumulative_risk_score
        == pytest.approx(resim_risk.cumulative_risk_score, rel=1e-9)
    )


def test_no_heating_energy_and_finals_equal_resimulation() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    resim = _resimulate(result.selected_action, NoHeating())
    assert result.energy_penalty_kwh == pytest.approx(
        resim.ventilation_heat_removed_kwh, rel=1e-9
    )
    assert result.heating_thermal_energy_supplied_kwh == pytest.approx(
        resim.heating_thermal_energy_supplied_kwh, abs=1e-12
    )
    assert result.heating_input_energy_purchased_kwh == pytest.approx(
        resim.heating_input_energy_purchased_kwh, abs=1e-12
    )
    assert result.final_indoor_temperature_c == pytest.approx(
        resim.trajectory.indoor_temperature_c[-1], rel=1e-9
    )
    assert result.final_indoor_absolute_humidity_g_m3 == pytest.approx(
        resim.trajectory.indoor_absolute_humidity_g_m3[-1], rel=1e-9
    )
    assert result.final_indoor_relative_humidity_pct == pytest.approx(
        resim.trajectory.indoor_relative_humidity_pct[-1], rel=1e-9
    )


# --- Thermostatic heating regime ----------------------------------------


def _heat_pump() -> ThermostaticHeating:
    return ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=2000.0,
        efficiency_or_cop=3.0,
        hysteresis_c=0.5,
    )


def test_heating_aware_selected_risk_equals_resimulation() -> None:
    heating = _heat_pump()
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating,
    )
    assert result.feasible
    resim = _resimulate(result.selected_action, heating)
    resim_risk = evaluate_moisture_risk(
        trajectory=resim.trajectory, surface=_surface(), config=RiskConfig()
    )
    assert result.selected_risk.cumulative_risk_score == pytest.approx(
        resim_risk.cumulative_risk_score, rel=1e-9
    )
    assert result.selected_risk.maximum_surface_rh_percent == pytest.approx(
        resim_risk.maximum_surface_rh_percent, rel=1e-9
    )
    assert result.selected_risk.time_in_condensation_hours == pytest.approx(
        resim_risk.time_in_condensation_hours, rel=1e-9
    )


def test_heating_aware_baseline_risk_equals_resimulation() -> None:
    heating = _heat_pump()
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating,
    )
    resim = _resimulate(ScheduledAction(0.0, 0.0), heating)
    resim_risk = evaluate_moisture_risk(
        trajectory=resim.trajectory, surface=_surface(), config=RiskConfig()
    )
    assert result.baseline_risk.cumulative_risk_score == pytest.approx(
        resim_risk.cumulative_risk_score, rel=1e-9
    )


def test_heating_aware_energy_and_finals_equal_resimulation() -> None:
    heating = _heat_pump()
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating,
    )
    assert result.feasible
    resim = _resimulate(result.selected_action, heating)
    assert result.energy_penalty_kwh == pytest.approx(
        resim.ventilation_heat_removed_kwh, rel=1e-9
    )
    assert result.heating_thermal_energy_supplied_kwh == pytest.approx(
        resim.heating_thermal_energy_supplied_kwh, rel=1e-9
    )
    assert result.heating_input_energy_purchased_kwh == pytest.approx(
        resim.heating_input_energy_purchased_kwh, rel=1e-9
    )
    assert result.final_indoor_temperature_c == pytest.approx(
        resim.trajectory.indoor_temperature_c[-1], rel=1e-9
    )
    assert result.final_indoor_absolute_humidity_g_m3 == pytest.approx(
        resim.trajectory.indoor_absolute_humidity_g_m3[-1], rel=1e-9
    )
    assert result.final_indoor_relative_humidity_pct == pytest.approx(
        resim.trajectory.indoor_relative_humidity_pct[-1], rel=1e-9
    )


def test_heating_aware_purchased_equals_supplied_divided_by_cop() -> None:
    """Bookkeeping cross-check: purchased == supplied / COP under a heat pump."""
    heating = _heat_pump()
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating,
    )
    assert result.feasible
    assert result.heating_input_energy_purchased_kwh == pytest.approx(
        result.heating_thermal_energy_supplied_kwh / 3.0, rel=1e-9
    )


# --- Pre-vent risk consistency ------------------------------------------


def test_pre_action_risk_equals_resimulated_pre_slice() -> None:
    """Pre-vent risk on the result matches the pre-slice of the re-simulation."""
    heating = _heat_pump()
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating,
    )
    assert result.feasible
    if result.selected_action.start_time_hours == 0.0:
        # Pre-vent slice is defined as t < 0, which is one sample
        # (t = 0 itself). That's the zero-exposure edge case.
        assert result.pre_action_risk.cumulative_risk_score == 0.0
        return

    # Slice the resim trajectory to t < start and re-evaluate.
    resim = _resimulate(result.selected_action, heating)
    times = resim.trajectory.times_hours
    start = result.selected_action.start_time_hours
    end_index = 0
    for i, t in enumerate(times):
        if t < start:
            end_index = i + 1
        else:
            break
    end_index = max(end_index, 1)
    # Build a sliced RoomTrajectory just like the optimiser does.
    from time_simulation import RoomTrajectory

    sliced = RoomTrajectory(
        times_hours=times[:end_index],
        indoor_temperature_c=resim.trajectory.indoor_temperature_c[:end_index],
        indoor_absolute_humidity_g_m3=resim.trajectory.indoor_absolute_humidity_g_m3[
            :end_index
        ],
        indoor_relative_humidity_pct=resim.trajectory.indoor_relative_humidity_pct[
            :end_index
        ],
        outdoor_temperature_c=resim.trajectory.outdoor_temperature_c[:end_index],
        outdoor_absolute_humidity_g_m3=resim.trajectory.outdoor_absolute_humidity_g_m3[
            :end_index
        ],
        window_open=resim.trajectory.window_open[:end_index],
        moisture_generation_g_per_hour=resim.trajectory.moisture_generation_g_per_hour[
            :end_index
        ],
    )
    resim_pre_risk = evaluate_moisture_risk(
        trajectory=sliced, surface=_surface(), config=RiskConfig()
    )
    assert result.pre_action_risk.cumulative_risk_score == pytest.approx(
        resim_pre_risk.cumulative_risk_score, rel=1e-9
    )


# --- Default heating_model behaviour ------------------------------------


def test_default_heating_model_is_no_heating() -> None:
    """Omitting ``heating_model`` produces the same result as passing NoHeating()."""
    default_result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    explicit_result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=3.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=NoHeating(),
    )
    assert (
        default_result.selected_action.start_time_hours
        == explicit_result.selected_action.start_time_hours
    )
    assert (
        default_result.selected_action.duration_minutes
        == explicit_result.selected_action.duration_minutes
    )
    assert default_result.selected_risk.cumulative_risk_score == pytest.approx(
        explicit_result.selected_risk.cumulative_risk_score, rel=1e-9
    )
    assert default_result.energy_penalty_kwh == pytest.approx(
        explicit_result.energy_penalty_kwh, rel=1e-9
    )


def test_heating_changes_the_predicted_risk_relative_to_no_heating() -> None:
    """Load-bearing: passing a heating model changes what the optimiser sees.

    If the optimiser silently ignored ``heating_model``, this test
    would produce identical numbers under NoHeating and
    ThermostaticHeating - the previous bug.
    """
    no_heat = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=8.0, max_cumulative_risk_score=100.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=NoHeating(),
    )
    with_heat = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=8.0, max_cumulative_risk_score=100.0
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=_heat_pump(),
    )
    assert no_heat.feasible and with_heat.feasible
    # The two trajectories differ: purchased energy is zero under
    # NoHeating and non-zero under the heat pump.
    assert no_heat.heating_input_energy_purchased_kwh == 0.0
    assert with_heat.heating_input_energy_purchased_kwh > 0.0
