"""Regression tests: baseline-subtracted energy and risk reduction bookkeeping.

After the "incremental energy" refactor, the scheduled risk-
constrained optimiser exposes six new fields on
``ScheduledActionResult``:

    baseline_heating_thermal_energy_supplied_kwh
    baseline_heating_input_energy_purchased_kwh
    incremental_heating_thermal_energy_supplied_kwh
    incremental_heating_input_energy_purchased_kwh
    risk_reduction
    condensation_time_reduction_hours

and the optimiser's primary energy objective is now the incremental
purchased energy (baseline-subtracted), not the total. These tests
prove the arithmetic and prove the ranking uses the new objective
under a heating model. Under ``NoHeating`` the earlier tests
(test_optimiser_scheduled*.py) already pin the ordering to the
secondary key ``ventilation_heat_removed_kwh``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from heating import NoHeating, ThermostaticHeating
from moisture import Room
from moisture_sources import MoistureSourceSchedule
from mould_risk import RiskConfig, evaluate_moisture_risk
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


def _room() -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.2,
        indoor_relative_humidity_pct=71.0,
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


def _schedule() -> MoistureSourceSchedule:
    return MoistureSourceSchedule(constant_background_rate_g_per_hour=80.0)


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


def _heat_pump() -> ThermostaticHeating:
    return ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=2000.0,
        efficiency_or_cop=3.0,
        hysteresis_c=0.5,
    )


def _candidates() -> tuple:
    actions = [ScheduledAction(0.0, 0.0)]
    for start in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for duration in (5.0, 10.0, 15.0, 20.0, 30.0):
            if start + duration / 60.0 <= 6.0:
                actions.append(ScheduledAction(start, duration))
    return tuple(actions)


def _resimulate_baseline(heating_model):
    return simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        ventilation_events=(),
        heating_model=heating_model,
        duration_hours=6.0,
        timestep_minutes=5.0,
    )


def _resimulate_action(action: ScheduledAction, heating_model):
    events = (
        ()
        if action.is_do_nothing
        else (
            VentilationEvent(
                start_time_hours=action.start_time_hours,
                end_time_hours=(
                    action.start_time_hours + action.duration_minutes / 60.0
                ),
            ),
        )
    )
    return simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        ventilation_events=events,
        heating_model=heating_model,
        duration_hours=6.0,
        timestep_minutes=5.0,
    )


def _optimise(heating_model, risk_ceiling: float = 3.0, comfort_cap: float = 3.0):
    return optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_forecast(),
        moisture_schedule=_schedule(),
        candidate_actions=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=comfort_cap,
            max_cumulative_risk_score=risk_ceiling,
        ),
        surface=_surface(),
        risk_config=RiskConfig(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        heating_model=heating_model,
    )


# --- Result shape --------------------------------------------------------


def test_result_carries_new_fields() -> None:
    result = _optimise(_heat_pump())
    for name in (
        "baseline_heating_thermal_energy_supplied_kwh",
        "baseline_heating_input_energy_purchased_kwh",
        "incremental_heating_thermal_energy_supplied_kwh",
        "incremental_heating_input_energy_purchased_kwh",
        "risk_reduction",
        "condensation_time_reduction_hours",
    ):
        assert hasattr(result, name), name


def test_objective_name_names_incremental_energy() -> None:
    result = _optimise(_heat_pump())
    assert "incremental" in result.objective_name.lower()


# --- Baseline totals match an independent baseline resim ---------------


def test_baseline_totals_match_independent_resimulation() -> None:
    heating = _heat_pump()
    result = _optimise(heating)
    baseline_resim = _resimulate_baseline(heating)
    assert result.baseline_heating_thermal_energy_supplied_kwh == pytest.approx(
        baseline_resim.heating_thermal_energy_supplied_kwh, rel=1e-9
    )
    assert result.baseline_heating_input_energy_purchased_kwh == pytest.approx(
        baseline_resim.heating_input_energy_purchased_kwh, rel=1e-9
    )


# --- Incremental = action - baseline arithmetic ------------------------


def test_incremental_thermal_equals_action_minus_baseline() -> None:
    heating = _heat_pump()
    result = _optimise(heating)
    diff = (
        result.heating_thermal_energy_supplied_kwh
        - result.baseline_heating_thermal_energy_supplied_kwh
    )
    assert result.incremental_heating_thermal_energy_supplied_kwh == pytest.approx(
        diff, rel=1e-9
    )


def test_incremental_input_equals_action_minus_baseline() -> None:
    heating = _heat_pump()
    result = _optimise(heating)
    diff = (
        result.heating_input_energy_purchased_kwh
        - result.baseline_heating_input_energy_purchased_kwh
    )
    assert result.incremental_heating_input_energy_purchased_kwh == pytest.approx(
        diff, rel=1e-9
    )


def test_incremental_input_equals_thermal_divided_by_cop() -> None:
    """Bookkeeping cross-check for a heat pump: input == supplied / COP."""
    heating = _heat_pump()
    result = _optimise(heating)
    assert result.incremental_heating_input_energy_purchased_kwh == pytest.approx(
        result.incremental_heating_thermal_energy_supplied_kwh / 3.0, rel=1e-9
    )


# --- Do-nothing edge cases ---------------------------------------------


def test_do_nothing_action_reports_zero_incrementals() -> None:
    """When do-nothing wins, every incremental / delta field is zero."""
    # Trivially loose ceiling -> baseline OK -> optimiser picks do-nothing.
    result = _optimise(_heat_pump(), risk_ceiling=1000.0)
    assert result.selected_action.is_do_nothing
    assert result.incremental_heating_thermal_energy_supplied_kwh == 0.0
    assert result.incremental_heating_input_energy_purchased_kwh == 0.0
    assert result.risk_reduction == 0.0
    assert result.condensation_time_reduction_hours == 0.0


def test_no_heating_regime_incrementals_are_zero() -> None:
    """Under NoHeating, incremental purchased energy is exactly zero.

    Both baseline and action deliver / purchase nothing, so the
    subtraction is zero for every candidate. The ranking then falls
    through to the secondary key (vent heat removed) - see the
    existing test_optimiser_scheduled tests.
    """
    result = _optimise(NoHeating())
    assert result.feasible
    assert result.baseline_heating_input_energy_purchased_kwh == 0.0
    assert result.heating_input_energy_purchased_kwh == 0.0
    assert result.incremental_heating_input_energy_purchased_kwh == 0.0
    assert result.baseline_heating_thermal_energy_supplied_kwh == 0.0
    assert result.heating_thermal_energy_supplied_kwh == 0.0
    assert result.incremental_heating_thermal_energy_supplied_kwh == 0.0


# --- Risk and condensation deltas --------------------------------------


def test_risk_reduction_equals_baseline_minus_action() -> None:
    result = _optimise(_heat_pump())
    expected = (
        result.baseline_risk.cumulative_risk_score
        - result.selected_risk.cumulative_risk_score
    )
    assert result.risk_reduction == pytest.approx(expected, rel=1e-9)


def test_condensation_time_reduction_equals_baseline_minus_action() -> None:
    result = _optimise(_heat_pump())
    expected = (
        result.baseline_risk.time_in_condensation_hours
        - result.selected_risk.time_in_condensation_hours
    )
    assert result.condensation_time_reduction_hours == pytest.approx(
        expected, rel=1e-9
    )


def test_ventilation_action_reduces_risk_relative_to_baseline() -> None:
    """A feasible ventilation action should not raise the risk score."""
    result = _optimise(_heat_pump())
    assert result.feasible
    # Feasibility means selected_risk <= risk_ceiling; but baseline
    # may exceed the ceiling. In practice risk_reduction is >= 0 for
    # any non-do-nothing action that was selected. We use >= 0.0 to
    # allow the exact do-nothing edge case (reduction == 0).
    assert result.risk_reduction >= -1e-9


# --- Ranking uses incremental purchased energy under a heat pump -------


def test_ranking_uses_incremental_purchased_energy_under_heating() -> None:
    """The winner has the smallest incremental purchased energy among feasible.

    Recompute every candidate's incremental purchased energy by
    independent resim and check the winner ties for the minimum.
    """
    heating = _heat_pump()
    result = _optimise(heating)
    assert result.feasible
    baseline_resim = _resimulate_baseline(heating)
    baseline_input_kwh = baseline_resim.heating_input_energy_purchased_kwh
    baseline_risk_state = evaluate_moisture_risk(
        trajectory=baseline_resim.trajectory,
        surface=_surface(),
        config=RiskConfig(),
    )
    baseline_risk_score = baseline_risk_state.cumulative_risk_score
    baseline_pre_risk_score = 0.0  # pre-slice at t < 0 is empty
    winner_incremental = (
        result.incremental_heating_input_energy_purchased_kwh
    )
    for action in _candidates():
        action_resim = _resimulate_action(action, heating)
        action_incremental = (
            action_resim.heating_input_energy_purchased_kwh - baseline_input_kwh
        )
        action_risk_state = evaluate_moisture_risk(
            trajectory=action_resim.trajectory,
            surface=_surface(),
            config=RiskConfig(),
        )
        action_risk = action_risk_state.cumulative_risk_score
        # Skip infeasible candidates - the ranking is over the
        # feasible subset only.
        if action_risk > 3.0 + 1e-9:
            continue
        # No comfort violation check here: the demo scenario's
        # candidates all satisfy the 3 K comfort cap under this
        # heating model. Any borderline candidate is caught by the
        # optimiser's own check; if it survives to this point, it
        # was feasible.
        if action_incremental + 1e-9 < winner_incremental:
            raise AssertionError(
                f"a feasible candidate {action} has smaller incremental "
                f"purchased energy ({action_incremental:.4f}) than the "
                f"winner ({winner_incremental:.4f})"
            )
