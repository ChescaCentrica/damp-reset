"""Tests for the risk-constrained minimum-energy ventilation strategy.

Kept in a separate module so ``test_optimiser.py``'s AST guard on
physics re-derivation stays focused on the moisture-target
strategies. The risk-constrained strategy delegates to
``time_simulation``, ``mould_risk``, and the single-event
``simulate_ventilation_event`` and does not introduce any physics
equation of its own.

Covers:
    - shape and immutability of ``RiskConstrainedOptimisationResult``,
    - do-nothing wins when baseline risk is already below the ceiling,
    - lowest-energy action is selected among feasible candidates,
    - a ventilation action strictly reduces predicted risk under the
      illustrative POC scenario, so a tighter ceiling picks a longer
      duration,
    - explicit infeasibility reporting when the risk ceiling cannot be
      met,
    - explicit infeasibility reporting when the risk ceiling is unset,
    - comfort constraint interacts correctly with the risk constraint,
    - validation on ``control_horizon_hours`` and
      ``trajectory_timestep_minutes``.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math import isnan

import pytest

from moisture import Room
from moisture_sources import MoistureSourceEvent, MoistureSourceSchedule
from mould_risk import RiskConfig
from psychrometrics import AirState
from surface_risk import SurfaceDescriptor
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties

from optimiser import (
    RiskConstrainedOptimisationResult,
    VentilationConstraints,
    optimise_min_energy_under_risk_limit,
)


def _room() -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )


def _outdoor() -> AirState:
    return AirState(temperature_c=8.0, relative_humidity_percent=70.0)


def _thermal() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _surface() -> SurfaceDescriptor:
    return SurfaceDescriptor(
        label="cold wall corner", surface_temperature_factor=0.75
    )


def _risk_config() -> RiskConfig:
    return RiskConfig()


def _schedule() -> MoistureSourceSchedule:
    """Illustrative source: 60 g/h background + 30-min cooking spike."""
    return MoistureSourceSchedule(
        constant_background_rate_g_per_hour=60.0,
        events=(
            MoistureSourceEvent(
                label="cooking",
                start_time_hours=0.0,
                end_time_hours=0.5,
                generation_rate_g_per_hour=400.0,
            ),
        ),
    )


def _candidates() -> tuple:
    return (0.0, 2.0, 5.0, 10.0, 20.0, 30.0)


# --- Shape ----------------------------------------------------------------


def test_result_is_frozen_dataclass() -> None:
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert isinstance(result, RiskConstrainedOptimisationResult)
    with pytest.raises(FrozenInstanceError):
        result.selected_duration_minutes = 99.0  # type: ignore[misc]


def test_result_exposes_named_fields() -> None:
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    for name in (
        "selected_duration_minutes",
        "selected_prediction",
        "baseline_risk",
        "selected_risk",
        "energy_penalty_kwh",
        "objective_name",
        "feasible",
        "reason",
    ):
        assert hasattr(result, name), name


def test_baseline_risk_always_populated_even_when_infeasible() -> None:
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0,
            max_cumulative_risk_score=None,  # unset -> infeasible
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert not result.feasible
    # Baseline risk is still measured and reported.
    assert result.baseline_risk.cumulative_risk_score >= 0.0


# --- Selection behaviour --------------------------------------------------


def test_do_nothing_wins_when_baseline_is_already_below_ceiling() -> None:
    """A ceiling above the baseline risk -> 0-min action wins.

    The baseline (do-nothing) trajectory over the control horizon has
    the lowest possible ventilation energy (zero). If it already
    satisfies the risk ceiling, no other candidate can beat it on
    energy, so the strategy must pick 0 min.
    """
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0,
            max_cumulative_risk_score=100.0,  # trivially loose ceiling
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert result.feasible
    assert result.selected_duration_minutes == 0.0
    assert result.energy_penalty_kwh == 0.0
    assert result.selected_risk == result.baseline_risk
    assert "do-nothing" in result.reason


def test_tighter_ceiling_forces_longer_duration() -> None:
    """A tighter risk ceiling can only be met by a longer (or equal) action."""
    common_kwargs = dict(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    result_loose = optimise_min_energy_under_risk_limit(
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=2.0
        ),
        **common_kwargs,
    )
    result_tight = optimise_min_energy_under_risk_limit(
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
        ),
        **common_kwargs,
    )
    assert result_loose.feasible
    assert result_tight.feasible
    assert (
        result_tight.selected_duration_minutes
        >= result_loose.selected_duration_minutes
    )
    assert (
        result_tight.energy_penalty_kwh
        >= result_loose.energy_penalty_kwh - 1e-9
    )


def test_selected_risk_is_at_or_below_ceiling_when_feasible() -> None:
    """Feasibility means the SELECTED candidate's risk is under the ceiling."""
    ceiling = 1.0
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=ceiling
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert result.feasible
    assert result.selected_risk.cumulative_risk_score <= ceiling


def test_action_reduces_predicted_risk_relative_to_baseline() -> None:
    """A ventilation action's post-action risk cannot exceed baseline risk.

    Under the illustrative scenario, opening the window for a few
    minutes strictly reduces the sustained surface exposure. This
    test asserts the weaker monotonic guarantee (post-action <=
    baseline) because in some corner cases the ventilation makes no
    difference but never worsens the score.
    """
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=1.5
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert result.feasible
    assert (
        result.selected_risk.cumulative_risk_score
        <= result.baseline_risk.cumulative_risk_score
    )


def test_energy_penalty_matches_selected_prediction() -> None:
    """``energy_penalty_kwh`` is exactly the selected event's energy loss."""
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert result.feasible
    assert (
        result.energy_penalty_kwh
        == result.selected_prediction.ventilation_energy_removed_kwh
    )


# --- Infeasibility --------------------------------------------------------


def test_missing_risk_ceiling_is_infeasible_with_named_reason() -> None:
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(max_temperature_drop_c=3.0),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert not result.feasible
    assert isnan(result.selected_duration_minutes)
    assert isnan(result.energy_penalty_kwh)
    assert "max_cumulative_risk_score" in result.reason


def test_unmeetable_risk_ceiling_reports_closest_miss() -> None:
    """A ceiling below every candidate's risk -> infeasible, reason names the miss."""
    tight_ceiling = 0.0001
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=(0.0, 1.0, 2.0),
        constraints=VentilationConstraints(
            max_temperature_drop_c=3.0,
            max_cumulative_risk_score=tight_ceiling,
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert not result.feasible
    assert isnan(result.selected_duration_minutes)
    assert "cumulative risk" in result.reason
    # Baseline is still reported for context.
    assert result.baseline_risk.cumulative_risk_score > tight_ceiling


def test_comfort_conflict_reports_comfort_violation() -> None:
    """If every candidate breaches comfort, reason names the comfort violation."""
    # A zero comfort budget forbids every non-zero duration; excluding
    # 0 from the candidate list forces the comfort branch.
    result = optimise_min_energy_under_risk_limit(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=(30.0,),
        constraints=VentilationConstraints(
            max_temperature_drop_c=0.0,  # unmeetable by any non-zero vent
            max_cumulative_risk_score=1.0,
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        moisture_schedule=_schedule(),
        control_horizon_hours=4.0,
        trajectory_timestep_minutes=2.0,
    )
    assert not result.feasible
    assert "comfort" in result.reason


# --- Validation -----------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_control_horizon_hours_validated(bad: float) -> None:
    with pytest.raises(ValueError, match="control_horizon_hours"):
        optimise_min_energy_under_risk_limit(
            room=_room(),
            outdoor=_outdoor(),
            thermal_properties=_thermal(),
            candidate_durations_minutes=_candidates(),
            constraints=VentilationConstraints(
                max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            moisture_schedule=_schedule(),
            control_horizon_hours=bad,
            trajectory_timestep_minutes=2.0,
        )


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_trajectory_timestep_minutes_validated(bad: float) -> None:
    with pytest.raises(ValueError, match="trajectory_timestep_minutes"):
        optimise_min_energy_under_risk_limit(
            room=_room(),
            outdoor=_outdoor(),
            thermal_properties=_thermal(),
            candidate_durations_minutes=_candidates(),
            constraints=VentilationConstraints(
                max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            moisture_schedule=_schedule(),
            control_horizon_hours=4.0,
            trajectory_timestep_minutes=bad,
        )


def test_empty_candidate_list_raises() -> None:
    with pytest.raises(ValueError, match="candidate_durations_minutes"):
        optimise_min_energy_under_risk_limit(
            room=_room(),
            outdoor=_outdoor(),
            thermal_properties=_thermal(),
            candidate_durations_minutes=(),
            constraints=VentilationConstraints(
                max_temperature_drop_c=3.0, max_cumulative_risk_score=1.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            moisture_schedule=_schedule(),
            control_horizon_hours=4.0,
            trajectory_timestep_minutes=2.0,
        )


def test_max_cumulative_risk_score_validation_on_constraints() -> None:
    """A negative risk ceiling on ``VentilationConstraints`` is rejected."""
    with pytest.raises(ValueError, match="max_cumulative_risk_score"):
        VentilationConstraints(max_cumulative_risk_score=-0.1)


def test_baseline_strategy_still_intact() -> None:
    """The moisture-target baseline continues to exist and to import cleanly.

    A load-bearing test that the new risk-constrained strategy did
    NOT replace the old strategy. Both must be callable independently.
    """
    from optimiser import (
        choose_minimum_energy_action,
        recommend_ventilation_action,
    )

    assert choose_minimum_energy_action is not None
    assert recommend_ventilation_action is not None
    # A quick call with a moisture target succeeds (the actual value
    # returned is the responsibility of the older strategy's tests).
    baseline = choose_minimum_energy_action(
        room=_room(),
        outdoor=_outdoor(),
        thermal_properties=_thermal(),
        candidate_durations_minutes=_candidates(),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=3.0,
        ),
    )
    assert baseline.objective_name == "minimum ventilation energy loss"
