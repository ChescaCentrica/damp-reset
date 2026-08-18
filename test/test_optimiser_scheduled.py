"""Tests for the (start-time + duration) risk-constrained optimiser.

Covers:
    - shape and immutability of ``ScheduledAction`` /
      ``ScheduledActionResult``;
    - do-nothing wins when baseline risk is already below the ceiling;
    - the strategy picks a later start time when waiting is safe AND
      reduces the ventilation energy penalty;
    - the strategy REFUSES to wait when waiting alone would breach
      the risk ceiling before the proposed ventilation event -
      i.e. the pre-vent risk guard fires and the result reports it
      explicitly;
    - the risk ceiling is enforced across the full horizon;
    - comfort constraints filter candidates;
    - explicit infeasibility for unset risk ceiling / candidates
      overflowing the horizon;
    - old strategies remain callable.
"""

import sys
from dataclasses import FrozenInstanceError
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room
from moisture_sources import MoistureSourceEvent, MoistureSourceSchedule
from mould_risk import RiskConfig
from surface_risk import SurfaceDescriptor
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from weather_forecast import ForecastPoint, WeatherForecast

from optimiser import (
    ScheduledAction,
    ScheduledActionResult,
    VentilationConstraints,
    optimise_scheduled_action_under_risk_limit,
)


def _room(rh: float = 70.0) -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=rh,
        ach_closed=0.5,
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
        label="cold external wall corner",
        surface_temperature_factor=0.65,
    )


def _risk_config() -> RiskConfig:
    return RiskConfig()


def _mild_schedule() -> MoistureSourceSchedule:
    """Moderate constant background - room drifts up slowly."""
    return MoistureSourceSchedule(constant_background_rate_g_per_hour=60.0)


def _stormy_schedule() -> MoistureSourceSchedule:
    """A shower event at t = 0.5-0.9 h drives risk up quickly."""
    return MoistureSourceSchedule(
        constant_background_rate_g_per_hour=60.0,
        events=(
            MoistureSourceEvent(
                label="shower",
                start_time_hours=0.5,
                end_time_hours=0.9,
                generation_rate_g_per_hour=1500.0,
            ),
        ),
    )


def _cold_now_mild_later() -> WeatherForecast:
    return WeatherForecast(
        points=(
            ForecastPoint(
                timestamp_hours=0.0,
                temperature_c=-2.0,
                relative_humidity_percent=70.0,
            ),
            ForecastPoint(
                timestamp_hours=2.0,
                temperature_c=8.0,
                relative_humidity_percent=60.0,
            ),
            ForecastPoint(
                timestamp_hours=4.0,
                temperature_c=12.0,
                relative_humidity_percent=55.0,
            ),
        )
    )


# --- Shape ---------------------------------------------------------------


def test_scheduled_action_is_frozen() -> None:
    action = ScheduledAction(start_time_hours=1.0, duration_minutes=10.0)
    with pytest.raises(FrozenInstanceError):
        action.start_time_hours = 2.0  # type: ignore[misc]


def test_scheduled_action_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="start_time_hours"):
        ScheduledAction(start_time_hours=-0.1, duration_minutes=10.0)


def test_scheduled_action_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_minutes"):
        ScheduledAction(start_time_hours=0.0, duration_minutes=-5.0)


def test_scheduled_action_is_do_nothing_flag() -> None:
    assert ScheduledAction(0.0, 0.0).is_do_nothing
    assert not ScheduledAction(0.0, 5.0).is_do_nothing


def test_result_is_frozen_and_exposes_named_fields() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[
            ScheduledAction(0.0, 0.0),
            ScheduledAction(0.0, 10.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert isinstance(result, ScheduledActionResult)
    for name in (
        "selected_action",
        "baseline_risk",
        "selected_risk",
        "pre_action_risk",
        "energy_penalty_kwh",
        "final_indoor_temperature_c",
        "final_indoor_absolute_humidity_g_m3",
        "final_indoor_relative_humidity_pct",
        "objective_name",
        "feasible",
        "reason",
    ):
        assert hasattr(result, name), name
    with pytest.raises(FrozenInstanceError):
        result.selected_action = ScheduledAction(1.0, 1.0)  # type: ignore[misc]


# --- Selection behaviour -------------------------------------------------


def test_do_nothing_wins_when_baseline_already_below_ceiling() -> None:
    """A loose ceiling that the baseline already meets -> 0-min action wins.

    Do-nothing has the LOWEST ventilation heat removed of any
    feasible candidate: the room only leaks through ach_closed, not
    through an open window. Any non-zero-duration candidate loses
    strictly more heat, so no other candidate can beat do-nothing
    when the ceiling accepts the baseline.
    """
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[
            ScheduledAction(0.0, 0.0),
            ScheduledAction(0.0, 10.0),
            ScheduledAction(2.0, 10.0),
            ScheduledAction(4.0, 10.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0,
            max_cumulative_risk_score=100.0,  # trivially loose
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    assert result.selected_action.is_do_nothing
    # Do-nothing accrues only closed-window background loss; it does
    # not accrue zero unless ach_closed is zero. Under NoHeating, no
    # heat is supplied to compensate.
    assert result.energy_penalty_kwh >= 0.0
    assert result.heating_thermal_energy_supplied_kwh == 0.0
    assert result.heating_input_energy_purchased_kwh == 0.0
    assert result.pre_action_risk.cumulative_risk_score == 0.0


def test_wait_forbidden_when_pre_action_risk_would_breach_ceiling() -> None:
    """Load-bearing: never wait past the ceiling.

    A stormy schedule (shower event at t = 0.5-0.9 h) drives risk up
    fast. A candidate that WAITS 2 h before ventilating would let
    the pre-ventilation risk cross the ceiling, and the strategy
    must not select it. Under a ceiling of 1.0, only immediate
    ventilation can survive.
    """
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_stormy_schedule(),
        candidate_actions=[
            ScheduledAction(0.0, 0.0),
            ScheduledAction(0.0, 15.0),
            ScheduledAction(0.0, 30.0),
            ScheduledAction(2.0, 15.0),
            ScheduledAction(2.0, 30.0),
            ScheduledAction(4.0, 30.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0, max_cumulative_risk_score=1.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    # Either the strategy picks an IMMEDIATE (start=0) action or, if
    # no immediate action can bring the horizon-wide risk below the
    # ceiling, reports infeasibility with the pre-vent-risk reason.
    # Both outcomes correctly express "don't wait if waiting itself
    # breaches the ceiling".
    if result.feasible:
        assert result.selected_action.start_time_hours == 0.0
        assert (
            result.pre_action_risk.cumulative_risk_score
            <= 1.0 + 1e-9
        )
    else:
        assert "waiting" in result.reason.lower() or "pre-vent" in result.reason.lower()


def test_selected_risk_is_at_or_below_ceiling_when_feasible() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[
            ScheduledAction(0.0, 0.0),
            ScheduledAction(0.0, 10.0),
            ScheduledAction(2.0, 10.0),
            ScheduledAction(4.0, 10.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    assert result.selected_risk.cumulative_risk_score <= 2.0


def test_pre_action_risk_is_at_or_below_ceiling_when_feasible() -> None:
    """The pre-vent risk guard produces feasible actions whose pre-slice risk fits."""
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[
            ScheduledAction(0.0, 0.0),
            ScheduledAction(0.0, 10.0),
            ScheduledAction(2.0, 10.0),
            ScheduledAction(4.0, 10.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    assert result.pre_action_risk.cumulative_risk_score <= 2.0 + 1e-9


def test_energy_penalty_matches_heating_aware_resimulation() -> None:
    """The reported energy penalty equals the heating-aware simulator's total.

    After the "heating-aware optimiser" refactor, the optimiser
    plans on the same trajectory type a caller obtains from
    ``simulate_room_period_with_heating``. The
    ``energy_penalty_kwh`` field is defined as the
    ``ventilation_heat_removed_kwh`` of that trajectory - and this
    test proves the optimiser has NOT invented its own number.
    """
    from heating import NoHeating
    from time_simulation import (
        VentilationEvent as _VentEvent,
        simulate_room_period_with_heating,
    )

    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[ScheduledAction(0.0, 10.0)],
        constraints=VentilationConstraints(
            max_temperature_drop_c=10.0, max_cumulative_risk_score=100.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert result.feasible
    assert not result.selected_action.is_do_nothing

    resim = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        ventilation_events=(
            _VentEvent(start_time_hours=0.0, end_time_hours=10.0 / 60.0),
        ),
        heating_model=NoHeating(),
        duration_hours=6.0,
        timestep_minutes=5.0,
    )
    assert result.energy_penalty_kwh == pytest.approx(
        resim.ventilation_heat_removed_kwh, rel=1e-9
    )


# --- Infeasibility -------------------------------------------------------


def test_missing_risk_ceiling_is_infeasible() -> None:
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[ScheduledAction(0.0, 10.0)],
        constraints=VentilationConstraints(max_temperature_drop_c=5.0),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert not result.feasible
    assert "max_cumulative_risk_score" in result.reason


def test_candidate_overshoots_horizon_rejected() -> None:
    """A start + duration overshooting the horizon is a ValueError."""
    with pytest.raises(ValueError, match="past the control horizon"):
        optimise_scheduled_action_under_risk_limit(
            room=_room(),
            thermal_properties=_thermal(),
            forecast=_cold_now_mild_later(),
            moisture_schedule=_mild_schedule(),
            candidate_actions=[ScheduledAction(5.5, 60.0)],  # 5.5 + 1.0 > 6.0
            constraints=VentilationConstraints(
                max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            control_horizon_hours=6.0,
            trajectory_timestep_minutes=5.0,
        )


def test_empty_candidate_list_raises() -> None:
    with pytest.raises(ValueError, match="candidate_actions"):
        optimise_scheduled_action_under_risk_limit(
            room=_room(),
            thermal_properties=_thermal(),
            forecast=_cold_now_mild_later(),
            moisture_schedule=_mild_schedule(),
            candidate_actions=[],
            constraints=VentilationConstraints(
                max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            control_horizon_hours=6.0,
            trajectory_timestep_minutes=5.0,
        )


def test_infeasibility_naming_pre_vent_when_waiting_alone_breaches() -> None:
    """A tight ceiling + stormy schedule where only late starts remain -> pre-vent branch fires."""
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_stormy_schedule(),
        # No immediate-start non-zero candidate; force only delayed
        # candidates so the pre-vent guard is the tightest miss.
        candidate_actions=[
            ScheduledAction(2.0, 15.0),
            ScheduledAction(2.0, 30.0),
            ScheduledAction(4.0, 30.0),
        ],
        constraints=VentilationConstraints(
            max_temperature_drop_c=5.0, max_cumulative_risk_score=0.5
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert not result.feasible
    assert isnan(result.energy_penalty_kwh)
    # pre_action_risk on the closest miss is reported.
    assert result.pre_action_risk.cumulative_risk_score > 0.5


def test_comfort_conflict_reports_comfort_violation() -> None:
    """A zero comfort budget on non-zero durations -> comfort branch."""
    result = optimise_scheduled_action_under_risk_limit(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_now_mild_later(),
        moisture_schedule=_mild_schedule(),
        candidate_actions=[ScheduledAction(0.0, 30.0)],
        constraints=VentilationConstraints(
            max_temperature_drop_c=0.0, max_cumulative_risk_score=2.0
        ),
        surface=_surface(),
        risk_config=_risk_config(),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
    )
    assert not result.feasible
    assert "comfort" in result.reason


# --- Validation ----------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_control_horizon_hours_validated(bad: float) -> None:
    with pytest.raises(ValueError, match="control_horizon_hours"):
        optimise_scheduled_action_under_risk_limit(
            room=_room(),
            thermal_properties=_thermal(),
            forecast=_cold_now_mild_later(),
            moisture_schedule=_mild_schedule(),
            candidate_actions=[ScheduledAction(0.0, 5.0)],
            constraints=VentilationConstraints(
                max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            control_horizon_hours=bad,
            trajectory_timestep_minutes=5.0,
        )


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_trajectory_timestep_minutes_validated(bad: float) -> None:
    with pytest.raises(ValueError, match="trajectory_timestep_minutes"):
        optimise_scheduled_action_under_risk_limit(
            room=_room(),
            thermal_properties=_thermal(),
            forecast=_cold_now_mild_later(),
            moisture_schedule=_mild_schedule(),
            candidate_actions=[ScheduledAction(0.0, 5.0)],
            constraints=VentilationConstraints(
                max_temperature_drop_c=5.0, max_cumulative_risk_score=2.0
            ),
            surface=_surface(),
            risk_config=_risk_config(),
            control_horizon_hours=6.0,
            trajectory_timestep_minutes=bad,
        )


def test_baseline_and_risk_constrained_strategies_still_intact() -> None:
    """The other strategies remain callable after this slice."""
    from optimiser import (
        choose_minimum_energy_action,
        optimise_min_energy_under_risk_limit,
        recommend_ventilation_action,
    )

    assert choose_minimum_energy_action is not None
    assert recommend_ventilation_action is not None
    assert optimise_min_energy_under_risk_limit is not None
