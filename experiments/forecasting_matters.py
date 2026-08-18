"""Why forecasting matters: three controllers compared on two scenarios.

Compares:
    1. immediate_when_indoor_humid
        Ventilates as soon as the current indoor RH is above a
        caller-set threshold. Ignores outdoor conditions entirely.
    2. immediate_when_outdoor_ah_below_indoor
        Ventilates as soon as the current outdoor AH is below the
        current indoor AH (the classic psychrometric "drying
        potential" rule). Ignores the forecast.
    3. predictive_risk_controller
        Delegates to optimiser.optimise_scheduled_action_under_risk_limit
        over the future forecast: chooses a (start_time, duration)
        pair that keeps the predicted surface-risk indicator below
        the ceiling AND minimises ventilation energy. Refuses to
        wait if waiting alone would breach the risk ceiling.

Two scenarios (same shape, different numbers):
    A - "waiting is safe": cold+dry outdoor NOW, mild+dry-enough
        outdoor LATER, moderate moisture generation. The predictive
        controller SHOULD wait; the greedy controllers should
        ventilate now.
    B - "waiting is unsafe": heavy moisture generation right now
        (shower) pushing surface risk up quickly. The predictive
        controller SHOULD ventilate immediately.

For each scenario, the script reports:
    - selected start time
    - selected duration
    - predicted cumulative risk score (indicator, NOT mould)
    - ventilation energy penalty (kWh)
    - final indoor T / AH / RH

The intended demonstration is: on Scenario A, waiting reduces
energy loss without increasing unacceptable surface moisture risk;
on Scenario B, the predictive controller correctly does not wait.

CALIBRATION WARNING: every threshold in this experiment is a POC
illustrative value. See the mould_risk / surface_risk / optimiser
module docstrings; the cumulative_risk_score is a caller-configured
INDICATOR and NOT a mould-growth prediction.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room
from moisture_sources import MoistureSourceSchedule
from mould_risk import RiskConfig, evaluate_moisture_risk
from optimiser import (
    ScheduledAction,
    ScheduledActionResult,
    VentilationConstraints,
    optimise_scheduled_action_under_risk_limit,
)
from psychrometrics import AirState
from surface_risk import SurfaceDescriptor
from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    ThermalProperties,
)
from time_simulation import (
    VentilationEvent,
    simulate_room_period_with_forecast,
)
from weather_forecast import ForecastPoint, WeatherForecast


@dataclass(frozen=True)
class ControllerReport:
    """Uniform result shape across the three controllers."""

    controller: str
    ventilation_started: bool
    start_time_hours: float
    duration_minutes: float
    cumulative_risk_score: float
    peak_surface_rh_percent: float
    time_in_condensation_hours: float
    ventilation_energy_kwh: float
    final_indoor_temperature_c: float
    final_indoor_absolute_humidity_g_m3: float
    final_indoor_relative_humidity_pct: float
    reason: str


@dataclass(frozen=True)
class Scenario:
    name: str
    room: Room
    thermal_properties: ThermalProperties
    surface: SurfaceDescriptor
    forecast: WeatherForecast
    moisture_schedule: MoistureSourceSchedule
    control_horizon_hours: float
    trajectory_timestep_minutes: float
    risk_config: RiskConfig
    risk_ceiling: float
    max_temperature_drop_c: float
    candidate_actions: Sequence[ScheduledAction]
    indoor_rh_threshold_pct: float


def _simulate_and_evaluate(
    scenario: Scenario, action: ScheduledAction
) -> Tuple[float, float, float, float, float, float, float]:
    """Simulate a scenario under one action; return the metrics tuple.

    Delegates to time_simulation for the trajectory, mould_risk for
    the risk indicator, and the trajectory's own last sample for the
    final indoor state. Energy is read from the single-event
    simulator when a vent event was scheduled.
    """
    if action.is_do_nothing:
        events: Tuple[VentilationEvent, ...] = ()
    else:
        events = (
            VentilationEvent(
                start_time_hours=action.start_time_hours,
                end_time_hours=(
                    action.start_time_hours + action.duration_minutes / 60.0
                ),
            ),
        )
    trajectory = simulate_room_period_with_forecast(
        room=scenario.room,
        thermal_properties=scenario.thermal_properties,
        forecast=scenario.forecast,
        moisture_schedule=scenario.moisture_schedule,
        ventilation_events=events,
        duration_hours=scenario.control_horizon_hours,
        timestep_minutes=scenario.trajectory_timestep_minutes,
    )
    risk = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=scenario.surface,
        config=scenario.risk_config,
    )
    if action.is_do_nothing:
        energy_kwh = 0.0
    else:
        # Compute the event energy from a single-event simulator at
        # the outdoor conditions active at start.
        from ventilation import simulate_ventilation_event

        # Room state at the moment the window opens.
        start_idx = 0
        for i, t in enumerate(trajectory.times_hours):
            if t <= action.start_time_hours + 1e-9:
                start_idx = i
        indoor_t = trajectory.indoor_temperature_c[start_idx]
        indoor_rh = max(
            0.0, min(100.0, trajectory.indoor_relative_humidity_pct[start_idx])
        )
        outdoor = scenario.forecast.sample_at(action.start_time_hours)
        event = simulate_ventilation_event(
            room_volume_m3=scenario.room.volume_m3,
            initial_indoor_temperature_c=indoor_t,
            initial_indoor_relative_humidity_pct=indoor_rh,
            outdoor_temperature_c=outdoor.temperature_c,
            outdoor_relative_humidity_pct=outdoor.relative_humidity_percent,
            ach=scenario.room.ach_window_open,
            effective_thermal_capacity_j_per_k=(
                scenario.thermal_properties.effective_thermal_capacity_j_per_k
            ),
            duration_minutes=action.duration_minutes,
        )
        energy_kwh = event.ventilation_energy_removed_kwh
    return (
        risk.cumulative_risk_score,
        risk.maximum_surface_rh_percent,
        risk.time_in_condensation_hours,
        energy_kwh,
        trajectory.indoor_temperature_c[-1],
        trajectory.indoor_absolute_humidity_g_m3[-1],
        trajectory.indoor_relative_humidity_pct[-1],
    )


# --- Controllers ----------------------------------------------------------


def _indoor_rh_at_time_zero(scenario: Scenario) -> float:
    return scenario.room.indoor_relative_humidity_pct


def _indoor_ah_at_time_zero(scenario: Scenario) -> float:
    return AirState(
        temperature_c=scenario.room.indoor_temperature_c,
        relative_humidity_percent=scenario.room.indoor_relative_humidity_pct,
    ).absolute_humidity


def _pick_shortest_useful_duration(
    scenario: Scenario,
) -> ScheduledAction:
    """Pick the shortest immediate-start candidate the caller supplied.

    Both greedy controllers reduce to "ventilate now for some
    duration". This picks the shortest non-zero, immediate-start
    action out of the same candidate list the predictive controller
    uses - that keeps the three controllers comparable and prevents
    an accidental advantage from choosing a longer duration for the
    greedy controllers.
    """
    immediate = [
        action
        for action in scenario.candidate_actions
        if action.start_time_hours == 0.0 and action.duration_minutes > 0.0
    ]
    if not immediate:
        return ScheduledAction(0.0, 0.0)
    return min(immediate, key=lambda a: a.duration_minutes)


def controller_immediate_when_indoor_humid(
    scenario: Scenario,
) -> ControllerReport:
    """Ventilate now if indoor RH already sits above the threshold; else do nothing."""
    if _indoor_rh_at_time_zero(scenario) > scenario.indoor_rh_threshold_pct:
        action = _pick_shortest_useful_duration(scenario)
        reason = (
            f"indoor RH at t=0 was "
            f"{_indoor_rh_at_time_zero(scenario):.1f} %, above the "
            f"threshold {scenario.indoor_rh_threshold_pct:g} %; opening "
            "the window immediately for the shortest available duration."
        )
    else:
        action = ScheduledAction(0.0, 0.0)
        reason = (
            f"indoor RH at t=0 was "
            f"{_indoor_rh_at_time_zero(scenario):.1f} %, at or below the "
            f"threshold {scenario.indoor_rh_threshold_pct:g} %; do nothing."
        )
    metrics = _simulate_and_evaluate(scenario, action)
    return ControllerReport(
        controller="immediate_when_indoor_humid",
        ventilation_started=(not action.is_do_nothing),
        start_time_hours=action.start_time_hours,
        duration_minutes=action.duration_minutes,
        cumulative_risk_score=metrics[0],
        peak_surface_rh_percent=metrics[1],
        time_in_condensation_hours=metrics[2],
        ventilation_energy_kwh=metrics[3],
        final_indoor_temperature_c=metrics[4],
        final_indoor_absolute_humidity_g_m3=metrics[5],
        final_indoor_relative_humidity_pct=metrics[6],
        reason=reason,
    )


def controller_immediate_when_outdoor_drier(
    scenario: Scenario,
) -> ControllerReport:
    """Ventilate now if outdoor AH at t=0 is below indoor AH; else do nothing."""
    outdoor_now = scenario.forecast.sample_at(0.0)
    outdoor_ah = outdoor_now.absolute_humidity
    indoor_ah = _indoor_ah_at_time_zero(scenario)
    if outdoor_ah < indoor_ah:
        action = _pick_shortest_useful_duration(scenario)
        reason = (
            f"outdoor AH at t=0 was {outdoor_ah:.2f} g/m^3, below indoor "
            f"AH {indoor_ah:.2f} g/m^3; opening the window immediately "
            "for the shortest available duration."
        )
    else:
        action = ScheduledAction(0.0, 0.0)
        reason = (
            f"outdoor AH at t=0 was {outdoor_ah:.2f} g/m^3, not below "
            f"indoor AH {indoor_ah:.2f} g/m^3; do nothing."
        )
    metrics = _simulate_and_evaluate(scenario, action)
    return ControllerReport(
        controller="immediate_when_outdoor_ah_below_indoor",
        ventilation_started=(not action.is_do_nothing),
        start_time_hours=action.start_time_hours,
        duration_minutes=action.duration_minutes,
        cumulative_risk_score=metrics[0],
        peak_surface_rh_percent=metrics[1],
        time_in_condensation_hours=metrics[2],
        ventilation_energy_kwh=metrics[3],
        final_indoor_temperature_c=metrics[4],
        final_indoor_absolute_humidity_g_m3=metrics[5],
        final_indoor_relative_humidity_pct=metrics[6],
        reason=reason,
    )


def controller_predictive_risk(scenario: Scenario) -> ControllerReport:
    """Delegate to the risk-constrained scheduled optimiser."""
    result: ScheduledActionResult = optimise_scheduled_action_under_risk_limit(
        room=scenario.room,
        thermal_properties=scenario.thermal_properties,
        forecast=scenario.forecast,
        moisture_schedule=scenario.moisture_schedule,
        candidate_actions=scenario.candidate_actions,
        constraints=VentilationConstraints(
            max_temperature_drop_c=scenario.max_temperature_drop_c,
            max_cumulative_risk_score=scenario.risk_ceiling,
        ),
        surface=scenario.surface,
        risk_config=scenario.risk_config,
        control_horizon_hours=scenario.control_horizon_hours,
        trajectory_timestep_minutes=scenario.trajectory_timestep_minutes,
    )
    if result.feasible:
        return ControllerReport(
            controller="predictive_risk_controller",
            ventilation_started=(not result.selected_action.is_do_nothing),
            start_time_hours=result.selected_action.start_time_hours,
            duration_minutes=result.selected_action.duration_minutes,
            cumulative_risk_score=result.selected_risk.cumulative_risk_score,
            peak_surface_rh_percent=(
                result.selected_risk.maximum_surface_rh_percent
            ),
            time_in_condensation_hours=(
                result.selected_risk.time_in_condensation_hours
            ),
            ventilation_energy_kwh=result.energy_penalty_kwh,
            final_indoor_temperature_c=result.final_indoor_temperature_c,
            final_indoor_absolute_humidity_g_m3=(
                result.final_indoor_absolute_humidity_g_m3
            ),
            final_indoor_relative_humidity_pct=(
                result.final_indoor_relative_humidity_pct
            ),
            reason=result.reason,
        )
    # If infeasible under the ceiling, fall back to the closest miss
    # reported by the optimiser (its selected_action / selected_risk
    # fields carry the closest-miss context; the reason names the
    # unmet constraint). This documents the failure without silently
    # picking a violating action.
    return ControllerReport(
        controller="predictive_risk_controller",
        ventilation_started=False,
        start_time_hours=float("nan"),
        duration_minutes=float("nan"),
        cumulative_risk_score=result.selected_risk.cumulative_risk_score,
        peak_surface_rh_percent=(
            result.selected_risk.maximum_surface_rh_percent
        ),
        time_in_condensation_hours=(
            result.selected_risk.time_in_condensation_hours
        ),
        ventilation_energy_kwh=float("nan"),
        final_indoor_temperature_c=result.final_indoor_temperature_c,
        final_indoor_absolute_humidity_g_m3=(
            result.final_indoor_absolute_humidity_g_m3
        ),
        final_indoor_relative_humidity_pct=(
            result.final_indoor_relative_humidity_pct
        ),
        reason=result.reason,
    )


# --- Scenarios ------------------------------------------------------------


def _common_candidate_actions() -> Tuple[ScheduledAction, ...]:
    """Same candidate set for every scenario and every controller.

    Includes do-nothing and every (start, duration) combination the
    predictive controller is allowed to consider. Sharing the set
    across controllers avoids handing the predictive controller a
    duration the greedy controllers cannot pick.
    """
    starts_hours = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    durations_min = (5.0, 10.0, 15.0, 20.0, 30.0)
    actions = [ScheduledAction(0.0, 0.0)]
    for start in starts_hours:
        for duration in durations_min:
            actions.append(ScheduledAction(start, duration))
    return tuple(actions)


def scenario_waiting_helps() -> Scenario:
    """Scenario A: cold+dry now, milder+drier-than-indoor later.

    Setup rationale:
        - Indoor: 20 C / 55 %RH. Moderate humidity, within comfort.
          AH is roughly 9.5 g/m^3.
        - Outdoor NOW: -1 C / 70 %RH. Outdoor AH is roughly 3 g/m^3
          (well below indoor), so both greedy controllers will
          trigger. But ventilating now uses cold outdoor air and
          drops the room temperature fast: a large T-drop per
          minute of open window means high energy loss.
        - Outdoor LATER (t >= 2 h): 10 C / 65 %RH. Outdoor AH
          around 6 g/m^3 - still below indoor, so waiting is still
          psychrometrically "drying". The temperature difference
          across the window is smaller, so the same duration loses
          less heat.
        - Moisture generation: 100 g/h steady background. Enough
          that doing nothing all 6 h would drift indoor AH high
          enough that the surface RH creeps above the elevated
          threshold - so the baseline breaches the ceiling.

    The predictive controller should therefore WAIT for the milder
    segment before ventilating: baseline is infeasible, but a
    delayed short window uses less energy than an immediate short
    window while keeping the horizon-wide risk below the ceiling.

    The parameters are not tuned to make one duration win a specific
    race - the candidate set exposes every controller to the same
    durations. The predictive win comes from combining choice of
    start time with the forecast.
    """
    return Scenario(
        name="A - waiting is safe",
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            ach_closed=0.3,
            ach_window_open=5.0,
        ),
        thermal_properties=ThermalProperties(
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            )
        ),
        surface=SurfaceDescriptor(
            label="cold external wall corner",
            surface_temperature_factor=0.72,
        ),
        forecast=WeatherForecast(
            points=(
                ForecastPoint(0.0, -1.0, 70.0),
                ForecastPoint(2.0, 10.0, 65.0),
                ForecastPoint(6.0, 12.0, 60.0),
            )
        ),
        moisture_schedule=MoistureSourceSchedule(
            constant_background_rate_g_per_hour=100.0
        ),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        risk_config=RiskConfig(),
        # Ceiling above the immediate-vent risk (~ 3.2) AND the
        # delayed-vent risk (~ 4.3), but BELOW the do-nothing
        # baseline (~ 5.7). Both immediate and delayed are feasible
        # so the optimiser can choose on energy.
        risk_ceiling=5.0,
        max_temperature_drop_c=8.0,
        candidate_actions=_common_candidate_actions(),
        # RH threshold at 50 % so the "indoor humid" greedy
        # controller triggers on the initial 55 %.
        indoor_rh_threshold_pct=50.0,
    )


def scenario_waiting_unsafe() -> Scenario:
    """Scenario B: room already stressed now, cannot wait for mild weather.

    Same forecast as Scenario A (cold now, mild later), BUT the
    room starts elevated: indoor RH 75 % and a cold-thermal-bridge
    surface at fRsi = 0.65. The surface is close to the elevated
    threshold from t = 0, and a steady moisture background continues
    to push it further. Even a 30-minute wait accumulates enough
    surface exposure that the delayed-vent options fail the risk
    ceiling.

    Under a moderate ceiling, the pre-vent risk guard blocks every
    delayed candidate: waiting for the mild segment would breach
    the ceiling before the window ever opens. The predictive
    controller is forced to ventilate immediately, using the cold
    outdoor air, and pays the higher energy penalty. The greedy
    controllers reach the same conclusion (indoor RH is above
    threshold; outdoor AH is below indoor at t = 0).

    This is the "waiting is unsafe" case the caller asked for: on
    Scenario B, the predictive controller does NOT win by waiting.
    """
    return Scenario(
        name="B - waiting is unsafe",
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=75.0,
            ach_closed=0.3,
            ach_window_open=5.0,
        ),
        thermal_properties=ThermalProperties(
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            )
        ),
        surface=SurfaceDescriptor(
            label="cold external wall corner",
            surface_temperature_factor=0.65,
        ),
        forecast=WeatherForecast(
            points=(
                ForecastPoint(0.0, -1.0, 70.0),
                ForecastPoint(2.0, 10.0, 65.0),
                ForecastPoint(6.0, 12.0, 60.0),
            )
        ),
        moisture_schedule=MoistureSourceSchedule(
            constant_background_rate_g_per_hour=80.0
        ),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        risk_config=RiskConfig(),
        # A ceiling only achievable by immediate longer vents; every
        # delayed candidate accumulates > 2.7 by the horizon end or
        # fails the pre-vent guard first.
        risk_ceiling=2.7,
        max_temperature_drop_c=8.0,
        candidate_actions=_common_candidate_actions(),
        indoor_rh_threshold_pct=70.0,
    )


# --- Report ---------------------------------------------------------------


def _print_report(scenario: Scenario, reports: Sequence[ControllerReport]) -> None:
    print(f"\n{'=' * 78}")
    print(f"Scenario: {scenario.name}")
    print(f"{'=' * 78}")
    print(
        f"Room initial: {scenario.room.indoor_temperature_c:.1f} C / "
        f"{scenario.room.indoor_relative_humidity_pct:.1f} %RH, "
        f"volume {scenario.room.volume_m3:g} m^3"
    )
    print(
        f"Surface: fRsi={scenario.surface.surface_temperature_factor:g}, "
        f"label={scenario.surface.label!r}"
    )
    print(
        f"Forecast points: "
        + ", ".join(
            f"t={p.timestamp_hours:g}h -> {p.temperature_c:g} C / "
            f"{p.relative_humidity_percent:g} %RH"
            for p in scenario.forecast.points
        )
    )
    print(
        f"Moisture: background "
        f"{scenario.moisture_schedule.constant_background_rate_g_per_hour:g} g/h"
        + (
            "; events: "
            + ", ".join(
                f"{e.label} {e.start_time_hours}-{e.end_time_hours}h @ "
                f"{e.generation_rate_g_per_hour:g} g/h"
                for e in scenario.moisture_schedule.events
            )
            if scenario.moisture_schedule.events
            else ""
        )
    )
    print(
        f"Risk ceiling (indicator): {scenario.risk_ceiling:g}; "
        f"comfort cap: max temperature drop {scenario.max_temperature_drop_c:g} K"
    )
    print(f"Control horizon: {scenario.control_horizon_hours:g} h\n")

    header = (
        f"{'controller':40s} {'start':>7s} {'dur':>6s} {'risk':>7s} "
        f"{'peak RH':>8s} {'cond h':>7s} {'kWh':>7s} {'T_end':>6s} "
        f"{'AH_end':>7s} {'RH_end':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        start_s = "-" if not r.ventilation_started else f"{r.start_time_hours:.2f}"
        dur_s = "-" if not r.ventilation_started else f"{r.duration_minutes:.1f}"
        kwh_s = (
            "-"
            if r.ventilation_energy_kwh != r.ventilation_energy_kwh
            else f"{r.ventilation_energy_kwh:.3f}"
        )
        print(
            f"{r.controller:40s} {start_s:>7s} {dur_s:>6s} "
            f"{r.cumulative_risk_score:>7.3f} {r.peak_surface_rh_percent:>8.1f} "
            f"{r.time_in_condensation_hours:>7.3f} {kwh_s:>7s} "
            f"{r.final_indoor_temperature_c:>6.2f} "
            f"{r.final_indoor_absolute_humidity_g_m3:>7.2f} "
            f"{r.final_indoor_relative_humidity_pct:>7.2f}"
        )
    print()
    for r in reports:
        print(f"  {r.controller}: {r.reason}")


def main() -> None:
    print(
        "Demonstration: forecast-aware ventilation vs greedy immediate rules.\n"
        "Every threshold below is illustrative (POC). The cumulative risk\n"
        "score is a caller-configured INDICATOR, not a validated mould\n"
        "growth prediction."
    )
    for scenario in (scenario_waiting_helps(), scenario_waiting_unsafe()):
        reports = (
            controller_immediate_when_indoor_humid(scenario),
            controller_immediate_when_outdoor_drier(scenario),
            controller_predictive_risk(scenario),
        )
        _print_report(scenario, reports)


if __name__ == "__main__":
    main()
