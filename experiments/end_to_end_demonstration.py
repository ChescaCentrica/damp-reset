"""End-to-end synthetic demonstration of the damp-reset POC pipeline.

Chains every layer the POC has built:

    * indoor sensor state
       -> psychrometric conversion (AirState) -> surface temperature +
          surface RH + condensation margin from the fRsi model.
    * a moisture-generation schedule
       (illustrative shower + steady background)
    * a future outdoor weather forecast
       (piecewise-constant, from weather_forecast.WeatherForecast)
    * caller-supplied "calibrated" room parameters
       (ACH and effective thermal capacity - in a real deployment
       these would come from calibration + thermal_calibration on
       measurements, per HYGROSCOPIC_BUFFERING_DESIGN_ASSESSMENT.md
       and MODEL_EVIDENCE_AND_LIMITATIONS.md)
    * the risk-constrained scheduled optimiser
       (optimise_scheduled_action_under_risk_limit) - chooses BOTH
       when and how long to ventilate
    * the heating-aware simulator
       (simulate_room_period_with_heating) - reports thermal energy
       supplied and input energy the occupant purchases for the
       selected action

The output is a plain-text narrative in the shape the caller
sketched:

    Current state: <T, RH, critical surface T, surface RH, margin>
    Without action: risk score, when limit crossed
    Recommendation: wait until HH:MM, open N min
    Why: <one-sentence reason from the optimiser>
    Predicted result: surface risk, temp drop, heating supplied +
                      purchased

Every threshold below is a caller POC assumption. See
MODEL_EVIDENCE_AND_LIMITATIONS.md for the full stocktake of what is
validated vs POC placeholder. This demo does NOT claim to prevent
mould; the "risk score" is a caller-configured indicator, not a
mould-growth prediction. See mould_risk.py module docstring.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heating import ThermostaticHeating
from moisture import Room
from moisture_sources import MoistureSourceSchedule
from mould_risk import RiskConfig, evaluate_moisture_risk
from optimiser import (
    ScheduledAction,
    ScheduledActionResult,
    VentilationConstraints,
    optimise_scheduled_action_under_risk_limit,
)
from psychrometrics import (
    AirState,
    G_PER_KG,
    M_WATER,
    R_UNIVERSAL,
    ZERO_CELSIUS_IN_KELVIN,
    saturation_vapour_pressure,
)
from surface_risk import SurfaceDescriptor, assess_surface, surface_temperature_c
from thermal import ThermalProperties
from time_simulation import (
    VentilationEvent,
    simulate_room_period_with_heating,
)
from weather_forecast import ForecastPoint, WeatherForecast


@dataclass(frozen=True)
class DemoInputs:
    """Every knob the demonstration exposes.

    Split from the physics-side value objects so the reader can see
    the caller's POC assumptions in one place before the pipeline
    runs.
    """

    # ---- Wall-clock context (used only for the human-readable output) ----
    current_wall_clock: str  # e.g. "08:00"; hours since this point index into the forecast.
    # ---- Current sensor state ----
    indoor_temperature_c: float
    indoor_relative_humidity_pct: float
    # ---- Room / calibrated parameters ----
    room_volume_m3: float
    ach_closed: float
    ach_window_open: float
    effective_thermal_capacity_j_per_k: float
    # ---- Critical surface ----
    surface_label: str
    surface_temperature_factor: float
    # ---- Moisture schedule over the horizon ----
    background_moisture_generation_g_per_hour: float
    scheduled_events: tuple  # tuple of MoistureSourceEvent
    # ---- Future outdoor weather ----
    forecast: WeatherForecast
    # ---- Optimiser constraints ----
    control_horizon_hours: float
    trajectory_timestep_minutes: float
    elevated_surface_rh_threshold_percent: float
    condensation_surface_rh_threshold_percent: float
    max_cumulative_risk_score: float
    max_temperature_drop_c: float
    # ---- Heating configuration ----
    heating_setpoint_c: float
    heating_max_thermal_power_w: float
    heating_efficiency_or_cop: float
    heating_hysteresis_c: float
    heating_appliance_label: str


def scenario() -> DemoInputs:
    """Illustrative POC scenario: cold-now, milder-later winter morning.

    Chosen so the wait-then-vent branch of the optimiser fires,
    because that is the interesting case for the reader. Not
    engineered to make the optimiser look good vs a strawman - the
    scenario simply exposes the fact that the same short vent is
    cheaper 3 h from now than immediately.
    """
    return DemoInputs(
        current_wall_clock="08:00",
        indoor_temperature_c=20.2,
        indoor_relative_humidity_pct=71.0,  # a shower has just finished
        room_volume_m3=40.0,
        ach_closed=0.3,
        ach_window_open=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        surface_label="cold external wall corner (kitchen)",
        surface_temperature_factor=0.72,
        background_moisture_generation_g_per_hour=80.0,
        scheduled_events=(),  # shower already ended before t = 0
        forecast=WeatherForecast(
            points=(
                ForecastPoint(0.0, -1.0, 70.0),  # cold + damp NOW
                ForecastPoint(0.5, 4.0, 65.0),   # mild spell arrives quickly
                ForecastPoint(2.0, 8.0, 60.0),   # milder still
                ForecastPoint(4.0, 10.0, 60.0),
                ForecastPoint(6.0, 12.0, 60.0),
            )
        ),
        control_horizon_hours=6.0,
        trajectory_timestep_minutes=5.0,
        elevated_surface_rh_threshold_percent=80.0,
        condensation_surface_rh_threshold_percent=100.0,
        max_cumulative_risk_score=3.0,
        max_temperature_drop_c=3.0,
        heating_setpoint_c=20.0,
        heating_max_thermal_power_w=2000.0,
        heating_efficiency_or_cop=3.0,  # heat pump COP - see docs
        heating_hysteresis_c=0.5,
        heating_appliance_label="heat pump (COP 3.0)",
    )


def _room_from(inputs: DemoInputs) -> Room:
    return Room(
        volume_m3=inputs.room_volume_m3,
        indoor_temperature_c=inputs.indoor_temperature_c,
        indoor_relative_humidity_pct=inputs.indoor_relative_humidity_pct,
        ach_closed=inputs.ach_closed,
        ach_window_open=inputs.ach_window_open,
    )


def _thermal_from(inputs: DemoInputs) -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            inputs.effective_thermal_capacity_j_per_k
        )
    )


def _surface_from(inputs: DemoInputs) -> SurfaceDescriptor:
    return SurfaceDescriptor(
        label=inputs.surface_label,
        surface_temperature_factor=inputs.surface_temperature_factor,
    )


def _schedule_from(inputs: DemoInputs) -> MoistureSourceSchedule:
    return MoistureSourceSchedule(
        constant_background_rate_g_per_hour=(
            inputs.background_moisture_generation_g_per_hour
        ),
        events=inputs.scheduled_events,
    )


def _risk_config_from(inputs: DemoInputs) -> RiskConfig:
    return RiskConfig(
        elevated_surface_rh_threshold_percent=(
            inputs.elevated_surface_rh_threshold_percent
        ),
        condensation_surface_rh_threshold_percent=(
            inputs.condensation_surface_rh_threshold_percent
        ),
    )


def _candidate_actions(inputs: DemoInputs) -> tuple:
    """Every (start_hours, duration_minutes) pair the optimiser considers.

    Do-nothing is always included. Non-zero starts step through the
    horizon in 30-min increments; durations span a residential vent
    range from 5 to 30 minutes. No caller-side sub-selection so the
    optimiser sees the full grid.
    """
    actions = [ScheduledAction(0.0, 0.0)]
    starts_hours = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5)
    durations_min = (5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
    for start in starts_hours:
        for duration in durations_min:
            if start + duration / 60.0 <= inputs.control_horizon_hours:
                actions.append(ScheduledAction(start, duration))
    return tuple(actions)


def _time_of_day_from_offset_hours(current_wall_clock: str, offset_hours: float) -> str:
    """Turn 'HH:MM' + hours offset into 'HH:MM'. No date handling."""
    hh, mm = current_wall_clock.split(":")
    total_minutes = int(hh) * 60 + int(mm) + int(round(offset_hours * 60.0))
    total_minutes = total_minutes % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _surface_state_now(
    indoor_t_c: float,
    outdoor_t_c: float,
    indoor_rh_pct: float,
    surface: SurfaceDescriptor,
) -> tuple:
    """Return (surface_T, surface_RH, condensation_margin_C) at t = 0."""
    result = assess_surface(
        indoor_air_state=AirState(
            temperature_c=indoor_t_c,
            relative_humidity_percent=indoor_rh_pct,
        ),
        outdoor_temperature_c=outdoor_t_c,
        surface=surface,
    )
    return (
        result.surface_temperature_c,
        result.surface_relative_humidity_pct,
        result.condensation_margin_c,
    )


def _find_time_of_threshold_crossing(
    trajectory,
    surface: SurfaceDescriptor,
    threshold_pct: float,
) -> Optional[float]:
    """First time (in hours) that surface RH crosses the elevated threshold.

    Uses the same ideal-gas P_v inversion mould_risk does, so a
    supersaturated moment (surface RH > 100) is still reported.
    Returns None if the threshold is never crossed within the
    trajectory.
    """
    times = trajectory.times_hours
    for i in range(len(times)):
        indoor_ah = trajectory.indoor_absolute_humidity_g_m3[i]
        indoor_t = trajectory.indoor_temperature_c[i]
        outdoor_t = trajectory.outdoor_temperature_c[i]
        indoor_p_v_pa = (
            (indoor_ah / G_PER_KG)
            * R_UNIVERSAL
            * (indoor_t + ZERO_CELSIUS_IN_KELVIN)
            / M_WATER
        )
        t_surface = surface_temperature_c(
            indoor_temperature_c=indoor_t,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        )
        surface_rh = 100.0 * indoor_p_v_pa / saturation_vapour_pressure(t_surface)
        if surface_rh > threshold_pct:
            return times[i]
    return None


def _predict_with_heating(
    inputs: DemoInputs,
    action: ScheduledAction,
):
    """Run the heating-aware simulator on the selected action."""
    heating_model = ThermostaticHeating(
        setpoint_temperature_c=inputs.heating_setpoint_c,
        max_thermal_power_w=inputs.heating_max_thermal_power_w,
        efficiency_or_cop=inputs.heating_efficiency_or_cop,
        hysteresis_c=inputs.heating_hysteresis_c,
    )
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
        room=_room_from(inputs),
        thermal_properties=_thermal_from(inputs),
        forecast=inputs.forecast,
        moisture_schedule=_schedule_from(inputs),
        ventilation_events=events,
        heating_model=heating_model,
        duration_hours=inputs.control_horizon_hours,
        timestep_minutes=inputs.trajectory_timestep_minutes,
    )


def _temperature_drop_during_action(traj, action: ScheduledAction) -> float:
    """Indoor T at start of vent minus indoor T at end of vent."""
    times = traj.trajectory.times_hours
    temperatures = traj.trajectory.indoor_temperature_c
    start_idx = 0
    end_idx = 0
    end_time = action.start_time_hours + action.duration_minutes / 60.0
    for i, t in enumerate(times):
        if t <= action.start_time_hours + 1e-9:
            start_idx = i
        if t <= end_time + 1e-9:
            end_idx = i
    return temperatures[start_idx] - temperatures[end_idx]


def _print_line(label: str, value: str, indent: int = 0) -> None:
    prefix = " " * indent
    print(f"{prefix}{label:<38s} {value}")


def _print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_poc_banner() -> None:
    print(
        "-" * 78 + "\n"
        "End-to-end damp-reset POC demonstration\n"
        "All thresholds and setpoints below are POC ASSUMPTIONS, not\n"
        "validated values. The 'cumulative risk score' is a caller-\n"
        "configured INDICATOR, NOT a mould-growth prediction. See\n"
        "MODEL_EVIDENCE_AND_LIMITATIONS.md and mould_risk.py for the\n"
        "full disclaimers.\n"
        + "-" * 78
    )


def run() -> None:
    inputs = scenario()

    _print_poc_banner()

    # --- Section: caller inputs ---
    _print_section("Caller inputs (POC assumptions)")
    _print_line("Room volume:", f"{inputs.room_volume_m3:g} m^3")
    _print_line(
        "Calibrated ACH (window closed / open):",
        f"{inputs.ach_closed:g} / {inputs.ach_window_open:g} h^-1  "
        "[POC placeholder; run calibration.py on real event]",
    )
    _print_line(
        "Calibrated effective thermal capacity:",
        f"{inputs.effective_thermal_capacity_j_per_k:.0f} J/K  "
        "[POC placeholder; run thermal_calibration.py]",
    )
    _print_line(
        "Critical surface:",
        f'"{inputs.surface_label}", fRsi={inputs.surface_temperature_factor:g}  '
        "[POC placeholder]",
    )
    _print_line(
        "Background moisture generation:",
        f"{inputs.background_moisture_generation_g_per_hour:g} g/h  "
        "[POC placeholder]",
    )
    _print_line(
        "Elevated surface RH threshold:",
        f"{inputs.elevated_surface_rh_threshold_percent:g} %  "
        "[POC placeholder; NOT a mould-growth threshold]",
    )
    _print_line(
        "Max cumulative risk indicator:",
        f"{inputs.max_cumulative_risk_score:g}  "
        "[POC placeholder]",
    )
    _print_line(
        "Comfort cap (max temp drop):",
        f"{inputs.max_temperature_drop_c:g} K  [POC placeholder]",
    )
    _print_line(
        "Heating appliance model:",
        f"{inputs.heating_appliance_label}, "
        f"setpoint {inputs.heating_setpoint_c:g} C, "
        f"max thermal power {inputs.heating_max_thermal_power_w:g} W  "
        "[POC placeholder]",
    )
    _print_line(
        "Control horizon:",
        f"{inputs.control_horizon_hours:g} h",
    )

    # --- Section: current state ---
    outdoor_now = inputs.forecast.sample_at(0.0)
    surface = _surface_from(inputs)
    surface_t_now, surface_rh_now, cond_margin_now = _surface_state_now(
        indoor_t_c=inputs.indoor_temperature_c,
        outdoor_t_c=outdoor_now.temperature_c,
        indoor_rh_pct=inputs.indoor_relative_humidity_pct,
        surface=surface,
    )
    indoor_ah_now = AirState(
        temperature_c=inputs.indoor_temperature_c,
        relative_humidity_percent=inputs.indoor_relative_humidity_pct,
    ).absolute_humidity

    _print_section(f"Current state (wall clock {inputs.current_wall_clock})")
    _print_line(
        "Indoor:",
        f"{inputs.indoor_temperature_c:.1f} C  "
        f"{inputs.indoor_relative_humidity_pct:.1f} %RH  "
        f"(AH = {indoor_ah_now:.2f} g/m^3)",
    )
    _print_line(
        "Outdoor now:",
        f"{outdoor_now.temperature_c:.1f} C  "
        f"{outdoor_now.relative_humidity_percent:.1f} %RH  "
        f"(AH = {outdoor_now.absolute_humidity:.2f} g/m^3)",
    )
    _print_line("Critical surface T (from fRsi):", f"{surface_t_now:.2f} C")
    _print_line(
        "Estimated surface RH at this surface:",
        f"{surface_rh_now:.1f} %",
    )
    margin_word = "clear" if cond_margin_now > 0 else "AT OR BELOW dew point"
    _print_line(
        "Condensation margin (T_surface - T_dew):",
        f"{cond_margin_now:+.2f} K  ({margin_word})",
    )

    # --- Section: without-action forecast ---
    do_nothing_traj = _predict_with_heating(inputs, ScheduledAction(0.0, 0.0))
    do_nothing_risk = evaluate_moisture_risk(
        trajectory=do_nothing_traj.trajectory,
        surface=surface,
        config=_risk_config_from(inputs),
    )
    threshold_crossing_h = _find_time_of_threshold_crossing(
        trajectory=do_nothing_traj.trajectory,
        surface=surface,
        threshold_pct=inputs.elevated_surface_rh_threshold_percent,
    )

    _print_section("Without any action (baseline projection)")
    _print_line(
        "Predicted baseline risk score over 6 h:",
        f"{do_nothing_risk.cumulative_risk_score:.3f}  "
        "(configured limit "
        f"{inputs.max_cumulative_risk_score:g})",
    )
    if threshold_crossing_h is None:
        _print_line(
            "Surface RH crosses elevated threshold:",
            "not within the 6 h horizon.",
        )
    else:
        crossing_wall = _time_of_day_from_offset_hours(
            inputs.current_wall_clock, threshold_crossing_h
        )
        _print_line(
            "Surface RH crosses elevated threshold at:",
            f"{crossing_wall} "
            f"(~{threshold_crossing_h:.1f} h from now)",
        )
    if (
        do_nothing_risk.cumulative_risk_score
        > inputs.max_cumulative_risk_score
    ):
        _print_line(
            "Verdict:",
            "baseline risk EXCEEDS the configured limit; action required.",
        )
    else:
        _print_line(
            "Verdict:",
            "baseline risk stays within the configured limit; do-nothing "
            "is acceptable.",
        )

    # --- Section: run the optimiser (heating-aware) ---
    candidate_actions = _candidate_actions(inputs)
    heating_model = ThermostaticHeating(
        setpoint_temperature_c=inputs.heating_setpoint_c,
        max_thermal_power_w=inputs.heating_max_thermal_power_w,
        efficiency_or_cop=inputs.heating_efficiency_or_cop,
        hysteresis_c=inputs.heating_hysteresis_c,
    )
    result: ScheduledActionResult = optimise_scheduled_action_under_risk_limit(
        room=_room_from(inputs),
        thermal_properties=_thermal_from(inputs),
        forecast=inputs.forecast,
        moisture_schedule=_schedule_from(inputs),
        candidate_actions=candidate_actions,
        constraints=VentilationConstraints(
            max_temperature_drop_c=inputs.max_temperature_drop_c,
            max_cumulative_risk_score=inputs.max_cumulative_risk_score,
        ),
        surface=surface,
        risk_config=_risk_config_from(inputs),
        control_horizon_hours=inputs.control_horizon_hours,
        trajectory_timestep_minutes=inputs.trajectory_timestep_minutes,
        heating_model=heating_model,
    )

    _print_section("Recommendation")
    if not result.feasible:
        _print_line(
            "Optimiser:",
            "no candidate satisfies every constraint.",
        )
        _print_line("Reason:", result.reason)
        return

    action = result.selected_action
    if action.is_do_nothing:
        _print_line("Action:", "do nothing over the 6 h horizon.")
        _print_line("Reason:", result.reason)
    else:
        start_wall = _time_of_day_from_offset_hours(
            inputs.current_wall_clock, action.start_time_hours
        )
        end_wall = _time_of_day_from_offset_hours(
            inputs.current_wall_clock,
            action.start_time_hours + action.duration_minutes / 60.0,
        )
        if action.start_time_hours == 0.0:
            _print_line(
                "Action:",
                f"open the window IMMEDIATELY for "
                f"{action.duration_minutes:.0f} min "
                f"(until {end_wall}).",
            )
        else:
            _print_line(
                "Action:",
                f"WAIT {action.start_time_hours:g} h (until {start_wall}), "
                f"then open the window for "
                f"{action.duration_minutes:.0f} min "
                f"(until {end_wall}).",
            )

        # Explain WHY the wait was safe (or immediate action required).
        # Each bullet is emitted only when the underlying fact holds
        # in the actual predicted trajectory - no boilerplate.
        why_lines = []
        outdoor_at_start = inputs.forecast.sample_at(action.start_time_hours)
        indoor_ah_g_m3 = indoor_ah_now
        outdoor_ah_at_start = outdoor_at_start.absolute_humidity
        if outdoor_ah_at_start < indoor_ah_g_m3:
            why_lines.append(
                f"outdoor AH at {start_wall} "
                f"({outdoor_ah_at_start:.2f} g/m^3) is below indoor AH "
                f"({indoor_ah_g_m3:.2f} g/m^3), so ventilating then still "
                "removes moisture"
            )
        if (
            action.start_time_hours > 0.0
            and outdoor_at_start.temperature_c > outdoor_now.temperature_c + 1e-6
        ):
            why_lines.append(
                f"outdoor T at {start_wall} "
                f"({outdoor_at_start.temperature_c:.1f} C) is warmer than "
                f"outdoor T now ({outdoor_now.temperature_c:.1f} C), so the "
                "same window length loses less heat later"
            )
        if action.start_time_hours > 0.0:
            pre_score = result.pre_action_risk.cumulative_risk_score
            why_lines.append(
                "the pre-vent risk accrued while waiting does not breach "
                "the configured surface risk limit before the window opens "
                f"(pre-action score {pre_score:.3f} <= "
                f"{inputs.max_cumulative_risk_score:g})"
            )
            why_lines.append(
                "among candidates that satisfy both the risk limit and the "
                "comfort cap, this (start, duration) has the lowest predicted "
                "ventilation-event energy penalty"
            )
        if action.start_time_hours == 0.0:
            why_lines.append(
                "immediate ventilation is the lowest-energy candidate that "
                "keeps the horizon-wide risk score within the configured limit"
            )
        _print_line("Why:", "")
        for line in why_lines:
            print(f"    - {line}")

    # --- Section: predicted result ---
    # Numbers here come DIRECTLY from the optimiser's result object.
    # No re-simulation is needed: the optimiser evaluated every
    # candidate on the same heating-aware trajectory a caller would
    # produce. Regression tests in
    # test/test_optimiser_scheduled_heating_consistency.py prove the
    # equality.
    #
    # We still call _predict_with_heating(inputs, action) for the
    # per-step "indoor T drop during vent" number, because that
    # metric isn't stored on the result. It is derived from the
    # same trajectory shape.
    with_action_traj = _predict_with_heating(inputs, action)
    if action.is_do_nothing:
        temp_drop_during_action = 0.0
    else:
        temp_drop_during_action = _temperature_drop_during_action(
            with_action_traj, action
        )
    _print_section("Predicted result under the recommended action")
    _print_line(
        "Cumulative risk score:",
        f"{result.selected_risk.cumulative_risk_score:.3f}   "
        f"(baseline: "
        f"{result.baseline_risk.cumulative_risk_score:.3f}; "
        f"limit {inputs.max_cumulative_risk_score:g})",
    )
    _print_line(
        "Risk reduction vs baseline:",
        f"{result.risk_reduction:+.3f}",
    )
    _print_line(
        "Peak surface RH:",
        f"{result.selected_risk.maximum_surface_rh_percent:.1f} %",
    )
    _print_line(
        "Time in condensation:",
        f"{result.selected_risk.time_in_condensation_hours:.2f} h",
    )
    _print_line(
        "Condensation time reduction vs baseline:",
        f"{result.condensation_time_reduction_hours:+.2f} h",
    )
    _print_line(
        "Indoor T drop during vent:",
        f"{temp_drop_during_action:.2f} K   "
        f"(comfort cap {inputs.max_temperature_drop_c:g} K)",
    )
    _print_line(
        "Ventilation heat removed (thermal):",
        f"{result.energy_penalty_kwh:.3f} kWh",
    )
    print()
    _print_line(
        "Baseline heating thermal supplied:",
        f"{result.baseline_heating_thermal_energy_supplied_kwh:.3f} kWh",
    )
    _print_line(
        "Baseline heating INPUT purchased:",
        f"{result.baseline_heating_input_energy_purchased_kwh:.3f} kWh   "
        "(what the occupant would have paid under do-nothing)",
    )
    _print_line(
        "Action heating thermal supplied:",
        f"{result.heating_thermal_energy_supplied_kwh:.3f} kWh",
    )
    _print_line(
        "Action heating INPUT purchased:",
        f"{result.heating_input_energy_purchased_kwh:.3f} kWh   "
        f"({inputs.heating_appliance_label})",
    )
    _print_line(
        "INCREMENTAL thermal delivered by action:",
        f"{result.incremental_heating_thermal_energy_supplied_kwh:+.3f} kWh",
    )
    _print_line(
        "INCREMENTAL purchased energy for action:",
        f"{result.incremental_heating_input_energy_purchased_kwh:+.3f} kWh   "
        "(action - baseline; the extra bill attributable to the vent)",
    )
    print()
    _print_line(
        "Final indoor state (end of horizon):",
        f"{result.final_indoor_temperature_c:.2f} C  "
        f"{result.final_indoor_relative_humidity_pct:.1f} %RH  "
        f"(AH {result.final_indoor_absolute_humidity_g_m3:.2f} g/m^3)",
    )

    print()
    print("REMINDER: every threshold above is a caller POC assumption. The")
    print("'cumulative risk score' is an indicator of sustained surface")
    print("exposure, not a mould-growth prediction. Before any real")
    print("deployment: replace every POC value with a defended one, run")
    print("calibration and thermal_calibration on measured events, and run")
    print("validation on a held-out event. See MODEL_EVIDENCE_AND_LIMITATIONS.md.")


if __name__ == "__main__":
    run()
