"""24-hour synthetic scenario comparing unventilated vs simple-ventilated risk.

Purpose: demonstrate that ventilation actions influence the
sustained surface moisture / condensation risk indicator - not
just the room's final AH. Runs the same synthetic 24-hour room
TWICE:

    1. No ventilation. Background moisture + morning shower +
       evening cooking + changing outdoor T and RH throughout the
       day. Nothing opens a window.

    2. Simple fixed ventilation strategy. Two short window-open
       events immediately after the biggest indoor moisture
       events.

For each case the script plots:
    room RH vs time
    indoor AH vs time
    surface temperature vs time
    surface RH vs time
    condensation margin vs time
    cumulative risk score vs time (integrated up to each moment)

Then prints a comparison table:
    total ventilation time
    total energy loss
    time in high surface moisture (surface RH > threshold)
    time in predicted condensation (surface RH >= 100%)

All rates, thresholds, and weights are ILLUSTRATIVE POC values.
This experiment does not optimise anything and does not claim any
threshold predicts mould growth. Its point is to show that a
ventilation decision moves the risk-indicator column, not just the
room-air column.

Saved to outputs/plots/risk_metric_*.png.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from moisture_sources import (
    MoistureSourceEvent,
    MoistureSourceSchedule,
)
from mould_risk import RiskConfig, evaluate_moisture_risk
from psychrometrics import AirState
from surface_risk import (
    SurfaceDescriptor,
    surface_temperature_c,
)
from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    ThermalProperties,
    ventilation_heat_loss_power,
)
from time_simulation import (
    RoomTrajectory,
    VentilationEvent,
    simulate_room_period,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"


# --- Time-varying outdoor conditions --------------------------------------
# The existing time_simulation module takes a CONSTANT outdoor
# AirState (see architecture proposal - weather.py is a separate
# future slice). To model time-varying outdoor conditions here we
# split the day into segments and run separate simulations back to
# back, stitching the results.


@dataclass(frozen=True)
class OutdoorSegment:
    start_time_hours: float
    end_time_hours: float
    outdoor: AirState


def _piecewise_outdoor_schedule() -> Tuple[OutdoorSegment, ...]:
    """Illustrative diurnal outdoor pattern.

    Cool damp night, mild morning rise, warmer drier afternoon,
    cool damp evening. Values chosen so the risk indicator has
    visibly different behaviour under the two strategies; NOT
    calibrated to any specific weather station.
    """
    return (
        OutdoorSegment(0.0, 6.0, AirState(3.0, 92.0)),
        OutdoorSegment(6.0, 12.0, AirState(8.0, 82.0)),
        OutdoorSegment(12.0, 18.0, AirState(14.0, 55.0)),
        OutdoorSegment(18.0, 24.0, AirState(6.0, 85.0)),
    )


def _run_piecewise_simulation(
    initial_room: Room,
    thermal_properties: ThermalProperties,
    outdoor_segments: Sequence[OutdoorSegment],
    moisture_schedule: MoistureSourceSchedule,
    ventilation_events: Sequence[VentilationEvent],
    timestep_minutes: float,
) -> RoomTrajectory:
    """Run simulate_room_period across each outdoor segment in turn.

    Each segment inherits the previous segment's ending indoor
    state as its new initial state, threading a piecewise-constant
    outdoor schedule through the existing constant-outdoor
    simulator. The moisture and ventilation schedules are passed
    intact to each segment - they use their own absolute-time
    boundaries.
    """
    stitched_times = []
    stitched_indoor_t = []
    stitched_indoor_ah = []
    stitched_indoor_rh = []
    stitched_outdoor_t = []
    stitched_outdoor_ah = []
    stitched_window = []
    stitched_source = []

    current_room = initial_room
    for i, segment in enumerate(outdoor_segments):
        # Ventilation events must be filtered / shifted to segment-local
        # coordinates. simulate_room_period uses times relative to its
        # own t=0. Shift starts and ends by -segment.start_time_hours;
        # keep only events that overlap this segment.
        segment_events = []
        for event in ventilation_events:
            overlap_start = max(event.start_time_hours, segment.start_time_hours)
            overlap_end = min(event.end_time_hours, segment.end_time_hours)
            if overlap_end > overlap_start:
                segment_events.append(
                    VentilationEvent(
                        start_time_hours=overlap_start - segment.start_time_hours,
                        end_time_hours=overlap_end - segment.start_time_hours,
                    )
                )
        segment_schedule = _shift_moisture_schedule(
            moisture_schedule,
            offset_hours=-segment.start_time_hours,
            segment_length_hours=segment.end_time_hours - segment.start_time_hours,
        )

        segment_duration = segment.end_time_hours - segment.start_time_hours
        segment_trajectory = simulate_room_period(
            room=current_room,
            thermal_properties=thermal_properties,
            outdoor=segment.outdoor,
            moisture_schedule=segment_schedule,
            ventilation_events=tuple(segment_events),
            duration_hours=segment_duration,
            timestep_minutes=timestep_minutes,
        )

        # Append segment samples to the stitched trajectory. Skip
        # the segment's first sample on subsequent segments to
        # avoid duplicating the boundary point.
        start_index = 0 if i == 0 else 1
        for j in range(start_index, len(segment_trajectory.times_hours)):
            absolute_time = (
                segment.start_time_hours + segment_trajectory.times_hours[j]
            )
            stitched_times.append(absolute_time)
            stitched_indoor_t.append(segment_trajectory.indoor_temperature_c[j])
            stitched_indoor_ah.append(
                segment_trajectory.indoor_absolute_humidity_g_m3[j]
            )
            stitched_indoor_rh.append(
                segment_trajectory.indoor_relative_humidity_pct[j]
            )
            stitched_outdoor_t.append(
                segment_trajectory.outdoor_temperature_c[j]
            )
            stitched_outdoor_ah.append(
                segment_trajectory.outdoor_absolute_humidity_g_m3[j]
            )
            stitched_window.append(segment_trajectory.window_open[j])
            stitched_source.append(
                segment_trajectory.moisture_generation_g_per_hour[j]
            )

        # Prepare the next segment's initial room.
        last_t = segment_trajectory.indoor_temperature_c[-1]
        # RH from the last sample; converts AirState back into a
        # Room for the next segment.
        last_rh = segment_trajectory.indoor_relative_humidity_pct[-1]
        current_room = Room(
            volume_m3=initial_room.volume_m3,
            indoor_temperature_c=last_t,
            indoor_relative_humidity_pct=max(0.0, min(100.0, last_rh)),
            ach_closed=initial_room.ach_closed,
            ach_window_open=initial_room.ach_window_open,
        )

    return RoomTrajectory(
        times_hours=tuple(stitched_times),
        indoor_temperature_c=tuple(stitched_indoor_t),
        indoor_absolute_humidity_g_m3=tuple(stitched_indoor_ah),
        indoor_relative_humidity_pct=tuple(stitched_indoor_rh),
        outdoor_temperature_c=tuple(stitched_outdoor_t),
        outdoor_absolute_humidity_g_m3=tuple(stitched_outdoor_ah),
        window_open=tuple(stitched_window),
        moisture_generation_g_per_hour=tuple(stitched_source),
    )


def _shift_moisture_schedule(
    schedule: MoistureSourceSchedule,
    offset_hours: float,
    segment_length_hours: float,
) -> MoistureSourceSchedule:
    """Return a schedule expressed in segment-local time.

    Events wholly outside the segment are dropped; events that
    overlap are clipped to the segment interval.
    """
    shifted_events = []
    for event in schedule.events:
        overlap_start = max(event.start_time_hours + offset_hours, 0.0)
        overlap_end = min(
            event.end_time_hours + offset_hours, segment_length_hours
        )
        if overlap_end > overlap_start:
            shifted_events.append(
                MoistureSourceEvent(
                    label=event.label,
                    start_time_hours=overlap_start,
                    end_time_hours=overlap_end,
                    generation_rate_g_per_hour=event.generation_rate_g_per_hour,
                )
            )
    return MoistureSourceSchedule(
        constant_background_rate_g_per_hour=schedule.constant_background_rate_g_per_hour,
        events=tuple(shifted_events),
    )


# --- Metric derivations ---------------------------------------------------


def _surface_series(
    trajectory: RoomTrajectory,
    surface: SurfaceDescriptor,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
    """Return (T_surface_c, surface_rh_pct, condensation_margin_c) tuples.

    Uses the indoor absolute humidity directly rather than routing
    through ``AirState(T, RH)``, because during a large indoor
    moisture event the trajectory can show RH > 100 % (physically
    supersaturated / dew forming), which ``AirState`` rightly
    rejects at construction. Computing indoor P_v via the ideal-
    gas relation from indoor AH avoids that boundary while still
    reusing the psychrometric ``saturation_vapour_pressure`` and
    ``dew_point_c`` helpers.
    """
    from math import log
    from psychrometrics import (
        M_WATER,
        R_UNIVERSAL,
        ZERO_CELSIUS_IN_KELVIN,
        G_PER_KG,
        MAGNUS_A,
        MAGNUS_B,
        P_SAT_0,
        saturation_vapour_pressure,
    )

    t_surface = []
    rh_surface = []
    margin = []
    for i in range(len(trajectory.times_hours)):
        indoor_t = trajectory.indoor_temperature_c[i]
        indoor_ah = trajectory.indoor_absolute_humidity_g_m3[i]
        outdoor_t = trajectory.outdoor_temperature_c[i]

        # Indoor P_v from indoor AH and indoor T via ideal gas.
        indoor_p_v_pa = (
            (indoor_ah / G_PER_KG)
            * R_UNIVERSAL
            * (indoor_t + ZERO_CELSIUS_IN_KELVIN)
            / M_WATER
        )

        t_s = surface_temperature_c(
            indoor_temperature_c=indoor_t,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        )
        t_surface.append(t_s)

        # Surface RH = 100 * P_v_indoor / P_sat(T_surface).
        rh_surface.append(
            100.0 * indoor_p_v_pa / saturation_vapour_pressure(t_s)
        )

        # Indoor dew point from indoor P_v via the analytic Magnus
        # inverse (same formula as psychrometrics.dew_point_c, but
        # applied on P_v directly so it works above saturation).
        alpha = log(indoor_p_v_pa / P_SAT_0)
        indoor_dew_point = MAGNUS_B * alpha / (MAGNUS_A - alpha)
        margin.append(t_s - indoor_dew_point)
    return tuple(t_surface), tuple(rh_surface), tuple(margin)


def _cumulative_risk_score(
    trajectory: RoomTrajectory,
    surface: SurfaceDescriptor,
    config: RiskConfig,
) -> Tuple[float, ...]:
    """Cumulative risk score up to each sample.

    Uses the same accumulation convention as
    ``mould_risk.evaluate_moisture_risk``.
    """
    n = len(trajectory.times_hours)
    if n == 0:
        return ()
    _, surface_rhs, _ = _surface_series(trajectory, surface)
    running_time_above = 0.0
    running_time_cond = 0.0
    running_max_rh = surface_rhs[0]
    cumulative = [
        (
            config.elevated_time_weight * running_time_above
            + config.condensation_time_weight * running_time_cond
            + config.peak_rh_excess_weight_hours_per_percent
            * max(
                0.0,
                running_max_rh
                - config.elevated_surface_rh_threshold_percent,
            )
        )
    ]
    for i in range(n - 1):
        dt = trajectory.times_hours[i + 1] - trajectory.times_hours[i]
        if surface_rhs[i] > config.elevated_surface_rh_threshold_percent:
            running_time_above += dt
        if surface_rhs[i] >= config.condensation_surface_rh_threshold_percent:
            running_time_cond += dt
        if surface_rhs[i + 1] > running_max_rh:
            running_max_rh = surface_rhs[i + 1]
        cumulative.append(
            config.elevated_time_weight * running_time_above
            + config.condensation_time_weight * running_time_cond
            + config.peak_rh_excess_weight_hours_per_percent
            * max(
                0.0,
                running_max_rh
                - config.elevated_surface_rh_threshold_percent,
            )
        )
    return tuple(cumulative)


def _ventilation_energy_loss_kwh(
    trajectory: RoomTrajectory,
    room: Room,
) -> float:
    """Integrated ventilation heat-loss energy over the trajectory.

    Sums ``instantaneous_heat_loss_power_w * dt_hours / 1000`` per
    step, using the existing thermal helper for the power at each
    step's START-of-step conditions. Returns kWh.
    """
    n = len(trajectory.times_hours)
    if n < 2:
        return 0.0
    total_wh = 0.0
    for i in range(n - 1):
        dt_hours = trajectory.times_hours[i + 1] - trajectory.times_hours[i]
        ach = room.ach_window_open if trajectory.window_open[i] else room.ach_closed
        power_w = ventilation_heat_loss_power(
            indoor_temperature_c=trajectory.indoor_temperature_c[i],
            outdoor_temperature_c=trajectory.outdoor_temperature_c[i],
            room_volume_m3=room.volume_m3,
            ach=ach,
        )
        total_wh += power_w * dt_hours
    return total_wh / 1000.0


def _total_ventilation_time_hours(
    trajectory: RoomTrajectory,
) -> float:
    """Sum of step durations where the window was open at step start."""
    n = len(trajectory.times_hours)
    total = 0.0
    for i in range(n - 1):
        if trajectory.window_open[i]:
            total += trajectory.times_hours[i + 1] - trajectory.times_hours[i]
    return total


# --- Plot helpers ---------------------------------------------------------


def _plot_series(
    trajectory: RoomTrajectory,
    y_values,
    y_label: str,
    title: str,
    output_path: Path,
    reference_line: Tuple[float, str] = None,
    ventilation_events: Sequence[VentilationEvent] = (),
) -> None:
    fig, ax = plt.subplots()
    ax.plot(trajectory.times_hours, y_values, color="tab:blue")
    if reference_line is not None:
        ref_value, ref_label = reference_line
        ax.axhline(
            ref_value,
            color="tab:gray",
            linestyle="--",
            label=ref_label,
        )
        ax.legend()
    for event in ventilation_events:
        ax.axvspan(
            event.start_time_hours,
            event.end_time_hours,
            color="tab:red",
            alpha=0.15,
        )
    ax.set_xlabel("time (hours)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# --- Main -----------------------------------------------------------------


def main() -> None:
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    thermal_props = ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )
    surface = SurfaceDescriptor(
        label="cold external wall corner",
        surface_temperature_factor=0.65,
    )
    risk_config = RiskConfig()  # POC defaults

    moisture_schedule = MoistureSourceSchedule(
        constant_background_rate_g_per_hour=60.0,
        events=(
            MoistureSourceEvent(
                label="morning shower",
                start_time_hours=7.0,
                end_time_hours=7.4,
                generation_rate_g_per_hour=1500.0,
            ),
            MoistureSourceEvent(
                label="evening cooking",
                start_time_hours=18.0,
                end_time_hours=19.5,
                generation_rate_g_per_hour=400.0,
            ),
        ),
    )
    outdoor_segments = _piecewise_outdoor_schedule()

    # Case 1: unventilated.
    unvent_trajectory = _run_piecewise_simulation(
        initial_room=room,
        thermal_properties=thermal_props,
        outdoor_segments=outdoor_segments,
        moisture_schedule=moisture_schedule,
        ventilation_events=(),
        timestep_minutes=2.0,
    )

    # Case 2: simple fixed ventilation immediately after big
    # moisture events.
    fixed_ventilation_events = (
        VentilationEvent(start_time_hours=7.4, end_time_hours=7.9),
        VentilationEvent(start_time_hours=19.5, end_time_hours=20.0),
    )
    vent_trajectory = _run_piecewise_simulation(
        initial_room=room,
        thermal_properties=thermal_props,
        outdoor_segments=outdoor_segments,
        moisture_schedule=moisture_schedule,
        ventilation_events=fixed_ventilation_events,
        timestep_minutes=2.0,
    )

    # --- Plots for the unventilated case -------------------------------------
    unvent_t_surface, unvent_rh_surface, unvent_margin = _surface_series(
        unvent_trajectory, surface
    )
    unvent_cumulative = _cumulative_risk_score(
        unvent_trajectory, surface, risk_config
    )

    _plot_series(
        unvent_trajectory,
        unvent_trajectory.indoor_relative_humidity_pct,
        "indoor RH (%)",
        "Unventilated: indoor RH vs time",
        OUTPUT_DIR / "risk_metric_unventilated_indoor_rh.png",
    )
    _plot_series(
        unvent_trajectory,
        unvent_trajectory.indoor_absolute_humidity_g_m3,
        "indoor AH (g/m^3)",
        "Unventilated: indoor AH vs time",
        OUTPUT_DIR / "risk_metric_unventilated_indoor_ah.png",
    )
    _plot_series(
        unvent_trajectory,
        unvent_t_surface,
        "surface T (C)",
        "Unventilated: critical surface temperature vs time",
        OUTPUT_DIR / "risk_metric_unventilated_surface_temperature.png",
    )
    _plot_series(
        unvent_trajectory,
        unvent_rh_surface,
        "surface RH (%)",
        "Unventilated: surface RH vs time",
        OUTPUT_DIR / "risk_metric_unventilated_surface_rh.png",
        reference_line=(
            risk_config.elevated_surface_rh_threshold_percent,
            f"threshold = {risk_config.elevated_surface_rh_threshold_percent:g} %",
        ),
    )
    _plot_series(
        unvent_trajectory,
        unvent_margin,
        "condensation margin (K)",
        "Unventilated: condensation margin vs time",
        OUTPUT_DIR / "risk_metric_unventilated_margin.png",
        reference_line=(0.0, "boundary = 0 K"),
    )
    _plot_series(
        unvent_trajectory,
        unvent_cumulative,
        "cumulative risk score",
        "Unventilated: cumulative risk score vs time",
        OUTPUT_DIR / "risk_metric_unventilated_cumulative.png",
    )

    # --- Plots for the ventilated case ------------------------------------
    vent_t_surface, vent_rh_surface, vent_margin = _surface_series(
        vent_trajectory, surface
    )
    vent_cumulative = _cumulative_risk_score(
        vent_trajectory, surface, risk_config
    )

    _plot_series(
        vent_trajectory,
        vent_trajectory.indoor_relative_humidity_pct,
        "indoor RH (%)",
        "Fixed vent: indoor RH vs time",
        OUTPUT_DIR / "risk_metric_ventilated_indoor_rh.png",
        ventilation_events=fixed_ventilation_events,
    )
    _plot_series(
        vent_trajectory,
        vent_trajectory.indoor_absolute_humidity_g_m3,
        "indoor AH (g/m^3)",
        "Fixed vent: indoor AH vs time",
        OUTPUT_DIR / "risk_metric_ventilated_indoor_ah.png",
        ventilation_events=fixed_ventilation_events,
    )
    _plot_series(
        vent_trajectory,
        vent_t_surface,
        "surface T (C)",
        "Fixed vent: critical surface temperature vs time",
        OUTPUT_DIR / "risk_metric_ventilated_surface_temperature.png",
        ventilation_events=fixed_ventilation_events,
    )
    _plot_series(
        vent_trajectory,
        vent_rh_surface,
        "surface RH (%)",
        "Fixed vent: surface RH vs time",
        OUTPUT_DIR / "risk_metric_ventilated_surface_rh.png",
        reference_line=(
            risk_config.elevated_surface_rh_threshold_percent,
            f"threshold = {risk_config.elevated_surface_rh_threshold_percent:g} %",
        ),
        ventilation_events=fixed_ventilation_events,
    )
    _plot_series(
        vent_trajectory,
        vent_margin,
        "condensation margin (K)",
        "Fixed vent: condensation margin vs time",
        OUTPUT_DIR / "risk_metric_ventilated_margin.png",
        reference_line=(0.0, "boundary = 0 K"),
        ventilation_events=fixed_ventilation_events,
    )
    _plot_series(
        vent_trajectory,
        vent_cumulative,
        "cumulative risk score",
        "Fixed vent: cumulative risk score vs time",
        OUTPUT_DIR / "risk_metric_ventilated_cumulative.png",
        ventilation_events=fixed_ventilation_events,
    )

    # --- Comparison table --------------------------------------------------
    unvent_risk = evaluate_moisture_risk(
        trajectory=unvent_trajectory, surface=surface, config=risk_config
    )
    vent_risk = evaluate_moisture_risk(
        trajectory=vent_trajectory, surface=surface, config=risk_config
    )
    unvent_total_time = _total_ventilation_time_hours(unvent_trajectory)
    vent_total_time = _total_ventilation_time_hours(vent_trajectory)
    unvent_energy = _ventilation_energy_loss_kwh(unvent_trajectory, room)
    vent_energy = _ventilation_energy_loss_kwh(vent_trajectory, room)

    print("Scenario")
    print("--------")
    print("indoor at t=0: 20 C / 55 %RH, 40 m^3")
    print("outdoor: piecewise diurnal - illustrative POC values")
    print("moisture: 60 g/h baseline + 1500 g/h shower 7.0-7.4 h + 400 g/h cooking 18.0-19.5 h")
    print("surface: fRsi = 0.65 ('cold external wall corner', illustrative)")
    print(
        f"risk config: elevated threshold = {risk_config.elevated_surface_rh_threshold_percent:g} %, "
        f"condensation threshold = {risk_config.condensation_surface_rh_threshold_percent:g} %"
    )
    print()
    print("Comparison")
    print("----------")
    header = (
        f"  {'metric':<45}  {'unventilated':>15}  {'fixed vent':>15}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'total ventilation time (h)':<45}  "
        f"{unvent_total_time:>15.3f}  {vent_total_time:>15.3f}"
    )
    print(
        f"  {'total ventilation energy loss (kWh)':<45}  "
        f"{unvent_energy:>15.4f}  {vent_energy:>15.4f}"
    )
    print(
        f"  {'time above surface RH threshold (h)':<45}  "
        f"{unvent_risk.time_above_surface_rh_threshold_hours:>15.3f}  "
        f"{vent_risk.time_above_surface_rh_threshold_hours:>15.3f}"
    )
    print(
        f"  {'time in predicted condensation (h)':<45}  "
        f"{unvent_risk.time_in_condensation_hours:>15.3f}  "
        f"{vent_risk.time_in_condensation_hours:>15.3f}"
    )
    print(
        f"  {'maximum surface RH (%)':<45}  "
        f"{unvent_risk.maximum_surface_rh_percent:>15.2f}  "
        f"{vent_risk.maximum_surface_rh_percent:>15.2f}"
    )
    print(
        f"  {'cumulative risk score (indicator)':<45}  "
        f"{unvent_risk.cumulative_risk_score:>15.3f}  "
        f"{vent_risk.cumulative_risk_score:>15.3f}"
    )
    print()
    print("Interpretation")
    print("--------------")
    print(
        "The fixed-ventilation strategy pays a small ventilation-time"
    )
    print(
        "and energy-loss cost to reduce sustained surface exposure"
    )
    print(
        "(time above threshold, time in condensation) and the"
    )
    print(
        "cumulative risk indicator. The indicator is illustrative"
    )
    print(
        "only - see mould_risk module docstring; it is NOT a validated"
    )
    print(
        "mould-growth prediction."
    )
    print()
    print(f"Plots saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
