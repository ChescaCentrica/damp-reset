"""End-to-end validation from a CSV of sensor observations.

Usage (recommended: run via python at the repo root):

    python experiments/validate_from_csv.py \\
        --csv path/to/sensor.csv \\
        --room-volume 40 \\
        --calibration-event-index 0 \\
        --output-dir outputs/plots

The script:
    1. Loads sensor observations from the CSV
       (``validation.load_observations_from_csv``).
    2. Identifies every window-open event
       (``validation.identify_ventilation_events``).
    3. If ``--calibration-event-index`` is given, uses THAT event
       to fit ACH (via ``calibration``) and, when the caller passes
       ``--calibrate-thermal``, also fits C_eff (via
       ``thermal_calibration``). The calibration event is
       EXCLUDED from the validation set - calibration and
       validation are logically separate.
    4. For every remaining event, runs the forward prediction of
       indoor AH and (when C_eff is available) indoor T and RH,
       and reports MAE per event.
    5. If ``matplotlib`` is available, writes one plot per event
       and quantity under ``--output-dir``. Plots are optional:
       the script does not fail if matplotlib is absent, it just
       prints the numeric summary.

The script does NOT touch model parameters using the validation
events. The fitter is only pointed at the caller-designated
calibration event; the ACH (and optionally C_eff) that comes out
of it is passed unchanged to the validation function.

CSV shape: see ``validation`` module docstring.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import CalibrationObservation, estimate_ach_from_observations
from thermal_calibration import (
    ThermalObservation,
    estimate_effective_thermal_capacity_from_observations,
)
from validation import (
    EventValidationResult,
    SensorObservation,
    VentilationEventWindow,
    identify_ventilation_events,
    load_observations_from_csv,
    validate_events,
)


def _calibration_observations_from(
    observations: Sequence[SensorObservation],
    event: VentilationEventWindow,
) -> list:
    return [
        CalibrationObservation(
            timestamp_hours=obs.timestamp_hours,
            indoor_absolute_humidity_g_m3=obs.indoor_absolute_humidity_g_m3,
            outdoor_absolute_humidity_g_m3=obs.outdoor_absolute_humidity_g_m3,
            window_open=obs.window_open,
        )
        for obs in observations[
            event.start_index : event.end_index_exclusive
        ]
    ]


def _thermal_observations_from(
    observations: Sequence[SensorObservation],
    event: VentilationEventWindow,
) -> list:
    return [
        ThermalObservation(
            timestamp_hours=obs.timestamp_hours,
            indoor_temperature_c=obs.indoor_temperature_c,
            outdoor_temperature_c=obs.outdoor_temperature_c,
            window_open=obs.window_open,
        )
        for obs in observations[
            event.start_index : event.end_index_exclusive
        ]
    ]


def _print_summary(
    events: Sequence[VentilationEventWindow],
    calibration_event_index: Optional[int],
    validation_results: Sequence[EventValidationResult],
    ach: float,
    c_eff: Optional[float],
) -> None:
    print(f"\nCalibration / validation summary")
    print("-" * 60)
    print(f"Identified {len(events)} window-open event(s).")
    if calibration_event_index is not None:
        cal = events[calibration_event_index]
        print(
            f"Calibration event: index {calibration_event_index}, "
            f"{cal.sample_count} samples."
        )
    else:
        print("Calibration event: none (using caller-supplied ACH).")
    print(f"ACH used for prediction: {ach:.4f} h^-1")
    if c_eff is not None:
        print(
            f"C_eff used for prediction: {c_eff:.0f} J/K "
            f"({c_eff / 1000:.1f} kJ/K)"
        )
    else:
        print("C_eff used: none (temperature validation skipped).")
    print()
    header = (
        f"{'validation event':>16s}  {'n':>4s}  "
        f"{'MAE AH (g/m^3)':>15s}  {'MAE T (C)':>10s}  "
        f"{'MAE RH (%)':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in validation_results:
        span = f"{r.start_time_hours:.2f}-{r.end_time_hours:.2f}h"
        t_str = (
            "-"
            if r.mae_temperature_c is None
            else f"{r.mae_temperature_c:10.3f}"
        )
        rh_str = (
            "-"
            if r.mae_relative_humidity_pct is None
            else f"{r.mae_relative_humidity_pct:10.3f}"
        )
        print(
            f"{span:>16s}  {r.n_samples:>4d}  "
            f"{r.mae_absolute_humidity_g_m3:>15.3f}  "
            f"{t_str}  {rh_str}"
        )
    print()


def _try_plot(
    validation_results: Sequence[EventValidationResult],
    output_dir: Path,
) -> None:
    """If matplotlib is available, save per-event predicted-vs-measured plots.

    Not a hard dependency: on ImportError the caller sees the
    numeric summary only.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib not available; skipping plots. "
            "Install with `pip install matplotlib` to enable them."
        )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(validation_results):
        for name, observed, predicted, ylabel in (
            (
                "absolute_humidity",
                r.observed_indoor_absolute_humidity_g_m3,
                r.predicted_indoor_absolute_humidity_g_m3,
                "indoor AH (g/m^3)",
            ),
            (
                "temperature",
                r.observed_indoor_temperature_c,
                r.predicted_indoor_temperature_c,
                "indoor T (C)",
            ),
            (
                "relative_humidity",
                r.observed_indoor_relative_humidity_pct,
                r.predicted_indoor_relative_humidity_pct,
                "indoor RH (%)",
            ),
        ):
            if predicted is None:
                continue
            fig, ax = plt.subplots(figsize=(6.0, 3.5))
            ax.plot(r.times_hours, observed, "o-", label="observed")
            ax.plot(
                r.times_hours, predicted, "x--", label="predicted"
            )
            ax.set_xlabel("time (hours)")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"Event {i}: {r.start_time_hours:.2f}-{r.end_time_hours:.2f}h, "
                f"n={r.n_samples}"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            filename = (
                f"validation_event{i}_{name}.png"
            )
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=120)
            plt.close(fig)
    print(f"Plots saved under {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the damp-reset physics model against measured "
            "sensor CSV data. Calibration and validation are kept "
            "logically separate."
        )
    )
    parser.add_argument(
        "--csv", required=True, type=Path, help="Path to the sensor CSV."
    )
    parser.add_argument(
        "--room-volume",
        required=True,
        type=float,
        help="Room volume in cubic metres.",
    )
    parser.add_argument(
        "--calibration-event-index",
        type=int,
        default=None,
        help=(
            "Index of the event to use for calibration. When set, the "
            "event is excluded from validation and its ACH (and, if "
            "--calibrate-thermal, its C_eff) are fitted and passed to "
            "the validation function."
        ),
    )
    parser.add_argument(
        "--ach",
        type=float,
        default=None,
        help=(
            "ACH to use if --calibration-event-index is not set. "
            "One of --ach or --calibration-event-index must be given."
        ),
    )
    parser.add_argument(
        "--effective-thermal-capacity-j-per-k",
        type=float,
        default=None,
        help=(
            "C_eff for the thermal prediction. Optional. Not used when "
            "--calibrate-thermal supplies its own value from a fit."
        ),
    )
    parser.add_argument(
        "--calibrate-thermal",
        action="store_true",
        help=(
            "Also fit C_eff on the calibration event. Requires "
            "--calibration-event-index."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory to write predicted-vs-measured plots.",
    )
    parser.add_argument(
        "--minimum-event-samples",
        type=int,
        default=3,
        help=(
            "Shortest event length (in samples) considered. Shorter "
            "runs (typically sensor bounce) are ignored."
        ),
    )
    args = parser.parse_args()

    if args.calibration_event_index is None and args.ach is None:
        parser.error(
            "one of --calibration-event-index or --ach is required."
        )
    if args.calibrate_thermal and args.calibration_event_index is None:
        parser.error(
            "--calibrate-thermal requires --calibration-event-index."
        )

    observations = load_observations_from_csv(args.csv)
    events = identify_ventilation_events(
        observations, minimum_samples=args.minimum_event_samples
    )
    if not events:
        parser.error("no window-open events identified in the CSV.")

    fitted_ach: Optional[float] = None
    fitted_c_eff: Optional[float] = None
    calibration_event_index = args.calibration_event_index
    if calibration_event_index is not None:
        if not (0 <= calibration_event_index < len(events)):
            parser.error(
                f"--calibration-event-index {calibration_event_index} out "
                f"of range 0..{len(events) - 1}."
            )
        cal_event = events[calibration_event_index]
        cal_result = estimate_ach_from_observations(
            _calibration_observations_from(observations, cal_event)
        )
        fitted_ach = cal_result.estimated_ach
        print(
            f"Fitted ACH on event {calibration_event_index}: "
            f"{fitted_ach:.4f} h^-1 "
            f"(RMS residual {cal_result.rms_error_g_m3:.4f} g/m^3, "
            f"{cal_result.n_observations} samples)."
        )
        if args.calibrate_thermal:
            thermal_result = (
                estimate_effective_thermal_capacity_from_observations(
                    observations=_thermal_observations_from(
                        observations, cal_event
                    ),
                    ach=fitted_ach,
                    room_volume_m3=args.room_volume,
                )
            )
            fitted_c_eff = (
                thermal_result.estimated_effective_thermal_capacity_j_per_k
            )
            print(
                f"Fitted C_eff on event {calibration_event_index}: "
                f"{fitted_c_eff:.0f} J/K "
                f"(RMS residual {thermal_result.rms_error_c:.3f} C, "
                f"{thermal_result.n_observations} samples)."
            )

    ach_to_use = (
        fitted_ach if fitted_ach is not None else args.ach
    )
    c_eff_to_use: Optional[float] = fitted_c_eff
    if c_eff_to_use is None:
        c_eff_to_use = args.effective_thermal_capacity_j_per_k

    validation_events = tuple(
        e
        for i, e in enumerate(events)
        if i != calibration_event_index
    )
    if not validation_events:
        parser.error(
            "no events left for validation after excluding the "
            "calibration event; a single-event CSV cannot both "
            "calibrate and validate."
        )
    results = validate_events(
        observations=observations,
        events=validation_events,
        ach=ach_to_use,
        room_volume_m3=args.room_volume,
        effective_thermal_capacity_j_per_k=c_eff_to_use,
    )
    _print_summary(
        events=events,
        calibration_event_index=calibration_event_index,
        validation_results=results,
        ach=ach_to_use,
        c_eff=c_eff_to_use,
    )
    _try_plot(results, args.output_dir)


if __name__ == "__main__":
    main()
