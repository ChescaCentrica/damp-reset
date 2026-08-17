"""Feasibility table across candidate window-open durations.

For the canonical scenario, evaluates each duration in a caller-set
list against a caller-set ``VentilationConstraints`` and prints one
row per candidate with:

    duration, water removed, energy loss, temp drop, final AH,
    feasible, reason if infeasible

The constraints used here are ILLUSTRATIVE control parameters. See
the ``VentilationConstraints`` docstring: they are NOT damp / mould /
health thresholds. Change them freely at the top of ``main()`` to run
different what-if experiments; the file is a template, not a policy.

No optimisation is done. The purpose is only to make the
feasibility-checking layer visible in a table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room
from optimiser import (
    VentilationConstraints,
    evaluate_candidate_durations_with_constraints,
)
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties

CANDIDATE_DURATIONS_MINUTES = [0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]


def main() -> None:
    """Run the constrained sweep and print the feasibility table."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)
    thermal_props = ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )

    # Illustrative control parameters. Edit freely; not thresholds.
    constraints = VentilationConstraints(
        max_temperature_drop_c=1.0,
        max_energy_loss_kwh=0.15,
        target_final_absolute_humidity_g_m3=8.0,
        target_moisture_reduction_g_m3=3.0,
    )

    print("Scenario")
    print("--------")
    print(
        f"indoor : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3, ACH = {room.ach_window_open:g} h^-1 (open)"
    )
    print(
        f"outdoor: T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} %"
    )
    print(
        f"C_eff  : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print()

    print("Illustrative control constraints")
    print("--------------------------------")
    print(
        f"  max_temperature_drop_c              = "
        f"{constraints.max_temperature_drop_c}"
    )
    print(
        f"  max_energy_loss_kwh                 = "
        f"{constraints.max_energy_loss_kwh}"
    )
    print(
        f"  target_final_absolute_humidity_g_m3 = "
        f"{constraints.target_final_absolute_humidity_g_m3}"
    )
    print(
        f"  target_moisture_reduction_g_m3      = "
        f"{constraints.target_moisture_reduction_g_m3}"
    )
    print()

    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
        constraints=constraints,
    )

    header = (
        f"  {'duration':>8}  "
        f"{'water rem':>10}  "
        f"{'energy':>9}  "
        f"{'T drop':>8}  "
        f"{'final AH':>10}  "
        f"{'feasible':>8}  "
        f"reason if infeasible"
    )
    units = (
        f"  {'(min)':>8}  "
        f"{'(g)':>10}  "
        f"{'(kWh)':>9}  "
        f"{'(K)':>8}  "
        f"{'(g/m^3)':>10}  "
        f"{'':>8}"
    )
    print(header)
    print(units)
    print("  " + "-" * (len(header) - 2))
    for duration_minutes, evaluation in zip(
        CANDIDATE_DURATIONS_MINUTES, evaluations
    ):
        prediction = evaluation.prediction
        reason = (
            ", ".join(evaluation.violated_constraints)
            if not evaluation.feasible
            else ""
        )
        print(
            f"  {duration_minutes:>8.1f}  "
            f"{prediction.water_removed_g:>+10.2f}  "
            f"{prediction.ventilation_energy_removed_kwh:>+9.4f}  "
            f"{prediction.temperature_drop_c:>+8.3f}  "
            f"{prediction.final_absolute_humidity_g_m3:>10.3f}  "
            f"{('yes' if evaluation.feasible else 'no'):>8}  "
            f"{reason}"
        )


if __name__ == "__main__":
    main()
