"""Pareto frontier: with and without an occupant-comfort constraint.

Runs the Pareto analysis TWICE on the same candidate list:

    1. Unconstrained. Every candidate duration participates.
    2. Comfort-constrained. Candidates whose predicted
       ``temperature_drop_c`` exceeds ``max_temperature_drop_c``
       are dropped before Pareto filtering.

Both surviving sets are plotted on the same (energy, water) axes so
the reader can see how the SHAPE of the reachable Pareto frontier
changes when comfort is enforced. A companion text summary lists
each candidate in both cases side by side.

The comfort cap used here is an ILLUSTRATIVE POC value chosen so
the two frontiers visibly differ. It is not a validated damp,
mould, health, or building-fabric threshold. Change it freely at
the top of ``main()`` for other what-if experiments.

Purpose: show that enforcing occupant comfort truncates the
reachable trade-off curve, and therefore constrains what the
optimisation layer can achieve. This experiment does NOT decide
which point on either frontier to select - it only characterises
the sets the strategies would pick from.

Saved to: outputs/plots/pareto_frontier_with_comfort.png.
"""

import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from optimiser import (
    VentilationConstraints,
    evaluate_candidate_durations_with_constraints,
    pareto_efficient_indices,
)
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from ventilation import VentilationSimulationResult

CANDIDATE_DURATIONS_MINUTES: Sequence[float] = [float(t) for t in range(0, 31)]
MAX_TEMPERATURE_DROP_C = 1.0  # illustrative comfort cap
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "plots"
    / "pareto_frontier_with_comfort.png"
)


def _feasible_predictions(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> List[VentilationSimulationResult]:
    """Return only the predictions whose hard constraints hold.

    Reuses the existing constraint-check pipeline; this file adds
    no physics of its own.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )
    return [e.prediction for e in evaluations if e.feasible]


def _feasible_durations(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> List[float]:
    """Return the durations that survive the constraint check."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )
    return [
        duration
        for duration, e in zip(candidate_durations_minutes, evaluations)
        if e.feasible
    ]


def main() -> None:
    """Run both Pareto analyses, print the summary, save the overlay plot."""
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

    unconstrained_predictions = _feasible_predictions(
        room, outdoor, thermal_props,
        CANDIDATE_DURATIONS_MINUTES,
        VentilationConstraints(),
    )
    unconstrained_durations = _feasible_durations(
        room, outdoor, thermal_props,
        CANDIDATE_DURATIONS_MINUTES,
        VentilationConstraints(),
    )
    unconstrained_efficient = pareto_efficient_indices(
        unconstrained_predictions
    )

    comfort_constraints = VentilationConstraints(
        max_temperature_drop_c=MAX_TEMPERATURE_DROP_C
    )
    comfort_predictions = _feasible_predictions(
        room, outdoor, thermal_props,
        CANDIDATE_DURATIONS_MINUTES,
        comfort_constraints,
    )
    comfort_durations = _feasible_durations(
        room, outdoor, thermal_props,
        CANDIDATE_DURATIONS_MINUTES,
        comfort_constraints,
    )
    comfort_efficient = pareto_efficient_indices(comfort_predictions)

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3"
    )
    print(
        f"outdoor : T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} %"
    )
    print(f"ACH open: {room.ach_window_open:g} h^-1")
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print(
        f"comfort cap: max_temperature_drop_c = "
        f"{MAX_TEMPERATURE_DROP_C:g} K (illustrative POC value)"
    )
    print(
        f"candidates: {len(CANDIDATE_DURATIONS_MINUTES)} durations "
        f"{int(CANDIDATE_DURATIONS_MINUTES[0])}-"
        f"{int(CANDIDATE_DURATIONS_MINUTES[-1])} min in 1-min steps"
    )
    print()

    print("Comparison summary")
    print("------------------")
    print(
        f"  unconstrained    : {len(unconstrained_predictions):>3} feasible, "
        f"{len(unconstrained_efficient):>3} Pareto-efficient"
    )
    print(
        f"  comfort <= {MAX_TEMPERATURE_DROP_C:g} K: "
        f"{len(comfort_predictions):>3} feasible, "
        f"{len(comfort_efficient):>3} Pareto-efficient"
    )
    unconstrained_max_water = max(
        p.water_removed_g for p in unconstrained_predictions
    )
    comfort_max_water = (
        max(p.water_removed_g for p in comfort_predictions)
        if comfort_predictions else 0.0
    )
    print(
        f"  max reachable water: unconstrained "
        f"{unconstrained_max_water:.2f} g   vs   "
        f"comfort-constrained {comfort_max_water:.2f} g"
    )
    unconstrained_max_energy = max(
        p.ventilation_energy_removed_kwh
        for p in unconstrained_predictions
    )
    comfort_max_energy = (
        max(
            p.ventilation_energy_removed_kwh for p in comfort_predictions
        )
        if comfort_predictions else 0.0
    )
    print(
        f"  max reachable energy: unconstrained "
        f"{unconstrained_max_energy:.4f} kWh   vs   "
        f"comfort-constrained {comfort_max_energy:.4f} kWh"
    )
    print()

    print("Pareto-efficient candidates")
    print("---------------------------")
    print(
        f"  {'duration':>8}  {'water':>10}  {'energy':>10}  "
        f"{'T drop':>8}  {'in unconstrained?':>18}  {'in comfort?':>12}"
    )
    print(
        f"  {'(min)':>8}  {'(g)':>10}  {'(kWh)':>10}  "
        f"{'(K)':>8}  {'':>18}  {'':>12}"
    )
    print("  " + "-" * 76)
    # Enumerate the full sweep so unconstrained-only candidates
    # (those the comfort cap would drop) are still visible in the
    # summary.
    for i, duration in enumerate(CANDIDATE_DURATIONS_MINUTES):
        # Any candidate that survives the unconstrained
        # (all-None-constraints) feasibility check is trivially all
        # of them here, since VentilationConstraints() imposes no
        # limits. So `unconstrained_predictions[i]` matches the raw
        # simulator output at index i.
        prediction = unconstrained_predictions[i]
        in_unconstrained_pareto = i in set(unconstrained_efficient)
        in_comfort_predictions = (
            prediction.temperature_drop_c <= MAX_TEMPERATURE_DROP_C
        )
        in_comfort_pareto = False
        if in_comfort_predictions:
            comfort_index = comfort_durations.index(duration)
            in_comfort_pareto = comfort_index in set(comfort_efficient)
        marker_unconstrained = "efficient" if in_unconstrained_pareto else " "
        marker_comfort = (
            "efficient" if in_comfort_pareto
            else ("dropped" if not in_comfort_predictions else " ")
        )
        print(
            f"  {duration:>8.1f}  "
            f"{prediction.water_removed_g:>+10.2f}  "
            f"{prediction.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{prediction.temperature_drop_c:>+8.3f}  "
            f"{marker_unconstrained:>18}  {marker_comfort:>12}"
        )
    print()

    # --- Plot: two frontiers overlaid ---------------------------------------
    fig, ax = plt.subplots()

    # Unconstrained: dominated candidates in light grey, efficient in blue.
    unconstrained_energies = [
        p.ventilation_energy_removed_kwh for p in unconstrained_predictions
    ]
    unconstrained_water = [
        p.water_removed_g for p in unconstrained_predictions
    ]
    unconstrained_efficient_set = set(unconstrained_efficient)
    dominated_indices = [
        i for i in range(len(unconstrained_predictions))
        if i not in unconstrained_efficient_set
    ]
    if dominated_indices:
        ax.scatter(
            [unconstrained_energies[i] for i in dominated_indices],
            [unconstrained_water[i] for i in dominated_indices],
            color="lightgray",
            zorder=1,
            label="unconstrained candidates (dominated)",
        )
    ax.plot(
        [unconstrained_energies[i] for i in unconstrained_efficient],
        [unconstrained_water[i] for i in unconstrained_efficient],
        color="tab:blue",
        linewidth=1,
        zorder=2,
    )
    ax.scatter(
        [unconstrained_energies[i] for i in unconstrained_efficient],
        [unconstrained_water[i] for i in unconstrained_efficient],
        color="tab:blue",
        zorder=3,
        label="unconstrained Pareto",
    )

    # Comfort-constrained frontier: red markers on top so the truncation
    # is visible.
    comfort_energies = [
        p.ventilation_energy_removed_kwh for p in comfort_predictions
    ]
    comfort_water = [p.water_removed_g for p in comfort_predictions]
    ax.plot(
        [comfort_energies[i] for i in comfort_efficient],
        [comfort_water[i] for i in comfort_efficient],
        color="tab:red",
        linewidth=1,
        zorder=4,
    )
    ax.scatter(
        [comfort_energies[i] for i in comfort_efficient],
        [comfort_water[i] for i in comfort_efficient],
        color="tab:red",
        zorder=5,
        label=f"comfort-constrained (ΔT ≤ {MAX_TEMPERATURE_DROP_C:g} K) Pareto",
    )

    ax.set_xlabel("ventilation energy removed (kWh)")
    ax.set_ylabel("water removed from room air (g)")
    ax.set_title(
        "Pareto frontier with and without a comfort constraint\n"
        f"candidate durations 0-30 min; comfort cap = "
        f"{MAX_TEMPERATURE_DROP_C:g} K (illustrative)"
    )
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
