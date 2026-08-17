"""Pareto frontier plot: water removed vs ventilation energy loss.

For every candidate ventilation duration, plot one marker at
(``ventilation_energy_removed_kwh``, ``water_removed_g``). Highlight
the Pareto-efficient candidates - the actions that no other candidate
dominates - and separately list them.

Definition (also documented on ``optimiser.pareto_efficient_indices``):
    Candidate A DOMINATES candidate B when A removes at least as
    much water AND uses no more energy than B, with at least one
    strict improvement. A candidate is PARETO-EFFICIENT when no
    other candidate dominates it.

Purpose:
    Show the trade-off space the optimisation layer sees. The
    Pareto frontier itself is NOT a decision rule - it is the set
    of candidates that are worth considering under some assumption
    about the moisture-vs-energy trade-off, and each of the
    strategies in ``optimiser.py`` picks a specific point on this
    frontier under its own additional assumption. Non-efficient
    candidates are the ones that no reasonable strategy should
    ever pick.

Saved to: outputs/plots/pareto_frontier.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from optimiser import pareto_efficient_indices
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from ventilation import simulate_ventilation_event

CANDIDATE_DURATIONS_MINUTES = [float(t) for t in range(0, 31)]
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "plots"
    / "pareto_frontier.png"
)


def main() -> None:
    """Run the sweep, identify the Pareto frontier, save the plot."""
    predictions = [
        simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=20.0,
            initial_indoor_relative_humidity_pct=70.0,
            outdoor_temperature_c=5.0,
            outdoor_relative_humidity_pct=85.0,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=t,
        )
        for t in CANDIDATE_DURATIONS_MINUTES
    ]
    efficient = pareto_efficient_indices(predictions)
    efficient_set = set(efficient)

    all_energies = [p.ventilation_energy_removed_kwh for p in predictions]
    all_water = [p.water_removed_g for p in predictions]
    efficient_energies = [all_energies[i] for i in efficient]
    efficient_water = [all_water[i] for i in efficient]

    fig, ax = plt.subplots()
    dominated_indices = [
        i for i in range(len(predictions)) if i not in efficient_set
    ]
    if dominated_indices:
        ax.scatter(
            [all_energies[i] for i in dominated_indices],
            [all_water[i] for i in dominated_indices],
            color="tab:gray",
            zorder=2,
            label="dominated candidates",
        )
    ax.plot(
        efficient_energies,
        efficient_water,
        color="tab:red",
        linewidth=1,
        zorder=3,
    )
    ax.scatter(
        efficient_energies,
        efficient_water,
        color="tab:red",
        zorder=4,
        label="Pareto-efficient candidates",
    )
    for i in efficient:
        ax.annotate(
            f"{CANDIDATE_DURATIONS_MINUTES[i]:g} min",
            xy=(all_energies[i], all_water[i]),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("ventilation energy removed (kWh)")
    ax.set_ylabel("water removed from room air (g)")
    ax.set_title(
        "Water removed vs ventilation energy: Pareto frontier\n"
        "canonical scenario, candidate durations 0-30 min in 1-min steps"
    )
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")
    print()

    print("Pareto-efficient candidate durations")
    print("------------------------------------")
    header = (
        f"  {'duration':>8}  {'water':>10}  {'energy':>10}"
    )
    print(header)
    print(f"  {'(min)':>8}  {'(g)':>10}  {'(kWh)':>10}")
    print("  " + "-" * (len(header) - 2))
    for i in efficient:
        print(
            f"  {CANDIDATE_DURATIONS_MINUTES[i]:>8.1f}  "
            f"{all_water[i]:>+10.2f}  {all_energies[i]:>+10.4f}"
        )
    print()
    print(f"{len(efficient)} of {len(predictions)} candidates are Pareto-efficient.")


if __name__ == "__main__":
    main()
