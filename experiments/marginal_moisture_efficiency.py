"""Marginal moisture-removal efficiency vs window-open duration.

For each pair of neighbouring durations in the sweep, compute:
    incremental_water_removed_g / incremental_ventilation_energy_removed_kwh
and plot one marker per interval. The marker's x-coordinate is the
MIDPOINT of the interval, because an incremental ratio describes the
average behaviour across an interval, not the instantaneous rate at
one duration. Every marker is annotated with the interval it
represents so the plot cannot be misread as an instantaneous curve.

No line connects the markers: the six intervals are non-overlapping
piecewise summaries, not samples of one continuous function of time.
Each point is a discrete "if you extend the event across this
interval, this is the marginal g/kWh you buy" statement.

Save location: outputs/plots/marginal_moisture_efficiency.png.

Physical reading (also included on the plot):

    The moisture ODE is dC/dt = n * (C_out - C). The instantaneous
    drying RATE is proportional to the indoor-outdoor absolute-humidity
    gap. As indoor AH falls toward outdoor AH, the gap shrinks and
    the drying rate shrinks with it. Meanwhile the thermal ODE runs
    on a much longer time constant (tau_thermal ~= 10 * tau_moisture
    for this scenario), so each 5-minute extension keeps costing
    roughly the same amount of energy while removing progressively
    less water. Together those two effects push the marginal g/kWh
    downward as the event runs. That is the diminishing-returns
    signal the earlier incremental table already carried; this plot
    shows it graphically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

from _metrics import safe_ratio

DURATIONS_MINUTES = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "plots"
    / "marginal_moisture_efficiency.png"
)


def main() -> None:
    """Compute per-interval marginal g/kWh and save the plot."""
    scenario_kwargs = dict(
        room_volume_m3=40.0,
        initial_indoor_temperature_c=20.0,
        initial_indoor_relative_humidity_pct=70.0,
        outdoor_temperature_c=5.0,
        outdoor_relative_humidity_pct=85.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        ),
    )
    results = [
        simulate_ventilation_event(duration_minutes=t, **scenario_kwargs)
        for t in DURATIONS_MINUTES
    ]

    midpoints_minutes = []
    marginal_g_per_kwh = []
    interval_labels = []
    for previous, current, previous_result, current_result in zip(
        DURATIONS_MINUTES[:-1],
        DURATIONS_MINUTES[1:],
        results[:-1],
        results[1:],
    ):
        delta_water_g = (
            current_result.water_removed_g - previous_result.water_removed_g
        )
        delta_energy_kwh = (
            current_result.ventilation_energy_removed_kwh
            - previous_result.ventilation_energy_removed_kwh
        )
        # safe_ratio returns NaN when the denominator is near zero;
        # every interval in this scenario has a meaningful positive
        # delta_energy so the guard is defensive rather than an
        # expected branch.
        ratio = safe_ratio(delta_water_g, delta_energy_kwh)
        if ratio != ratio:  # NaN check
            continue
        midpoints_minutes.append(0.5 * (previous + current))
        marginal_g_per_kwh.append(ratio)
        interval_labels.append(f"{previous:g} -> {current:g} min")

    fig, ax = plt.subplots()
    ax.scatter(midpoints_minutes, marginal_g_per_kwh, color="tab:blue", zorder=3)
    for x, y, label in zip(midpoints_minutes, marginal_g_per_kwh, interval_labels):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("midpoint of the interval (minutes)")
    ax.set_ylabel("marginal moisture removed per energy lost (g / kWh)")
    ax.set_title(
        "Marginal moisture-removal efficiency vs duration\n"
        "each point = one non-overlapping interval; no line drawn between points"
    )
    ax.set_ylim(bottom=0.0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
