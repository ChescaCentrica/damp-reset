"""Trade-off plot: water removed vs ventilation energy loss.

Each point is one row of the duration sweep already produced by
``combined_ventilation_dataframe.py`` (durations 0, 2, 3, 5, 10, 15,
20 min for the shared default scenario). The x-axis is the DYNAMIC
ventilation energy removed in kWh (from the thermal model,
``C_eff * (T_0 - T_f)``); the y-axis is the water removed from the
room air in grams (from the moisture model, ``(C_0 - C_f) * V``).

Why a line between the points is drawn:

    The seven points are NOT independent measurements. They are
    samples of one continuous model trajectory parameterised by
    ventilation duration - as duration grows from 0 to 20 min, the
    (energy, water) pair traces out a curve in the plane. A light
    connecting line reflects that continuity honestly and helps the
    eye see the shape of the trade-off. The marker for each duration
    is labelled so a reader can tell where along the trajectory each
    sample sits. Nothing about the line implies a measured trend or a
    physical rate that has not been computed; it is just the model
    curve interpolated between our chosen sample points.

Save location: outputs/plots/moisture_removed_vs_energy_loss.png.

No optimum is chosen. This is a visualisation of the trade-off, not
a decision rule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

DURATIONS_MINUTES = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "plots"
    / "moisture_removed_vs_energy_loss.png"
)


def main() -> None:
    """Compute the sweep, plot the trade-off, save the figure."""
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
    energies_kwh = [r.ventilation_energy_removed_kwh for r in results]
    water_g = [r.water_removed_g for r in results]

    fig, ax = plt.subplots()
    ax.plot(
        energies_kwh,
        water_g,
        color="tab:gray",
        linewidth=1,
        label="model trajectory as duration varies",
    )
    ax.scatter(
        energies_kwh,
        water_g,
        color="tab:blue",
        zorder=3,
        label="sampled durations",
    )
    for t, x, y in zip(DURATIONS_MINUTES, energies_kwh, water_g):
        ax.annotate(
            f"{t:g} min",
            xy=(x, y),
            xytext=(6, -4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("ventilation energy removed (kWh)")
    ax.set_ylabel("water removed from room air (g)")
    ax.set_title(
        "Water removed vs ventilation energy loss\n"
        "each point = one window-open duration; line = model trajectory as duration varies"
    )
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
