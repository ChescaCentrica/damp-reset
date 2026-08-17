"""Compare ventilation rates: indoor absolute humidity vs time for several ACH.

Same scenario as ``indoor_ah_vs_time.py`` (indoor 20 C / 70 %RH, outdoor
5 C / 85 %RH, 40 m^3 room), sweeping ACH over [0.5, 1, 2, 5, 10] h^-1 and
plotting one indoor-AH curve per rate against the outdoor-AH reference
line. Also prints a comparison table of final indoor AH at 5, 10, and 15
minutes for each ACH.

Purpose:
    - show that HIGHER ACH removes moisture faster (steeper initial slope);
    - show that LOWER ACH converges slowly toward outdoor AH;
    - show that ALL curves approach the same outdoor AH asymptote when
      outdoor conditions are constant, no matter the exchange rate.

The ACH values here are ILLUSTRATIVE model inputs, not calibrated
representations of specific window configurations, gaps, wind pressures,
or trickle-vent settings. Real-room ACH depends on window opening area,
stack effect, wind speed and direction, cross-ventilation paths, and
building envelope leakage - none of which are measured or claimed here.

This is exactly why the ventilation model will later need an ESTIMATED or
LEARNED effective ACH per room / window state: the same 15-minute
window-open event moves the room to a very different final AH depending on
what the real ACH is (see the table). Any control decision that recommends
"open the window for X minutes" is only as good as the ACH value it is
built on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room, predict_room_moisture
from psychrometrics import AirState

ACH_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0)  # h^-1, illustrative only
DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121  # 0.25-min resolution across 0-30 min
TABLE_MINUTES = (5.0, 10.0, 15.0)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "indoor_ah_vs_time_ach_sweep.png"
)


def _make_room(ach_window_open: float) -> Room:
    """Return the shared default room with the requested window-open ACH."""
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=ach_window_open,
    )


def main() -> None:
    """Run the sweep, save the plot, and print the comparison table."""
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)
    outdoor_ah = outdoor.absolute_humidity

    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations_minutes = [i * step for i in range(DURATION_STEPS)]

    fig, ax = plt.subplots()
    for ach in ACH_VALUES:
        room = _make_room(ach_window_open=ach)
        indoor_ah_series = [
            predict_room_moisture(room, outdoor, duration_minutes=t).final_absolute_humidity_g_m3
            for t in durations_minutes
        ]
        ax.plot(durations_minutes, indoor_ah_series, label=f"ACH = {ach:g} h^-1")

    ax.axhline(
        outdoor_ah,
        linestyle="--",
        color="tab:red",
        label=f"outdoor AH = {outdoor_ah:.2f} g/m^3",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor absolute humidity (g/m^3)")
    ax.set_title(
        "Indoor absolute humidity vs ventilation duration for several ACH\n"
        "indoor 20 C / 70 %RH, outdoor 5 C / 85 %RH, 40 m^3"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")
    print()

    print("Final indoor AH (g/m^3) after fixed durations, by ACH")
    print("-----------------------------------------------------")
    header = f"{'ACH (h^-1)':>10}  " + "  ".join(
        f"{f'{m:g} min':>10}" for m in TABLE_MINUTES
    )
    print(header)
    print("-" * len(header))
    for ach in ACH_VALUES:
        room = _make_room(ach_window_open=ach)
        row_values = [
            predict_room_moisture(
                room, outdoor, duration_minutes=t
            ).final_absolute_humidity_g_m3
            for t in TABLE_MINUTES
        ]
        row = f"{ach:>10g}  " + "  ".join(f"{v:>10.3f}" for v in row_values)
        print(row)
    print(f"\noutdoor AH asymptote = {outdoor_ah:.3f} g/m^3")


if __name__ == "__main__":
    main()
