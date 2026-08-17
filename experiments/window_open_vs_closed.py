"""Quantify the moisture-removal benefit of opening the window vs leaving it shut.

Same room and outdoor conditions in both cases; the only thing that changes
is the air-change rate:

    window closed (background infiltration only) : ACH = 0.4 h^-1
    window open  (active ventilation)            : ACH = 5.0 h^-1

Sweeps window-open duration over 0-30 minutes with the analytic well-mixed
moisture model, then reports:

    1. A table of indoor AH and water removed at 5, 15, and 30 minutes for
       both cases.
    2. The DELTA at each checkpoint - i.e. how much extra moisture the
       active choice of opening the window pulls out of the room air over
       and above what background infiltration would have removed anyway.
    3. A two-line plot (outputs/window_open_vs_closed.png) so the shape of
       the two decay curves can be compared visually.

Caveats to hold in mind while reading the numbers:
    - No thermal model. Opening a window in winter also cools the room; the
      moisture savings shown here ignore the heat loss that always
      accompanies them. A control decision has to weigh both.
    - No moisture buffering from walls, furniture, textiles - see the
      MoisturePrediction docstring. The 'water removed' number is water
      that left the AIR, not water that left the building's fabric.
    - The two ACH values here are illustrative model inputs, not
      calibrated measurements of a specific window or building.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room, predict_room_moisture
from psychrometrics import AirState

TABLE_MINUTES = (5.0, 15.0, 30.0)
DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "outputs" / "window_open_vs_closed.png"
)


def main() -> None:
    """Run the comparison, print the tables, and save the plot."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.4,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T={room.indoor_temperature_c:g} C, "
        f"RH={room.indoor_relative_humidity_pct:g} %, V={room.volume_m3:g} m^3"
    )
    print(
        f"outdoor : T={outdoor.temperature_c:g} C, "
        f"RH={outdoor.relative_humidity_percent:g} %"
    )
    print(
        f"vent    : window closed ACH={room.ach_closed:g} h^-1 vs "
        f"window open ACH={room.ach_window_open:g} h^-1"
    )
    print()

    header = (
        f"{'minutes':>8}  {'closed AH':>10}  {'open AH':>10}  "
        f"{'closed removed':>15}  {'open removed':>13}  {'extra removed':>14}"
    )
    units = (
        f"{'(min)':>8}  {'(g/m^3)':>10}  {'(g/m^3)':>10}  "
        f"{'(g)':>15}  {'(g)':>13}  {'(g)':>14}"
    )
    print(header)
    print(units)
    print("-" * len(header))
    for minutes in TABLE_MINUTES:
        closed = predict_room_moisture(room, outdoor, duration_minutes=minutes, window_open=False)
        opened = predict_room_moisture(room, outdoor, duration_minutes=minutes, window_open=True)
        extra_removed = opened.water_removed_g - closed.water_removed_g
        print(
            f"{minutes:>8.0f}  "
            f"{closed.final_absolute_humidity_g_m3:>10.3f}  "
            f"{opened.final_absolute_humidity_g_m3:>10.3f}  "
            f"{closed.water_removed_g:>+15.2f}  "
            f"{opened.water_removed_g:>+13.2f}  "
            f"{extra_removed:>+14.2f}"
        )

    print()
    print("'extra removed' is the additional water pulled out of the room AIR by")
    print("actively opening the window, over and above what background infiltration")
    print("would have removed on its own in the same interval.")

    # --- Plot ---------------------------------------------------------------
    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations = [i * step for i in range(DURATION_STEPS)]
    closed_series = [
        predict_room_moisture(room, outdoor, duration_minutes=t, window_open=False).final_absolute_humidity_g_m3
        for t in durations
    ]
    open_series = [
        predict_room_moisture(room, outdoor, duration_minutes=t, window_open=True).final_absolute_humidity_g_m3
        for t in durations
    ]
    outdoor_ah = outdoor.absolute_humidity

    fig, ax = plt.subplots()
    ax.plot(durations, closed_series, label=f"window closed (ACH = {room.ach_closed:g} h^-1)")
    ax.plot(durations, open_series, label=f"window open (ACH = {room.ach_window_open:g} h^-1)")
    ax.axhline(
        outdoor_ah,
        linestyle="--",
        color="tab:red",
        label=f"outdoor AH = {outdoor_ah:.2f} g/m^3",
    )
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("indoor absolute humidity (g/m^3)")
    ax.set_title(
        "Window open vs closed: indoor absolute humidity over 30 minutes\n"
        "indoor 20 C / 70 %RH, outdoor 5 C / 85 %RH, 40 m^3"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print()
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
