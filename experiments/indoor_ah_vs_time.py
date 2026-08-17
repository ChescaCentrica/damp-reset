"""Plot indoor absolute humidity vs window-open duration for the default room.

Uses the same scenario as ``examples/moisture_duration_comparison.py``:
indoor 20 C / 70 %RH, outdoor 5 C / 85 %RH, 40 m^3 room, ACH = 5 h^-1.
Sweeps window-open duration from 0 to 30 minutes and plots the resulting
indoor absolute humidity, with the outdoor absolute humidity drawn as a
horizontal reference. The plot is saved to ``outputs/indoor_ah_vs_time.png``.

Why the curve is exponential and why returns diminish as indoor AH
approaches outdoor AH:

The well-mixed moisture balance with no internal sources is
    dC/dt = n * (C_out - C)                         (C is indoor AH, n is ACH)
The instantaneous drying RATE is proportional to the gap between indoor
and outdoor absolute humidity. As indoor AH falls toward outdoor AH the
gap shrinks, so the rate itself shrinks proportionally - a first-order
linear ODE, whose solution is
    C(t) = C_out + (C_0 - C_out) * exp(-n * t).
The exponential shape comes directly from that self-similar decay. Two
consequences for ventilation strategy:

  * A ventilation event closes the gap by a factor 1/e every 1/n hours
    (~12 minutes at n=5). The first few minutes remove the most water
    per minute because the driving gap is largest.
  * Beyond ~3 time constants (~36 minutes at n=5) more than 95 % of
    the reachable improvement has already happened - additional
    window-open minutes buy less and less moisture removal until the
    room asymptotes at outdoor AH. If outdoor air isn't dry enough,
    no amount of ventilation can push the room below it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room, predict_room_moisture
from psychrometrics import AirState

DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121  # 0.25-min resolution across 0-30 min
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "indoor_ah_vs_time.png"


def main() -> None:
    """Sweep window-open duration, compute indoor AH, and save the plot."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations_minutes = [i * step for i in range(DURATION_STEPS)]
    indoor_ah_series = [
        predict_room_moisture(room, outdoor, duration_minutes=t).final_absolute_humidity_g_m3
        for t in durations_minutes
    ]
    outdoor_ah = outdoor.absolute_humidity

    fig, ax = plt.subplots()
    ax.plot(durations_minutes, indoor_ah_series, label="indoor AH")
    ax.axhline(
        outdoor_ah,
        linestyle="--",
        color="tab:red",
        label=f"outdoor AH = {outdoor_ah:.2f} g/m^3",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor absolute humidity (g/m^3)")
    ax.set_title(
        "Indoor absolute humidity vs ventilation duration\n"
        "indoor 20 C / 70 %RH, outdoor 5 C / 85 %RH, 40 m^3, ACH = 5 h^-1"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
