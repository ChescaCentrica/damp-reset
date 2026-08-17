"""Three separate plots of the combined moisture + thermal + RH model.

For the shared default scenario (indoor 20 C / 70 %RH, outdoor 5 C /
85 %RH, 40 m^3, ACH = 5 h^-1, illustrative C_eff = 500 000 J/K),
sweeps window-open duration from 0 to 30 minutes and produces three
independent figures, one per moisture / thermal / RH coordinate.

Output files (in ``outputs/``):
    combined_ah_vs_time.png
    combined_temperature_vs_time.png
    combined_rh_vs_time.png

Why the AH and RH curves are NOT the same shape:

    AH depends only on the moisture content of the room air; the
    moisture model gives it as a first-order exponential decay from
    12.07 g/m^3 toward the outdoor value of 5.77 g/m^3 with time
    constant tau_moisture = 1 / ACH = 1/5 h = 12 min.

    RH is 100 * P_v / P_sat(T). Both the numerator (P_v, which is
    proportional to AH) AND the denominator (P_sat, which is a strong
    non-linear function of T) change during a ventilation event. The
    room cools while it dries, and cooler air holds less water at
    saturation, so P_sat(T) shrinks. RH therefore falls LESS than AH
    does in relative terms, because part of the drop in AH is
    cancelled by the drop in P_sat.

    The two curves are on different time constants as well:
    tau_moisture ~= 12 min in this scenario, while
    tau_thermal   = C_eff / H_vent ~= 124 min. Moisture equilibrates
    an order of magnitude faster than temperature, so on the 0-30
    min window plotted here you see almost the full moisture
    response but only the leading edge of the thermal response.
    That asymmetry is the whole reason a smart ventilation
    controller can trade moisture against thermal cost.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from ventilation import predict_ventilation

DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121  # 0.25-min resolution
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
AH_PLOT_PATH = OUTPUT_DIR / "combined_ah_vs_time.png"
TEMPERATURE_PLOT_PATH = OUTPUT_DIR / "combined_temperature_vs_time.png"
RH_PLOT_PATH = OUTPUT_DIR / "combined_rh_vs_time.png"


def main() -> None:
    """Sweep durations through the combined model and save three plots."""
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

    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations = [i * step for i in range(DURATION_STEPS)]
    predictions = [
        predict_ventilation(
            room=room,
            thermal_properties=thermal_props,
            outdoor=outdoor,
            duration_minutes=t,
        )
        for t in durations
    ]
    ah_series = [p.moisture.final_absolute_humidity_g_m3 for p in predictions]
    temperature_series = [p.thermal.final_temperature_c for p in predictions]
    rh_series = [p.final_relative_humidity_pct for p in predictions]

    outdoor_ah = outdoor.absolute_humidity

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: absolute humidity -----------------------------------------
    fig, ax = plt.subplots()
    ax.plot(durations, ah_series, label="indoor AH")
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
        "combined moisture / thermal model, ACH = 5 h^-1"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()
    fig.savefig(AH_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {AH_PLOT_PATH}")

    # --- Plot 2: indoor temperature ----------------------------------------
    fig, ax = plt.subplots()
    ax.plot(durations, temperature_series, label="indoor T")
    ax.axhline(
        outdoor.temperature_c,
        linestyle="--",
        color="tab:red",
        label=f"outdoor T = {outdoor.temperature_c:g} C",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor temperature (C)")
    ax.set_title(
        "Indoor temperature vs ventilation duration\n"
        f"combined model, ACH = 5 h^-1, C_eff = "
        f"{thermal_props.effective_thermal_capacity_j_per_k:,.0f} J/K (illustrative)"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()
    fig.savefig(TEMPERATURE_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {TEMPERATURE_PLOT_PATH}")

    # --- Plot 3: relative humidity -----------------------------------------
    fig, ax = plt.subplots()
    ax.plot(durations, rh_series, label="indoor RH (from final T and final AH)")
    ax.axhline(
        outdoor.relative_humidity_percent,
        linestyle="--",
        color="tab:red",
        label=f"outdoor RH = {outdoor.relative_humidity_percent:g} %",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor relative humidity (%)")
    ax.set_title(
        "Indoor relative humidity vs ventilation duration\n"
        "combined model; RH derived from PREDICTED final T and final AH"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()
    fig.savefig(RH_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {RH_PLOT_PATH}")


if __name__ == "__main__":
    main()
