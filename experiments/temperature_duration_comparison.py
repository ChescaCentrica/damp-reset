"""Plot indoor temperature vs window-open duration for the default room.

Same scenario as the moisture experiments (indoor 20 C / 70 %RH,
outdoor 5 C / 85 %RH, 40 m^3, ACH = 5 h^-1) plus a single illustrative
lumped effective thermal capacitance from the thermal module. Sweeps
window-open duration from 0 to 30 minutes and plots the resulting
indoor temperature, with the outdoor temperature drawn as a horizontal
reference. Saved to ``outputs/temperature_duration_comparison.png``.

Note on the C_eff value used
============================
This experiment uses ``ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K``
straight from the thermal module (documented as a placeholder for
demos; not calibrated to any specific building). The value is NOT
tuned to make the resulting curve visually punchy. In particular the
30-minute drop at this C_eff is only a few kelvin - well short of the
outdoor asymptote - because the thermal time constant tau = C_eff /
H_vent is about two hours for this room. That is the physically
honest picture of a furnished residential room: opening a window for
half an hour barely closes any of the indoor-outdoor gap. The illustrative
constant is printed prominently at the top of the console output so a
reader can see what value the plot depends on before drawing any
conclusions.

The moisture experiments in this directory tell the complementary
story: the moisture gap closes far faster than the temperature gap in
the same 30 minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    predict_thermal_response,
    ventilation_heat_loss_coefficient,
)

INITIAL_INDOOR_TEMPERATURE_C = 20.0
OUTDOOR_TEMPERATURE_C = 5.0
ROOM_VOLUME_M3 = 40.0
ACH = 5.0
DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121  # 0.25-min resolution across 0-30 min
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "temperature_duration_comparison.png"
)


def main() -> None:
    """Sweep window-open duration, compute indoor temperature, save the plot."""
    c_eff_j_per_k = ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
    h_vent = ventilation_heat_loss_coefficient(ROOM_VOLUME_M3, ACH)
    tau_minutes = c_eff_j_per_k / h_vent / 60.0

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T_0 = {INITIAL_INDOOR_TEMPERATURE_C:g} C, V = {ROOM_VOLUME_M3:g} m^3"
    )
    print(f"outdoor : T   = {OUTDOOR_TEMPERATURE_C:g} C (constant across the event)")
    print(f"vent    : ACH = {ACH:g} h^-1 (window open)")
    print()
    print(
        f"Illustrative effective thermal capacitance = "
        f"{c_eff_j_per_k:,.0f} J/K"
    )
    print("(Placeholder value from thermal.py; not calibrated to any specific")
    print(" building. See ThermalProperties docstring for the identification")
    print(" procedure a real deployment would use.)")
    print()
    print(f"Derived time constant tau = C_eff / H_vent = {tau_minutes:.1f} min")
    print("(30 minutes is far below one tau, so the room only cools slightly.)")
    print()

    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations = [i * step for i in range(DURATION_STEPS)]
    indoor_temperature_series = [
        predict_thermal_response(
            initial_indoor_temperature_c=INITIAL_INDOOR_TEMPERATURE_C,
            outdoor_temperature_c=OUTDOOR_TEMPERATURE_C,
            room_volume_m3=ROOM_VOLUME_M3,
            ach=ACH,
            effective_thermal_capacity_j_per_k=c_eff_j_per_k,
            duration_minutes=t,
        ).final_temperature_c
        for t in durations
    ]

    fig, ax = plt.subplots()
    ax.plot(durations, indoor_temperature_series, label="indoor T")
    ax.axhline(
        OUTDOOR_TEMPERATURE_C,
        linestyle="--",
        color="tab:red",
        label=f"outdoor T = {OUTDOOR_TEMPERATURE_C:g} C",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor temperature (C)")
    ax.set_title(
        "Indoor temperature vs ventilation duration\n"
        f"C_eff = {c_eff_j_per_k:,.0f} J/K (illustrative), "
        f"tau = {tau_minutes:.0f} min"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
