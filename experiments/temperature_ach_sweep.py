"""Compare ventilation rates: indoor temperature vs time for several ACH.

Mirror of ``indoor_ah_vs_time_ach_sweep.py`` on the thermal side. Same
default room (V = 40 m^3, indoor 20 C, outdoor 5 C, illustrative
lumped C_eff = 500 000 J/K) with ACH swept over [0.5, 1, 2, 5, 10]
h^-1. Plots one indoor-T curve per rate against the outdoor-T
reference line, and prints a comparison table of indoor T at 5, 10,
and 15 minutes plus the dynamic ventilation energy removed after
15 minutes.

Purpose:
    - show that a real ventilation choice has TWO consequences that
      live on very different timescales: moisture (fast, minutes) and
      temperature (slow, hours);
    - illustrate the thermal cost of ventilation ACH by ACH, so the
      trade-off against the moisture benefit is visible per rate.

The moisture and thermal models are NOT combined here - two separate
sweeps in this directory show the two consequences side by side. A
combined moisture-thermal predictor is deferred to a later slice.

Illustrative note on C_eff (also printed on stdout):
    500 000 J/K is a placeholder from the thermal module, not
    calibrated to any specific building. A real deployment would
    identify C_eff from a measured indoor-temperature response to a
    controlled ventilation event; see ThermalProperties docstring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    predict_thermal_response,
)

INITIAL_INDOOR_TEMPERATURE_C = 20.0
OUTDOOR_TEMPERATURE_C = 5.0
ROOM_VOLUME_M3 = 40.0
ACH_VALUES = (0.5, 1.0, 2.0, 5.0, 10.0)
DURATION_MINUTES_MAX = 30.0
DURATION_STEPS = 121
TABLE_MINUTES = (5.0, 10.0, 15.0)
DYNAMIC_ENERGY_CHECKPOINT_MINUTES = 15.0
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "temperature_ach_sweep.png"
)


def main() -> None:
    """Run the sweep, save the plot, print the comparison table."""
    c_eff = ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
    print("Scenario")
    print("--------")
    print(
        f"indoor  : T_0 = {INITIAL_INDOOR_TEMPERATURE_C:g} C, "
        f"V = {ROOM_VOLUME_M3:g} m^3"
    )
    print(f"outdoor : T   = {OUTDOOR_TEMPERATURE_C:g} C (constant across the event)")
    print(f"ACH     : sweep over {list(ACH_VALUES)} h^-1")
    print()
    print(f"Illustrative effective thermal capacitance = {c_eff:,.0f} J/K")
    print("(Placeholder from thermal.py; not calibrated. See ThermalProperties.)")
    print()

    # --- Plot ---------------------------------------------------------------
    step = DURATION_MINUTES_MAX / (DURATION_STEPS - 1)
    durations = [i * step for i in range(DURATION_STEPS)]

    fig, ax = plt.subplots()
    for ach in ACH_VALUES:
        indoor_t_series = [
            predict_thermal_response(
                initial_indoor_temperature_c=INITIAL_INDOOR_TEMPERATURE_C,
                outdoor_temperature_c=OUTDOOR_TEMPERATURE_C,
                room_volume_m3=ROOM_VOLUME_M3,
                ach=ach,
                effective_thermal_capacity_j_per_k=c_eff,
                duration_minutes=t,
            ).final_temperature_c
            for t in durations
        ]
        ax.plot(durations, indoor_t_series, label=f"ACH = {ach:g} h^-1")

    ax.axhline(
        OUTDOOR_TEMPERATURE_C,
        linestyle="--",
        color="tab:red",
        label=f"outdoor T = {OUTDOOR_TEMPERATURE_C:g} C",
    )
    ax.set_xlabel("window-open duration (minutes)")
    ax.set_ylabel("indoor temperature (C)")
    ax.set_title(
        "Indoor temperature vs ventilation duration for several ACH\n"
        f"indoor {INITIAL_INDOOR_TEMPERATURE_C:g} C, outdoor "
        f"{OUTDOOR_TEMPERATURE_C:g} C, {ROOM_VOLUME_M3:g} m^3, "
        f"C_eff = {c_eff:,.0f} J/K (illustrative)"
    )
    ax.set_xlim(0.0, DURATION_MINUTES_MAX)
    ax.legend()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    print(f"saved plot to {OUTPUT_PATH}")
    print()

    # --- Table --------------------------------------------------------------
    print(
        f"Indoor T (C) after fixed durations, plus dynamic energy removed "
        f"at {DYNAMIC_ENERGY_CHECKPOINT_MINUTES:g} min, by ACH"
    )
    print("-" * 78)
    time_columns = "  ".join(f"{f'T @ {m:g} min':>12}" for m in TABLE_MINUTES)
    energy_col = (
        f"energy @ {DYNAMIC_ENERGY_CHECKPOINT_MINUTES:g} min"
    )
    header = f"{'ACH (h^-1)':>10}  " + time_columns + f"  {energy_col:>18}"
    units = f"{'':>10}  " + "  ".join(f"{'(C)':>12}" for _ in TABLE_MINUTES) + f"  {'(kWh)':>18}"
    print(header)
    print(units)
    print("-" * len(header))
    for ach in ACH_VALUES:
        row_temperatures = [
            predict_thermal_response(
                initial_indoor_temperature_c=INITIAL_INDOOR_TEMPERATURE_C,
                outdoor_temperature_c=OUTDOOR_TEMPERATURE_C,
                room_volume_m3=ROOM_VOLUME_M3,
                ach=ach,
                effective_thermal_capacity_j_per_k=c_eff,
                duration_minutes=t,
            ).final_temperature_c
            for t in TABLE_MINUTES
        ]
        energy_kwh = predict_thermal_response(
            initial_indoor_temperature_c=INITIAL_INDOOR_TEMPERATURE_C,
            outdoor_temperature_c=OUTDOOR_TEMPERATURE_C,
            room_volume_m3=ROOM_VOLUME_M3,
            ach=ach,
            effective_thermal_capacity_j_per_k=c_eff,
            duration_minutes=DYNAMIC_ENERGY_CHECKPOINT_MINUTES,
        ).energy_removed_kwh
        temp_cells = "  ".join(f"{t:>12.3f}" for t in row_temperatures)
        print(f"{ach:>10g}  " + temp_cells + f"  {energy_kwh:>18.4f}")

    print()
    print(
        "Note: this is only the THERMAL consequence of ventilation. The\n"
        "companion moisture experiment shows that the room's absolute\n"
        "humidity drops much faster than its temperature over the same\n"
        "durations - which is the whole point of considering moisture and\n"
        "thermal costs together in a future controller. The two models\n"
        "are not combined yet."
    )


if __name__ == "__main__":
    main()
