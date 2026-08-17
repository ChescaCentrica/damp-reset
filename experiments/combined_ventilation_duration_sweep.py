"""Combined moisture + thermal + RH sweep across window-open durations.

Runs the composed ventilation predictor at 0, 2, 3, 5, 10, 15, and 20
minutes for the shared default scenario and prints one row per duration
with:
    - final indoor absolute humidity            (g/m^3)
    - water mass removed from the room air      (g, positive = drying)
    - final indoor temperature                  (C)
    - temperature drop (initial - final)        (K, positive = cooled)
    - final relative humidity                   (%, from final T + final AH)
    - dynamic ventilation energy removed        (kWh, from C_eff * dT)

No optimisation - the aim is only to make the joint moisture/thermal/RH
picture readable at a glance.

Illustrative C_eff is the placeholder from thermal.py; not tuned, not
calibrated to any specific building.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from ventilation import predict_ventilation

DURATIONS_MINUTES = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)


def main() -> None:
    """Run the sweep and print the combined comparison table."""
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

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, RH = "
        f"{room.indoor_relative_humidity_pct:g} %, V = {room.volume_m3:g} m^3"
    )
    print(
        f"outdoor : T = {outdoor.temperature_c:g} C, RH = "
        f"{outdoor.relative_humidity_percent:g} %"
    )
    print(f"vent    : window open, ACH = {room.ach_window_open:g} h^-1")
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} J/K "
        "(illustrative)"
    )
    print()

    header = (
        f"{'minutes':>8}  "
        f"{'final AH':>10}  "
        f"{'water rem':>10}  "
        f"{'final T':>9}  "
        f"{'T drop':>8}  "
        f"{'final RH':>9}  "
        f"{'energy':>9}"
    )
    units = (
        f"{'(min)':>8}  "
        f"{'(g/m^3)':>10}  "
        f"{'(g)':>10}  "
        f"{'(C)':>9}  "
        f"{'(K)':>8}  "
        f"{'(%)':>9}  "
        f"{'(kWh)':>9}"
    )
    print(header)
    print(units)
    print("-" * len(header))
    for minutes in DURATIONS_MINUTES:
        r = predict_ventilation(
            room=room,
            thermal_properties=thermal_props,
            outdoor=outdoor,
            duration_minutes=minutes,
        )
        temperature_drop_k = (
            r.thermal.initial_temperature_c - r.thermal.final_temperature_c
        )
        print(
            f"{minutes:>8.1f}  "
            f"{r.moisture.final_absolute_humidity_g_m3:>10.3f}  "
            f"{r.moisture.water_removed_g:>+10.2f}  "
            f"{r.thermal.final_temperature_c:>9.3f}  "
            f"{temperature_drop_k:>+8.3f}  "
            f"{r.final_relative_humidity_pct:>9.2f}  "
            f"{r.thermal.energy_removed_kwh:>9.4f}"
        )


if __name__ == "__main__":
    main()
