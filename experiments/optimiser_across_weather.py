"""Preferred optimiser run across several outdoor weather scenarios.

Holds the room state constant (indoor 20 C / 70 %RH, V = 40 m^3,
ACH_open = 5 h^-1, illustrative C_eff) and varies only the outdoor
conditions. For each scenario the script:

    1. Computes outdoor absolute humidity via ``AirState``.
    2. Computes indoor - outdoor absolute humidity difference (the
       drying potential from the psychrometric layer).
    3. Runs the preferred constraint-based strategy
       ``choose_minimum_energy_action`` with the SAME moisture
       target and comfort cap across every scenario.

The purpose is to sanity-check that the optimiser's recommendation
depends on the outdoor conditions - identical inputs would produce
identical recommendations, but different weather should produce
different picks.

Illustrative POC control constraints (fixed across scenarios so the
recommendation reflects the weather change, not a change in the
caller's preferences):

    target_final_absolute_humidity_g_m3 = 8.0
    max_temperature_drop_c              = 2.0

The primary optimiser reports ``feasible=True`` when a candidate
achieves the moisture target under the comfort cap, and
``feasible=False`` otherwise (with an explicit fallback pick). The
"target achieved?" column reads that flag directly.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room
from optimiser import (
    VentilationConstraints,
    choose_minimum_energy_action,
)
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties

CANDIDATE_DURATIONS_MINUTES: List[float] = [float(t) for t in range(0, 31)]


@dataclass(frozen=True)
class OutdoorScenario:
    """Named outdoor condition for the sweep."""

    label: str
    temperature_c: float
    relative_humidity_pct: float


SCENARIOS = (
    OutdoorScenario("cold & humid", 5.0, 85.0),
    OutdoorScenario("cold & dry", 5.0, 40.0),
    OutdoorScenario("cool & humid", 10.0, 85.0),
    OutdoorScenario("mild & humid", 15.0, 85.0),
    OutdoorScenario("mild & dry", 15.0, 40.0),
    OutdoorScenario("mild & moderate", 12.0, 60.0),
    OutdoorScenario("warm & dry", 22.0, 40.0),
    OutdoorScenario("warm & humid", 22.0, 85.0),
)


def main() -> None:
    """Run every scenario through the primary optimiser and print the table."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    thermal_props = ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )
    constraints = VentilationConstraints(
        target_final_absolute_humidity_g_m3=8.0,
        max_temperature_drop_c=2.0,
    )

    indoor_ah = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    ).absolute_humidity

    print("Room (held fixed)")
    print("-----------------")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3, ACH_open = {room.ach_window_open:g} h^-1"
    )
    print(f"indoor AH: {indoor_ah:.3f} g/m^3")
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print()
    print(
        "Preferred optimiser: choose_minimum_energy_action "
        "(min E_loss under moisture target + comfort cap)"
    )
    print(
        f"  target_final_absolute_humidity_g_m3 = "
        f"{constraints.target_final_absolute_humidity_g_m3:g}  "
        f"(illustrative POC value)"
    )
    print(
        f"  max_temperature_drop_c              = "
        f"{constraints.max_temperature_drop_c:g}   "
        f"(illustrative POC value)"
    )
    print(
        f"  candidates: 0-{int(CANDIDATE_DURATIONS_MINUTES[-1])} min in 1-min steps"
    )
    print()

    header = (
        f"  {'scenario':<15}  {'out T':>6}  {'out RH':>7}  "
        f"{'out AH':>8}  {'drying pot.':>11}  "
        f"{'duration':>10}  {'water':>10}  "
        f"{'energy':>10}  {'T drop':>8}  {'target?':>8}"
    )
    units = (
        f"  {'':<15}  {'(C)':>6}  {'(%)':>7}  "
        f"{'(g/m^3)':>8}  {'(g/m^3)':>11}  "
        f"{'(min)':>10}  {'(g)':>10}  "
        f"{'(kWh)':>10}  {'(K)':>8}  {'':>8}"
    )
    print(header)
    print(units)
    print("  " + "-" * (len(header) - 2))

    selected_durations = []
    for scenario in SCENARIOS:
        outdoor = AirState(
            temperature_c=scenario.temperature_c,
            relative_humidity_percent=scenario.relative_humidity_pct,
        )
        outdoor_ah = outdoor.absolute_humidity
        drying_potential = indoor_ah - outdoor_ah

        result = choose_minimum_energy_action(
            room=room,
            outdoor=outdoor,
            thermal_properties=thermal_props,
            candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
            constraints=constraints,
        )
        if result.feasible:
            duration_str = f"{result.selected_duration_minutes:.1f}"
        elif (
            result.selected_duration_minutes
            != result.selected_duration_minutes  # NaN check
        ):
            duration_str = "n/a"
        else:
            duration_str = f"{result.selected_duration_minutes:.1f}*"
        target_str = "yes" if result.feasible else "no"

        print(
            f"  {scenario.label:<15}  "
            f"{scenario.temperature_c:>6g}  "
            f"{scenario.relative_humidity_pct:>7g}  "
            f"{outdoor_ah:>8.3f}  "
            f"{drying_potential:>+11.3f}  "
            f"{duration_str:>10}  "
            f"{result.selected_prediction.water_removed_g:>+10.2f}  "
            f"{result.selected_prediction.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{result.selected_prediction.temperature_drop_c:>+8.3f}  "
            f"{target_str:>8}"
        )
        if (
            result.feasible
            or result.selected_duration_minutes
            == result.selected_duration_minutes
        ):
            selected_durations.append(result.selected_duration_minutes)

    print()
    print(
        "  * duration reported but the target could not be achieved; "
        "the optimiser fell back to the maximum-drying candidate that "
        "stays within the comfort cap. See the strategy docstring."
    )
    print()

    # Verification: the optimiser must produce different picks across
    # different weather scenarios, otherwise this whole layer is doing
    # nothing new relative to a fixed-time timer.
    if selected_durations:
        unique_picks = sorted(set(selected_durations))
        print(
            f"Verification: "
            f"{len(unique_picks)} distinct duration(s) selected across "
            f"{len(SCENARIOS)} scenarios; range = "
            f"{min(selected_durations):g} to "
            f"{max(selected_durations):g} min."
        )
        if len(unique_picks) == 1:
            print(
                "  WARNING: every scenario produced the same duration. "
                "The optimiser is not sensitive to the outdoor conditions "
                "in this sweep - a design regression."
            )
        else:
            print(
                "  Multiple distinct picks confirm that the optimiser's "
                "recommendation genuinely depends on the outdoor conditions."
            )


if __name__ == "__main__":
    main()
