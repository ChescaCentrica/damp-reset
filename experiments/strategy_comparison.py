"""Side-by-side comparison of every optimiser strategy.

Runs the same indoor / outdoor / room / duration-set scenario through
each of the five strategies the optimiser exposes and prints one row
per strategy with:

    selected duration, water removed, final AH, final RH, final T,
    temperature drop, energy loss, reason for selection

The five strategies live in ``optimiser.py``:

    A) choose_minimum_energy_action
       min E_loss subject to a moisture target (and any comfort caps).

    B) optimise_max_moisture_under_energy_budget
       max water removed subject to an energy budget.

    C) optimise_max_moisture_under_comfort_limit
       max water removed subject to a temperature-drop cap.

    D) optimise_weighted_tradeoff (research only)
       max (water - lambda_energy * energy). Lambda has units of
       g/kWh and expresses the caller's exchange rate.

    E) optimise_marginal_efficiency_threshold
       select the last duration whose extension buys marginal
       efficiency (Δwater / Δenergy) strictly above a caller-set
       floor.

No strategy is declared "best". The purpose of this experiment is
to make each strategy's INPUT ASSUMPTIONS visible next to its
OUTPUT recommendation, so the reader can see what value judgement
drives the pick.

The specific numeric constraints and threshold values in this
experiment are ILLUSTRATIVE POC values chosen so the strategies can
be seen to differ. They are not validated damp / mould / health
thresholds.
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room
from optimiser import (
    OptimisationResult,
    VentilationConstraints,
    choose_minimum_energy_action,
    optimise_marginal_efficiency_threshold,
    optimise_max_moisture_under_comfort_limit,
    optimise_max_moisture_under_energy_budget,
    optimise_weighted_tradeoff,
)
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties


def _describe(result: OptimisationResult, strategy_label: str) -> None:
    """Print a fixed-width summary line for one strategy's result."""
    if result.feasible:
        duration = f"{result.selected_duration_minutes:.1f} min"
    else:
        duration = "infeasible"
    print(
        f"  {strategy_label:<32}  {duration:>11}  "
        f"{result.selected_prediction.water_removed_g:>+10.2f}  "
        f"{result.selected_prediction.final_absolute_humidity_g_m3:>9.3f}  "
        f"{result.selected_prediction.final_relative_humidity_pct:>8.2f}  "
        f"{result.selected_prediction.final_temperature_c:>8.3f}  "
        f"{result.selected_prediction.temperature_drop_c:>+8.3f}  "
        f"{result.selected_prediction.ventilation_energy_removed_kwh:>+9.4f}"
    )


def main() -> None:
    """Run every strategy on one shared scenario and print the comparison."""
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
    candidates: List[float] = [float(t) for t in range(0, 31)]

    print("Shared scenario")
    print("---------------")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3"
    )
    print(
        f"outdoor : T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} %"
    )
    print(f"ACH open: {room.ach_window_open:g} h^-1")
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print(f"candidates: durations {int(candidates[0])} to {int(candidates[-1])} min "
          f"in 1-min steps")
    print()
    print("Strategy inputs (illustrative POC values, not validated thresholds)")
    print("-------------------------------------------------------------------")
    print("  A) min energy         : target_final_absolute_humidity_g_m3 = 8.0")
    print("  B) max water / budget : max_energy_loss_kwh = 0.15")
    print("  C) max water / comfort: max_temperature_drop_c = 1.0")
    print("  D) weighted trade-off : lambda_energy = 500.0 g/kWh")
    print("  E) marginal threshold : minimum_marginal_g_per_kwh = 600.0")
    print()

    strategy_args = dict(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=candidates,
    )

    results = [
        (
            "A) min energy",
            choose_minimum_energy_action(
                **strategy_args,
                constraints=VentilationConstraints(
                    target_final_absolute_humidity_g_m3=8.0
                ),
            ),
        ),
        (
            "B) max water / budget",
            optimise_max_moisture_under_energy_budget(
                **strategy_args,
                constraints=VentilationConstraints(
                    max_energy_loss_kwh=0.15
                ),
            ),
        ),
        (
            "C) max water / comfort",
            optimise_max_moisture_under_comfort_limit(
                **strategy_args,
                constraints=VentilationConstraints(
                    max_temperature_drop_c=1.0
                ),
            ),
        ),
        (
            "D) weighted (lambda=500)",
            optimise_weighted_tradeoff(
                **strategy_args,
                constraints=VentilationConstraints(),
                lambda_energy=500.0,
            ),
        ),
        (
            "E) marginal threshold",
            optimise_marginal_efficiency_threshold(
                **strategy_args,
                constraints=VentilationConstraints(
                    minimum_marginal_g_per_kwh=600.0
                ),
            ),
        ),
    ]

    header = (
        f"  {'strategy':<32}  {'duration':>11}  "
        f"{'water':>10}  {'final AH':>9}  "
        f"{'final RH':>8}  {'final T':>8}  "
        f"{'T drop':>8}  {'energy':>9}"
    )
    units = (
        f"  {'':<32}  {'(min)':>11}  "
        f"{'(g)':>10}  {'(g/m^3)':>9}  "
        f"{'(%)':>8}  {'(C)':>8}  "
        f"{'(K)':>8}  {'(kWh)':>9}"
    )
    print("Results")
    print("-------")
    print(header)
    print(units)
    print("  " + "-" * (len(header) - 2))
    for label, result in results:
        _describe(result, label)
    print()

    print("Reason strings (what each strategy is saying)")
    print("---------------------------------------------")
    for label, result in results:
        print(f"  {label}")
        print(f"    {result.reason}")
        print()

    print("What assumption or value judgement drives each strategy?")
    print("--------------------------------------------------------")
    print(
        "  A) Min energy under moisture target"
    )
    print(
        "     Assumption: the caller has a specific moisture goal that must"
    )
    print(
        "     be met; any candidate that meets it is acceptable. Minimises"
    )
    print(
        "     heating cost within that acceptance set. Requires a MOISTURE"
    )
    print(
        "     TARGET to be specified - the caller must state what 'dry"
    )
    print(
        "     enough' means."
    )
    print()
    print("  B) Max water under energy budget")
    print(
        "     Assumption: the caller has a fixed heating budget for the"
    )
    print(
        "     event. Extracts as much water as possible without exceeding"
    )
    print(
        "     it. Requires an ENERGY BUDGET - the caller must state how"
    )
    print(
        "     much heat loss they will tolerate per event."
    )
    print()
    print("  C) Max water under comfort limit")
    print(
        "     Assumption: the temperature drop is what occupants notice"
    )
    print(
        "     and dislike, not the energy meter. Requires a COMFORT CAP -"
    )
    print(
        "     the caller must state how far the room may cool per event."
    )
    print()
    print("  D) Weighted trade-off (research only)")
    print(
        "     Assumption: moisture benefit and thermal cost can be reduced"
    )
    print(
        "     to a single scalar via a linear exchange rate. Lambda has"
    )
    print(
        "     UNITS of g/kWh; its numeric value is an economic / preference"
    )
    print(
        "     judgement, NOT a physics constant. There is no universally"
    )
    print(
        "     valid lambda; this strategy is exposed to make the choice of"
    )
    print(
        "     exchange rate explicit, not to hide it."
    )
    print()
    print("  E) Marginal efficiency threshold")
    print(
        "     Assumption: extending the event is worthwhile only while the"
    )
    print(
        "     MARGINAL water-per-kWh stays above a caller-set floor. Picks"
    )
    print(
        "     the last duration before diminishing returns cross the floor."
    )
    print(
        "     Requires a MARGINAL THRESHOLD - the caller must state how"
    )
    print(
        "     efficient an extension has to be to justify itself."
    )
    print()
    print(
        "No strategy is universally best. Each requires the caller to state"
    )
    print(
        "a different kind of preference; the reasonable choice depends on"
    )
    print(
        "which of those preferences the caller can defend with an evidence"
    )
    print(
        "base. This experiment does NOT rank the strategies."
    )


if __name__ == "__main__":
    main()
