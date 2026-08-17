"""Sensitivity of each optimiser strategy to its own control parameter.

Sweeps one caller-set parameter per strategy over a plausible range
and records the selected duration at every setpoint. Produces one
small table per sweep so the reader can see, at a glance:

    * which parameters barely move the recommendation;
    * which parameters produce large swings in the recommendation;
    * which parameters would need strong justification before any
      production use, because arbitrary changes at similar-looking
      values produce very different actions.

No strategy is declared best. No numeric setpoint in this file is
declared correct. The purpose is to expose the instability inside
the control logic rather than hide it - every strategy in the
optimiser layer depends on a caller-set value, and this experiment
shows how much of the decision each of those values actually owns.

The scenario is fixed (indoor 20 C / 70 %RH, outdoor 5 C / 85 %RH,
40 m^3 room, ACH = 5 h^-1, illustrative C_eff) so the tables read
as apples-to-apples comparisons across the five strategies.
"""

import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

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

CANDIDATE_DURATIONS_MINUTES: List[float] = [float(t) for t in range(0, 31)]


def _fmt_duration(result: OptimisationResult) -> str:
    """Format the selected duration for the summary table."""
    if not result.feasible or (
        result.selected_duration_minutes
        != result.selected_duration_minutes
    ):
        return "infeasible"
    return f"{result.selected_duration_minutes:.1f} min"


def _fmt_water_energy(result: OptimisationResult) -> Tuple[str, str]:
    """Format water and energy for the summary table."""
    if not result.feasible or (
        result.selected_duration_minutes
        != result.selected_duration_minutes
    ):
        return ("-", "-")
    return (
        f"{result.selected_prediction.water_removed_g:+.2f} g",
        f"{result.selected_prediction.ventilation_energy_removed_kwh:+.4f} kWh",
    )


def _run_sweep(
    parameter_label: str,
    parameter_unit: str,
    values: Iterable[float],
    strategy_call: Callable[[float], OptimisationResult],
    stability_interpretation: str,
) -> None:
    """Print one sensitivity table for a single strategy."""
    print(f"Sensitivity: {parameter_label}")
    print("-" * (len(parameter_label) + 13))
    header = (
        f"  {parameter_label + ' ' + parameter_unit:>26}  "
        f"{'selected':>11}  {'water':>10}  {'energy':>13}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    picks: List[Optional[float]] = []
    for value in values:
        result = strategy_call(value)
        duration_str = _fmt_duration(result)
        water_str, energy_str = _fmt_water_energy(result)
        print(
            f"  {value:>26g}  {duration_str:>11}  "
            f"{water_str:>10}  {energy_str:>13}"
        )
        if (
            result.feasible
            and result.selected_duration_minutes
            == result.selected_duration_minutes
        ):
            picks.append(result.selected_duration_minutes)
        else:
            picks.append(None)

    # Summarise sensitivity: unique picks + range + stability verdict.
    finite_picks = [p for p in picks if p is not None]
    if not finite_picks:
        print("  (no feasible pick at any value in the sweep)")
    else:
        unique = sorted(set(finite_picks))
        min_pick, max_pick = min(finite_picks), max(finite_picks)
        print()
        print(
            f"  {len(unique)} distinct pick(s) across "
            f"{len(list(values))} setpoints; "
            f"range = {min_pick:g} to {max_pick:g} min "
            f"(spread {max_pick - min_pick:g} min)."
        )
    print(f"  Interpretation: {stability_interpretation}")
    print()


def main() -> None:
    """Run all five sensitivity sweeps and print the summary at the end."""
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
    common_args = dict(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
    )

    print("Scenario (held fixed across every sweep)")
    print("----------------------------------------")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3, ACH = {room.ach_window_open:g} h^-1"
    )
    print(
        f"outdoor : T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} %"
    )
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print(
        f"candidates: durations {int(CANDIDATE_DURATIONS_MINUTES[0])}-"
        f"{int(CANDIDATE_DURATIONS_MINUTES[-1])} min in 1-min steps"
    )
    print()

    # A) Moisture target (final indoor AH ceiling). Strategy: min energy.
    #    At this scenario final AH ranges from 12.07 (t=0) down to
    #    ~6.28 (t=30). Sweep the target across values that produce
    #    achievable and unachievable cases.
    moisture_targets = [12.0, 10.0, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0]
    _run_sweep(
        parameter_label="target_final_absolute_humidity_g_m3",
        parameter_unit="",
        values=moisture_targets,
        strategy_call=lambda target: choose_minimum_energy_action(
            **common_args,
            constraints=VentilationConstraints(
                target_final_absolute_humidity_g_m3=target
            ),
        ),
        stability_interpretation=(
            "moving the target closer to outdoor AH (~5.77 g/m^3) "
            "pushes the pick toward longer durations quickly; targets "
            "already met at t = 0 collapse the pick to 0 min. This "
            "parameter shapes the pick strongly across its plausible "
            "residential range."
        ),
    )

    # B) Comfort cap: max temperature drop. Strategy: max water under
    #    comfort limit. At this scenario T drop varies from 0 (t=0) up
    #    to ~3.22 K (t=30). Sweep the cap across residentially
    #    plausible values.
    comfort_caps = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    _run_sweep(
        parameter_label="max_temperature_drop_c",
        parameter_unit="",
        values=comfort_caps,
        strategy_call=lambda cap: optimise_max_moisture_under_comfort_limit(
            **common_args,
            constraints=VentilationConstraints(max_temperature_drop_c=cap),
        ),
        stability_interpretation=(
            "the comfort cap gates the last-feasible duration and the "
            "pick moves roughly linearly with the cap in this scenario "
            "(the thermal ODE is on its near-linear leading edge over "
            "0-30 min). This parameter has a strong, roughly proportional "
            "effect."
        ),
    )

    # C) Energy budget. Strategy: max water under energy budget. At
    #    this scenario energy ranges from 0 (t=0) up to ~0.45 kWh
    #    (t=30). Sweep the budget across a mix of tight and loose
    #    values.
    energy_budgets = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    _run_sweep(
        parameter_label="max_energy_loss_kwh",
        parameter_unit="",
        values=energy_budgets,
        strategy_call=lambda budget: optimise_max_moisture_under_energy_budget(
            **common_args,
            constraints=VentilationConstraints(max_energy_loss_kwh=budget),
        ),
        stability_interpretation=(
            "the budget selects the longest duration whose energy still "
            "fits inside it; the pick moves in ~5-minute steps as the "
            "budget rises. Strong, near-linear sensitivity across the "
            "range."
        ),
    )

    # D) Weighted trade-off: lambda. Research-only strategy. Sweep
    #    lambda across three orders of magnitude to expose how quickly
    #    the pick moves as the exchange rate changes.
    lambda_values = [0.0, 50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 20_000.0]
    _run_sweep(
        parameter_label="lambda_energy",
        parameter_unit="(g/kWh)",
        values=lambda_values,
        strategy_call=lambda lam: optimise_weighted_tradeoff(
            **common_args,
            constraints=VentilationConstraints(),
            lambda_energy=lam,
        ),
        stability_interpretation=(
            "the pick is highly sensitive to lambda around the g/kWh "
            "scale of the ventilation itself (~300-1200 g/kWh across "
            "the sweep). A change from 500 to 1000 alone can move the "
            "pick by 5+ minutes; lambda values chosen without an "
            "explicit economic basis would drive very different "
            "recommendations. Requires strong justification before "
            "production use."
        ),
    )

    # E) Marginal-efficiency threshold. Strategy: last duration before
    #    marginal g/kWh crosses the floor.
    marginal_thresholds = [100.0, 300.0, 500.0, 600.0, 700.0, 900.0, 1200.0, 2000.0]
    _run_sweep(
        parameter_label="minimum_marginal_g_per_kwh",
        parameter_unit="",
        values=marginal_thresholds,
        strategy_call=lambda threshold: optimise_marginal_efficiency_threshold(
            **common_args,
            constraints=VentilationConstraints(
                minimum_marginal_g_per_kwh=threshold
            ),
        ),
        stability_interpretation=(
            "the marginal efficiency across candidate intervals in this "
            "scenario spans roughly 300-1200 g/kWh, so thresholds inside "
            "that band shift the pick in large discrete steps; thresholds "
            "outside the band pin the pick to either 0 min or the longest "
            "candidate. High sensitivity in the band a caller would most "
            "plausibly set."
        ),
    )

    print("Overall observations (from this scenario only)")
    print("----------------------------------------------")
    print(
        "- Every strategy's recommendation is sensitive to its own control"
    )
    print(
        "  parameter over the plausible residential range."
    )
    print(
        "- No parameter is a 'set-and-forget' knob: changing a moisture"
    )
    print(
        "  target by 1 g/m^3, or a comfort cap by 0.5 K, or an energy"
    )
    print(
        "  budget by 0.05 kWh, moves the pick by several minutes."
    )
    print(
        "- Lambda in the weighted trade-off has the sharpest sensitivity"
    )
    print(
        "  because it directly rates two incommensurable quantities;"
    )
    print(
        "  small changes in the value flip the balance."
    )
    print(
        "- The marginal-threshold strategy moves in discrete steps as"
    )
    print(
        "  the threshold crosses each interval's efficiency, so its"
    )
    print(
        "  recommendation is stable inside those windows but jumps at"
    )
    print(
        "  their boundaries."
    )
    print()
    print(
        "None of these parameters have a universally correct value. Every"
    )
    print(
        "one of them is a control preference the caller must be able to"
    )
    print(
        "defend on evidence. This experiment does NOT pick preferred"
    )
    print(
        "settings; it shows how much of the decision each parameter owns."
    )


if __name__ == "__main__":
    main()
