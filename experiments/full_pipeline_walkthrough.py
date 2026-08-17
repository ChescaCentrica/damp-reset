"""End-to-end walk-through of the current damp-reset POC pipeline.

Runs a single scenario through every layer of the system and prints
each layer's output:

    inputs                (indoor T/RH, outdoor T/RH, room, candidates)
      -> psychrometric    (indoor AH, outdoor AH, drying potential)
      -> moisture         (final AH per candidate)
      -> thermal          (final T, temperature drop, energy loss per candidate)
      -> composed         (final RH per candidate, water removed)
      -> constraints      (feasibility of each candidate)
      -> optimiser        (recommended duration under the POC default)
      -> presentation     (human-readable explanation)

Sections produced:
    1. Scenario inputs.
    2. Candidate outcome table (all durations).
    3. Feasibility table under illustrative constraints.
    4. Pareto frontier plot (moisture vs energy).
    5. Selected action and human-readable explanation.
    6. Comparison table across all five optimisation strategies.
    7. Sensitivity table on the primary strategy's moisture target.
    8. Unit test result summary (from the test suite).
    9. Assumptions and limitations still standing.
   10. What the optimiser currently LACKS before it can be called a
       damp / mould prevention optimiser rather than a moisture-
       reduction optimiser.

No new physics, no new optimisation logic. Every table and figure
is a driver over existing modules.

The scenario, moisture target, comfort cap, and other setpoints
are illustrative POC values. See the module docstrings on
``optimiser`` and ``VentilationConstraints`` for the calibration
warning.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from optimiser import (
    OptimisationResult,
    VentilationConstraints,
    choose_minimum_energy_action,
    evaluate_candidate_durations,
    evaluate_candidate_durations_with_constraints,
    optimise_marginal_efficiency_threshold,
    optimise_max_moisture_under_comfort_limit,
    optimise_max_moisture_under_energy_budget,
    optimise_weighted_tradeoff,
    pareto_efficient_indices,
    recommend_ventilation_action,
)
from presentation import format_recommendation
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "outputs" / "plots" / "full_pipeline_pareto.png"
CANDIDATE_DURATIONS_MINUTES = [float(t) for t in range(0, 31)]


def section(title: str) -> None:
    """Print a labelled section header."""
    print()
    print(title)
    print("=" * len(title))


def main() -> None:
    """Run the full pipeline walkthrough and print every section."""
    # ---- Inputs -----------------------------------------------------------
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
    constraints = VentilationConstraints(
        target_final_absolute_humidity_g_m3=8.0,
        max_temperature_drop_c=2.0,
    )

    section("1) Scenario inputs")
    print(
        f"indoor  : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3, ACH_open = {room.ach_window_open:g} h^-1"
    )
    print(
        f"outdoor : T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} %"
    )
    print(
        f"C_eff   : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative POC value)"
    )
    print(
        f"candidates: durations {int(CANDIDATE_DURATIONS_MINUTES[0])}-"
        f"{int(CANDIDATE_DURATIONS_MINUTES[-1])} min in 1-min steps "
        f"({len(CANDIDATE_DURATIONS_MINUTES)} candidates)"
    )
    print(
        "POC constraints (illustrative, not damp/mould/health thresholds):"
    )
    print(
        f"  target_final_absolute_humidity_g_m3 = "
        f"{constraints.target_final_absolute_humidity_g_m3}"
    )
    print(
        f"  max_temperature_drop_c              = "
        f"{constraints.max_temperature_drop_c}"
    )

    # ---- Psychrometric --------------------------------------------------
    indoor_ah = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    ).absolute_humidity
    outdoor_ah = outdoor.absolute_humidity
    section("2) Psychrometric layer")
    print(f"initial indoor AH = {indoor_ah:.3f} g/m^3")
    print(f"outdoor AH        = {outdoor_ah:.3f} g/m^3")
    print(f"drying potential  = {indoor_ah - outdoor_ah:+.3f} g/m^3")

    # ---- Candidate outcomes (moisture + thermal composed via facade) ----
    predictions = evaluate_candidate_durations(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
    )
    section("3) Candidate outcome table")
    header = (
        f"  {'duration':>8}  {'final AH':>9}  {'water':>10}  "
        f"{'final T':>8}  {'T drop':>8}  {'final RH':>9}  {'energy':>9}"
    )
    units = (
        f"  {'(min)':>8}  {'(g/m^3)':>9}  {'(g)':>10}  "
        f"{'(C)':>8}  {'(K)':>8}  {'(%)':>9}  {'(kWh)':>9}"
    )
    print(header)
    print(units)
    print("  " + "-" * (len(header) - 2))
    for duration, prediction in zip(CANDIDATE_DURATIONS_MINUTES, predictions):
        print(
            f"  {duration:>8.0f}  "
            f"{prediction.final_absolute_humidity_g_m3:>9.3f}  "
            f"{prediction.water_removed_g:>+10.2f}  "
            f"{prediction.final_temperature_c:>8.3f}  "
            f"{prediction.temperature_drop_c:>+8.3f}  "
            f"{prediction.final_relative_humidity_pct:>9.2f}  "
            f"{prediction.ventilation_energy_removed_kwh:>+9.4f}"
        )

    # ---- Feasibility ----------------------------------------------------
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
        constraints=constraints,
    )
    section("4) Feasibility table")
    print(
        f"  {'duration':>8}  {'feasible?':>10}  reason if infeasible"
    )
    print("  " + "-" * 60)
    for duration, evaluation in zip(CANDIDATE_DURATIONS_MINUTES, evaluations):
        marker = "yes" if evaluation.feasible else "no"
        reason = (
            ", ".join(evaluation.violated_constraints)
            if not evaluation.feasible
            else ""
        )
        print(f"  {duration:>8.0f}  {marker:>10}  {reason}")

    # ---- Pareto frontier plot ------------------------------------------
    efficient = pareto_efficient_indices(predictions)
    energies = [p.ventilation_energy_removed_kwh for p in predictions]
    water = [p.water_removed_g for p in predictions]

    fig, ax = plt.subplots()
    ax.scatter(
        [energies[i] for i in range(len(predictions)) if i not in efficient],
        [water[i] for i in range(len(predictions)) if i not in efficient],
        color="lightgray",
        zorder=1,
        label="dominated",
    )
    ax.plot(
        [energies[i] for i in efficient],
        [water[i] for i in efficient],
        color="tab:blue",
        linewidth=1,
        zorder=2,
    )
    ax.scatter(
        [energies[i] for i in efficient],
        [water[i] for i in efficient],
        color="tab:blue",
        zorder=3,
        label="Pareto-efficient",
    )
    ax.set_xlabel("ventilation energy removed (kWh)")
    ax.set_ylabel("water removed from room air (g)")
    ax.set_title(
        "Pareto frontier: water removed vs ventilation energy\n"
        "canonical POC scenario"
    )
    ax.legend()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH)
    plt.close(fig)
    section("5) Pareto frontier plot")
    print(f"  saved to {OUTPUT_PATH}")
    print(
        f"  {len(efficient)} of {len(predictions)} candidates are "
        "Pareto-efficient under (water, energy)."
    )

    # ---- Selected action (POC default) ---------------------------------
    result = recommend_ventilation_action(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
        constraints=constraints,
    )
    section("6) Selected action (POC default strategy)")
    print(
        "  Strategy: choose_minimum_energy_action - lowest ventilation "
        "energy loss achieving the moisture target without exceeding "
        "the temperature-drop cap."
    )
    print(f"  selected duration : {result.selected_duration_minutes:g} min")
    print(
        f"  water removed     : "
        f"{result.selected_prediction.water_removed_g:+.2f} g"
    )
    print(
        f"  final indoor AH   : "
        f"{result.selected_prediction.final_absolute_humidity_g_m3:.3f} g/m^3"
    )
    print(
        f"  final indoor RH   : "
        f"{result.selected_prediction.final_relative_humidity_pct:.2f} %"
    )
    print(
        f"  final indoor T    : "
        f"{result.selected_prediction.final_temperature_c:.3f} C"
    )
    print(
        f"  temperature drop  : "
        f"{result.selected_prediction.temperature_drop_c:+.3f} K"
    )
    print(
        f"  energy loss       : "
        f"{result.selected_prediction.ventilation_energy_removed_kwh:+.4f} kWh"
    )
    print(f"  target achieved?  : {'yes' if result.feasible else 'no'}")

    section("7) Human-readable explanation (presentation layer)")
    print(format_recommendation(result))

    # ---- Comparison across every strategy ------------------------------
    section("8) All five optimisation strategies on this scenario")
    common = dict(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
    )
    strategies = [
        (
            "A) min energy under moisture target (POC default)",
            choose_minimum_energy_action(
                **common,
                constraints=VentilationConstraints(
                    target_final_absolute_humidity_g_m3=8.0,
                    max_temperature_drop_c=2.0,
                ),
            ),
        ),
        (
            "B) max water under energy budget",
            optimise_max_moisture_under_energy_budget(
                **common,
                constraints=VentilationConstraints(max_energy_loss_kwh=0.15),
            ),
        ),
        (
            "C) max water under comfort cap",
            optimise_max_moisture_under_comfort_limit(
                **common,
                constraints=VentilationConstraints(
                    max_temperature_drop_c=1.0
                ),
            ),
        ),
        (
            "D) weighted trade-off (lambda = 500 g/kWh)",
            optimise_weighted_tradeoff(
                **common,
                constraints=VentilationConstraints(),
                lambda_energy=500.0,
            ),
        ),
        (
            "E) marginal efficiency threshold (>= 600 g/kWh)",
            optimise_marginal_efficiency_threshold(
                **common,
                constraints=VentilationConstraints(
                    minimum_marginal_g_per_kwh=600.0
                ),
            ),
        ),
    ]
    header = (
        f"  {'strategy':<50}  {'duration':>10}  "
        f"{'water':>10}  {'energy':>10}  {'T drop':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, res in strategies:
        if res.feasible:
            duration_str = f"{res.selected_duration_minutes:.1f} min"
        elif (
            res.selected_duration_minutes != res.selected_duration_minutes
        ):
            duration_str = "infeasible"
        else:
            duration_str = f"{res.selected_duration_minutes:.1f} min*"
        print(
            f"  {label:<50}  {duration_str:>10}  "
            f"{res.selected_prediction.water_removed_g:>+10.2f}  "
            f"{res.selected_prediction.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{res.selected_prediction.temperature_drop_c:>+8.3f}"
        )
    print(
        "  * duration reported but the moisture target could not be "
        "achieved; the strategy fell back to a comfort-respecting "
        "alternative. See its reason string."
    )

    # ---- Sensitivity: moisture target on the POC default ---------------
    section("9) Sensitivity of the POC default to the moisture target")
    print(
        f"  {'target (g/m^3)':>16}  {'selected':>10}  "
        f"{'water':>10}  {'energy':>10}  {'feasible?':>10}"
    )
    print("  " + "-" * 62)
    for target in (12.0, 10.0, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0):
        res = recommend_ventilation_action(
            room=room,
            outdoor=outdoor,
            thermal_properties=thermal_props,
            candidate_durations_minutes=CANDIDATE_DURATIONS_MINUTES,
            constraints=VentilationConstraints(
                target_final_absolute_humidity_g_m3=target,
                max_temperature_drop_c=2.0,
            ),
        )
        if res.feasible:
            duration_str = f"{res.selected_duration_minutes:.1f} min"
        else:
            duration_str = "n/a"
        print(
            f"  {target:>16g}  {duration_str:>10}  "
            f"{res.selected_prediction.water_removed_g:>+10.2f}  "
            f"{res.selected_prediction.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{('yes' if res.feasible else 'no'):>10}"
        )
    print(
        "  Small shifts in the caller's moisture target move the pick by "
        "several minutes. See experiments/strategy_sensitivity.py for the "
        "full four-parameter sensitivity study."
    )

    # ---- Test-suite result ---------------------------------------------
    section("10) Unit test result summary")
    try:
        completed = subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "pytest",
             "test/", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        summary_line = completed.stdout.strip().splitlines()[-1]
        print(f"  pytest: {summary_line}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  could not run pytest: {exc}")

    # ---- Assumptions and limitations -----------------------------------
    section("11) Assumptions and limitations still standing")
    print(
        "  Physics model:\n"
        "    * Well-mixed single-zone air; no stratification, no dead\n"
        "      corners, no jets from openings.\n"
        "    * Ideal-gas behaviour; Magnus over-water saturation curve.\n"
        "    * Constant outdoor T and RH across each candidate event.\n"
        "    * Constant ACH while a window state is unchanged.\n"
        "    * Lumped effective thermal capacitance; a single scalar\n"
        "      C_eff represents the fast-responding fabric plus air.\n"
        "    * No moisture buffering from walls, textiles, or furniture.\n"
        "    * No moisture generation from occupants, cooking, showers,\n"
        "      laundry, plants.\n"
        "    * No solar gains, no equipment gains, no heating system.\n"
        "    * No inter-room airflow; single zone only.\n"
        "  Control layer:\n"
        "    * The moisture target and comfort cap are POC assumptions\n"
        "      and require an evidence base before any production use\n"
        "      (see optimiser module docstring).\n"
        "    * ACH_open is a caller-set constant; no learning from\n"
        "      measured room response.\n"
        "    * No time-domain planning: one event at a time, no\n"
        "      sequences over hours or days.\n"
        "    * No sensors, no dashboard, no actuators, no forecast.\n"
    )

    # ---- What the optimiser lacks before it can call itself
    #      damp/mould-preventive ---------------------------------------
    section(
        "12) What the optimiser lacks to claim damp/mould prevention"
    )
    print(
        "  The current system MINIMISES ENERGY SUBJECT TO A CALLER-SET\n"
        "  MOISTURE TARGET. That is a moisture-reduction optimiser. To\n"
        "  claim damp / mould prevention, at minimum the following are\n"
        "  missing:\n"
        "\n"
        "  A. SURFACE-LEVEL moisture, not room-air moisture.\n"
        "     Mould grows on cold surfaces where local RH stays high\n"
        "     for long periods (~30-90 days at RH > 80 %). Room-air AH\n"
        "     is a proxy for surface RH only when surface temperatures\n"
        "     are known. The POC does not model surface temperatures\n"
        "     (a thermal-bridge / building-envelope model is required)\n"
        "     and therefore does not know the quantity mould actually\n"
        "     responds to.\n"
        "\n"
        "  B. TIME-INTEGRATED exposure, not single-event outcomes.\n"
        "     Mould-growth models (VTT, isopleth, ASHRAE 160, WUFI-\n"
        "     Bio) integrate temperature and RH over days / weeks.\n"
        "     One ventilation event does not by itself cause or\n"
        "     prevent mould; a controller must reason over sustained\n"
        "     conditions. The optimiser currently reasons per event.\n"
        "\n"
        "  C. MOISTURE BUFFERING from walls, textiles, and furniture.\n"
        "     The fabric holds orders of magnitude more water than\n"
        "     the air and slowly re-wets the air after ventilation.\n"
        "     Damp risk depends on that reservoir's state and its\n"
        "     coupling to surfaces, neither of which is in the model.\n"
        "\n"
        "  D. MOISTURE GENERATION from occupants and activities.\n"
        "     A real room is a continuous moisture source (people,\n"
        "     cooking, showers, drying laundry, plants). The current\n"
        "     model has zero source term, so its \"target achieved\"\n"
        "     verdict does not survive contact with a lived-in room.\n"
        "\n"
        "  E. HEATING-SYSTEM RESPONSE during and after the event.\n"
        "     Opening a window while the heating is on shifts energy\n"
        "     onto the fuel bill rather than cooling the room. The\n"
        "     energy the optimiser reads is un-compensated heat loss;\n"
        "     the real economic cost depends on heating behaviour.\n"
        "\n"
        "  F. WEATHER FORECAST.\n"
        "     Preferring a mild-dry hour over the current cold-humid\n"
        "     one requires knowing what the next hours look like.\n"
        "     Every strategy in this POC assumes conditions are as\n"
        "     they are right now.\n"
        "\n"
        "  G. LEARNED / MEASURED effective ACH and C_eff.\n"
        "     Both are caller-set POC inputs. Without identification\n"
        "     from measured room response, the numeric recommendations\n"
        "     are illustrative rather than calibrated.\n"
        "\n"
        "  H. VALIDATED THRESHOLDS.\n"
        "     The moisture target, comfort cap, usefulness floors,\n"
        "     marginal-efficiency floor, and lambda are all caller-\n"
        "     supplied preferences with no independent evidence base\n"
        "     in this repo. Every plot and table in this repository\n"
        "     is honest about that; no deployment can be.\n"
        "\n"
        "  I. INDEPENDENT VERIFICATION against real measurements.\n"
        "     Every prediction in this repository is model-only. No\n"
        "     recommendation has been checked against a real room's\n"
        "     temperature or humidity trajectory during and after a\n"
        "     controlled ventilation event.\n"
        "\n"
        "  Until at least items A-D are addressed and the system has\n"
        "  been validated against measured data (item I), the POC is\n"
        "  a moisture-reduction and heat-loss trade-off tool. Calling\n"
        "  it a damp / mould prevention system would misrepresent what\n"
        "  it actually models."
    )


if __name__ == "__main__":
    main()
