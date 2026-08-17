"""Moisture-energy trade-off across four outdoor weather scenarios.

Keeps the indoor room identical (20 C, 70 %RH, 40 m^3, ACH = 5 h^-1
when the window is open, C_eff = 500 kJ/K illustrative) and runs the
same window-open duration sweep for four labelled outdoor scenarios:

    cold & humid   :  5 C, 85 %RH
    cold & dry     :  5 C, 40 %RH
    mild & humid   : 15 C, 85 %RH
    mild & dry     : 15 C, 40 %RH

For each scenario the script prints:
    - a per-duration cumulative table (final AH / T / RH, water removed,
      dynamic ventilation energy);
    - an interval-by-interval table with additional water, energy, T
      drop, marginal g/kWh, and marginal g/kWh relative to that
      scenario's first-interval efficiency.

The aim is not to pick a best duration - it is to show that the
SHAPE of the trade-off (how much moisture you buy per unit of
thermal cost, and how fast that efficiency degrades) depends on
BOTH outdoor absolute humidity AND outdoor temperature. The
scenarios above cover the four sign quadrants of that dependence.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import (
    VentilationSimulationResult,
    simulate_ventilation_event,
)

from _metrics import safe_ratio

DURATIONS_MINUTES: Tuple[float, ...] = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class OutdoorScenario:
    """Named outdoor condition for the comparison sweep."""

    label: str
    outdoor_temperature_c: float
    outdoor_relative_humidity_pct: float


SCENARIOS: Tuple[OutdoorScenario, ...] = (
    OutdoorScenario("cold & humid", 5.0, 85.0),
    OutdoorScenario("cold & dry", 5.0, 40.0),
    OutdoorScenario("mild & humid", 15.0, 85.0),
    OutdoorScenario("mild & dry", 15.0, 40.0),
)


def _run_scenario(
    scenario: OutdoorScenario,
) -> List[VentilationSimulationResult]:
    """Run the shared duration sweep for one outdoor scenario."""
    return [
        simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=20.0,
            initial_indoor_relative_humidity_pct=70.0,
            outdoor_temperature_c=scenario.outdoor_temperature_c,
            outdoor_relative_humidity_pct=scenario.outdoor_relative_humidity_pct,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=t,
        )
        for t in DURATIONS_MINUTES
    ]


def _print_cumulative_table(
    durations_minutes: Iterable[float],
    results: Iterable[VentilationSimulationResult],
) -> None:
    """Print the per-duration table for one scenario."""
    print(
        f"  {'minutes':>7}  {'final AH':>9}  {'water rem':>10}  "
        f"{'final T':>8}  {'T drop':>7}  {'final RH':>9}  {'energy':>8}"
    )
    print(
        f"  {'(min)':>7}  {'(g/m^3)':>9}  {'(g)':>10}  "
        f"{'(C)':>8}  {'(K)':>7}  {'(%)':>9}  {'(kWh)':>8}"
    )
    print("  " + "-" * 74)
    for duration_minutes, r in zip(durations_minutes, results):
        print(
            f"  {duration_minutes:>7.1f}  "
            f"{r.final_absolute_humidity_g_m3:>9.3f}  "
            f"{r.water_removed_g:>+10.2f}  "
            f"{r.final_temperature_c:>8.3f}  "
            f"{r.temperature_drop_c:>+7.3f}  "
            f"{r.final_relative_humidity_pct:>9.2f}  "
            f"{r.ventilation_energy_removed_kwh:>+8.4f}"
        )


def _print_interval_table(
    durations_minutes: Tuple[float, ...],
    results: List[VentilationSimulationResult],
) -> None:
    """Print the interval-by-interval marginal-efficiency table."""
    intervals = []
    for previous_duration, current_duration, previous, current in zip(
        durations_minutes[:-1],
        durations_minutes[1:],
        results[:-1],
        results[1:],
    ):
        delta_water_g = current.water_removed_g - previous.water_removed_g
        delta_energy_kwh = (
            current.ventilation_energy_removed_kwh
            - previous.ventilation_energy_removed_kwh
        )
        delta_t_drop_k = current.temperature_drop_c - previous.temperature_drop_c
        marginal = safe_ratio(delta_water_g, delta_energy_kwh)
        intervals.append(
            (
                previous_duration,
                current_duration,
                delta_water_g,
                delta_energy_kwh,
                delta_t_drop_k,
                marginal,
            )
        )
    baseline = intervals[0][5]
    print(
        f"  {'interval':>15}  {'Δwater':>10}  {'Δenergy':>10}  "
        f"{'ΔT drop':>10}  {'marginal':>10}  {'% of first':>11}"
    )
    print(
        f"  {'(min)':>15}  {'(g)':>10}  {'(kWh)':>10}  "
        f"{'(K)':>10}  {'(g/kWh)':>10}  {'':>11}"
    )
    print("  " + "-" * 76)
    for previous, current, delta_water, delta_energy, delta_t, marginal in intervals:
        interval_label = f"{previous:g} -> {current:g}"
        relative_pct = 100.0 * safe_ratio(marginal, baseline)
        print(
            f"  {interval_label:>15}  "
            f"{delta_water:>+10.2f}  "
            f"{delta_energy:>+10.4f}  "
            f"{delta_t:>+10.3f}  "
            f"{marginal:>10.1f}  "
            f"{relative_pct:>10.1f} %"
        )


def _print_scenario_summary(
    scenario: OutdoorScenario,
    results: List[VentilationSimulationResult],
) -> None:
    """Print a one-line summary of a scenario for easy cross-comparison."""
    initial_ah = results[0].initial_absolute_humidity_g_m3
    outdoor_ah = AirState(
        temperature_c=scenario.outdoor_temperature_c,
        relative_humidity_percent=scenario.outdoor_relative_humidity_pct,
    ).absolute_humidity
    ah_gap = initial_ah - outdoor_ah
    t_gap = 20.0 - scenario.outdoor_temperature_c  # indoor is fixed at 20 C
    # 15-min row -> results[5] (0, 2, 3, 5, 10, 15, 20)
    r15 = results[5]
    energy_15 = r15.ventilation_energy_removed_kwh
    water_15 = r15.water_removed_g
    print(
        f"  moisture gap (C_0 - C_out) = {ah_gap:+6.3f} g/m^3;  "
        f"thermal gap (T_0 - T_out) = {t_gap:+.1f} K"
    )
    print(
        f"  at 15 min:  water removed = {water_15:+7.2f} g;  "
        f"energy removed = {energy_15:+7.4f} kWh"
    )


def main() -> None:
    """Run the four scenarios and print their comparison tables."""
    print("Fixed indoor: T = 20 C, RH = 70 %, V = 40 m^3, ACH = 5 h^-1 (open),")
    print("              C_eff = 500,000 J/K (illustrative)")
    print()
    for scenario in SCENARIOS:
        results = _run_scenario(scenario)
        print("=" * 82)
        print(
            f"Scenario: {scenario.label}"
            f"  (outdoor T = {scenario.outdoor_temperature_c:g} C, "
            f"outdoor RH = {scenario.outdoor_relative_humidity_pct:g} %)"
        )
        print("=" * 82)
        _print_scenario_summary(scenario, results)
        print()
        print("  Cumulative view")
        _print_cumulative_table(DURATIONS_MINUTES, results)
        print()
        print("  Interval view")
        _print_interval_table(DURATIONS_MINUTES, results)
        print()


if __name__ == "__main__":
    main()
