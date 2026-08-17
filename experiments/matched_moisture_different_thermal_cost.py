"""Two ventilation opportunities with matched moisture benefit, different thermal cost.

Constructs two outdoor scenarios that have IDENTICAL outdoor absolute
humidity but very different outdoor temperatures. In the model,
moisture removal depends only on the indoor-outdoor absolute-humidity
gap and ACH, while thermal cost depends on the indoor-outdoor
temperature gap. Fixing the moisture side lets the thermal side move
independently, so both scenarios yield the same water_removed_g at
each duration but very different energy_removed_kwh and temperature
drops.

Method:
    * Fix scenario B directly: T_out = 12 C, RH_out = 50 %.
    * Compute B's outdoor AH via ``AirState(12, 50).absolute_humidity``.
    * Choose scenario A's temperature (2 C) and back-solve for the RH
      that gives the SAME outdoor AH, using
      ``relative_humidity_from_absolute_humidity`` (the AH-to-RH
      inverse already in the psychrometric module).

The lesson: two ventilation opportunities can offer identical moisture
removal but incur very different heating penalties. Later slices of
the POC will be able to justify preferring the milder opportunity
even though "outdoor air is dry enough" is true in both. This
experiment does NOT choose an optimum and does NOT use weather
forecasts - it only compares two concrete matched-moisture opportunities.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math import isnan

from psychrometrics import AirState, relative_humidity_from_absolute_humidity
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

from _metrics import safe_ratio

DURATIONS_MINUTES: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)


@dataclass(frozen=True)
class MatchedScenario:
    """Named outdoor condition, labelled for the comparison."""

    label: str
    outdoor_temperature_c: float
    outdoor_relative_humidity_pct: float

    @property
    def outdoor_absolute_humidity_g_m3(self) -> float:
        return AirState(
            temperature_c=self.outdoor_temperature_c,
            relative_humidity_percent=self.outdoor_relative_humidity_pct,
        ).absolute_humidity


def _build_matched_pair() -> Tuple[MatchedScenario, MatchedScenario]:
    """Return (cold, mild) scenarios with identical outdoor AH."""
    scenario_mild = MatchedScenario(
        label="mild & moderately dry",
        outdoor_temperature_c=12.0,
        outdoor_relative_humidity_pct=50.0,
    )
    target_outdoor_ah = scenario_mild.outdoor_absolute_humidity_g_m3
    cold_temperature_c = 2.0
    cold_relative_humidity_pct = relative_humidity_from_absolute_humidity(
        temperature_c=cold_temperature_c,
        absolute_humidity_g_m3=target_outdoor_ah,
    )
    scenario_cold = MatchedScenario(
        label="cold & nearly saturated",
        outdoor_temperature_c=cold_temperature_c,
        outdoor_relative_humidity_pct=cold_relative_humidity_pct,
    )
    return scenario_cold, scenario_mild


def main() -> None:
    """Run the matched comparison and print the side-by-side table."""
    scenario_cold, scenario_mild = _build_matched_pair()

    indoor_ah = AirState(
        temperature_c=20.0, relative_humidity_percent=70.0
    ).absolute_humidity

    print("Fixed indoor : T = 20 C, RH = 70 %, V = 40 m^3, ACH = 5 h^-1 (open),")
    print("               C_eff = 500,000 J/K (illustrative)")
    print(f"initial indoor AH = {indoor_ah:.3f} g/m^3")
    print()
    print("Matched outdoor scenarios (same outdoor AH, different outdoor T)")
    print("-" * 70)
    for scenario in (scenario_cold, scenario_mild):
        print(
            f"  {scenario.label:>22}: "
            f"T = {scenario.outdoor_temperature_c:5.2f} C, "
            f"RH = {scenario.outdoor_relative_humidity_pct:5.2f} %  ->  "
            f"AH = {scenario.outdoor_absolute_humidity_g_m3:.4f} g/m^3"
        )
    print()
    print(
        "  moisture gap (indoor AH - outdoor AH) = "
        f"{indoor_ah - scenario_cold.outdoor_absolute_humidity_g_m3:.3f} g/m^3 "
        "for BOTH scenarios (matched by construction)."
    )
    print(
        f"  thermal gap  (indoor T - outdoor T)   = "
        f"{20.0 - scenario_cold.outdoor_temperature_c:+.1f} K for cold, "
        f"{20.0 - scenario_mild.outdoor_temperature_c:+.1f} K for mild."
    )
    print()

    print("Side-by-side duration sweep")
    print("-" * 92)
    print(
        f"  {'minutes':>7}  "
        f"{'water removed (g)':>21}  "
        f"{'energy removed (kWh)':>22}  "
        f"{'T drop (K)':>16}"
    )
    print(
        f"  {'(min)':>7}  "
        f"{'cold':>9}  {'mild':>9}  "
        f"{'cold':>10}  {'mild':>10}  "
        f"{'cold':>7}  {'mild':>7}"
    )
    print("  " + "-" * 88)
    energy_ratios = []
    for duration_minutes in DURATIONS_MINUTES:
        cold_result = simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=20.0,
            initial_indoor_relative_humidity_pct=70.0,
            outdoor_temperature_c=scenario_cold.outdoor_temperature_c,
            outdoor_relative_humidity_pct=scenario_cold.outdoor_relative_humidity_pct,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=duration_minutes,
        )
        mild_result = simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=20.0,
            initial_indoor_relative_humidity_pct=70.0,
            outdoor_temperature_c=scenario_mild.outdoor_temperature_c,
            outdoor_relative_humidity_pct=scenario_mild.outdoor_relative_humidity_pct,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=duration_minutes,
        )
        ratio = safe_ratio(
            cold_result.ventilation_energy_removed_kwh,
            mild_result.ventilation_energy_removed_kwh,
        )
        if not isnan(ratio):
            energy_ratios.append(ratio)
        print(
            f"  {duration_minutes:>7.1f}  "
            f"{cold_result.water_removed_g:>+9.2f}  "
            f"{mild_result.water_removed_g:>+9.2f}  "
            f"{cold_result.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{mild_result.ventilation_energy_removed_kwh:>+10.4f}  "
            f"{cold_result.temperature_drop_c:>+7.3f}  "
            f"{mild_result.temperature_drop_c:>+7.3f}"
        )
    print()

    if energy_ratios:
        mean_ratio = sum(energy_ratios) / len(energy_ratios)
        print(
            f"Across the sweep, the cold scenario costs on average "
            f"{mean_ratio:.2f}x the thermal energy of the mild scenario "
            "for essentially the same water removed."
        )


if __name__ == "__main__":
    main()
