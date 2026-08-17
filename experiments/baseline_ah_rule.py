"""Baseline AH rule vs the full moisture / thermal model.

Implements the simplest useful ventilation decision rule for later
comparison against a future optimiser:

    if indoor_absolute_humidity > outdoor_absolute_humidity:
        ventilation is BENEFICIAL
    else:
        ventilation is NOT beneficial

For each of the four outdoor scenarios from the earlier weather sweep
(cold & humid, cold & dry, mild & humid, mild & dry), this experiment
reports what the baseline says AND what a fixed 10-minute ventilation
event would actually produce under the model - water removed, energy
lost, temperature drop, marginal g/kWh.

The point is to show what the baseline sees and what it CANNOT see.
The baseline is retained as a controller-under-test in later slices
(so the eventual optimiser can be scored against it), not as a
recommendation. The baseline does NOT choose a duration; it only
labels the outdoor state as "beneficial or not for drying".
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

from _metrics import safe_ratio


@dataclass(frozen=True)
class OutdoorScenario:
    """Named outdoor condition, matching the earlier weather sweep."""

    label: str
    outdoor_temperature_c: float
    outdoor_relative_humidity_pct: float


SCENARIOS: Tuple[OutdoorScenario, ...] = (
    OutdoorScenario("cold & humid", 5.0, 85.0),
    OutdoorScenario("cold & dry", 5.0, 40.0),
    OutdoorScenario("mild & humid", 15.0, 85.0),
    OutdoorScenario("mild & dry", 15.0, 40.0),
)

FIXED_VENTILATION_DURATION_MINUTES = 10.0
INDOOR_TEMPERATURE_C = 20.0
INDOOR_RELATIVE_HUMIDITY_PCT = 70.0


def baseline_ah_rule_says_beneficial(
    indoor_absolute_humidity_g_m3: float,
    outdoor_absolute_humidity_g_m3: float,
) -> bool:
    """Baseline rule: indoor AH > outdoor AH -> ventilation is beneficial.

    Purely a moisture-content comparison; does not consider temperature,
    ACH, event duration, or any thermal cost. Kept simple on purpose:
    this is the strawman controller the eventual optimiser will be
    scored against.

    Boundary case (indoor AH exactly equal to outdoor AH):
        Uses a strict ``>`` comparator, so the equal case returns
        False - the rule says "not beneficial". At the equal-AH
        boundary the model's ``water_removed_g`` is exactly zero, so
        the strict-``>`` verdict is physically consistent (no net
        drying either way). ``>=`` would be equally defensible, since
        both the water and its opportunity cost round to zero. The
        strict form is retained because it keeps the rule's output a
        strict superset of the "definitely beneficial" cases.
    """
    return indoor_absolute_humidity_g_m3 > outdoor_absolute_humidity_g_m3


def main() -> None:
    """Score the four scenarios under the baseline and against the model."""
    indoor_ah = AirState(
        temperature_c=INDOOR_TEMPERATURE_C,
        relative_humidity_percent=INDOOR_RELATIVE_HUMIDITY_PCT,
    ).absolute_humidity

    print(
        f"Fixed indoor : T = {INDOOR_TEMPERATURE_C:g} C, "
        f"RH = {INDOOR_RELATIVE_HUMIDITY_PCT:g} %, V = 40 m^3, ACH = 5 h^-1 (open),"
    )
    print("               C_eff = 500,000 J/K (illustrative)")
    print(f"initial indoor AH = {indoor_ah:.3f} g/m^3")
    print(
        f"Comparison duration: {FIXED_VENTILATION_DURATION_MINUTES:g} minutes "
        "(same across all scenarios so the baseline decision is separate from the "
        "duration question)."
    )
    print()

    header = (
        f"  {'scenario':<15}  "
        f"{'outdoor AH':>10}  "
        f"{'baseline':>10}  "
        f"{'AH gap':>8}  "
        f"{'water rem':>10}  "
        f"{'energy':>9}  "
        f"{'T drop':>8}  "
        f"{'marginal':>10}"
    )
    units = (
        f"  {'':<15}  "
        f"{'(g/m^3)':>10}  "
        f"{'says':>10}  "
        f"{'(g/m^3)':>8}  "
        f"{'(g)':>10}  "
        f"{'(kWh)':>9}  "
        f"{'(K)':>8}  "
        f"{'(g/kWh)':>10}"
    )
    print(header)
    print(units)
    print("  " + "-" * (len(header) - 2))

    for scenario in SCENARIOS:
        outdoor_ah = AirState(
            temperature_c=scenario.outdoor_temperature_c,
            relative_humidity_percent=scenario.outdoor_relative_humidity_pct,
        ).absolute_humidity
        ah_gap = indoor_ah - outdoor_ah
        rule_says = baseline_ah_rule_says_beneficial(indoor_ah, outdoor_ah)
        result = simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=INDOOR_TEMPERATURE_C,
            initial_indoor_relative_humidity_pct=INDOOR_RELATIVE_HUMIDITY_PCT,
            outdoor_temperature_c=scenario.outdoor_temperature_c,
            outdoor_relative_humidity_pct=scenario.outdoor_relative_humidity_pct,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=FIXED_VENTILATION_DURATION_MINUTES,
        )
        marginal = safe_ratio(
            result.water_removed_g,
            result.ventilation_energy_removed_kwh,
        )
        rule_label = "beneficial" if rule_says else "not benef."
        print(
            f"  {scenario.label:<15}  "
            f"{outdoor_ah:>10.3f}  "
            f"{rule_label:>10}  "
            f"{ah_gap:>+8.3f}  "
            f"{result.water_removed_g:>+10.2f}  "
            f"{result.ventilation_energy_removed_kwh:>+9.4f}  "
            f"{result.temperature_drop_c:>+8.3f}  "
            f"{marginal:>10.1f}"
        )


if __name__ == "__main__":
    main()
