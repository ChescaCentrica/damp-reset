"""Diminishing-returns report on the ventilation-duration sweep.

Reads the same duration sweep as the other combined-model experiments
and prints one paragraph per interval in the shape:

    5 -> 10 min: +Δwater g, +Δenergy kWh, +ΔT drop K.
                 marginal Δwater/Δenergy = X g/kWh
                 (= Y % of the first interval's marginal efficiency)

The intent is to describe the SHAPE of the moisture-vs-energy trade-off
before any optimisation algorithm is designed. Two derived quantities
make the "diminishing" statement quantitative:

  * incremental efficiency (marginal g/kWh) for each interval, and
  * that same efficiency normalised to the FIRST interval so drops
    across the sweep are visible as percentages.

The script then flags the first interval where the marginal efficiency
falls below a stated fraction (50 %) of the initial value - a
BOUNDARY, not an optimum. Choosing an "optimal" duration would
require an objective function that trades moisture, energy, comfort,
and cost against each other; that is out of scope for this slice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

from _metrics import safe_ratio

DURATIONS_MINUTES = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)
DIMINISHING_RETURNS_FRACTION = 0.5  # 50 % of the first interval's marginal efficiency


def main() -> None:
    """Run the sweep and print the diminishing-returns interpretation."""
    scenario_kwargs = dict(
        room_volume_m3=40.0,
        initial_indoor_temperature_c=20.0,
        initial_indoor_relative_humidity_pct=70.0,
        outdoor_temperature_c=5.0,
        outdoor_relative_humidity_pct=85.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        ),
    )
    results = [
        simulate_ventilation_event(duration_minutes=t, **scenario_kwargs)
        for t in DURATIONS_MINUTES
    ]

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T = {scenario_kwargs['initial_indoor_temperature_c']:g} C, "
        f"RH = {scenario_kwargs['initial_indoor_relative_humidity_pct']:g} %, "
        f"V = {scenario_kwargs['room_volume_m3']:g} m^3"
    )
    print(
        f"outdoor : T = {scenario_kwargs['outdoor_temperature_c']:g} C, "
        f"RH = {scenario_kwargs['outdoor_relative_humidity_pct']:g} %"
    )
    print(f"vent    : ACH = {scenario_kwargs['ach']:g} h^-1 (window open)")
    print(
        f"C_eff   : {scenario_kwargs['effective_thermal_capacity_j_per_k']:,.0f} J/K "
        "(illustrative)"
    )
    print()

    print("Interval-by-interval marginal changes")
    print("-------------------------------------")

    intervals = []  # (previous, current, delta_water, delta_energy, delta_t, marginal_g_per_kwh)
    for previous, current, previous_result, current_result in zip(
        DURATIONS_MINUTES[:-1],
        DURATIONS_MINUTES[1:],
        results[:-1],
        results[1:],
    ):
        delta_water_g = (
            current_result.water_removed_g - previous_result.water_removed_g
        )
        delta_energy_kwh = (
            current_result.ventilation_energy_removed_kwh
            - previous_result.ventilation_energy_removed_kwh
        )
        delta_t_drop_k = (
            current_result.temperature_drop_c
            - previous_result.temperature_drop_c
        )
        marginal_g_per_kwh = safe_ratio(delta_water_g, delta_energy_kwh)
        intervals.append(
            (
                previous,
                current,
                delta_water_g,
                delta_energy_kwh,
                delta_t_drop_k,
                marginal_g_per_kwh,
            )
        )

    baseline_efficiency_g_per_kwh = intervals[0][5]
    for previous, current, delta_water, delta_energy, delta_t, marginal in intervals:
        relative_pct = 100.0 * safe_ratio(marginal, baseline_efficiency_g_per_kwh)
        print(
            f"  {previous:>4g} -> {current:>4g} min: "
            f"+{delta_water:6.2f} g water, "
            f"+{delta_energy:6.4f} kWh energy, "
            f"+{delta_t:5.3f} K T drop"
        )
        print(
            f"                marginal = {marginal:7.1f} g/kWh "
            f"({relative_pct:5.1f} % of the first-interval efficiency)"
        )

    print()
    print("Diminishing-returns boundary (reported, not chosen)")
    print("---------------------------------------------------")
    print(
        f"Threshold: the first interval whose marginal g/kWh falls below "
        f"{DIMINISHING_RETURNS_FRACTION * 100:g} % of the "
        f"first-interval efficiency ({baseline_efficiency_g_per_kwh:.1f} g/kWh)."
    )
    threshold_g_per_kwh = DIMINISHING_RETURNS_FRACTION * baseline_efficiency_g_per_kwh
    boundary_interval = None
    for previous, current, _, _, _, marginal in intervals:
        if marginal < threshold_g_per_kwh:
            boundary_interval = (previous, current, marginal)
            break
    if boundary_interval is None:
        print("  No interval in the sweep falls below the threshold.")
    else:
        previous, current, marginal = boundary_interval
        print(
            f"  First interval below threshold: {previous:g} -> {current:g} min "
            f"(marginal = {marginal:.1f} g/kWh, "
            f"which is {100.0 * safe_ratio(marginal, baseline_efficiency_g_per_kwh):.1f} % "
            "of the first-interval efficiency)."
        )
        print(
            "  Interpretation: extending the event past this interval buys water "
            "at less than half the moisture-per-kWh rate the first interval "
            "achieved. This is a BOUNDARY on the trade-off curve, not an "
            "optimal duration - the correct trade-off between moisture, "
            "energy, comfort, and cost is an optimisation objective this "
            "POC does not yet define."
        )


if __name__ == "__main__":
    main()
