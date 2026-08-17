"""Combined ventilation sweep -> tidy pandas DataFrame.

Runs ``simulate_ventilation_event`` at a fixed set of window-open
durations for the shared default scenario, assembles the display
columns the top-level POC report is expected to show, prints the
cumulative and incremental tables, and saves three CSVs:

    outputs/results/ventilation_duration_comparison.csv
        The eight cumulative display columns, one row per duration.
        Values are stored UNROUNDED (raw float64) - the CSV is
        machine-readable input for downstream work.

    outputs/results/ventilation_duration_comparison_incremental.csv
        Incremental change of each cumulative quantity, computed as
        ``row - previous_row`` via pandas ``.diff()``. The first row
        (duration 0) has no predecessor, so its increment values are
        left as NaN in the CSV.

    outputs/results/ventilation_duration_comparison_full.csv
        The full ten-field ``VentilationSimulationResult`` for each
        duration, so nothing is thrown away.

Rounding is applied only when tables are PRINTED to the console -
never during any calculation, and never when writing to disk.

Purpose of the incremental view: identify diminishing returns. For
each step from one duration to the next, the incremental columns
report how much MORE water is removed, how much MORE energy is lost,
etc. If a five-minute extension buys only a small extra amount of
moisture removal while costing a lot more thermal energy, the
incremental table makes that comparison visible directly. This
experiment stops at reporting the increments; deciding an "optimal"
duration is out of scope for this slice.
"""

import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
from ventilation import simulate_ventilation_event

from _metrics import safe_ratio

DURATIONS_MINUTES = (0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "results"
DISPLAY_CSV_PATH = OUTPUT_DIR / "ventilation_duration_comparison.csv"
INCREMENTAL_CSV_PATH = (
    OUTPUT_DIR / "ventilation_duration_comparison_incremental.csv"
)
FULL_CSV_PATH = OUTPUT_DIR / "ventilation_duration_comparison_full.csv"

# Cumulative columns that make sense to difference for the
# incremental view. All of these are monotone-cumulative for a
# ventilation event that runs a consistent direction (cooling +
# drying, or warming + wetting): larger duration -> larger absolute
# value in the same sign.
INCREMENTAL_SOURCE_COLUMNS = {
    "absolute_humidity_reduction_g_m3": "incremental_absolute_humidity_reduction_g_m3",
    "water_removed_g": "incremental_water_removed_g",
    "temperature_drop_c": "incremental_temperature_drop_c",
    "ventilation_energy_removed_kwh": "incremental_ventilation_energy_removed_kwh",
    "relative_humidity_reduction_pct": "incremental_relative_humidity_reduction_pct",
}

# Columns for the display / primary CSV, in the order requested.
DISPLAY_COLUMNS = [
    "duration_minutes",
    "final_absolute_humidity_g_m3",
    "absolute_humidity_reduction_g_m3",
    "water_removed_g",
    "final_relative_humidity_pct",
    "final_temperature_c",
    "temperature_drop_c",
    "ventilation_energy_removed_kwh",
    "grams_of_water_removed_per_kwh",
    "grams_of_water_removed_per_degree_temperature_drop",
]

# Per-column display precision. Applied only when the DataFrame is
# printed to the console; the underlying floats keep full precision
# in memory and on disk.
DISPLAY_DECIMALS = {
    "duration_minutes": 1,
    "final_absolute_humidity_g_m3": 3,
    "absolute_humidity_reduction_g_m3": 3,
    "water_removed_g": 2,
    "final_relative_humidity_pct": 2,
    "final_temperature_c": 3,
    "temperature_drop_c": 3,
    "ventilation_energy_removed_kwh": 4,
    "grams_of_water_removed_per_kwh": 1,
    "grams_of_water_removed_per_degree_temperature_drop": 2,
}


def main() -> None:
    """Build the DataFrame, print it, and save both CSVs."""
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

    rows_full = []
    for duration_minutes in DURATIONS_MINUTES:
        result = simulate_ventilation_event(
            duration_minutes=duration_minutes, **scenario_kwargs
        )
        row = asdict(result)
        row["duration_minutes"] = duration_minutes
        rows_full.append(row)

    full_dataframe = pd.DataFrame(rows_full)

    # Derived cumulative columns not stored on the scalar result:
    #   AH reduction (initial - final) - already a display column.
    #   RH reduction (initial - final) - source for the incremental
    #     view; monotone-cumulative for a consistent-direction event.
    # Both computed here from the raw floats with no rounding applied.
    full_dataframe["absolute_humidity_reduction_g_m3"] = (
        full_dataframe["initial_absolute_humidity_g_m3"]
        - full_dataframe["final_absolute_humidity_g_m3"]
    )
    full_dataframe["relative_humidity_reduction_pct"] = (
        full_dataframe["initial_relative_humidity_pct"]
        - full_dataframe["final_relative_humidity_pct"]
    )

    # Engineering-indication metrics (NOT optimisation objectives):
    #   grams_of_water_removed_per_kwh
    #       = water_removed_g / ventilation_energy_removed_kwh
    #     Moisture benefit divided by thermal cost. Higher = better in
    #     the intuitive sense that we're pulling more water out per unit
    #     of heat we lose to the exchange. Undefined (NaN) when the
    #     energy denominator is zero (no gradient, ACH = 0, or duration
    #     = 0). For a summer wetting event both numerator and
    #     denominator are negative, so the ratio is still positive but
    #     now means "grams of water ADDED per kWh of heat GAINED".
    #     LIMITATIONS: kWh alone is not a monetary or health cost;
    #     ignores heating-system recovery, comfort, occupancy needs,
    #     mould risk, and the moisture buffering of walls / textiles;
    #     assumes the lumped-C, single-zone, isothermal-vent model. Do
    #     not treat as a complete decision rule.
    #
    #   grams_of_water_removed_per_degree_temperature_drop
    #       = water_removed_g / temperature_drop_c
    #     A more interpretable companion: how many grams of water leave
    #     the air per kelvin the room cools. Same NaN rule when the
    #     denominator is zero. Same list of limitations.
    full_dataframe["grams_of_water_removed_per_kwh"] = safe_ratio(
        full_dataframe["water_removed_g"],
        full_dataframe["ventilation_energy_removed_kwh"],
    )
    full_dataframe["grams_of_water_removed_per_degree_temperature_drop"] = (
        safe_ratio(
            full_dataframe["water_removed_g"],
            full_dataframe["temperature_drop_c"],
        )
    )

    display_dataframe = full_dataframe[DISPLAY_COLUMNS].copy()

    # Incremental view: change between consecutive rows. .diff() on a
    # monotone-cumulative column gives per-step "additional X" values
    # exactly. The first row (duration 0) has no predecessor so its
    # increments are NaN by construction.
    incremental_dataframe = pd.DataFrame(
        {"duration_minutes": full_dataframe["duration_minutes"]}
    )
    incremental_dataframe["previous_duration_minutes"] = (
        incremental_dataframe["duration_minutes"].shift(1)
    )
    for source_col, incremental_col in INCREMENTAL_SOURCE_COLUMNS.items():
        incremental_dataframe[incremental_col] = full_dataframe[source_col].diff()

    # Incremental efficiency metric: per-step moisture benefit divided
    # by per-step thermal cost. Same undefined-when-denominator-is-zero
    # rule as the cumulative version. Answers the more actionable
    # question "if I extend by another few minutes, what's the marginal
    # moisture / kWh ratio?" - which is more informative for spotting
    # diminishing returns than the cumulative-so-far ratio.
    incremental_dataframe["incremental_grams_removed_per_incremental_kwh"] = (
        safe_ratio(
            incremental_dataframe["incremental_water_removed_g"],
            incremental_dataframe["incremental_ventilation_energy_removed_kwh"],
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    display_dataframe.to_csv(DISPLAY_CSV_PATH, index=False)
    incremental_dataframe.to_csv(INCREMENTAL_CSV_PATH, index=False)
    full_dataframe.to_csv(FULL_CSV_PATH, index=False)

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

    # Display-only rounding. Original DataFrames keep full precision.
    print("Cumulative view (from t = 0 up to each duration)")
    print("------------------------------------------------")
    formatted = display_dataframe.round(DISPLAY_DECIMALS)
    print(formatted.to_string(index=False))
    print()

    incremental_display_decimals = {
        "duration_minutes": 1,
        "previous_duration_minutes": 1,
        "incremental_absolute_humidity_reduction_g_m3": 3,
        "incremental_water_removed_g": 2,
        "incremental_temperature_drop_c": 3,
        "incremental_ventilation_energy_removed_kwh": 4,
        "incremental_relative_humidity_reduction_pct": 2,
        "incremental_grams_removed_per_incremental_kwh": 1,
    }
    print(
        "Incremental view (change relative to the previous duration)"
    )
    print("-" * 60)
    incremental_formatted = incremental_dataframe.round(
        incremental_display_decimals
    )
    print(incremental_formatted.to_string(index=False))
    print()
    print(f"saved display table    -> {DISPLAY_CSV_PATH}")
    print(f"saved incremental view -> {INCREMENTAL_CSV_PATH}")
    print(f"saved full results     -> {FULL_CSV_PATH}")


if __name__ == "__main__":
    main()
