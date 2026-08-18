"""Tests for the sensor-CSV validation module.

Covers:
    - CSV ingestion: missing columns, missing header, unparseable
      numbers, unparseable booleans, non-monotone timestamps.
    - Optional ``heater_on`` column round-trips.
    - Event detection: contiguous runs, minimum-length filter,
      leading/trailing closed observations ignored.
    - Forward-simulation validation: synthetic data generated from
      the same physics model reduces MAE to numerical noise (i.e.
      the validation function faithfully reproduces the model when
      the model is exact).
    - Thermal validation optional: when C_eff is not passed, the T
      / RH fields are None and no MAE is computed; when it is
      passed, MAE is computed.
    - Validation NEVER mutates its inputs (frozen dataclasses).
    - Refusal cases: empty events, invalid ACH / volume / C_eff.
"""

import sys
from io import StringIO
from math import exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from psychrometrics import AirState, relative_humidity_from_absolute_humidity
from thermal import ventilation_heat_loss_coefficient

from validation import (
    CSV_HEATER_ON_COLUMN,
    CSV_INDOOR_RH_COLUMN,
    CSV_INDOOR_TEMPERATURE_COLUMN,
    CSV_OUTDOOR_RH_COLUMN,
    CSV_OUTDOOR_TEMPERATURE_COLUMN,
    CSV_TIMESTAMP_COLUMN,
    CSV_WINDOW_OPEN_COLUMN,
    EventValidationResult,
    SensorObservation,
    identify_ventilation_events,
    load_observations_from_csv,
    validate_events,
)


def _linspace(start, stop, count):
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def _csv_header() -> str:
    return ",".join(
        [
            CSV_TIMESTAMP_COLUMN,
            CSV_INDOOR_TEMPERATURE_COLUMN,
            CSV_INDOOR_RH_COLUMN,
            CSV_OUTDOOR_TEMPERATURE_COLUMN,
            CSV_OUTDOOR_RH_COLUMN,
            CSV_WINDOW_OPEN_COLUMN,
        ]
    )


def _synthetic_event_rows(
    true_ach: float,
    room_volume_m3: float,
    initial_indoor_t: float,
    initial_indoor_rh: float,
    outdoor_t: float,
    outdoor_rh: float,
    times_hours,
    window_open: bool = True,
    true_c_eff_j_per_k: float = None,
):
    """Rows exactly satisfying moisture and (optionally) thermal ODEs.

    Returns a list of dicts keyed by CSV column names.
    """
    outdoor_ah = AirState(outdoor_t, outdoor_rh).absolute_humidity
    initial_indoor_ah = AirState(
        initial_indoor_t, initial_indoor_rh
    ).absolute_humidity
    h_vent = (
        ventilation_heat_loss_coefficient(room_volume_m3, true_ach)
        if true_c_eff_j_per_k is not None
        else None
    )

    rows = []
    indoor_ah = initial_indoor_ah
    indoor_t = initial_indoor_t
    for i, t in enumerate(times_hours):
        if i > 0:
            dt = t - times_hours[i - 1]
            indoor_ah = outdoor_ah + (
                indoor_ah - outdoor_ah
            ) * exp(-true_ach * dt)
            if true_c_eff_j_per_k is not None:
                indoor_t = outdoor_t + (
                    indoor_t - outdoor_t
                ) * exp(-h_vent / true_c_eff_j_per_k * dt * 3600.0)
        rh = relative_humidity_from_absolute_humidity(
            temperature_c=indoor_t,
            absolute_humidity_g_m3=indoor_ah,
        )
        rows.append(
            {
                CSV_TIMESTAMP_COLUMN: t,
                CSV_INDOOR_TEMPERATURE_COLUMN: indoor_t,
                CSV_INDOOR_RH_COLUMN: rh,
                CSV_OUTDOOR_TEMPERATURE_COLUMN: outdoor_t,
                CSV_OUTDOOR_RH_COLUMN: outdoor_rh,
                CSV_WINDOW_OPEN_COLUMN: str(window_open).lower(),
            }
        )
    return rows


def _rows_to_csv(rows, include_heater=False):
    columns = list(rows[0].keys())
    if include_heater and CSV_HEATER_ON_COLUMN not in columns:
        columns.append(CSV_HEATER_ON_COLUMN)
    header = ",".join(columns)
    lines = [header]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in columns))
    return "\n".join(lines) + "\n"


# --- CSV ingestion --------------------------------------------------------


def test_load_valid_csv_returns_observations() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.5, 6),
    )
    csv_text = _rows_to_csv(rows)
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    assert len(observations) == 6
    assert observations[0].timestamp_hours == pytest.approx(0.0)
    assert observations[0].window_open is True
    assert observations[0].heater_on is None


def test_load_csv_missing_header_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        load_observations_from_csv(StringIO("").readlines())


def test_load_csv_missing_required_column_raises() -> None:
    csv_text = (
        f"{CSV_TIMESTAMP_COLUMN},{CSV_INDOOR_TEMPERATURE_COLUMN},"
        f"{CSV_INDOOR_RH_COLUMN},{CSV_OUTDOOR_TEMPERATURE_COLUMN}\n"
        "0.0,20.0,55.0,-2.0\n"
    )
    with pytest.raises(ValueError, match=CSV_OUTDOOR_RH_COLUMN):
        load_observations_from_csv(StringIO(csv_text).readlines())


def test_load_csv_unparseable_number_raises() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.2, 3),
    )
    rows[1][CSV_INDOOR_TEMPERATURE_COLUMN] = "not-a-number"
    with pytest.raises(ValueError, match="could not convert"):
        load_observations_from_csv(
            StringIO(_rows_to_csv(rows)).readlines()
        )


def test_load_csv_unparseable_boolean_raises() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.2, 3),
    )
    rows[1][CSV_WINDOW_OPEN_COLUMN] = "maybe"
    with pytest.raises(ValueError, match="bool"):
        load_observations_from_csv(
            StringIO(_rows_to_csv(rows)).readlines()
        )


def test_load_csv_non_monotone_timestamps_raise() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=[0.0, 0.1, 0.2],
    )
    rows[2][CSV_TIMESTAMP_COLUMN] = 0.05  # goes backward
    with pytest.raises(ValueError, match="time-sorted"):
        load_observations_from_csv(
            StringIO(_rows_to_csv(rows)).readlines()
        )


def test_load_csv_optional_heater_column_parsed() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.2, 3),
    )
    for row in rows:
        row[CSV_HEATER_ON_COLUMN] = "true"
    csv_text = _rows_to_csv(rows, include_heater=True)
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    for obs in observations:
        assert obs.heater_on is True


def test_load_csv_from_filesystem(tmp_path: Path) -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 4),
    )
    csv_file = tmp_path / "sensor.csv"
    csv_file.write_text(_rows_to_csv(rows))
    observations = load_observations_from_csv(csv_file)
    assert len(observations) == 4


def test_various_boolean_spellings_accepted() -> None:
    csv_lines = [_csv_header()]
    for i, w in enumerate(("true", "FALSE", "1", "0", "yes", "no")):
        csv_lines.append(f"{i * 0.05},20.0,55.0,-2.0,70.0,{w}")
    csv_text = "\n".join(csv_lines) + "\n"
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    assert [o.window_open for o in observations] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


# --- Event detection -----------------------------------------------------


def test_detects_contiguous_open_run() -> None:
    obs = tuple(
        SensorObservation(
            timestamp_hours=i * 0.05,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            outdoor_temperature_c=-2.0,
            outdoor_relative_humidity_pct=70.0,
            window_open=(2 <= i <= 6),
        )
        for i in range(10)
    )
    events = identify_ventilation_events(obs, minimum_samples=3)
    assert len(events) == 1
    assert events[0].start_index == 2
    assert events[0].end_index_exclusive == 7


def test_multiple_disjoint_open_runs_detected() -> None:
    window_flags = [False, True, True, True, False, False, True, True, True, True]
    obs = tuple(
        SensorObservation(
            timestamp_hours=i * 0.05,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            outdoor_temperature_c=-2.0,
            outdoor_relative_humidity_pct=70.0,
            window_open=w,
        )
        for i, w in enumerate(window_flags)
    )
    events = identify_ventilation_events(obs, minimum_samples=3)
    assert len(events) == 2


def test_short_runs_filtered_by_minimum_samples() -> None:
    window_flags = [False, True, True, False, True, True, True, True, False]
    obs = tuple(
        SensorObservation(
            timestamp_hours=i * 0.05,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            outdoor_temperature_c=-2.0,
            outdoor_relative_humidity_pct=70.0,
            window_open=w,
        )
        for i, w in enumerate(window_flags)
    )
    events = identify_ventilation_events(obs, minimum_samples=4)
    # Only the 4-long run survives.
    assert len(events) == 1


def test_event_extends_to_end_of_series() -> None:
    window_flags = [False, False, True, True, True, True]
    obs = tuple(
        SensorObservation(
            timestamp_hours=i * 0.05,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            outdoor_temperature_c=-2.0,
            outdoor_relative_humidity_pct=70.0,
            window_open=w,
        )
        for i, w in enumerate(window_flags)
    )
    events = identify_ventilation_events(obs, minimum_samples=3)
    assert len(events) == 1
    assert events[0].end_index_exclusive == len(obs)


def test_minimum_samples_below_two_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_samples"):
        identify_ventilation_events((), minimum_samples=1)


# --- Validation on synthetic data ----------------------------------------


def test_ah_mae_near_zero_on_clean_synthetic_data() -> None:
    """The physics model reproduces itself: MAE ~ float noise."""
    true_ach = 5.0
    rows = _synthetic_event_rows(
        true_ach=true_ach,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.5, 21),
    )
    csv_text = _rows_to_csv(rows)
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    events = identify_ventilation_events(observations)
    assert len(events) == 1
    results = validate_events(
        observations=observations,
        events=events,
        ach=true_ach,
        room_volume_m3=40.0,
    )
    assert results[0].mae_absolute_humidity_g_m3 < 1e-9
    assert results[0].predicted_indoor_temperature_c is None
    assert results[0].mae_temperature_c is None


def test_wrong_ach_produces_measurable_mae() -> None:
    """Passing a badly-wrong ACH increases MAE (basic sanity check)."""
    true_ach = 5.0
    rows = _synthetic_event_rows(
        true_ach=true_ach,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.5, 21),
    )
    csv_text = _rows_to_csv(rows)
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    events = identify_ventilation_events(observations)
    correct = validate_events(
        observations=observations, events=events, ach=true_ach, room_volume_m3=40.0
    )
    wrong = validate_events(
        observations=observations, events=events, ach=1.0, room_volume_m3=40.0
    )
    assert (
        wrong[0].mae_absolute_humidity_g_m3
        > correct[0].mae_absolute_humidity_g_m3
    )


def test_thermal_validation_produces_predictions_and_mae() -> None:
    true_ach = 5.0
    true_c_eff = 500_000.0
    rows = _synthetic_event_rows(
        true_ach=true_ach,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.5, 21),
        true_c_eff_j_per_k=true_c_eff,
    )
    csv_text = _rows_to_csv(rows)
    observations = load_observations_from_csv(StringIO(csv_text).readlines())
    events = identify_ventilation_events(observations)
    results = validate_events(
        observations=observations,
        events=events,
        ach=true_ach,
        room_volume_m3=40.0,
        effective_thermal_capacity_j_per_k=true_c_eff,
    )
    r = results[0]
    assert r.predicted_indoor_temperature_c is not None
    assert r.predicted_indoor_relative_humidity_pct is not None
    assert r.mae_temperature_c < 1e-9
    assert r.mae_relative_humidity_pct < 1e-6


def test_observations_and_events_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    obs = SensorObservation(
        timestamp_hours=0.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        outdoor_temperature_c=-2.0,
        outdoor_relative_humidity_pct=70.0,
        window_open=True,
    )
    with pytest.raises(FrozenInstanceError):
        obs.window_open = False  # type: ignore[misc]


def test_result_is_frozen_and_carries_named_fields() -> None:
    from dataclasses import FrozenInstanceError

    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 8),
    )
    obs = load_observations_from_csv(
        StringIO(_rows_to_csv(rows)).readlines()
    )
    events = identify_ventilation_events(obs)
    results = validate_events(
        observations=obs, events=events, ach=5.0, room_volume_m3=40.0
    )
    r = results[0]
    assert isinstance(r, EventValidationResult)
    for name in (
        "event",
        "start_time_hours",
        "end_time_hours",
        "n_samples",
        "times_hours",
        "observed_indoor_absolute_humidity_g_m3",
        "predicted_indoor_absolute_humidity_g_m3",
        "observed_indoor_temperature_c",
        "predicted_indoor_temperature_c",
        "observed_indoor_relative_humidity_pct",
        "predicted_indoor_relative_humidity_pct",
        "mae_absolute_humidity_g_m3",
        "mae_temperature_c",
        "mae_relative_humidity_pct",
        "ach_used",
        "effective_thermal_capacity_j_per_k",
        "room_volume_m3",
    ):
        assert hasattr(r, name), name
    with pytest.raises(FrozenInstanceError):
        r.mae_absolute_humidity_g_m3 = 0.0  # type: ignore[misc]


def test_predicted_first_ah_equals_observed_first_ah() -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 8),
    )
    obs = load_observations_from_csv(
        StringIO(_rows_to_csv(rows)).readlines()
    )
    events = identify_ventilation_events(obs)
    results = validate_events(
        observations=obs, events=events, ach=5.0, room_volume_m3=40.0
    )
    r = results[0]
    assert (
        r.predicted_indoor_absolute_humidity_g_m3[0]
        == r.observed_indoor_absolute_humidity_g_m3[0]
    )


# --- Refusal cases -------------------------------------------------------


@pytest.mark.parametrize("bad_ach", [-1.0, float("nan"), float("inf")])
def test_invalid_ach_rejected(bad_ach: float) -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 4),
    )
    obs = load_observations_from_csv(
        StringIO(_rows_to_csv(rows)).readlines()
    )
    events = identify_ventilation_events(obs)
    with pytest.raises(ValueError, match="ach"):
        validate_events(
            observations=obs,
            events=events,
            ach=bad_ach,
            room_volume_m3=40.0,
        )


@pytest.mark.parametrize("bad_vol", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_room_volume_rejected(bad_vol: float) -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 4),
    )
    obs = load_observations_from_csv(
        StringIO(_rows_to_csv(rows)).readlines()
    )
    events = identify_ventilation_events(obs)
    with pytest.raises(ValueError, match="room_volume_m3"):
        validate_events(
            observations=obs,
            events=events,
            ach=5.0,
            room_volume_m3=bad_vol,
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_c_eff_rejected_when_supplied(bad: float) -> None:
    rows = _synthetic_event_rows(
        true_ach=5.0,
        room_volume_m3=40.0,
        initial_indoor_t=20.0,
        initial_indoor_rh=55.0,
        outdoor_t=-2.0,
        outdoor_rh=70.0,
        times_hours=_linspace(0.0, 0.3, 4),
    )
    obs = load_observations_from_csv(
        StringIO(_rows_to_csv(rows)).readlines()
    )
    events = identify_ventilation_events(obs)
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        validate_events(
            observations=obs,
            events=events,
            ach=5.0,
            room_volume_m3=40.0,
            effective_thermal_capacity_j_per_k=bad,
        )
