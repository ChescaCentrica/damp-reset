"""Tests for the outdoor weather-forecast abstraction.

Covers:
    - Changing outdoor temperature across the horizon.
    - Changing outdoor RH across the horizon.
    - Missing timestamps / missing columns rejected clearly.
    - Invalid RH values rejected by the psychrometric layer.
    - Irregular forecast intervals accepted; monotone order enforced.
    - Origin-agnosticism: the same time series ingested via a Python
      data structure and via CSV produces identical ``WeatherForecast``
      state, so the rest of the model does not care which route was
      used.
"""

import sys
from dataclasses import FrozenInstanceError
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from psychrometrics import AirState

from weather_forecast import (
    CSV_RELATIVE_HUMIDITY_COLUMN,
    CSV_TEMPERATURE_COLUMN,
    CSV_TIMESTAMP_COLUMN,
    ForecastPoint,
    WeatherForecast,
    forecast_from_csv,
    forecast_from_dicts,
    outdoor_air_states,
)


def _rows(triples):
    """Compact helper: list of (ts, T, RH) tuples -> list of dict rows."""
    return [
        {
            CSV_TIMESTAMP_COLUMN: ts,
            CSV_TEMPERATURE_COLUMN: t,
            CSV_RELATIVE_HUMIDITY_COLUMN: rh,
        }
        for ts, t, rh in triples
    ]


def _csv_text(triples, header=True) -> str:
    """Same shape as _rows but rendered as CSV text."""
    lines = []
    if header:
        lines.append(
            ",".join(
                [
                    CSV_TIMESTAMP_COLUMN,
                    CSV_TEMPERATURE_COLUMN,
                    CSV_RELATIVE_HUMIDITY_COLUMN,
                ]
            )
        )
    for ts, t, rh in triples:
        lines.append(f"{ts},{t},{rh}")
    return "\n".join(lines) + "\n"


# --- Basic shape and immutability ----------------------------------------


def test_forecast_point_is_frozen() -> None:
    point = ForecastPoint(
        timestamp_hours=0.0, temperature_c=10.0, relative_humidity_percent=60.0
    )
    with pytest.raises(FrozenInstanceError):
        point.temperature_c = 11.0  # type: ignore[misc]


def test_weather_forecast_is_frozen() -> None:
    forecast = forecast_from_dicts(_rows([(0.0, 10.0, 60.0)]))
    with pytest.raises(FrozenInstanceError):
        forecast.points = tuple()  # type: ignore[misc]


def test_forecast_horizon_hours_matches_first_and_last() -> None:
    forecast = forecast_from_dicts(
        _rows([(0.0, 10.0, 60.0), (2.5, 11.0, 65.0), (5.0, 12.0, 70.0)])
    )
    assert forecast.horizon_hours == pytest.approx(5.0)


# --- Changing outdoor temperature ----------------------------------------


def test_changing_outdoor_temperature_reflected_at_each_sample() -> None:
    """Consecutive samples of a forecast with rising T track the input."""
    forecast = forecast_from_dicts(
        _rows(
            [
                (0.0, 5.0, 80.0),
                (1.0, 8.0, 80.0),
                (2.0, 12.0, 80.0),
                (3.0, 15.0, 80.0),
            ]
        )
    )
    for query, expected_t in ((0.0, 5.0), (1.0, 8.0), (2.0, 12.0), (3.0, 15.0)):
        assert forecast.sample_at(query).temperature_c == pytest.approx(
            expected_t
        )


def test_falling_outdoor_temperature_reflected_in_derived_absolute_humidity() -> None:
    """At fixed RH, colder outdoor air has LOWER absolute humidity.

    Load-bearing: the whole point of consuming outdoor data through
    psychrometrics is that comparisons downstream use AH, not raw RH.
    """
    forecast = forecast_from_dicts(
        _rows(
            [
                (0.0, 15.0, 70.0),
                (1.0, 10.0, 70.0),
                (2.0, 5.0, 70.0),
                (3.0, 0.0, 70.0),
            ]
        )
    )
    ahs = [forecast.sample_at(t).absolute_humidity for t in (0.0, 1.0, 2.0, 3.0)]
    # Strictly decreasing.
    for earlier, later in zip(ahs, ahs[1:]):
        assert later < earlier


# --- Changing outdoor RH -------------------------------------------------


def test_changing_outdoor_rh_reflected_at_each_sample() -> None:
    forecast = forecast_from_dicts(
        _rows(
            [
                (0.0, 15.0, 40.0),
                (1.0, 15.0, 55.0),
                (2.0, 15.0, 70.0),
                (3.0, 15.0, 90.0),
            ]
        )
    )
    for query, expected_rh in (
        (0.0, 40.0),
        (1.0, 55.0),
        (2.0, 70.0),
        (3.0, 90.0),
    ):
        state = forecast.sample_at(query)
        assert state.relative_humidity_percent == pytest.approx(expected_rh)


def test_rising_rh_at_fixed_temperature_increases_absolute_humidity() -> None:
    """At fixed T, higher RH -> higher AH."""
    forecast = forecast_from_dicts(
        _rows(
            [
                (0.0, 15.0, 30.0),
                (1.0, 15.0, 50.0),
                (2.0, 15.0, 70.0),
                (3.0, 15.0, 90.0),
            ]
        )
    )
    ahs = [forecast.sample_at(t).absolute_humidity for t in (0.0, 1.0, 2.0, 3.0)]
    for earlier, later in zip(ahs, ahs[1:]):
        assert later > earlier


# --- Missing timestamps / columns -----------------------------------------


def test_missing_timestamp_column_in_dict_rows_rejected() -> None:
    """A row without ``timestamp_hours`` is rejected with a clear message."""
    rows = [
        {CSV_TEMPERATURE_COLUMN: 10.0, CSV_RELATIVE_HUMIDITY_COLUMN: 60.0}
    ]
    with pytest.raises(ValueError, match=CSV_TIMESTAMP_COLUMN):
        forecast_from_dicts(rows)


def test_missing_temperature_column_in_dict_rows_rejected() -> None:
    rows = [
        {CSV_TIMESTAMP_COLUMN: 0.0, CSV_RELATIVE_HUMIDITY_COLUMN: 60.0}
    ]
    with pytest.raises(ValueError, match=CSV_TEMPERATURE_COLUMN):
        forecast_from_dicts(rows)


def test_missing_rh_column_in_dict_rows_rejected() -> None:
    rows = [{CSV_TIMESTAMP_COLUMN: 0.0, CSV_TEMPERATURE_COLUMN: 10.0}]
    with pytest.raises(ValueError, match=CSV_RELATIVE_HUMIDITY_COLUMN):
        forecast_from_dicts(rows)


def test_empty_forecast_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        forecast_from_dicts([])


def test_missing_timestamp_value_in_csv_rejected() -> None:
    """A CSV row with a blank timestamp cell is rejected."""
    csv_text = (
        f"{CSV_TIMESTAMP_COLUMN},{CSV_TEMPERATURE_COLUMN},"
        f"{CSV_RELATIVE_HUMIDITY_COLUMN}\n"
        ",10.0,60.0\n"
    )
    with pytest.raises(ValueError, match=CSV_TIMESTAMP_COLUMN):
        forecast_from_csv(StringIO(csv_text).readlines())


def test_missing_header_in_csv_rejected() -> None:
    """A CSV with no header at all is caught before parsing rows."""
    with pytest.raises(ValueError, match="empty"):
        forecast_from_csv(StringIO("").readlines())


def test_missing_required_column_in_csv_header_rejected() -> None:
    csv_text = (
        f"{CSV_TIMESTAMP_COLUMN},{CSV_TEMPERATURE_COLUMN}\n0.0,10.0\n"
    )
    with pytest.raises(ValueError, match=CSV_RELATIVE_HUMIDITY_COLUMN):
        forecast_from_csv(StringIO(csv_text).readlines())


# --- Invalid RH values ---------------------------------------------------


def test_negative_rh_rejected_when_queried() -> None:
    """A row with RH < 0 is rejected via AirState validation on sample.

    The forecast stores the raw pair; the residential-range check
    lives on ``AirState`` (exactly one owner). Constructing a
    ``ForecastPoint`` with RH = -5 does not itself raise, but
    ``sample_at`` on that point does (via the AirState property call
    inside the psychrometric functions).
    """
    forecast = forecast_from_dicts(_rows([(0.0, 10.0, -5.0)]))
    with pytest.raises(ValueError, match="relative_humidity"):
        forecast.sample_at(0.0).absolute_humidity


def test_rh_above_100_rejected_when_queried() -> None:
    forecast = forecast_from_dicts(_rows([(0.0, 10.0, 110.0)]))
    with pytest.raises(ValueError, match="relative_humidity"):
        forecast.sample_at(0.0).absolute_humidity


def test_non_numeric_rh_rejected_at_ingestion() -> None:
    """A row whose RH cannot be parsed as a number is rejected at ingestion."""
    rows = [
        {
            CSV_TIMESTAMP_COLUMN: 0.0,
            CSV_TEMPERATURE_COLUMN: 10.0,
            CSV_RELATIVE_HUMIDITY_COLUMN: "not-a-number",
        }
    ]
    with pytest.raises(ValueError, match="could not convert"):
        forecast_from_dicts(rows)


# --- Irregular forecast intervals ---------------------------------------


def test_irregular_intervals_accepted() -> None:
    """Non-uniform spacing between timestamps is legal."""
    forecast = forecast_from_dicts(
        _rows(
            [
                (0.0, 5.0, 80.0),
                (0.25, 7.0, 78.0),  # 15 min later
                (1.5, 10.0, 70.0),  # 1 h 15 min later
                (6.0, 3.0, 90.0),  # 4 h 30 min later
            ]
        )
    )
    # Between 1.5 and 6.0 h, everything holds the t=1.5 reading.
    for query in (1.5, 2.0, 4.9, 5.9999):
        state = forecast.sample_at(query)
        assert state.temperature_c == pytest.approx(10.0)
        assert state.relative_humidity_percent == pytest.approx(70.0)
    # From t=6.0 onwards, the last reading holds.
    for query in (6.0, 7.0, 100.0):
        state = forecast.sample_at(query)
        assert state.temperature_c == pytest.approx(3.0)


def test_non_monotone_timestamps_rejected() -> None:
    """Timestamps must strictly increase."""
    with pytest.raises(ValueError, match="increasing"):
        forecast_from_dicts(_rows([(0.0, 10.0, 60.0), (0.0, 11.0, 61.0)]))
    with pytest.raises(ValueError, match="increasing"):
        forecast_from_dicts(
            _rows(
                [(0.0, 10.0, 60.0), (2.0, 11.0, 61.0), (1.5, 12.0, 62.0)]
            )
        )


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValueError, match="timestamp_hours"):
        ForecastPoint(
            timestamp_hours=-0.1,
            temperature_c=10.0,
            relative_humidity_percent=60.0,
        )


# --- Boundary behaviour on sample_at ------------------------------------


def test_query_before_first_point_returns_first_point() -> None:
    """A query time earlier than the first timestamp holds the first reading.

    The forecast is treated as valid from t = 0 onwards for POC
    convenience. Callers who want a stricter interpretation can
    range-check before sampling.
    """
    forecast = forecast_from_dicts(
        _rows([(2.0, 5.0, 80.0), (4.0, 6.0, 82.0)])
    )
    state = forecast.sample_at(0.0)
    assert state.temperature_c == pytest.approx(5.0)


def test_query_after_last_point_returns_last_point() -> None:
    """Queries beyond the horizon hold the final reading (no extrapolation)."""
    forecast = forecast_from_dicts(
        _rows([(0.0, 5.0, 80.0), (2.0, 8.0, 75.0)])
    )
    state = forecast.sample_at(10.0)
    assert state.temperature_c == pytest.approx(8.0)
    assert state.relative_humidity_percent == pytest.approx(75.0)


def test_query_at_boundary_uses_new_segment() -> None:
    """At t equal to a point's timestamp, that point becomes active."""
    forecast = forecast_from_dicts(
        _rows([(0.0, 5.0, 80.0), (1.0, 15.0, 40.0)])
    )
    just_before = forecast.sample_at(0.999)
    at_boundary = forecast.sample_at(1.0)
    assert just_before.temperature_c == pytest.approx(5.0)
    assert at_boundary.temperature_c == pytest.approx(15.0)


def test_non_finite_query_time_rejected() -> None:
    forecast = forecast_from_dicts(_rows([(0.0, 5.0, 80.0)]))
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="time_hours"):
            forecast.sample_at(bad)


# --- Origin-agnosticism --------------------------------------------------


def test_dict_ingestion_and_csv_ingestion_produce_identical_forecasts() -> None:
    """Same time series, two routes -> identical ``WeatherForecast`` value.

    Load-bearing: the module docstring claims the rest of the model
    does not care where a forecast came from. This test verifies that
    claim structurally.
    """
    triples = [
        (0.0, 5.0, 80.0),
        (1.0, 8.0, 75.0),
        (2.5, 12.0, 60.0),
        (6.0, 3.0, 90.0),
    ]
    from_dicts = forecast_from_dicts(_rows(triples))
    from_csv = forecast_from_csv(StringIO(_csv_text(triples)).readlines())
    assert from_dicts == from_csv


def test_csv_ingestion_from_file_path(tmp_path: Path) -> None:
    """Full round-trip through the filesystem."""
    csv_file = tmp_path / "forecast.csv"
    csv_file.write_text(
        _csv_text(
            [
                (0.0, 5.0, 80.0),
                (1.0, 8.0, 75.0),
                (2.0, 12.0, 60.0),
            ]
        )
    )
    forecast = forecast_from_csv(csv_file)
    assert forecast.horizon_hours == pytest.approx(2.0)
    assert forecast.sample_at(1.0).temperature_c == pytest.approx(8.0)


def test_extra_columns_in_csv_are_ignored() -> None:
    """A CSV with additional columns beyond the required three is still valid."""
    csv_text = (
        f"{CSV_TIMESTAMP_COLUMN},"
        f"{CSV_TEMPERATURE_COLUMN},"
        f"{CSV_RELATIVE_HUMIDITY_COLUMN},"
        "wind_speed_m_s\n"
        "0.0,5.0,80.0,3.5\n"
        "1.0,8.0,75.0,2.1\n"
    )
    forecast = forecast_from_csv(StringIO(csv_text).readlines())
    assert len(forecast.points) == 2
    assert forecast.sample_at(0.5).temperature_c == pytest.approx(5.0)


def test_outdoor_air_states_helper_returns_state_per_time() -> None:
    """The convenience helper returns one ``AirState`` per query time."""
    forecast = forecast_from_dicts(
        _rows([(0.0, 5.0, 80.0), (2.0, 10.0, 70.0), (4.0, 8.0, 75.0)])
    )
    times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    states = outdoor_air_states(forecast, times)
    assert len(states) == len(times)
    for state in states:
        assert isinstance(state, AirState)
    # Piecewise-constant boundaries at t=2.0 and t=4.0.
    assert states[0].temperature_c == pytest.approx(5.0)
    assert states[1].temperature_c == pytest.approx(5.0)
    assert states[2].temperature_c == pytest.approx(10.0)
    assert states[3].temperature_c == pytest.approx(10.0)
    assert states[4].temperature_c == pytest.approx(8.0)
    assert states[5].temperature_c == pytest.approx(8.0)
