"""Outdoor weather forecast abstraction for the ventilation model.

The rest of the model consumes outdoor conditions through
``psychrometrics.AirState`` values. This module owns the ingestion of
a future-looking outdoor time series (temperature and relative
humidity per timestamp) and exposes a single query interface -
``WeatherForecast.sample_at(time_hours)`` - that returns the outdoor
``AirState`` for any time in the horizon. Every derived psychrometric
quantity (vapour pressure, absolute humidity, dew point) is available
on the returned ``AirState`` via its existing properties, so no
consumer needs to know where the forecast came from or how it was
stored.

Two ingestion shapes are supported for the POC:

    1. Python data structures - a sequence of dicts (or any iterable
       of dict-like rows) via ``forecast_from_dicts``.
    2. CSV files - via ``forecast_from_csv``, using the stdlib ``csv``
       module. No pandas, no external dependencies.

Both routes flow into the same ``WeatherForecast`` value, so the
downstream model is agnostic to origin.

Explicitly NOT in this slice:
    - Live weather-API clients.
    - Statistical forecasting or downscaling.
    - Any physics (this module owns unit-tagged storage and lookup;
      every conversion delegates to ``psychrometrics``).

Timestamp convention:
    Timestamps on incoming rows are expressed in hours since the
    forecast's origin (t = 0.0 hours). Callers that receive absolute
    timestamps from a data source can convert to relative hours at
    the boundary; keeping the module's time unit consistent with
    ``time_simulation.RoomTrajectory.times_hours`` avoids a second
    time convention in the codebase. Timestamps must be strictly
    monotone increasing; the ``sample_at`` lookup relies on the
    ordering.

Sampling convention:
    Piecewise-constant, START-of-interval. If the forecast contains
    a point at t = 1.0 h and the next at t = 4.0 h, every query
    time in [1.0, 4.0) returns the t = 1.0 h reading. Queries before
    the first point return the first point (the forecast is treated
    as valid from t = 0 onwards for POC convenience); queries after
    the last point return the last point (no extrapolation of trend
    - the last known reading is held). This matches the piecewise-
    constant conventions used elsewhere in the model (window state,
    moisture-generation rate).
"""

import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple, Union

from psychrometrics import AirState

CSV_TIMESTAMP_COLUMN: str = "timestamp_hours"
CSV_TEMPERATURE_COLUMN: str = "temperature_c"
CSV_RELATIVE_HUMIDITY_COLUMN: str = "relative_humidity_percent"

_REQUIRED_COLUMNS: Tuple[str, str, str] = (
    CSV_TIMESTAMP_COLUMN,
    CSV_TEMPERATURE_COLUMN,
    CSV_RELATIVE_HUMIDITY_COLUMN,
)


@dataclass(frozen=True)
class ForecastPoint:
    """One outdoor reading at one timestamp.

    Fields:
        timestamp_hours: hours since the forecast's origin, strictly
            non-negative and finite. The comparison-adjacent tests use
            this as an ordering key.
        temperature_c: outdoor air temperature in degrees Celsius.
            Validated by ``AirState`` when the point is queried, so
            the residential range check lives in exactly one place.
        relative_humidity_percent: outdoor RH in percent (0-100).
            Also validated by ``AirState`` on query.

    Rationale for storing raw T + RH rather than pre-computing AH /
    dew point / vapour pressure: the psychrometrics module owns the
    equations; every ``ForecastPoint`` produces its ``AirState`` on
    demand, which recomputes every derived quantity from the two
    stored fields. That keeps the physics in exactly one owner and
    matches the immutability the rest of the value objects rely on.
    """

    timestamp_hours: float
    temperature_c: float
    relative_humidity_percent: float

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_hours):
            raise ValueError(
                f"timestamp_hours must be finite, got {self.timestamp_hours!r}"
            )
        if self.timestamp_hours < 0.0:
            raise ValueError(
                "timestamp_hours must be non-negative, got "
                f"{self.timestamp_hours}"
            )
        # T and RH are NOT validated here - AirState owns the
        # residential-range checks so those messages appear in
        # exactly one place. A ForecastPoint holds the raw pair; a
        # consumer that reads it constructs an AirState and pays
        # AirState's validation cost.

    def to_air_state(self) -> AirState:
        """Return the AirState for this point's temperature and RH.

        Delegates to ``psychrometrics.AirState`` for validation and
        for every derived psychrometric quantity. Callers who want
        absolute humidity, dew point, vapour pressure, or humidity
        ratio read them off the returned state's properties - no
        physics happens here.
        """
        return AirState(
            temperature_c=self.temperature_c,
            relative_humidity_percent=self.relative_humidity_percent,
        )


@dataclass(frozen=True)
class WeatherForecast:
    """Ordered outdoor forecast with a piecewise-constant lookup.

    Fields:
        points: tuple of ``ForecastPoint``. Must be non-empty; every
            timestamp must be strictly greater than its predecessor
            (irregular spacing is allowed, but duplicates and
            reversals are not).

    Query interface: ``sample_at(time_hours)`` returns the outdoor
    ``AirState`` corresponding to the forecast segment containing
    ``time_hours``. Segments are half-open, START-of-interval; see
    the module docstring for the exact semantics on the boundaries.

    The forecast does NOT store any pre-computed psychrometric
    quantities - it stores raw T/RH per timestamp and lets the query
    interface produce ``AirState`` values on demand. Consumers who
    want outdoor AH per point can call
    ``forecast.sample_at(t).absolute_humidity``.
    """

    points: Tuple[ForecastPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) == 0:
            raise ValueError(
                "WeatherForecast must contain at least one ForecastPoint."
            )
        for i in range(1, len(self.points)):
            previous_ts = self.points[i - 1].timestamp_hours
            current_ts = self.points[i].timestamp_hours
            if current_ts <= previous_ts:
                raise ValueError(
                    "ForecastPoints must have strictly increasing "
                    f"timestamps; entry at index {i} has "
                    f"timestamp_hours={current_ts} which is not greater "
                    f"than the previous timestamp_hours={previous_ts}."
                )

    @property
    def horizon_hours(self) -> float:
        """Duration from the first to the last forecast timestamp."""
        return self.points[-1].timestamp_hours - self.points[0].timestamp_hours

    def sample_at(self, time_hours: float) -> AirState:
        """Return the outdoor AirState valid at ``time_hours``.

        Piecewise-constant, start-of-interval lookup. See the module
        docstring for the boundary handling.
        """
        if not isfinite(time_hours):
            raise ValueError(f"time_hours must be finite, got {time_hours!r}")
        selected = self.points[0]
        for point in self.points:
            if point.timestamp_hours <= time_hours:
                selected = point
            else:
                break
        return selected.to_air_state()


def _validate_row(
    row: Mapping[str, object], row_index: int
) -> Tuple[float, float, float]:
    """Extract (ts, T, RH) from a row-like mapping with clear errors.

    Missing columns and non-numeric values are the two ingestion
    failure modes this function names explicitly, so callers get a
    line-number-aware error rather than a KeyError or a bare
    ValueError from a downstream numeric conversion.
    """
    for column in _REQUIRED_COLUMNS:
        if column not in row or row[column] is None or row[column] == "":
            raise ValueError(
                f"row {row_index}: missing value for required column "
                f"{column!r}. Every forecast row must supply a "
                f"timestamp, a temperature, and a relative humidity."
            )
    try:
        ts = float(row[CSV_TIMESTAMP_COLUMN])
        t = float(row[CSV_TEMPERATURE_COLUMN])
        rh = float(row[CSV_RELATIVE_HUMIDITY_COLUMN])
    except (TypeError, ValueError) as cause:
        raise ValueError(
            f"row {row_index}: could not convert every field to a number "
            f"({cause})."
        ) from cause
    return ts, t, rh


def forecast_from_dicts(
    rows: Iterable[Mapping[str, object]],
) -> WeatherForecast:
    """Build a ``WeatherForecast`` from an iterable of dict-like rows.

    Each row must expose ``timestamp_hours``, ``temperature_c``, and
    ``relative_humidity_percent`` as string or numeric values. Row
    ordering must be monotone in ``timestamp_hours`` (irregular
    spacing is fine); non-monotone or duplicate timestamps are
    rejected by ``WeatherForecast.__post_init__``.

    Args:
        rows: any iterable of mapping-like rows.

    Returns:
        A ``WeatherForecast`` composed of the ingested points.

    Raises:
        ValueError: on missing columns, non-numeric fields, empty
            input, or non-monotone timestamps. Downstream range
            errors on temperature and RH surface when the point is
            queried via ``sample_at``.
    """
    points: List[ForecastPoint] = []
    for row_index, row in enumerate(rows):
        ts, t, rh = _validate_row(row, row_index)
        points.append(
            ForecastPoint(
                timestamp_hours=ts,
                temperature_c=t,
                relative_humidity_percent=rh,
            )
        )
    return WeatherForecast(points=tuple(points))


def forecast_from_csv(
    source: Union[str, Path, Iterable[str]],
) -> WeatherForecast:
    """Build a ``WeatherForecast`` from a CSV source.

    The CSV must expose three named columns (case-sensitive):
    ``timestamp_hours``, ``temperature_c``,
    ``relative_humidity_percent``. Column order is flexible; extra
    columns are ignored.

    Args:
        source: either a file path (str or Path) that will be opened
            for reading with the default encoding, or any iterable of
            CSV-formatted lines (already opened file, StringIO, etc.).
            The iterable-of-lines route is provided so tests do not
            need to touch the filesystem.

    Returns:
        A ``WeatherForecast`` composed of the ingested points.

    Raises:
        ValueError: on missing required columns, non-numeric fields,
            empty input, or non-monotone timestamps.
        FileNotFoundError: if a str / Path source cannot be opened.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", newline="") as handle:
            return _forecast_from_csv_lines(handle)
    return _forecast_from_csv_lines(source)


def _forecast_from_csv_lines(lines: Iterable[str]) -> WeatherForecast:
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(
            "CSV source is empty; expected a header row with columns "
            f"{_REQUIRED_COLUMNS}."
        )
    missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(
            "CSV source is missing required column(s) "
            f"{missing}. Present columns were {list(reader.fieldnames)}."
        )
    return forecast_from_dicts(reader)


def outdoor_air_states(
    forecast: WeatherForecast,
    times_hours: Sequence[float],
) -> Tuple[AirState, ...]:
    """Sample the forecast at each caller-supplied time.

    Convenience wrapper for callers that already have a time grid
    (typically the ``times_hours`` field of a ``RoomTrajectory``) and
    want the outdoor ``AirState`` at every step. Order is preserved.

    Args:
        forecast: the source forecast.
        times_hours: sequence of query times in hours.

    Returns:
        A tuple of ``AirState`` values, one per query time, in the
        same order as ``times_hours``.
    """
    return tuple(forecast.sample_at(t) for t in times_hours)
