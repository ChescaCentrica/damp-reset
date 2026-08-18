"""Validation of the POC physics against measured sensor data.

Reads a CSV (or an in-memory sequence of dicts) of sensor
observations and, for each identified window-open event:

    1. Converts every measured (T, RH) pair into absolute humidity
       via ``psychrometrics.AirState`` (single owner of the
       conversion).
    2. Runs forward simulation of indoor T and indoor AH using a
       CALLER-SUPPLIED ACH and (optionally) C_eff.
    3. Compares predicted vs measured indoor T, AH, and RH.
    4. Returns validation metrics: mean absolute error per
       quantity, per event, and (for a caller who plots) the
       time-aligned predicted and measured series.

Calibration and validation are kept LOGICALLY SEPARATE. This
module does NOT re-fit model parameters using validation data. A
caller who wants to calibrate first runs ``calibration`` and
``thermal_calibration`` on a caller-designated calibration event
(a completely separate CSV, or a specific event index in a shared
file), then passes the resulting ACH and C_eff into
``validate_events``. The validation function itself never touches
the fitter.

CSV contract:
    Header row required. Columns (case sensitive):
        timestamp_hours
        indoor_temperature_c
        indoor_relative_humidity_pct
        outdoor_temperature_c
        outdoor_relative_humidity_pct
        window_open              (bool-like: 'true'/'false', '1'/'0')
        heater_on                (optional; same bool-like format)
    Extra columns are ignored. Rows must be time-sorted; that is
    checked at ingestion. See ``load_observations_from_csv`` for
    the exact accepted boolean spellings.

Explicitly NOT in this module:
    - Machine-learning based validation.
    - Any parameter fitting on the validation set.
    - Plot rendering (kept in the caller-side experiment script so
      matplotlib is not a hard dependency of the module).
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from moisture import predict_final_absolute_humidity
from psychrometrics import AirState, relative_humidity_from_absolute_humidity
from thermal import predict_indoor_temperature


CSV_TIMESTAMP_COLUMN: str = "timestamp_hours"
CSV_INDOOR_TEMPERATURE_COLUMN: str = "indoor_temperature_c"
CSV_INDOOR_RH_COLUMN: str = "indoor_relative_humidity_pct"
CSV_OUTDOOR_TEMPERATURE_COLUMN: str = "outdoor_temperature_c"
CSV_OUTDOOR_RH_COLUMN: str = "outdoor_relative_humidity_pct"
CSV_WINDOW_OPEN_COLUMN: str = "window_open"
CSV_HEATER_ON_COLUMN: str = "heater_on"

_REQUIRED_COLUMNS: Tuple[str, ...] = (
    CSV_TIMESTAMP_COLUMN,
    CSV_INDOOR_TEMPERATURE_COLUMN,
    CSV_INDOOR_RH_COLUMN,
    CSV_OUTDOOR_TEMPERATURE_COLUMN,
    CSV_OUTDOOR_RH_COLUMN,
    CSV_WINDOW_OPEN_COLUMN,
)

_TRUE_LITERALS = {"true", "1", "yes", "on", "t", "y"}
_FALSE_LITERALS = {"false", "0", "no", "off", "f", "n"}


@dataclass(frozen=True)
class SensorObservation:
    """One measured sample.

    Fields:
        timestamp_hours: sample time in hours since the series
            origin. Monotone non-decreasing across the loaded set.
        indoor_temperature_c: indoor air temperature.
        indoor_relative_humidity_pct: indoor RH, 0-100.
        outdoor_temperature_c: outdoor air temperature at the same
            instant.
        outdoor_relative_humidity_pct: outdoor RH, 0-100.
        window_open: whether the caller marks the window open at
            this instant.
        heater_on: optional record of heating-system state. Not
            consumed by the validation forward model (this POC
            treats heating as absent during controlled events);
            propagated for audit only.

    Validation: fields validated by ``AirState`` when the caller
    converts, and by ``__post_init__`` here for timestamp finiteness
    and non-negativity. RH range is validated later on demand
    (indoor RH can occasionally exceed 100 in transient sensor
    dropouts; the caller decides whether to reject or clip).
    """

    timestamp_hours: float
    indoor_temperature_c: float
    indoor_relative_humidity_pct: float
    outdoor_temperature_c: float
    outdoor_relative_humidity_pct: float
    window_open: bool
    heater_on: Optional[bool] = None

    def __post_init__(self) -> None:
        if not _isfinite(self.timestamp_hours):
            raise ValueError(
                "timestamp_hours must be finite, got "
                f"{self.timestamp_hours!r}"
            )
        if self.timestamp_hours < 0.0:
            raise ValueError(
                "timestamp_hours must be non-negative, got "
                f"{self.timestamp_hours}"
            )
        for name, value in (
            ("indoor_temperature_c", self.indoor_temperature_c),
            (
                "indoor_relative_humidity_pct",
                self.indoor_relative_humidity_pct,
            ),
            ("outdoor_temperature_c", self.outdoor_temperature_c),
            (
                "outdoor_relative_humidity_pct",
                self.outdoor_relative_humidity_pct,
            ),
        ):
            if not _isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

    @property
    def indoor_absolute_humidity_g_m3(self) -> float:
        """Indoor AH computed from (T, RH) via the psychrometric owner."""
        return AirState(
            temperature_c=self.indoor_temperature_c,
            relative_humidity_percent=self.indoor_relative_humidity_pct,
        ).absolute_humidity

    @property
    def outdoor_absolute_humidity_g_m3(self) -> float:
        return AirState(
            temperature_c=self.outdoor_temperature_c,
            relative_humidity_percent=self.outdoor_relative_humidity_pct,
        ).absolute_humidity


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), -float("inf"))


def _parse_bool(raw: object, column: str, row_index: int) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        raise ValueError(
            f"row {row_index}: missing value for boolean column {column!r}."
        )
    text = str(raw).strip().lower()
    if text in _TRUE_LITERALS:
        return True
    if text in _FALSE_LITERALS:
        return False
    raise ValueError(
        f"row {row_index}: could not interpret {column}={raw!r} as bool; "
        f"expected one of {sorted(_TRUE_LITERALS | _FALSE_LITERALS)}."
    )


def load_observations_from_csv(
    source: Union[str, Path, Iterable[str]],
) -> Tuple[SensorObservation, ...]:
    """Ingest CSV rows into ``SensorObservation`` values.

    Accepts either a filesystem path (opened for reading with the
    default encoding) or any iterable of CSV-formatted lines
    (already-open file, ``io.StringIO``, ...). Enforces the
    required-column contract, monotone non-decreasing timestamps,
    and bool-like ``window_open`` / ``heater_on`` parsing.

    Args:
        source: str/Path (opens the file), or iterable of lines.

    Returns:
        Tuple of ``SensorObservation``.

    Raises:
        ValueError: on missing columns, non-numeric fields,
            unparseable booleans, or timestamps going backwards.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", newline="") as handle:
            return _load_from_lines(handle)
    return _load_from_lines(source)


def _load_from_lines(lines: Iterable[str]) -> Tuple[SensorObservation, ...]:
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(
            "CSV source is empty; expected a header row with columns "
            f"{_REQUIRED_COLUMNS}."
        )
    missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(
            f"CSV source is missing required column(s) {missing}. "
            f"Present columns were {list(reader.fieldnames)}."
        )
    heater_present = CSV_HEATER_ON_COLUMN in reader.fieldnames
    return _rows_to_observations(reader, heater_present)


def _rows_to_observations(
    rows: Iterable[Mapping[str, object]], heater_present: bool
) -> Tuple[SensorObservation, ...]:
    observations: List[SensorObservation] = []
    previous_timestamp: Optional[float] = None
    for row_index, row in enumerate(rows):
        try:
            ts = float(row[CSV_TIMESTAMP_COLUMN])
            ti = float(row[CSV_INDOOR_TEMPERATURE_COLUMN])
            rhi = float(row[CSV_INDOOR_RH_COLUMN])
            to = float(row[CSV_OUTDOOR_TEMPERATURE_COLUMN])
            rho = float(row[CSV_OUTDOOR_RH_COLUMN])
        except (KeyError, TypeError, ValueError) as cause:
            raise ValueError(
                f"row {row_index}: could not convert every numeric field "
                f"({cause})."
            ) from cause
        window_open = _parse_bool(
            row.get(CSV_WINDOW_OPEN_COLUMN),
            CSV_WINDOW_OPEN_COLUMN,
            row_index,
        )
        heater_on: Optional[bool] = None
        if heater_present:
            raw = row.get(CSV_HEATER_ON_COLUMN)
            if raw is not None and str(raw).strip() != "":
                heater_on = _parse_bool(
                    raw, CSV_HEATER_ON_COLUMN, row_index
                )
        if previous_timestamp is not None and ts < previous_timestamp:
            raise ValueError(
                f"row {row_index}: timestamp_hours={ts} is earlier than the "
                f"previous timestamp {previous_timestamp}; observations "
                "must be time-sorted."
            )
        previous_timestamp = ts
        observations.append(
            SensorObservation(
                timestamp_hours=ts,
                indoor_temperature_c=ti,
                indoor_relative_humidity_pct=rhi,
                outdoor_temperature_c=to,
                outdoor_relative_humidity_pct=rho,
                window_open=window_open,
                heater_on=heater_on,
            )
        )
    return tuple(observations)


@dataclass(frozen=True)
class VentilationEventWindow:
    """A contiguous window-open segment as a (start_index, end_index+1) slice.

    ``end_index_exclusive`` is one past the last window-open
    observation so ``observations[start_index:end_index_exclusive]``
    is the event's sample list.
    """

    start_index: int
    end_index_exclusive: int

    @property
    def sample_count(self) -> int:
        return self.end_index_exclusive - self.start_index


def identify_ventilation_events(
    observations: Sequence[SensorObservation],
    minimum_samples: int = 3,
) -> Tuple[VentilationEventWindow, ...]:
    """Return every contiguous run of window-open observations.

    Args:
        observations: sensor series.
        minimum_samples: shortest event length (in samples) to
            report. Very short runs are often sensor bounce and not
            useful for validation. Defaults to 3.

    Returns:
        Tuple of ``VentilationEventWindow`` in time order.
    """
    if minimum_samples < 2:
        raise ValueError(
            f"minimum_samples must be >= 2, got {minimum_samples}"
        )
    events: List[VentilationEventWindow] = []
    start: Optional[int] = None
    for i, obs in enumerate(observations):
        if obs.window_open:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= minimum_samples:
                    events.append(
                        VentilationEventWindow(
                            start_index=start, end_index_exclusive=i
                        )
                    )
                start = None
    if start is not None:
        length = len(observations) - start
        if length >= minimum_samples:
            events.append(
                VentilationEventWindow(
                    start_index=start,
                    end_index_exclusive=len(observations),
                )
            )
    return tuple(events)


@dataclass(frozen=True)
class EventValidationResult:
    """Validation output for one window-open event.

    Fields:
        event: the event's index window in the source observations.
        start_time_hours / end_time_hours: bounds of the event.
        n_samples: number of samples used.
        times_hours: per-sample timestamps.
        observed_indoor_absolute_humidity_g_m3: measured AH per
            sample.
        predicted_indoor_absolute_humidity_g_m3: forward-simulated
            AH per sample, using the caller-supplied ACH.
        observed_indoor_temperature_c: measured T per sample.
        predicted_indoor_temperature_c: forward-simulated T per
            sample, using ACH plus C_eff, or ``None`` when the
            caller did not supply thermal parameters (in which case
            the T MAE is also ``None``).
        observed_indoor_relative_humidity_pct: measured RH.
        predicted_indoor_relative_humidity_pct: RH computed from
            the predicted (T, AH). ``None`` in tandem with the
            temperature prediction when thermal parameters are
            absent.
        mae_absolute_humidity_g_m3: mean absolute error of the
            AH prediction across the event.
        mae_temperature_c: MAE of the T prediction, or ``None``.
        mae_relative_humidity_pct: MAE of the RH prediction, or
            ``None``.
        ach_used, effective_thermal_capacity_j_per_k, room_volume_m3:
            echoed for audit.
    """

    event: VentilationEventWindow
    start_time_hours: float
    end_time_hours: float
    n_samples: int
    times_hours: Tuple[float, ...]
    observed_indoor_absolute_humidity_g_m3: Tuple[float, ...]
    predicted_indoor_absolute_humidity_g_m3: Tuple[float, ...]
    observed_indoor_temperature_c: Tuple[float, ...]
    predicted_indoor_temperature_c: Optional[Tuple[float, ...]]
    observed_indoor_relative_humidity_pct: Tuple[float, ...]
    predicted_indoor_relative_humidity_pct: Optional[Tuple[float, ...]]
    mae_absolute_humidity_g_m3: float
    mae_temperature_c: Optional[float]
    mae_relative_humidity_pct: Optional[float]
    ach_used: float
    effective_thermal_capacity_j_per_k: Optional[float]
    room_volume_m3: float


def _mean_absolute_error(
    predicted: Sequence[float], observed: Sequence[float]
) -> float:
    return sum(
        abs(p - o) for p, o in zip(predicted, observed)
    ) / len(predicted)


def validate_events(
    observations: Sequence[SensorObservation],
    events: Sequence[VentilationEventWindow],
    ach: float,
    room_volume_m3: float,
    effective_thermal_capacity_j_per_k: Optional[float] = None,
) -> Tuple[EventValidationResult, ...]:
    """Predict indoor AH (and optionally T / RH) for each event; compute MAE.

    For each event, forward-simulates the moisture ODE from the
    first observation's AH using the caller-supplied ``ach`` and
    the observed outdoor AH per interval. When
    ``effective_thermal_capacity_j_per_k`` is provided, the same
    per-step loop also forward-simulates the temperature ODE and
    reconstructs RH via
    ``psychrometrics.relative_humidity_from_absolute_humidity``.

    Args:
        observations: full sensor series.
        events: window-open events to validate against. Every event
            must have at least two samples.
        ach: air-change rate in hours^-1 to use for the forward
            simulation. Non-negative and finite. Callers who
            calibrated on a SEPARATE dataset pass the fitted ACH.
        room_volume_m3: room volume for the thermal H_vent term.
            Strictly positive.
        effective_thermal_capacity_j_per_k: caller's C_eff. When
            ``None`` the thermal forward simulation is skipped and
            the T / RH fields are also ``None`` on every result.

    Returns:
        Tuple of ``EventValidationResult`` in the same order as
        ``events``.

    Raises:
        ValueError: on invalid ACH / volume / C_eff, on empty
            events, on events shorter than two samples.
    """
    if not _isfinite(ach):
        raise ValueError(f"ach must be finite, got {ach!r}")
    if ach < 0.0:
        raise ValueError(f"ach must be non-negative, got {ach}")
    if not _isfinite(room_volume_m3):
        raise ValueError(
            f"room_volume_m3 must be finite, got {room_volume_m3!r}"
        )
    if room_volume_m3 <= 0.0:
        raise ValueError(
            f"room_volume_m3 must be strictly positive, got {room_volume_m3}"
        )
    if effective_thermal_capacity_j_per_k is not None:
        if not _isfinite(effective_thermal_capacity_j_per_k):
            raise ValueError(
                "effective_thermal_capacity_j_per_k must be finite when set, "
                f"got {effective_thermal_capacity_j_per_k!r}"
            )
        if effective_thermal_capacity_j_per_k <= 0.0:
            raise ValueError(
                "effective_thermal_capacity_j_per_k must be strictly "
                f"positive when set, got {effective_thermal_capacity_j_per_k}"
            )

    results: List[EventValidationResult] = []
    for event in events:
        segment = observations[event.start_index : event.end_index_exclusive]
        if len(segment) < 2:
            raise ValueError(
                f"event {event} has only {len(segment)} samples; "
                "need at least 2."
            )

        times = tuple(obs.timestamp_hours for obs in segment)
        observed_ah = tuple(
            obs.indoor_absolute_humidity_g_m3 for obs in segment
        )
        observed_t = tuple(obs.indoor_temperature_c for obs in segment)
        observed_rh = tuple(
            obs.indoor_relative_humidity_pct for obs in segment
        )

        # AH forward simulation.
        predicted_ah = [observed_ah[0]]
        for i in range(len(segment) - 1):
            dt_hours = segment[i + 1].timestamp_hours - segment[i].timestamp_hours
            if dt_hours <= 0.0:
                raise ValueError(
                    f"event {event}: non-monotone timestamps within event "
                    f"at internal index {i}."
                )
            next_ah = predict_final_absolute_humidity(
                indoor_ah_g_m3=predicted_ah[-1],
                outdoor_ah_g_m3=segment[i].outdoor_absolute_humidity_g_m3,
                ach=ach,
                duration_minutes=dt_hours * 60.0,
            )
            predicted_ah.append(next_ah)

        predicted_t: Optional[List[float]] = None
        predicted_rh: Optional[List[float]] = None
        if effective_thermal_capacity_j_per_k is not None:
            predicted_t = [observed_t[0]]
            predicted_rh = [observed_rh[0]]
            for i in range(len(segment) - 1):
                dt_hours = (
                    segment[i + 1].timestamp_hours
                    - segment[i].timestamp_hours
                )
                next_t = predict_indoor_temperature(
                    initial_indoor_temperature_c=predicted_t[-1],
                    outdoor_temperature_c=segment[i].outdoor_temperature_c,
                    room_volume_m3=room_volume_m3,
                    ach=ach,
                    effective_thermal_capacity_j_per_k=(
                        effective_thermal_capacity_j_per_k
                    ),
                    duration_minutes=dt_hours * 60.0,
                )
                predicted_t.append(next_t)
                # RH from predicted (T, AH) via the psychrometric inverse.
                # A supersaturated forward-simulated point can exceed 100
                # RH; we report the raw arithmetic value.
                predicted_rh.append(
                    relative_humidity_from_absolute_humidity(
                        temperature_c=next_t,
                        absolute_humidity_g_m3=predicted_ah[i + 1],
                    )
                )

        mae_ah = _mean_absolute_error(predicted_ah, observed_ah)
        mae_t = (
            _mean_absolute_error(predicted_t, observed_t)
            if predicted_t is not None
            else None
        )
        mae_rh = (
            _mean_absolute_error(predicted_rh, observed_rh)
            if predicted_rh is not None
            else None
        )
        results.append(
            EventValidationResult(
                event=event,
                start_time_hours=times[0],
                end_time_hours=times[-1],
                n_samples=len(segment),
                times_hours=times,
                observed_indoor_absolute_humidity_g_m3=observed_ah,
                predicted_indoor_absolute_humidity_g_m3=tuple(predicted_ah),
                observed_indoor_temperature_c=observed_t,
                predicted_indoor_temperature_c=(
                    tuple(predicted_t) if predicted_t is not None else None
                ),
                observed_indoor_relative_humidity_pct=observed_rh,
                predicted_indoor_relative_humidity_pct=(
                    tuple(predicted_rh) if predicted_rh is not None else None
                ),
                mae_absolute_humidity_g_m3=mae_ah,
                mae_temperature_c=mae_t,
                mae_relative_humidity_pct=mae_rh,
                ach_used=ach,
                effective_thermal_capacity_j_per_k=(
                    effective_thermal_capacity_j_per_k
                ),
                room_volume_m3=room_volume_m3,
            )
        )
    return tuple(results)
