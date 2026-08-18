"""Estimate effective ACH from a measured ventilation event.

Every downstream model in this repo receives ACH as a caller-set
input (see the module comment in ``thermal.py``: a room's
effective ACH depends on wind, stack effect, opening geometry, and
the actual gap when the window is on the latch, so there is no
universal default). This module lets a caller REPLACE that guess
with a value fitted to real measurements.

Ingestion contract:
    A time series of observations captured during a window-open
    event, each a ``CalibrationObservation`` with:
        - timestamp_hours (monotone, non-negative)
        - indoor_absolute_humidity_g_m3 (>= 0)
        - outdoor_absolute_humidity_g_m3 (>= 0)
        - window_open (bool)
    Timestamps are relative hours - the same time unit
    ``time_simulation.RoomTrajectory`` uses. Only consecutive
    observations with BOTH endpoints having ``window_open = True``
    contribute residuals; the calibration is deliberately scoped to
    the window-open segment.

Model:
    Between two consecutive window-open samples with a piecewise-
    constant outdoor AH over the interval, the well-mixed ODE
    ``dC/dt = ach * (C_out - C)`` integrates to
        C(t + dt) = C_out(t) + (C(t) - C_out(t)) * exp(-ach * dt)
    (Same expression the physics simulator uses; the calibration
    module does not redefine it - it consumes it.)

Objective:
    Sum-of-squared prediction errors. For each consecutive window-
    open pair the residual is
        r_i = C_observed(t_{i+1})
              - (C_out(t_i) + (C_observed(t_i) - C_out(t_i)) * exp(-ach * dt_i))
    minimised over a single scalar ``ach``. Solved by golden-section
    search on a bounded interval - transparent, dependency-free, no
    machine-learning machinery. Bounds are caller-configurable;
    defaults span the residential range from a very tightly sealed
    room (0.05 h^-1) to a fully open large opening on a windy day
    (50 h^-1).

Returned trajectory:
    In addition to the point estimate, the result carries the
    OBSERVED indoor-AH series over the window-open segment and the
    PREDICTED indoor-AH series obtained by SIMULATING FORWARD from
    the first observation with the fitted ACH (rather than the
    one-step-ahead predictions the objective uses). Callers who
    want a visual overlay of "how well did the fit reproduce the
    event?" plot these two arrays.

Assumptions and caveats:
    - Moisture generation during the event is neglected. Real
      residential ventilation events include modest background
      generation (occupants, plants, evaporation); when the
      ventilation term dominates the generation term (i.e. window
      open, cool weather), the estimate is close to the physical
      ACH. When it does not, the fit absorbs the generation into
      the ACH estimate as a systematic bias.
    - Outdoor AH is treated as piecewise-constant across each
      interval, evaluated at the START of the interval. Real
      outdoor AH drifts; over short (~10-30 min) events the drift
      is small.
    - Well-mixed room. Same well-mixed assumption every other
      layer already carries.
    - No handling of measurement noise beyond the least-squares
      objective. A future slice can extend to weighted least
      squares or robust regression if callers report
      heteroscedastic or outlier-prone measurements.

Explicitly NOT in this module:
    - Any machine-learning fitter.
    - Fitting ach_window_open AND ach_closed jointly (a caller who
      wants that runs this on two different events).
    - Building-fabric physics (U-values, air-leakage tests).
"""

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Sequence, Tuple


DEFAULT_ACH_SEARCH_MIN: float = 0.05
"""Lower bound on the ACH search interval, in hours^-1.

Below 0.05 the room is effectively sealed (blower-door "tight"
range) and a window-open event should not produce values there.
Callers with very tight buildings can override.
"""

DEFAULT_ACH_SEARCH_MAX: float = 50.0
"""Upper bound on the ACH search interval, in hours^-1.

A wide-open opening on a windy day rarely exceeds ~30. 50 leaves
headroom without opening the search to physically unreasonable
values.
"""

DEFAULT_GOLDEN_SECTION_TOLERANCE: float = 1e-5
"""Bracket-width tolerance at which the golden-section search stops.

Empirical: 1e-5 hours^-1 is well below the resolution any real
measurement can support and terminates in ~30 iterations across
the default bracket.
"""

MAX_GOLDEN_SECTION_ITERATIONS: int = 200
"""Hard cap on golden-section iterations.

A protective backstop against pathological objective surfaces;
the default tolerance normally converges well within this cap.
"""


@dataclass(frozen=True)
class CalibrationObservation:
    """One measured point during a candidate ventilation event.

    Fields:
        timestamp_hours: measurement time in hours since the
            observation series origin. Non-negative and finite.
        indoor_absolute_humidity_g_m3: measured indoor AH.
        outdoor_absolute_humidity_g_m3: measured outdoor AH at the
            same instant.
        window_open: True if the window is open at this observation.

    Validation:
        Every numeric field must be finite and non-negative. The
        module docstring names the ingestion contract.
    """

    timestamp_hours: float
    indoor_absolute_humidity_g_m3: float
    outdoor_absolute_humidity_g_m3: float
    window_open: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("timestamp_hours", self.timestamp_hours),
            (
                "indoor_absolute_humidity_g_m3",
                self.indoor_absolute_humidity_g_m3,
            ),
            (
                "outdoor_absolute_humidity_g_m3",
                self.outdoor_absolute_humidity_g_m3,
            ),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class CalibrationResult:
    """Fitted ACH plus fit diagnostics and reconstructed trajectory.

    Fields:
        estimated_ach: fitted air-change rate in hours^-1. This is
            the constant ACH that MINIMISES the sum-of-squared
            prediction errors across every consecutive window-open
            pair.
        rms_error_g_m3: root-mean-square residual of the one-step
            prediction objective, in g/m^3. Callers can compare
            this against the noise level of their measurement to
            judge whether the fit is trustworthy.
        n_observations: number of window-open observations
            (points, not intervals) the fit used. The number of
            residuals is one less.
        times_hours: window-open segment timestamps used by the
            fit, in hours.
        observed_indoor_ah_g_m3: observed indoor AH at each of the
            window-open timestamps.
        predicted_indoor_ah_g_m3: indoor AH obtained by SIMULATING
            FORWARD from the first window-open observation, using
            ``estimated_ach`` and the observed outdoor AH per
            interval. Predicted[0] equals observed[0]; predicted[i]
            for i >= 1 is model output rather than a measurement.
        ach_search_lower_bound / ach_search_upper_bound: the bounds
            the golden-section search used. Reported so a caller
            can spot cases where the estimate saturated at a
            bracket edge (typically a sign the true ACH lies
            outside the search interval).

    Sign / bookkeeping notes:
        RMS error is non-negative by construction. Number of
        observations is >= 2 whenever the fit was performed (a
        single-point series cannot form a residual). The
        ``observed_indoor_ah_g_m3`` and ``predicted_indoor_ah_g_m3``
        arrays share the same length and time axis as
        ``times_hours``.
    """

    estimated_ach: float
    rms_error_g_m3: float
    n_observations: int
    times_hours: Tuple[float, ...]
    observed_indoor_ah_g_m3: Tuple[float, ...]
    predicted_indoor_ah_g_m3: Tuple[float, ...]
    ach_search_lower_bound: float
    ach_search_upper_bound: float


def _window_open_run(
    observations: Sequence[CalibrationObservation],
) -> Sequence[CalibrationObservation]:
    """Extract the longest CONTIGUOUS run of window-open observations.

    Real measurement sessions often contain closed periods before
    and after the event of interest; the fit is scoped to the
    window-open segment. If several disjoint runs are present the
    LONGEST is returned; a future slice can add explicit selection.
    """
    best_run: list = []
    current: list = []
    for obs in observations:
        if obs.window_open:
            current.append(obs)
        else:
            if len(current) > len(best_run):
                best_run = current
            current = []
    if len(current) > len(best_run):
        best_run = current
    return best_run


def _predict_next_indoor_ah(
    current_indoor_ah_g_m3: float,
    outdoor_ah_g_m3: float,
    ach: float,
    dt_hours: float,
) -> float:
    """One step of the well-mixed ventilation ODE.

    Same integrator ``moisture.predict_final_absolute_humidity``
    uses; kept inline here to avoid coupling calibration to the
    caller-facing wrapper's argument list.
    """
    return outdoor_ah_g_m3 + (
        current_indoor_ah_g_m3 - outdoor_ah_g_m3
    ) * exp(-ach * dt_hours)


def _sum_squared_prediction_error(
    run: Sequence[CalibrationObservation], ach: float
) -> float:
    """Sum over consecutive pairs of (observed_next - predicted_next)^2."""
    total = 0.0
    for i in range(len(run) - 1):
        current = run[i]
        nxt = run[i + 1]
        dt_hours = nxt.timestamp_hours - current.timestamp_hours
        if dt_hours <= 0.0:
            raise ValueError(
                "observation timestamps must be strictly increasing; "
                f"got dt = {dt_hours} between index {i} and {i + 1}."
            )
        predicted = _predict_next_indoor_ah(
            current_indoor_ah_g_m3=current.indoor_absolute_humidity_g_m3,
            outdoor_ah_g_m3=current.outdoor_absolute_humidity_g_m3,
            ach=ach,
            dt_hours=dt_hours,
        )
        residual = nxt.indoor_absolute_humidity_g_m3 - predicted
        total += residual * residual
    return total


def _golden_section_minimise(
    objective,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Minimise a unimodal 1-D objective on [lower_bound, upper_bound].

    Standard golden-section search: no derivatives, transparent
    step rule, guaranteed convergence for a unimodal objective on
    a bounded interval. The prediction-error surface for a first-
    order-decay model in one parameter is unimodal in ACH under
    the assumptions above.
    """
    golden_ratio = (sqrt(5.0) - 1.0) / 2.0  # ~0.618
    a, b = lower_bound, upper_bound
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    fc = objective(c)
    fd = objective(d)
    for _ in range(max_iterations):
        if abs(b - a) < tolerance:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - golden_ratio * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden_ratio * (b - a)
            fd = objective(d)
    return (a + b) / 2.0


def estimate_ach_from_observations(
    observations: Sequence[CalibrationObservation],
    ach_search_lower_bound: float = DEFAULT_ACH_SEARCH_MIN,
    ach_search_upper_bound: float = DEFAULT_ACH_SEARCH_MAX,
    tolerance: float = DEFAULT_GOLDEN_SECTION_TOLERANCE,
    max_iterations: int = MAX_GOLDEN_SECTION_ITERATIONS,
) -> CalibrationResult:
    """Fit a single constant ACH to a measured window-open segment.

    Solves conceptually:

        minimise sum over consecutive window-open pairs
                 (C_observed(t_{i+1})
                  - [C_out(t_i)
                     + (C_observed(t_i) - C_out(t_i)) * exp(-ach * dt_i)])^2

    for ``ach`` on [``ach_search_lower_bound``,
    ``ach_search_upper_bound``] via golden-section search.

    Args:
        observations: iterable of ``CalibrationObservation`` in
            monotone time order. Must contain at least two
            observations flagged ``window_open = True``, or the
            call raises ``ValueError``.
        ach_search_lower_bound / ach_search_upper_bound: search
            bracket in hours^-1. Defaults span the residential
            range; callers with an unusual room can widen or
            narrow them. Must satisfy
            ``lower_bound < upper_bound`` and both non-negative.
        tolerance: golden-section bracket-width tolerance
            (hours^-1). Defaults to ``1e-5``.
        max_iterations: hard cap on golden-section iterations.

    Returns:
        A ``CalibrationResult`` with the fitted ACH, RMS
        prediction error, the number of window-open observations,
        and both the observed and forward-simulated indoor-AH
        arrays over the segment. A caller inspecting the fit plots
        these two arrays.

    Raises:
        ValueError: on missing / short window-open segments, on
            non-monotone timestamps, on invalid bracket bounds.
    """
    if not isfinite(ach_search_lower_bound):
        raise ValueError(
            f"ach_search_lower_bound must be finite, got {ach_search_lower_bound!r}"
        )
    if not isfinite(ach_search_upper_bound):
        raise ValueError(
            f"ach_search_upper_bound must be finite, got {ach_search_upper_bound!r}"
        )
    if ach_search_lower_bound < 0.0:
        raise ValueError(
            "ach_search_lower_bound must be non-negative, got "
            f"{ach_search_lower_bound}"
        )
    if ach_search_upper_bound <= ach_search_lower_bound:
        raise ValueError(
            "ach_search_upper_bound must be strictly greater than "
            "ach_search_lower_bound, got "
            f"({ach_search_lower_bound}, {ach_search_upper_bound})"
        )

    run = _window_open_run(observations)
    if len(run) < 2:
        raise ValueError(
            "at least two consecutive window-open observations are required "
            f"to fit an ACH; got {len(run)}."
        )
    # Timestamp monotonicity is checked inside
    # _sum_squared_prediction_error (per pair) - a clearer error
    # location than a separate pre-scan.

    def objective(ach: float) -> float:
        return _sum_squared_prediction_error(run, ach)

    estimated_ach = _golden_section_minimise(
        objective=objective,
        lower_bound=ach_search_lower_bound,
        upper_bound=ach_search_upper_bound,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    n_pairs = len(run) - 1
    rms_error = sqrt(_sum_squared_prediction_error(run, estimated_ach) / n_pairs)

    # Forward-simulate the entire segment from the first observation
    # using the fitted ACH; this gives a whole-trajectory reconstruction
    # that a caller can overlay on the measurement for a visual check.
    times = tuple(obs.timestamp_hours for obs in run)
    observed = tuple(obs.indoor_absolute_humidity_g_m3 for obs in run)
    predicted_list = [observed[0]]
    for i in range(len(run) - 1):
        current_predicted = predicted_list[-1]
        outdoor_ah = run[i].outdoor_absolute_humidity_g_m3
        dt_hours = run[i + 1].timestamp_hours - run[i].timestamp_hours
        predicted_list.append(
            _predict_next_indoor_ah(
                current_indoor_ah_g_m3=current_predicted,
                outdoor_ah_g_m3=outdoor_ah,
                ach=estimated_ach,
                dt_hours=dt_hours,
            )
        )
    predicted = tuple(predicted_list)

    return CalibrationResult(
        estimated_ach=estimated_ach,
        rms_error_g_m3=rms_error,
        n_observations=len(run),
        times_hours=times,
        observed_indoor_ah_g_m3=observed,
        predicted_indoor_ah_g_m3=predicted,
        ach_search_lower_bound=ach_search_lower_bound,
        ach_search_upper_bound=ach_search_upper_bound,
    )
