"""Estimate effective thermal capacitance from a measured ventilation event.

Companion to ``calibration`` (which fits ACH). The thermal
capacitance ``C_eff`` is the other caller-set thermal knob the POC
currently requires as an input; this module lets a caller REPLACE
that guess with a value fitted to real measurements from the same
window-open event they used for the ACH fit.

Ingestion contract:
    A time series of observations captured during a window-open
    event, each a ``ThermalObservation`` with:
        - timestamp_hours (monotone, non-negative)
        - indoor_temperature_c
        - outdoor_temperature_c
        - window_open (bool)
    Timestamps are relative hours. The caller ALSO supplies
    ``ach`` (the effective air-change rate during the event) and
    ``room_volume_m3``; both should already be known - typically
    the ACH comes from ``calibration.estimate_ach_from_observations``
    run on the same event's humidity trace, and the volume from a
    room measurement.

Model:
    The single-zone ventilation-only thermal ODE lives in
    ``thermal.predict_indoor_temperature``:
        T(t + dt) = T_out(t) + (T(t) - T_out(t))
                    * exp(-H_vent / C_eff * dt_seconds)
    where H_vent (W/K) comes from
    ``thermal.ventilation_heat_loss_coefficient`` given room
    volume and ACH. This module CONSUMES those two helpers; no
    thermodynamic equation is reimplemented here.

Objective:
    Sum-of-squared prediction errors on the indoor temperature.
    For each consecutive window-open pair with dt = t_{i+1} - t_i,
        r_i = T_observed(t_{i+1})
              - [T_out(t_i)
                 + (T_observed(t_i) - T_out(t_i))
                   * exp(-H_vent / C_eff * dt_seconds)]
    minimised over the scalar ``C_eff`` on a caller-configurable
    bracket via golden-section search. Same transparent, no-ML,
    no-external-solver strategy as the ACH fitter.

Returned trajectory:
    Alongside the point estimate the result carries the observed
    indoor-T series over the window-open segment and a forward-
    simulated series produced from the first observation using
    the fitted ``C_eff``. Callers who want a visual overlay of
    "how well did the fit reproduce the temperature drop?" plot
    these two arrays.

Assumptions and caveats (inherited from ``thermal.py``):
    - Ventilation is the ONLY heat-transfer mechanism during the
      fit interval. No conduction through walls, no radiation, no
      active heating, no occupant / equipment / solar gains. The
      fit therefore assumes the CALLER ARRANGED a controlled
      ventilation event (no heater running, no occupants doing
      thermally significant things during the window-open period).
      Real deployments must satisfy this before trusting the
      estimate.
    - ACH is treated as constant across the event. If the real ACH
      drifted (wind gusts, door state changing) the fit absorbs the
      variability into C_eff.
    - Outdoor T is piecewise-constant across each interval,
      evaluated at the START of the interval. Fine for the short
      events the POC targets.
    - Well-mixed room. Real rooms stratify; the well-mixed
      assumption is inherited from ``thermal.py``.
    - Constant air density and c_p (the constants are on
      ``thermal.py``); the fit does not compensate for their
      temperature dependence.

Explicitly NOT in this module:
    - Any neural-network / gradient-based fitter.
    - Fitting C_eff AND ACH jointly. Callers who want that run the
      ACH fitter first, then this one; that avoids fitting two
      degenerate parameters against a single decay time constant.
    - Building-fabric parameter identification (U-values, area,
      surface areas).
"""

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Sequence, Tuple

from thermal import ventilation_heat_loss_coefficient


DEFAULT_C_EFF_SEARCH_MIN_J_PER_K: float = 10_000.0
"""Lower bound on the C_eff search interval, in J/K.

Below 10 kJ/K the room is thermally smaller than a bathtub of
water. Air-only capacity of a modest room is around 50 kJ/K, and
real furnished rooms are substantially higher; a lower value than
10 kJ/K almost certainly reflects a bad event or non-ventilation
losses that dwarf the ventilation term.
"""

DEFAULT_C_EFF_SEARCH_MAX_J_PER_K: float = 50_000_000.0
"""Upper bound on the C_eff search interval, in J/K.

50 MJ/K is a large heavy building; residential rooms are almost
always well below this. Leaves plenty of headroom on the upper
side of the residential range.
"""

DEFAULT_GOLDEN_SECTION_TOLERANCE_J_PER_K: float = 1.0
"""Bracket-width tolerance at which the golden-section search stops.

1 J/K is well below the resolution any real measurement can
support and terminates well within the iteration cap.
"""

MAX_GOLDEN_SECTION_ITERATIONS: int = 200
"""Hard cap on golden-section iterations."""


@dataclass(frozen=True)
class ThermalObservation:
    """One measured point during a controlled ventilation event.

    Fields:
        timestamp_hours: measurement time in hours since the
            observation series origin. Non-negative and finite.
        indoor_temperature_c: measured indoor air temperature at
            this instant.
        outdoor_temperature_c: outdoor temperature at this instant.
        window_open: True if the window is open at this observation.

    Validation:
        Every numeric field must be finite. Temperatures may be
        negative (winter measurements) but not NaN or infinite.
    """

    timestamp_hours: float
    indoor_temperature_c: float
    outdoor_temperature_c: float
    window_open: bool

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_hours):
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
            ("outdoor_temperature_c", self.outdoor_temperature_c),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class ThermalCalibrationResult:
    """Fitted C_eff plus fit diagnostics and reconstructed trajectory.

    Fields:
        estimated_effective_thermal_capacity_j_per_k: fitted lumped
            effective heat capacity in J/K.
        rms_error_c: root-mean-square residual of the one-step
            prediction objective, in degrees Celsius. Compare
            against the sensor's noise level to judge fit quality.
        n_observations: number of window-open observations used;
            the number of residuals is one less.
        times_hours: window-open segment timestamps used by the
            fit.
        observed_indoor_temperature_c: observed indoor T at each
            of the window-open timestamps.
        predicted_indoor_temperature_c: indoor T obtained by
            SIMULATING FORWARD from the first window-open
            observation using ``estimated_effective_thermal_capacity_j_per_k``,
            ``ach``, ``room_volume_m3``, and the observed outdoor
            T per interval. Predicted[0] equals observed[0];
            subsequent values are model output.
        ach: the ACH used by the fit (echoed for audit).
        room_volume_m3: the room volume used by the fit (echoed).
        c_eff_search_lower_bound_j_per_k / c_eff_search_upper_bound_j_per_k:
            the bounds the golden-section search used. A caller
            can spot a saturated-at-edge estimate.
    """

    estimated_effective_thermal_capacity_j_per_k: float
    rms_error_c: float
    n_observations: int
    times_hours: Tuple[float, ...]
    observed_indoor_temperature_c: Tuple[float, ...]
    predicted_indoor_temperature_c: Tuple[float, ...]
    ach: float
    room_volume_m3: float
    c_eff_search_lower_bound_j_per_k: float
    c_eff_search_upper_bound_j_per_k: float


def _window_open_run(
    observations: Sequence[ThermalObservation],
) -> Sequence[ThermalObservation]:
    """Return the longest contiguous run of window-open observations."""
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


def _predict_next_indoor_temperature_c(
    current_indoor_temperature_c: float,
    outdoor_temperature_c: float,
    h_vent_w_per_k: float,
    effective_thermal_capacity_j_per_k: float,
    dt_seconds: float,
) -> float:
    """One step of the well-mixed ventilation thermal ODE.

    Same closed-form as ``thermal.predict_indoor_temperature`` (kept
    inline here so the fit's inner loop does not repeatedly re-
    derive H_vent for the same fixed ACH and room). H_vent itself
    comes from ``thermal.ventilation_heat_loss_coefficient`` -
    the fit consumes the validated helper.
    """
    decay_factor = exp(
        -h_vent_w_per_k / effective_thermal_capacity_j_per_k * dt_seconds
    )
    return outdoor_temperature_c + (
        current_indoor_temperature_c - outdoor_temperature_c
    ) * decay_factor


def _sum_squared_prediction_error(
    run: Sequence[ThermalObservation],
    h_vent_w_per_k: float,
    effective_thermal_capacity_j_per_k: float,
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
        dt_seconds = dt_hours * 3600.0
        predicted = _predict_next_indoor_temperature_c(
            current_indoor_temperature_c=current.indoor_temperature_c,
            outdoor_temperature_c=current.outdoor_temperature_c,
            h_vent_w_per_k=h_vent_w_per_k,
            effective_thermal_capacity_j_per_k=(
                effective_thermal_capacity_j_per_k
            ),
            dt_seconds=dt_seconds,
        )
        residual = nxt.indoor_temperature_c - predicted
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

    Same golden-section routine as ``calibration._golden_section_minimise``.
    Kept inline here so this module stays self-contained; the
    routine is a dozen lines and has no owner-of-truth in the repo.
    """
    golden_ratio = (sqrt(5.0) - 1.0) / 2.0
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


def estimate_effective_thermal_capacity_from_observations(
    observations: Sequence[ThermalObservation],
    ach: float,
    room_volume_m3: float,
    c_eff_search_lower_bound_j_per_k: float = DEFAULT_C_EFF_SEARCH_MIN_J_PER_K,
    c_eff_search_upper_bound_j_per_k: float = DEFAULT_C_EFF_SEARCH_MAX_J_PER_K,
    tolerance: float = DEFAULT_GOLDEN_SECTION_TOLERANCE_J_PER_K,
    max_iterations: int = MAX_GOLDEN_SECTION_ITERATIONS,
) -> ThermalCalibrationResult:
    """Fit a single constant C_eff to a measured window-open temperature trace.

    Solves conceptually:

        minimise sum over consecutive window-open pairs
                 (T_observed(t_{i+1})
                  - [T_out(t_i)
                     + (T_observed(t_i) - T_out(t_i))
                       * exp(-H_vent / C_eff * dt_seconds)])^2

    for ``C_eff`` on
    [``c_eff_search_lower_bound_j_per_k``,
    ``c_eff_search_upper_bound_j_per_k``]
    via golden-section search. H_vent is computed once from ``ach``
    and ``room_volume_m3`` via
    ``thermal.ventilation_heat_loss_coefficient`` (no physics
    duplication).

    Args:
        observations: iterable of ``ThermalObservation`` in monotone
            time order. Must contain at least two window-open
            observations, or the call raises ``ValueError``.
        ach: air-change rate during the event, in hours^-1. Must be
            strictly positive (an ACH of zero produces H_vent = 0
            and makes C_eff unidentifiable from the temperature
            trace). Typically the value returned by
            ``calibration.estimate_ach_from_observations`` on the
            same event's humidity trace.
        room_volume_m3: room volume in cubic metres. Strictly
            positive.
        c_eff_search_lower_bound_j_per_k / c_eff_search_upper_bound_j_per_k:
            search bracket in J/K. Must satisfy
            ``lower_bound < upper_bound`` and both strictly
            positive.
        tolerance: golden-section bracket-width tolerance, in J/K.
            Defaults to 1 J/K.
        max_iterations: hard cap on golden-section iterations.

    Returns:
        A ``ThermalCalibrationResult`` carrying the fitted
        capacity, RMS prediction error, the number of window-open
        observations used, both observed and forward-simulated
        indoor-T arrays over the segment, the ACH / volume echoed
        for audit, and the bracket that was searched.

    Raises:
        ValueError: on missing / short window-open segments, on
            non-monotone timestamps, on invalid ACH / volume /
            bracket, on non-finite fields.
    """
    if not isfinite(ach):
        raise ValueError(f"ach must be finite, got {ach!r}")
    if ach <= 0.0:
        raise ValueError(
            "ach must be strictly positive so H_vent > 0 and C_eff is "
            f"identifiable, got {ach}"
        )
    if not isfinite(room_volume_m3):
        raise ValueError(
            f"room_volume_m3 must be finite, got {room_volume_m3!r}"
        )
    if room_volume_m3 <= 0.0:
        raise ValueError(
            f"room_volume_m3 must be strictly positive, got {room_volume_m3}"
        )
    if not isfinite(c_eff_search_lower_bound_j_per_k):
        raise ValueError(
            "c_eff_search_lower_bound_j_per_k must be finite, got "
            f"{c_eff_search_lower_bound_j_per_k!r}"
        )
    if not isfinite(c_eff_search_upper_bound_j_per_k):
        raise ValueError(
            "c_eff_search_upper_bound_j_per_k must be finite, got "
            f"{c_eff_search_upper_bound_j_per_k!r}"
        )
    if c_eff_search_lower_bound_j_per_k <= 0.0:
        raise ValueError(
            "c_eff_search_lower_bound_j_per_k must be strictly positive, "
            f"got {c_eff_search_lower_bound_j_per_k}"
        )
    if (
        c_eff_search_upper_bound_j_per_k
        <= c_eff_search_lower_bound_j_per_k
    ):
        raise ValueError(
            "c_eff_search_upper_bound_j_per_k must be strictly greater "
            "than c_eff_search_lower_bound_j_per_k, got "
            f"({c_eff_search_lower_bound_j_per_k}, "
            f"{c_eff_search_upper_bound_j_per_k})"
        )

    run = _window_open_run(observations)
    if len(run) < 2:
        raise ValueError(
            "at least two consecutive window-open observations are required "
            f"to fit C_eff; got {len(run)}."
        )

    h_vent_w_per_k = ventilation_heat_loss_coefficient(
        room_volume_m3=room_volume_m3, ach=ach
    )

    def objective(c_eff: float) -> float:
        return _sum_squared_prediction_error(
            run=run,
            h_vent_w_per_k=h_vent_w_per_k,
            effective_thermal_capacity_j_per_k=c_eff,
        )

    estimated_c_eff = _golden_section_minimise(
        objective=objective,
        lower_bound=c_eff_search_lower_bound_j_per_k,
        upper_bound=c_eff_search_upper_bound_j_per_k,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    n_pairs = len(run) - 1
    rms_error_c = sqrt(
        _sum_squared_prediction_error(
            run=run,
            h_vent_w_per_k=h_vent_w_per_k,
            effective_thermal_capacity_j_per_k=estimated_c_eff,
        )
        / n_pairs
    )

    # Forward-simulate the whole window-open segment from the first
    # observation. Predicted[0] equals observed[0]; subsequent values
    # are model output at the same timestamps as the observations.
    times = tuple(obs.timestamp_hours for obs in run)
    observed = tuple(obs.indoor_temperature_c for obs in run)
    predicted_list = [observed[0]]
    for i in range(len(run) - 1):
        outdoor_t = run[i].outdoor_temperature_c
        dt_hours = run[i + 1].timestamp_hours - run[i].timestamp_hours
        predicted_list.append(
            _predict_next_indoor_temperature_c(
                current_indoor_temperature_c=predicted_list[-1],
                outdoor_temperature_c=outdoor_t,
                h_vent_w_per_k=h_vent_w_per_k,
                effective_thermal_capacity_j_per_k=estimated_c_eff,
                dt_seconds=dt_hours * 3600.0,
            )
        )
    predicted = tuple(predicted_list)

    return ThermalCalibrationResult(
        estimated_effective_thermal_capacity_j_per_k=estimated_c_eff,
        rms_error_c=rms_error_c,
        n_observations=len(run),
        times_hours=times,
        observed_indoor_temperature_c=observed,
        predicted_indoor_temperature_c=predicted,
        ach=ach,
        room_volume_m3=room_volume_m3,
        c_eff_search_lower_bound_j_per_k=c_eff_search_lower_bound_j_per_k,
        c_eff_search_upper_bound_j_per_k=c_eff_search_upper_bound_j_per_k,
    )
