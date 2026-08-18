"""Tests for the effective-ACH calibration module.

Covers:
    - Synthetic-data recovery: given data generated from the exact
      first-order-decay model at a KNOWN ACH, the calibration
      recovers that ACH to well under one percent.
    - Recovery across a sweep of true ACH values from very tight
      (0.2 h^-1) to wide-open (30 h^-1).
    - Robustness to modest Gaussian noise: recovery still within
      a small fraction of the true ACH.
    - CalibrationObservation validation (finite, non-negative
      fields).
    - Refusal cases: empty input, all-closed-window input, single-
      point window-open segment, non-monotone timestamps, bad
      bracket bounds.
    - Segment extraction: only the window-open run is fitted; a
      leading and trailing closed segment is ignored.
    - Reconstruction: the forward-simulated trajectory matches the
      observed one closely under clean synthetic data.
"""

import sys
from math import exp
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from calibration import (
    CalibrationObservation,
    CalibrationResult,
    DEFAULT_ACH_SEARCH_MAX,
    DEFAULT_ACH_SEARCH_MIN,
    estimate_ach_from_observations,
)


def _synthetic_series(
    true_ach: float,
    initial_indoor_ah: float,
    outdoor_ah: float,
    times_hours: List[float],
    window_open: bool = True,
) -> List[CalibrationObservation]:
    """Build noise-free observations that exactly satisfy the physics model."""
    obs = []
    for t in times_hours:
        indoor = outdoor_ah + (initial_indoor_ah - outdoor_ah) * exp(
            -true_ach * t
        )
        obs.append(
            CalibrationObservation(
                timestamp_hours=t,
                indoor_absolute_humidity_g_m3=indoor,
                outdoor_absolute_humidity_g_m3=outdoor_ah,
                window_open=window_open,
            )
        )
    return obs


def _linspace(start: float, stop: float, count: int) -> List[float]:
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


# --- Basic recovery ------------------------------------------------------


def test_recovers_true_ach_on_clean_synthetic_data() -> None:
    """With noise-free data, the fitted ACH matches the truth tightly."""
    true_ach = 4.0
    obs = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=12.0,
        outdoor_ah=3.0,
        times_hours=_linspace(0.0, 0.5, 21),
    )
    result = estimate_ach_from_observations(obs)
    assert isinstance(result, CalibrationResult)
    assert result.estimated_ach == pytest.approx(true_ach, rel=1e-3)
    assert result.rms_error_g_m3 < 1e-4
    assert result.n_observations == 21


@pytest.mark.parametrize(
    "true_ach", [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)
def test_recovers_across_wide_range_of_true_ach(true_ach: float) -> None:
    obs = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.5, 31),
    )
    result = estimate_ach_from_observations(obs)
    assert result.estimated_ach == pytest.approx(true_ach, rel=5e-3)


def test_rms_error_is_tiny_for_clean_data() -> None:
    obs = _synthetic_series(
        true_ach=3.5,
        initial_indoor_ah=11.0,
        outdoor_ah=3.5,
        times_hours=_linspace(0.0, 0.5, 21),
    )
    result = estimate_ach_from_observations(obs)
    assert result.rms_error_g_m3 < 1e-4


# --- Reconstruction ------------------------------------------------------


def test_predicted_trajectory_matches_observed_on_clean_data() -> None:
    """Forward simulation from the fit reproduces the observed trace."""
    true_ach = 3.0
    obs = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=10.5,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.5, 16),
    )
    result = estimate_ach_from_observations(obs)
    for expected, predicted in zip(
        result.observed_indoor_ah_g_m3, result.predicted_indoor_ah_g_m3
    ):
        assert predicted == pytest.approx(expected, abs=1e-3)


def test_predicted_first_point_equals_observed_first_point() -> None:
    """The forward simulation starts from the first measurement.

    Predicted[0] must equal Observed[0] by construction; subsequent
    predictions are model output.
    """
    obs = _synthetic_series(
        true_ach=2.0,
        initial_indoor_ah=9.0,
        outdoor_ah=4.5,
        times_hours=_linspace(0.0, 0.5, 11),
    )
    result = estimate_ach_from_observations(obs)
    assert (
        result.predicted_indoor_ah_g_m3[0]
        == result.observed_indoor_ah_g_m3[0]
    )


# --- Segment extraction --------------------------------------------------


def test_ignores_leading_and_trailing_closed_observations() -> None:
    """Closed-window observations before/after the event are dropped."""
    true_ach = 4.0
    open_run = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=11.0,
        outdoor_ah=3.5,
        times_hours=_linspace(0.5, 1.0, 21),
    )
    # Leading closed: same indoor AH held steady before the event.
    leading_closed = [
        CalibrationObservation(
            timestamp_hours=t,
            indoor_absolute_humidity_g_m3=11.0,
            outdoor_absolute_humidity_g_m3=3.5,
            window_open=False,
        )
        for t in _linspace(0.0, 0.45, 4)
    ]
    trailing_closed = [
        CalibrationObservation(
            timestamp_hours=t,
            indoor_absolute_humidity_g_m3=open_run[-1].indoor_absolute_humidity_g_m3,
            outdoor_absolute_humidity_g_m3=3.5,
            window_open=False,
        )
        for t in _linspace(1.05, 1.5, 4)
    ]
    result = estimate_ach_from_observations(
        leading_closed + open_run + trailing_closed
    )
    assert result.n_observations == len(open_run)
    assert result.estimated_ach == pytest.approx(true_ach, rel=1e-3)


def test_selects_longest_contiguous_open_run() -> None:
    """When two disjoint open runs are present, the longer one is used."""
    true_ach = 3.0
    short_run = _synthetic_series(
        true_ach=1.0,  # different ACH; if this run were used, the fit would drift
        initial_indoor_ah=8.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.1, 3),  # 3 points
    )
    long_run = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.5, 1.0, 21),  # 21 points
    )
    closed_gap = [
        CalibrationObservation(
            timestamp_hours=0.3,
            indoor_absolute_humidity_g_m3=8.0,
            outdoor_absolute_humidity_g_m3=4.0,
            window_open=False,
        )
    ]
    result = estimate_ach_from_observations(
        short_run + closed_gap + long_run
    )
    assert result.n_observations == len(long_run)
    assert result.estimated_ach == pytest.approx(true_ach, rel=1e-3)


# --- Noise robustness ----------------------------------------------------


def test_noisy_data_recovers_true_ach_within_a_few_percent() -> None:
    """Small deterministic pseudo-noise perturbs the estimate only mildly.

    Uses a repeatable pseudo-random sequence (no Random(), no
    external module state) so the test is deterministic. Noise
    amplitude ~ 0.05 g/m^3, well below the ~5 g/m^3 signal.
    """
    true_ach = 3.0
    times_hours = _linspace(0.0, 0.5, 31)
    clean = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=times_hours,
    )
    # Deterministic sawtooth noise.
    noise_amplitude = 0.05
    noisy = []
    for i, c in enumerate(clean):
        # Alternating perturbation: +eps, -eps, +eps, ... .
        eps = noise_amplitude * (1.0 if i % 2 == 0 else -1.0)
        noisy.append(
            CalibrationObservation(
                timestamp_hours=c.timestamp_hours,
                indoor_absolute_humidity_g_m3=max(
                    0.0, c.indoor_absolute_humidity_g_m3 + eps
                ),
                outdoor_absolute_humidity_g_m3=(
                    c.outdoor_absolute_humidity_g_m3
                ),
                window_open=c.window_open,
            )
        )
    result = estimate_ach_from_observations(noisy)
    assert result.estimated_ach == pytest.approx(true_ach, rel=0.05)
    # RMS error should reflect the noise (order 0.05 g/m^3), not
    # blow up.
    assert result.rms_error_g_m3 < 0.2


# --- Time-varying outdoor AH --------------------------------------------


def test_handles_time_varying_outdoor_ah() -> None:
    """The residual sums over piecewise-constant outdoor per interval.

    Generates a synthetic trajectory using a stepwise-changing
    outdoor AH schedule (each interval has its own outdoor value)
    and checks the fit recovers the true ACH.
    """
    true_ach = 4.0
    times_hours = _linspace(0.0, 0.5, 21)
    initial_indoor = 10.0
    outdoor_schedule = [
        4.0 if t < 0.25 else 5.0 for t in times_hours
    ]  # step change mid-way
    indoor = [initial_indoor]
    for i in range(len(times_hours) - 1):
        dt = times_hours[i + 1] - times_hours[i]
        outdoor = outdoor_schedule[i]
        next_indoor = outdoor + (indoor[-1] - outdoor) * exp(-true_ach * dt)
        indoor.append(next_indoor)
    obs = [
        CalibrationObservation(
            timestamp_hours=times_hours[i],
            indoor_absolute_humidity_g_m3=indoor[i],
            outdoor_absolute_humidity_g_m3=outdoor_schedule[i],
            window_open=True,
        )
        for i in range(len(times_hours))
    ]
    result = estimate_ach_from_observations(obs)
    assert result.estimated_ach == pytest.approx(true_ach, rel=1e-3)


# --- Validation & refusals ----------------------------------------------


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        estimate_ach_from_observations([])


def test_all_closed_input_raises() -> None:
    obs = [
        CalibrationObservation(
            timestamp_hours=t,
            indoor_absolute_humidity_g_m3=10.0,
            outdoor_absolute_humidity_g_m3=4.0,
            window_open=False,
        )
        for t in _linspace(0.0, 0.5, 11)
    ]
    with pytest.raises(ValueError, match="at least two"):
        estimate_ach_from_observations(obs)


def test_single_open_observation_raises() -> None:
    obs = [
        CalibrationObservation(0.0, 10.0, 4.0, True),
        CalibrationObservation(0.1, 10.0, 4.0, False),
    ]
    with pytest.raises(ValueError, match="at least two"):
        estimate_ach_from_observations(obs)


def test_non_monotone_timestamps_within_open_run_raise() -> None:
    obs = [
        CalibrationObservation(0.0, 10.0, 4.0, True),
        CalibrationObservation(0.1, 9.0, 4.0, True),
        CalibrationObservation(0.05, 8.0, 4.0, True),  # goes backward
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_ach_from_observations(obs)


def test_calibration_observation_rejects_negative_ah() -> None:
    with pytest.raises(ValueError, match="indoor_absolute_humidity_g_m3"):
        CalibrationObservation(0.0, -1.0, 4.0, True)
    with pytest.raises(ValueError, match="outdoor_absolute_humidity_g_m3"):
        CalibrationObservation(0.0, 10.0, -1.0, True)


def test_calibration_observation_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_hours"):
        CalibrationObservation(-0.1, 10.0, 4.0, True)


def test_calibration_observation_rejects_non_finite_fields() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            CalibrationObservation(bad, 10.0, 4.0, True)
        with pytest.raises(ValueError):
            CalibrationObservation(0.0, bad, 4.0, True)
        with pytest.raises(ValueError):
            CalibrationObservation(0.0, 10.0, bad, True)


def test_bracket_bounds_must_be_ordered() -> None:
    obs = _synthetic_series(
        true_ach=3.0,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.3, 7),
    )
    with pytest.raises(ValueError, match="ach_search_upper_bound"):
        estimate_ach_from_observations(
            obs,
            ach_search_lower_bound=5.0,
            ach_search_upper_bound=1.0,
        )


def test_bracket_lower_bound_non_negative() -> None:
    obs = _synthetic_series(
        true_ach=3.0,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.3, 7),
    )
    with pytest.raises(ValueError, match="ach_search_lower_bound"):
        estimate_ach_from_observations(
            obs, ach_search_lower_bound=-0.1, ach_search_upper_bound=10.0
        )


def test_result_reports_bracket_used() -> None:
    obs = _synthetic_series(
        true_ach=3.0,
        initial_indoor_ah=10.0,
        outdoor_ah=4.0,
        times_hours=_linspace(0.0, 0.3, 11),
    )
    result = estimate_ach_from_observations(
        obs, ach_search_lower_bound=0.1, ach_search_upper_bound=20.0
    )
    assert result.ach_search_lower_bound == 0.1
    assert result.ach_search_upper_bound == 20.0


def test_default_bracket_values_are_reasonable() -> None:
    """The exported defaults sit inside plausible residential range."""
    assert 0.0 <= DEFAULT_ACH_SEARCH_MIN < 1.0
    assert DEFAULT_ACH_SEARCH_MAX > 20.0


# --- Cross-check: a fit reproduces the driver in the simulator ---------


def test_fitted_ach_reproduces_final_indoor_ah_via_forward_simulation() -> None:
    """Given a true ACH, ingest data from the pure ODE, refit, and reforward.

    Loops the whole pipeline: generate observations with true ACH,
    fit, use the fitted ACH to forward-simulate, and check the
    forward simulation ends at the same indoor AH as the last
    observation.
    """
    true_ach = 5.0
    times_hours = _linspace(0.0, 0.4, 25)
    obs = _synthetic_series(
        true_ach=true_ach,
        initial_indoor_ah=13.0,
        outdoor_ah=4.5,
        times_hours=times_hours,
    )
    result = estimate_ach_from_observations(obs)
    assert result.predicted_indoor_ah_g_m3[-1] == pytest.approx(
        obs[-1].indoor_absolute_humidity_g_m3, rel=1e-3
    )
