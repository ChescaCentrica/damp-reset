"""Tests for the effective thermal capacitance calibration module.

Covers:
    - Synthetic-data recovery: given data generated from the exact
      first-order-decay temperature model at a KNOWN C_eff, the
      calibration recovers that C_eff to well under one percent.
    - Recovery across a sweep of true C_eff values from a very
      light room (50 kJ/K) to a heavy one (5 MJ/K).
    - Recovery across a sweep of ACH values.
    - Robustness to modest deterministic noise: recovery stays
      within a small fraction of truth.
    - ThermalObservation validation.
    - Refusal cases: empty input, all-closed-window input, single
      window-open observation, non-monotone timestamps, invalid
      ACH / volume / bracket bounds.
    - Segment extraction: only the window-open run is fitted.
    - Reconstruction: forward-simulated trajectory matches observed
      closely on clean synthetic data.
"""

import sys
from math import exp
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from thermal import ventilation_heat_loss_coefficient

from thermal_calibration import (
    DEFAULT_C_EFF_SEARCH_MAX_J_PER_K,
    DEFAULT_C_EFF_SEARCH_MIN_J_PER_K,
    ThermalCalibrationResult,
    ThermalObservation,
    estimate_effective_thermal_capacity_from_observations,
)


def _synthetic_series(
    true_c_eff: float,
    initial_indoor_t: float,
    outdoor_t: float,
    times_hours: List[float],
    ach: float,
    room_volume_m3: float,
    window_open: bool = True,
) -> List[ThermalObservation]:
    """Noise-free observations that exactly satisfy the physics model."""
    h_vent = ventilation_heat_loss_coefficient(
        room_volume_m3=room_volume_m3, ach=ach
    )
    obs = []
    for t in times_hours:
        indoor = outdoor_t + (initial_indoor_t - outdoor_t) * exp(
            -h_vent / true_c_eff * t * 3600.0
        )
        obs.append(
            ThermalObservation(
                timestamp_hours=t,
                indoor_temperature_c=indoor,
                outdoor_temperature_c=outdoor_t,
                window_open=window_open,
            )
        )
    return obs


def _linspace(start: float, stop: float, count: int) -> List[float]:
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


# --- Basic recovery ------------------------------------------------------


def test_recovers_true_c_eff_on_clean_synthetic_data() -> None:
    true_c_eff = 500_000.0
    obs = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.5, 21),
        ach=5.0,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=5.0, room_volume_m3=40.0
    )
    assert isinstance(result, ThermalCalibrationResult)
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=1e-3)
    )
    assert result.rms_error_c < 1e-3
    assert result.n_observations == 21


@pytest.mark.parametrize(
    "true_c_eff",
    [50_000.0, 100_000.0, 300_000.0, 500_000.0, 1_000_000.0, 2_500_000.0, 5_000_000.0],
)
def test_recovers_across_wide_range_of_true_c_eff(true_c_eff: float) -> None:
    obs = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.5, 31),
        ach=5.0,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=5.0, room_volume_m3=40.0
    )
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=5e-3)
    )


@pytest.mark.parametrize("ach", [1.0, 3.0, 5.0, 10.0, 15.0])
def test_recovers_across_wide_range_of_ach(ach: float) -> None:
    true_c_eff = 500_000.0
    obs = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.5, 31),
        ach=ach,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=ach, room_volume_m3=40.0
    )
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=5e-3)
    )


def test_rms_error_is_tiny_for_clean_data() -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.5, 21),
        ach=5.0,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=5.0, room_volume_m3=40.0
    )
    assert result.rms_error_c < 1e-3


# --- Reconstruction ------------------------------------------------------


def test_predicted_trajectory_matches_observed_on_clean_data() -> None:
    obs = _synthetic_series(
        true_c_eff=750_000.0,
        initial_indoor_t=21.0,
        outdoor_t=-1.0,
        times_hours=_linspace(0.0, 0.5, 16),
        ach=4.0,
        room_volume_m3=45.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=4.0, room_volume_m3=45.0
    )
    for expected, predicted in zip(
        result.observed_indoor_temperature_c,
        result.predicted_indoor_temperature_c,
    ):
        assert predicted == pytest.approx(expected, abs=1e-3)


def test_predicted_first_point_equals_observed_first_point() -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=19.5,
        outdoor_t=0.0,
        times_hours=_linspace(0.0, 0.4, 11),
        ach=4.0,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=4.0, room_volume_m3=40.0
    )
    assert (
        result.predicted_indoor_temperature_c[0]
        == result.observed_indoor_temperature_c[0]
    )


# --- Segment extraction --------------------------------------------------


def test_ignores_leading_and_trailing_closed_observations() -> None:
    true_c_eff = 500_000.0
    ach = 5.0
    volume = 40.0
    open_run = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.5, 1.0, 21),
        ach=ach,
        room_volume_m3=volume,
    )
    leading_closed = [
        ThermalObservation(
            timestamp_hours=t,
            indoor_temperature_c=20.0,
            outdoor_temperature_c=-2.0,
            window_open=False,
        )
        for t in _linspace(0.0, 0.45, 4)
    ]
    trailing_closed = [
        ThermalObservation(
            timestamp_hours=t,
            indoor_temperature_c=open_run[-1].indoor_temperature_c,
            outdoor_temperature_c=-2.0,
            window_open=False,
        )
        for t in _linspace(1.05, 1.5, 4)
    ]
    result = estimate_effective_thermal_capacity_from_observations(
        observations=leading_closed + open_run + trailing_closed,
        ach=ach,
        room_volume_m3=volume,
    )
    assert result.n_observations == len(open_run)
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=1e-3)
    )


def test_selects_longest_contiguous_open_run() -> None:
    true_c_eff = 500_000.0
    ach = 5.0
    volume = 40.0
    short_run = _synthetic_series(
        true_c_eff=1_500_000.0,  # different C_eff; would drift the fit if used
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.1, 3),
        ach=ach,
        room_volume_m3=volume,
    )
    long_run = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.5, 1.0, 21),
        ach=ach,
        room_volume_m3=volume,
    )
    closed_gap = [
        ThermalObservation(0.3, 20.0, -2.0, False)
    ]
    result = estimate_effective_thermal_capacity_from_observations(
        observations=short_run + closed_gap + long_run,
        ach=ach,
        room_volume_m3=volume,
    )
    assert result.n_observations == len(long_run)
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=1e-3)
    )


# --- Noise robustness ----------------------------------------------------


def test_noisy_data_recovers_true_c_eff_within_a_few_percent() -> None:
    """Deterministic sawtooth noise of amplitude 0.05 K perturbs only mildly."""
    true_c_eff = 500_000.0
    ach = 5.0
    volume = 40.0
    times_hours = _linspace(0.0, 0.5, 31)
    clean = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=times_hours,
        ach=ach,
        room_volume_m3=volume,
    )
    noise_amplitude = 0.05
    noisy = []
    for i, c in enumerate(clean):
        eps = noise_amplitude * (1.0 if i % 2 == 0 else -1.0)
        noisy.append(
            ThermalObservation(
                timestamp_hours=c.timestamp_hours,
                indoor_temperature_c=c.indoor_temperature_c + eps,
                outdoor_temperature_c=c.outdoor_temperature_c,
                window_open=c.window_open,
            )
        )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=noisy, ach=ach, room_volume_m3=volume
    )
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=0.05)
    )
    assert result.rms_error_c < 0.2


# --- Time-varying outdoor T ---------------------------------------------


def test_handles_time_varying_outdoor_temperature() -> None:
    """Piecewise-constant outdoor T with a step change still identifies C_eff."""
    true_c_eff = 500_000.0
    ach = 5.0
    volume = 40.0
    h_vent = ventilation_heat_loss_coefficient(volume, ach)
    times_hours = _linspace(0.0, 0.5, 21)
    outdoor_schedule = [
        -2.0 if t < 0.25 else 2.0 for t in times_hours
    ]
    indoor = [20.0]
    for i in range(len(times_hours) - 1):
        dt_seconds = (times_hours[i + 1] - times_hours[i]) * 3600.0
        outdoor = outdoor_schedule[i]
        next_indoor = outdoor + (indoor[-1] - outdoor) * exp(
            -h_vent / true_c_eff * dt_seconds
        )
        indoor.append(next_indoor)
    obs = [
        ThermalObservation(
            timestamp_hours=times_hours[i],
            indoor_temperature_c=indoor[i],
            outdoor_temperature_c=outdoor_schedule[i],
            window_open=True,
        )
        for i in range(len(times_hours))
    ]
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=ach, room_volume_m3=volume
    )
    assert (
        result.estimated_effective_thermal_capacity_j_per_k
        == pytest.approx(true_c_eff, rel=1e-3)
    )


# --- Validation & refusals ----------------------------------------------


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least two"):
        estimate_effective_thermal_capacity_from_observations(
            observations=[], ach=5.0, room_volume_m3=40.0
        )


def test_all_closed_input_raises() -> None:
    obs = [
        ThermalObservation(t, 20.0, -2.0, False)
        for t in _linspace(0.0, 0.5, 11)
    ]
    with pytest.raises(ValueError, match="at least two"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs, ach=5.0, room_volume_m3=40.0
        )


def test_single_open_observation_raises() -> None:
    obs = [
        ThermalObservation(0.0, 20.0, -2.0, True),
        ThermalObservation(0.1, 20.0, -2.0, False),
    ]
    with pytest.raises(ValueError, match="at least two"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs, ach=5.0, room_volume_m3=40.0
        )


def test_non_monotone_timestamps_within_open_run_raise() -> None:
    obs = [
        ThermalObservation(0.0, 20.0, -2.0, True),
        ThermalObservation(0.1, 19.0, -2.0, True),
        ThermalObservation(0.05, 18.0, -2.0, True),
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs, ach=5.0, room_volume_m3=40.0
        )


def test_thermal_observation_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_hours"):
        ThermalObservation(-0.1, 20.0, -2.0, True)


def test_thermal_observation_rejects_non_finite_fields() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ThermalObservation(bad, 20.0, -2.0, True)
        with pytest.raises(ValueError):
            ThermalObservation(0.0, bad, -2.0, True)
        with pytest.raises(ValueError):
            ThermalObservation(0.0, 20.0, bad, True)


def test_thermal_observation_allows_negative_temperatures() -> None:
    """Winter outdoor observations can and should be below freezing."""
    obs = ThermalObservation(0.0, 15.0, -10.0, True)
    assert obs.outdoor_temperature_c == -10.0


@pytest.mark.parametrize("bad_ach", [0.0, -1.0, float("nan"), float("inf")])
def test_zero_or_negative_ach_rejected(bad_ach: float) -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.3, 7),
        ach=5.0,
        room_volume_m3=40.0,
    )
    with pytest.raises(ValueError, match="ach"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs, ach=bad_ach, room_volume_m3=40.0
        )


@pytest.mark.parametrize("bad_vol", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_room_volume_rejected(bad_vol: float) -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.3, 7),
        ach=5.0,
        room_volume_m3=40.0,
    )
    with pytest.raises(ValueError, match="room_volume_m3"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs, ach=5.0, room_volume_m3=bad_vol
        )


def test_bracket_bounds_must_be_ordered() -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.3, 7),
        ach=5.0,
        room_volume_m3=40.0,
    )
    with pytest.raises(ValueError, match="c_eff_search_upper_bound_j_per_k"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs,
            ach=5.0,
            room_volume_m3=40.0,
            c_eff_search_lower_bound_j_per_k=1_000_000.0,
            c_eff_search_upper_bound_j_per_k=100_000.0,
        )


def test_bracket_lower_bound_must_be_positive() -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.3, 7),
        ach=5.0,
        room_volume_m3=40.0,
    )
    with pytest.raises(ValueError, match="c_eff_search_lower_bound_j_per_k"):
        estimate_effective_thermal_capacity_from_observations(
            observations=obs,
            ach=5.0,
            room_volume_m3=40.0,
            c_eff_search_lower_bound_j_per_k=0.0,
            c_eff_search_upper_bound_j_per_k=1_000_000.0,
        )


def test_result_reports_bracket_used() -> None:
    obs = _synthetic_series(
        true_c_eff=500_000.0,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.3, 11),
        ach=5.0,
        room_volume_m3=40.0,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs,
        ach=5.0,
        room_volume_m3=40.0,
        c_eff_search_lower_bound_j_per_k=100_000.0,
        c_eff_search_upper_bound_j_per_k=5_000_000.0,
    )
    assert result.c_eff_search_lower_bound_j_per_k == 100_000.0
    assert result.c_eff_search_upper_bound_j_per_k == 5_000_000.0
    assert result.ach == 5.0
    assert result.room_volume_m3 == 40.0


def test_default_bracket_values_are_reasonable() -> None:
    assert DEFAULT_C_EFF_SEARCH_MIN_J_PER_K > 0.0
    assert DEFAULT_C_EFF_SEARCH_MAX_J_PER_K > 1_000_000.0
    assert (
        DEFAULT_C_EFF_SEARCH_MIN_J_PER_K < DEFAULT_C_EFF_SEARCH_MAX_J_PER_K
    )


# --- Cross-check: composed with the physics simulator -------------------


def test_fitted_c_eff_reproduces_final_indoor_temperature() -> None:
    """A refit + forward simulation lands at the observed final T."""
    true_c_eff = 500_000.0
    ach = 5.0
    volume = 40.0
    obs = _synthetic_series(
        true_c_eff=true_c_eff,
        initial_indoor_t=20.0,
        outdoor_t=-2.0,
        times_hours=_linspace(0.0, 0.5, 25),
        ach=ach,
        room_volume_m3=volume,
    )
    result = estimate_effective_thermal_capacity_from_observations(
        observations=obs, ach=ach, room_volume_m3=volume
    )
    assert result.predicted_indoor_temperature_c[-1] == pytest.approx(
        obs[-1].indoor_temperature_c, rel=1e-3
    )
