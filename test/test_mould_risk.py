"""Tests for the time-integrated moisture / condensation risk indicator.

Covers:
    - ``RiskConfig`` validation.
    - Zero-exposure trajectory -> all-zero state.
    - Constant elevated exposure -> full-duration ``time_above``.
    - Constant condensation exposure -> both accumulations = duration.
    - Score composes with caller-set weights.
    - Peak RH tracks the true max.
    - Two ventilation strategies produce different risk on the same
      room + source (the actual purpose of the module).
    - Single-sample and empty trajectories are handled without
      arithmetic errors.
    - Module never labels its output as a mould prediction (AST /
      grep guard).
"""

import ast
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room
from moisture_sources import (
    MoistureSourceEvent,
    MoistureSourceSchedule,
)
from mould_risk import (
    MoistureRiskState,
    RiskConfig,
    evaluate_moisture_risk,
)
from psychrometrics import AirState
from surface_risk import SurfaceDescriptor
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from time_simulation import (
    RoomTrajectory,
    VentilationEvent,
    simulate_room_period,
)


def _default_thermal_properties() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _make_synthetic_trajectory(
    times_hours: tuple,
    indoor_temperatures_c: tuple,
    indoor_relative_humidities_pct: tuple,
    outdoor_temperature_c: float = 5.0,
) -> RoomTrajectory:
    """Build a RoomTrajectory with caller-controlled indoor conditions.

    Computes each sample's indoor AH from (T, RH) via AirState so
    the trajectory is internally consistent for consumers that
    read indoor_absolute_humidity_g_m3 directly (like
    ``evaluate_moisture_risk``).
    """
    n = len(times_hours)
    indoor_ahs = tuple(
        AirState(
            temperature_c=indoor_temperatures_c[i],
            relative_humidity_percent=indoor_relative_humidities_pct[i],
        ).absolute_humidity
        for i in range(n)
    )
    return RoomTrajectory(
        times_hours=times_hours,
        indoor_temperature_c=indoor_temperatures_c,
        indoor_absolute_humidity_g_m3=indoor_ahs,
        indoor_relative_humidity_pct=indoor_relative_humidities_pct,
        outdoor_temperature_c=(outdoor_temperature_c,) * n,
        outdoor_absolute_humidity_g_m3=(0.0,) * n,  # unused by evaluator
        window_open=(False,) * n,
        moisture_generation_g_per_hour=(0.0,) * n,
    )


# --- RiskConfig validation --------------------------------------------------


def test_risk_config_defaults_are_sensible_but_not_authoritative() -> None:
    """RiskConfig() constructs; defaults are documented as caller-configurable."""
    config = RiskConfig()
    assert config.elevated_surface_rh_threshold_percent == 80.0
    assert config.condensation_surface_rh_threshold_percent == 100.0
    assert config.elevated_time_weight == 1.0
    assert config.condensation_time_weight == 1.0
    assert config.peak_rh_excess_weight_hours_per_percent == 0.0


def test_risk_config_accepts_custom_values() -> None:
    """Every field is caller-settable."""
    RiskConfig(
        elevated_surface_rh_threshold_percent=75.0,
        condensation_surface_rh_threshold_percent=98.0,
        elevated_time_weight=0.5,
        condensation_time_weight=5.0,
        peak_rh_excess_weight_hours_per_percent=0.2,
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "elevated_surface_rh_threshold_percent",
        "condensation_surface_rh_threshold_percent",
        "elevated_time_weight",
        "condensation_time_weight",
        "peak_rh_excess_weight_hours_per_percent",
    ],
)
def test_risk_config_rejects_negative_values(field_name: str) -> None:
    """Every numeric field must be non-negative."""
    kwargs = {field_name: -0.1}
    with pytest.raises(ValueError, match=field_name):
        RiskConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("elevated_surface_rh_threshold_percent", float("nan")),
        ("condensation_surface_rh_threshold_percent", float("inf")),
        ("elevated_time_weight", float("nan")),
        ("peak_rh_excess_weight_hours_per_percent", float("inf")),
    ],
)
def test_risk_config_rejects_non_finite_values(
    field_name: str, bad_value: float
) -> None:
    """NaN / inf on any field is rejected."""
    kwargs = {field_name: bad_value}
    with pytest.raises(ValueError, match=field_name):
        RiskConfig(**kwargs)  # type: ignore[arg-type]


def test_risk_config_rejects_rh_thresholds_above_200() -> None:
    """Upper bound of 200 % catches gross caller mistakes."""
    with pytest.raises(ValueError, match="elevated_surface_rh_threshold_percent"):
        RiskConfig(elevated_surface_rh_threshold_percent=250.0)


def test_risk_config_is_frozen() -> None:
    """RiskConfig is a frozen dataclass."""
    config = RiskConfig()
    with pytest.raises(FrozenInstanceError):
        config.elevated_time_weight = 5.0  # type: ignore[misc]


# --- Zero-exposure trajectory ---------------------------------------------


def test_dry_room_never_triggers_thresholds() -> None:
    """A room at 40 %RH on a warm outdoor never crosses 80 % on a warm surface."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0, 4.0),
        indoor_temperatures_c=(20.0, 20.0, 20.0, 20.0, 20.0),
        indoor_relative_humidities_pct=(40.0, 40.0, 40.0, 40.0, 40.0),
        outdoor_temperature_c=18.0,
    )
    surface = SurfaceDescriptor(
        label="warm wall", surface_temperature_factor=0.95
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    assert result.time_above_surface_rh_threshold_hours == 0.0
    assert result.time_in_condensation_hours == 0.0
    assert result.cumulative_risk_score == 0.0


# --- Constant above-threshold trajectory ----------------------------------


def test_constant_elevated_surface_accumulates_full_duration() -> None:
    """When every step has surface RH > threshold, time_above = total duration.

    Warm-humid room + cold surface: 25 C indoor / 70 %RH with a
    severe thermal bridge (fRsi = 0.3) under a 2 C outdoor. Surface
    T = 2 + 0.3 * (25-2) = 8.9 C, well below the room's ~19 C dew
    point -> surface RH > 100 across every sample. Both time
    counters should equal the trajectory duration (3 h).
    """
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0),
        indoor_temperatures_c=(25.0,) * 4,
        indoor_relative_humidities_pct=(70.0,) * 4,
        outdoor_temperature_c=2.0,
    )
    surface = SurfaceDescriptor(
        label="thermal bridge", surface_temperature_factor=0.3
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    # Trajectory has 4 samples; 3 step intervals of 1 hour each
    # contribute if the START-of-step is above threshold; every
    # start is above, so full-duration accumulation = 3.0 h.
    assert result.time_above_surface_rh_threshold_hours == pytest.approx(3.0)
    assert result.time_in_condensation_hours == pytest.approx(3.0)
    assert result.maximum_surface_rh_percent > 100.0
    # Default config: score = 1.0 * time_above + 1.0 * time_cond.
    assert result.cumulative_risk_score == pytest.approx(6.0)


# --- Partial exposure ------------------------------------------------------


def test_partial_exposure_only_counts_affected_intervals() -> None:
    """Only steps starting above threshold contribute to time_above.

    Build a synthetic trajectory of six equal 1-hour steps where
    the caller controls the indoor RH per sample and thereby the
    surface RH at each start-of-step.
    """
    # Surface fRsi = 1.0 -> surface RH = indoor RH exactly. Then
    # threshold triggering is trivial to reason about by inspection.
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        indoor_temperatures_c=(20.0,) * 6,
        # Only steps starting at t=2 (85 %) and t=3 (85 %) exceed
        # the default 80 % threshold. Steps starting at t=4 (75 %)
        # and t=5 do not - and the t=5 sample doesn't start a step
        # anyway (last sample).
        indoor_relative_humidities_pct=(50.0, 50.0, 85.0, 85.0, 75.0, 60.0),
        outdoor_temperature_c=15.0,
    )
    surface = SurfaceDescriptor(
        label="uniform", surface_temperature_factor=1.0
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    # Steps starting at t=2 and t=3 each contribute 1.0 h. No
    # condensation anywhere (max is 85 %).
    assert result.time_above_surface_rh_threshold_hours == pytest.approx(2.0)
    assert result.time_in_condensation_hours == 0.0
    assert result.maximum_surface_rh_percent == pytest.approx(85.0)


def test_configurable_threshold_changes_the_accumulation() -> None:
    """Lowering the elevated threshold to 70 % should catch more time.

    Same trajectory as the previous test. With threshold 70, the
    steps starting at t=2 (85 %), t=3 (85 %), and t=4 (75 %)
    contribute. Total = 3.0 h.
    """
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        indoor_temperatures_c=(20.0,) * 6,
        indoor_relative_humidities_pct=(50.0, 50.0, 85.0, 85.0, 75.0, 60.0),
        outdoor_temperature_c=15.0,
    )
    surface = SurfaceDescriptor(
        label="uniform", surface_temperature_factor=1.0
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(elevated_surface_rh_threshold_percent=70.0),
    )
    assert result.time_above_surface_rh_threshold_hours == pytest.approx(3.0)


# --- Score composition -----------------------------------------------------


def test_cumulative_score_uses_caller_weights() -> None:
    """Score is a weighted sum of the three accumulations."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0),
        indoor_temperatures_c=(25.0,) * 4,
        indoor_relative_humidities_pct=(70.0,) * 4,
        outdoor_temperature_c=2.0,
    )
    surface = SurfaceDescriptor(
        label="bridge", surface_temperature_factor=0.3
    )
    config = RiskConfig(
        elevated_time_weight=2.0,
        condensation_time_weight=10.0,
        peak_rh_excess_weight_hours_per_percent=0.5,
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=config,
    )
    # time_above = 3.0, time_cond = 3.0, peak > 100 so peak_excess
    # is large. Score = 2 * 3 + 10 * 3 + 0.5 * (max - 80).
    expected = (
        2.0 * result.time_above_surface_rh_threshold_hours
        + 10.0 * result.time_in_condensation_hours
        + 0.5
        * max(
            0.0,
            result.maximum_surface_rh_percent
            - config.elevated_surface_rh_threshold_percent,
        )
    )
    assert result.cumulative_risk_score == pytest.approx(expected, rel=1e-12)


def test_zero_weights_zero_out_the_score_but_preserve_accumulations() -> None:
    """Score can be zeroed by weights while the raw accumulations remain."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0),
        indoor_temperatures_c=(25.0,) * 3,
        indoor_relative_humidities_pct=(70.0,) * 3,
        outdoor_temperature_c=2.0,
    )
    surface = SurfaceDescriptor(
        label="bridge", surface_temperature_factor=0.3
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(
            elevated_time_weight=0.0,
            condensation_time_weight=0.0,
        ),
    )
    assert result.cumulative_risk_score == 0.0
    # But the raw accumulations are unchanged.
    assert result.time_above_surface_rh_threshold_hours > 0.0
    assert result.time_in_condensation_hours > 0.0


# --- Peak RH tracking -----------------------------------------------------


def test_maximum_surface_rh_tracks_the_strict_max() -> None:
    """Peak surface RH equals the maximum across every sample."""
    # Warm surface (fRsi = 1) -> surface RH = indoor RH each step.
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0, 2.0, 3.0),
        indoor_temperatures_c=(20.0,) * 4,
        indoor_relative_humidities_pct=(30.0, 92.0, 60.0, 45.0),
        outdoor_temperature_c=15.0,
    )
    surface = SurfaceDescriptor(
        label="uniform", surface_temperature_factor=1.0
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    assert result.maximum_surface_rh_percent == pytest.approx(92.0)


# --- The actual purpose: compare two ventilation strategies --------------


def test_two_ventilation_strategies_produce_different_risk_states() -> None:
    """A ventilated schedule and an unventilated schedule differ on risk.

    Runs the ACTUAL time_simulation pipeline over 4 hours with a
    persistent moisture source in a room facing a cold surface.
    Compares the risk state under (a) window closed all 4 h and
    (b) window open for the middle 30 min. The ventilated case
    must show LOWER accumulated exposure - that's the whole point
    of the module.
    """
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.3,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=2.0, relative_humidity_percent=80.0)
    thermal_props = _default_thermal_properties()
    moisture_schedule = MoistureSourceSchedule(
        constant_background_rate_g_per_hour=120.0
    )
    surface = SurfaceDescriptor(
        label="cold wall", surface_temperature_factor=0.6
    )
    config = RiskConfig()

    closed_trajectory = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=moisture_schedule,
        ventilation_events=(),
        duration_hours=4.0,
        timestep_minutes=5.0,
    )
    open_trajectory = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=moisture_schedule,
        ventilation_events=(
            VentilationEvent(start_time_hours=1.5, end_time_hours=2.0),
        ),
        duration_hours=4.0,
        timestep_minutes=5.0,
    )

    closed_risk = evaluate_moisture_risk(
        trajectory=closed_trajectory,
        surface=surface,
        config=config,
    )
    open_risk = evaluate_moisture_risk(
        trajectory=open_trajectory,
        surface=surface,
        config=config,
    )

    # The ventilated case must have a strictly lower risk score
    # AND less accumulated time above threshold. The peak might
    # not differ because the peak can occur before the vent.
    assert (
        open_risk.cumulative_risk_score < closed_risk.cumulative_risk_score
    )
    assert (
        open_risk.time_above_surface_rh_threshold_hours
        < closed_risk.time_above_surface_rh_threshold_hours
    )


# --- Edge cases ------------------------------------------------------------


def test_single_sample_trajectory_returns_zero_time_accumulations() -> None:
    """A one-sample trajectory has no step intervals to accumulate over."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0,),
        indoor_temperatures_c=(20.0,),
        indoor_relative_humidities_pct=(90.0,),
        outdoor_temperature_c=15.0,
    )
    surface = SurfaceDescriptor(
        label="uniform", surface_temperature_factor=1.0
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    assert result.time_above_surface_rh_threshold_hours == 0.0
    assert result.time_in_condensation_hours == 0.0
    # But the single sample counts for the peak.
    assert result.maximum_surface_rh_percent == pytest.approx(90.0)


def test_empty_trajectory_returns_all_zeros() -> None:
    """A zero-sample trajectory returns an all-zero state."""
    trajectory = RoomTrajectory(
        times_hours=(),
        indoor_temperature_c=(),
        indoor_absolute_humidity_g_m3=(),
        indoor_relative_humidity_pct=(),
        outdoor_temperature_c=(),
        outdoor_absolute_humidity_g_m3=(),
        window_open=(),
        moisture_generation_g_per_hour=(),
    )
    surface = SurfaceDescriptor(
        label="test", surface_temperature_factor=0.75
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=surface,
        config=RiskConfig(),
    )
    assert result.time_above_surface_rh_threshold_hours == 0.0
    assert result.time_in_condensation_hours == 0.0
    assert result.maximum_surface_rh_percent == 0.0
    assert result.cumulative_risk_score == 0.0


# --- Result dataclass hygiene --------------------------------------------


def test_moisture_risk_state_is_frozen() -> None:
    """MoistureRiskState is immutable."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0),
        indoor_temperatures_c=(20.0, 20.0),
        indoor_relative_humidities_pct=(50.0, 50.0),
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=SurfaceDescriptor(
            label="test", surface_temperature_factor=1.0
        ),
        config=RiskConfig(),
    )
    with pytest.raises(FrozenInstanceError):
        result.cumulative_risk_score = 99.0  # type: ignore[misc]


def test_moisture_risk_state_echoes_the_surface_label() -> None:
    """The surface's label is preserved on the result for audit."""
    trajectory = _make_synthetic_trajectory(
        times_hours=(0.0, 1.0),
        indoor_temperatures_c=(20.0, 20.0),
        indoor_relative_humidities_pct=(50.0, 50.0),
    )
    result = evaluate_moisture_risk(
        trajectory=trajectory,
        surface=SurfaceDescriptor(
            label="kitchen north wall behind fridge",
            surface_temperature_factor=0.75,
        ),
        config=RiskConfig(),
    )
    assert result.surface_label == "kitchen north wall behind fridge"


# --- Design contract: module does NOT claim to predict mould -------------


def test_module_source_does_not_claim_mould_prediction() -> None:
    """AST + text guard: no string in mould_risk.py positively asserts
    "this predicts mould growth". Comments and docstrings that
    explicitly disclaim the claim are allowed.

    The rule: any occurrence of the words 'mould' or 'mold' near a
    'predicts' / 'predicted' / 'guarantee' / 'causes' must be
    negated ('does not predict', 'not a prediction', etc.).
    """
    source_path = Path(__file__).resolve().parent.parent / "mould_risk.py"
    source = source_path.read_text()
    # Rough textual guard: any line containing a positive assertion
    # of prediction must not appear. Compile a small negative list.
    forbidden_phrases = (
        "predicts mould",
        "predicts mold",
        "predicted mould growth",
        "predicted mold growth",
        "guarantees mould",
        "guarantees mold",
        "causes mould",
        "causes mold",
    )
    lowered = source.lower()
    for phrase in forbidden_phrases:
        # A phrase is allowed only if immediately preceded by "not"
        # or "no ". We do a coarse check: forbid ANY appearance and
        # rely on the negated disclaimer language the module uses.
        assert phrase not in lowered, (
            f"mould_risk.py contains the phrase '{phrase}', which positively "
            "claims mould growth prediction. This module is a risk INDICATOR, "
            "not a validated growth model."
        )