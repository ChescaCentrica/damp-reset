"""Tests for the internal-surface temperature and humidity model.

Covers:
    - Dataclass validation on ``SurfaceDescriptor``.
    - The fRsi temperature formulation: boundary cases and monotone
      behaviour.
    - Surface RH derived via the existing psychrometric inverse:
        * colder surfaces produce higher surface RH,
        * surface RH reaches 100 % at exactly the room's dew point,
        * warmer surfaces reduce condensation risk.
    - Behaviour above saturation (surface RH > 100 without clamp).
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from psychrometrics import AirState, dew_point_c
from surface_risk import (
    SurfaceDescriptor,
    SurfaceRiskResult,
    assess_surface,
    condensation_margin_c,
    surface_relative_humidity,
    surface_relative_humidity_pct,
    surface_temperature_c,
)


# --- SurfaceDescriptor -----------------------------------------------------


def test_surface_descriptor_accepts_reasonable_values() -> None:
    """Every value inside [0, 1] is accepted."""
    for factor in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        SurfaceDescriptor(label="test", surface_temperature_factor=factor)


def test_surface_descriptor_rejects_negative_factor() -> None:
    """Values below 0 are outside the physical interpretation of fRsi."""
    with pytest.raises(ValueError, match="surface_temperature_factor"):
        SurfaceDescriptor(label="bad", surface_temperature_factor=-0.01)


def test_surface_descriptor_rejects_factor_above_one() -> None:
    """Values above 1 imply the surface is warmer than indoor air, not modelled."""
    with pytest.raises(ValueError, match="surface_temperature_factor"):
        SurfaceDescriptor(label="bad", surface_temperature_factor=1.01)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_surface_descriptor_rejects_non_finite_factor(bad_value: float) -> None:
    """NaN / inf factor is rejected."""
    with pytest.raises(ValueError, match="surface_temperature_factor"):
        SurfaceDescriptor(label="bad", surface_temperature_factor=bad_value)


def test_surface_descriptor_is_frozen() -> None:
    """The descriptor is immutable."""
    descriptor = SurfaceDescriptor(label="wall", surface_temperature_factor=0.75)
    with pytest.raises(FrozenInstanceError):
        descriptor.surface_temperature_factor = 0.5  # type: ignore[misc]


# --- Surface temperature formulation ---------------------------------------


def test_temperature_factor_one_gives_indoor_temperature() -> None:
    """fRsi = 1 -> surface at indoor air temperature."""
    surface = SurfaceDescriptor(label="ideal", surface_temperature_factor=1.0)
    assert surface_temperature_c(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        surface=surface,
    ) == pytest.approx(20.0, rel=1e-12)


def test_temperature_factor_zero_gives_outdoor_temperature() -> None:
    """fRsi = 0 -> surface at outdoor temperature."""
    surface = SurfaceDescriptor(
        label="uninsulated", surface_temperature_factor=0.0
    )
    assert surface_temperature_c(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        surface=surface,
    ) == pytest.approx(5.0, rel=1e-12)


def test_temperature_factor_half_interpolates_linearly() -> None:
    """fRsi = 0.5 -> surface at the arithmetic midpoint."""
    surface = SurfaceDescriptor(
        label="thermal bridge", surface_temperature_factor=0.5
    )
    result = surface_temperature_c(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        surface=surface,
    )
    assert result == pytest.approx(12.5, rel=1e-12)


def test_temperature_factor_is_monotone_in_factor() -> None:
    """At fixed indoor > outdoor, higher fRsi -> warmer surface."""
    indoor_t, outdoor_t = 20.0, 5.0
    temperatures = []
    for factor in (0.1, 0.3, 0.5, 0.7, 0.9):
        surface = SurfaceDescriptor(
            label="test", surface_temperature_factor=factor
        )
        temperatures.append(
            surface_temperature_c(
                indoor_temperature_c=indoor_t,
                outdoor_temperature_c=outdoor_t,
                surface=surface,
            )
        )
    for earlier, later in zip(temperatures, temperatures[1:]):
        assert later > earlier


def test_temperature_reversal_when_outdoor_warmer_than_indoor() -> None:
    """Formulation still holds when outdoor > indoor (summer case).

    At fRsi = 0.5 the surface should still sit at the midpoint,
    which is now GREATER than indoor.
    """
    surface = SurfaceDescriptor(label="test", surface_temperature_factor=0.5)
    result = surface_temperature_c(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=30.0,
        surface=surface,
    )
    assert result == pytest.approx(25.0, rel=1e-12)


def test_temperature_rejects_non_finite_inputs() -> None:
    """Both temperatures are validated."""
    surface = SurfaceDescriptor(label="x", surface_temperature_factor=0.5)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="indoor_temperature_c"):
            surface_temperature_c(
                indoor_temperature_c=bad,
                outdoor_temperature_c=5.0,
                surface=surface,
            )
        with pytest.raises(ValueError, match="outdoor_temperature_c"):
            surface_temperature_c(
                indoor_temperature_c=20.0,
                outdoor_temperature_c=bad,
                surface=surface,
            )


# --- Surface RH: the three physical guarantees the user asked for ---------


def test_colder_surface_produces_higher_surface_rh() -> None:
    """Load-bearing test: as fRsi decreases (surface gets colder), RH rises.

    Winter-like scenario: indoor 20 C / 60 %RH, outdoor 5 C. Sweep
    the fRsi from 1.0 (surface at indoor T) down to 0.4 (severe
    bridge) and confirm the surface RH increases monotonically.
    """
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=60.0)
    outdoor_t = 5.0
    surface_rhs = []
    for factor in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        surface = SurfaceDescriptor(
            label="wall", surface_temperature_factor=factor
        )
        rh = surface_relative_humidity_pct(
            indoor_air_state=indoor_state,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        )
        surface_rhs.append(rh)
    # Monotone-increasing as fRsi drops.
    for earlier, later in zip(surface_rhs, surface_rhs[1:]):
        assert later > earlier
    # At fRsi = 1 the surface is at indoor air T, so its RH is the
    # room's indoor RH.
    assert surface_rhs[0] == pytest.approx(60.0, rel=1e-6)


def test_surface_rh_reaches_100_pct_at_room_dew_point() -> None:
    """Load-bearing test: when T_surface equals the room's dew point,
    surface RH is exactly 100 %.

    Constructs an indoor state and finds the fRsi that makes
    T_surface equal to the room's dew point (via
    ``psychrometrics.dew_point_c``, independently). The surface
    RH at that fRsi must be 100 %.
    """
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=60.0)
    outdoor_t = 5.0
    room_dew_point = dew_point_c(20.0, 60.0)
    # Solve for fRsi so T_surface = room_dew_point.
    #   T_surface = outdoor_t + f * (indoor_t - outdoor_t)
    # =>       f = (T_surface - outdoor_t) / (indoor_t - outdoor_t)
    critical_factor = (
        (room_dew_point - outdoor_t) / (20.0 - outdoor_t)
    )
    # Sanity: this f is inside [0, 1] for this scenario.
    assert 0.0 <= critical_factor <= 1.0
    surface = SurfaceDescriptor(
        label="dew-point surface",
        surface_temperature_factor=critical_factor,
    )
    result = surface_relative_humidity_pct(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    )
    assert result == pytest.approx(100.0, rel=1e-9, abs=1e-9)


def test_warmer_surface_reduces_condensation_risk() -> None:
    """Load-bearing test: a warmer surface (higher fRsi) has LOWER RH.

    Directly the flip side of the "colder surface -> higher RH"
    test. If a caller compares two surfaces of the same room, the
    warmer one carries less risk.
    """
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=60.0)
    outdoor_t = 5.0
    cold_surface = SurfaceDescriptor(
        label="thermal bridge", surface_temperature_factor=0.6
    )
    warm_surface = SurfaceDescriptor(
        label="well-insulated wall", surface_temperature_factor=0.9
    )
    cold_rh = surface_relative_humidity_pct(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=cold_surface,
    )
    warm_rh = surface_relative_humidity_pct(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=warm_surface,
    )
    assert warm_rh < cold_rh


def test_surface_rh_matches_room_rh_when_surface_is_at_indoor_temperature() -> None:
    """fRsi = 1 -> surface at indoor T -> surface RH equals room RH."""
    for indoor_rh in (30.0, 55.0, 70.0, 90.0):
        indoor_state = AirState(
            temperature_c=20.0, relative_humidity_percent=indoor_rh
        )
        surface = SurfaceDescriptor(
            label="ideal", surface_temperature_factor=1.0
        )
        result = surface_relative_humidity_pct(
            indoor_air_state=indoor_state,
            outdoor_temperature_c=5.0,
            surface=surface,
        )
        assert result == pytest.approx(indoor_rh, rel=1e-6)


def test_surface_rh_exceeds_100_below_dew_point() -> None:
    """A surface strictly below the room's dew point sees RH > 100 %.

    Physical meaning: dew is forming (condensation). The module
    returns the raw arithmetic value rather than clamping, so
    callers can see the supersaturation.
    """
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
    outdoor_t = 0.0  # very cold outdoor
    # A severe thermal bridge, well below the room's dew point of ~14 C.
    surface = SurfaceDescriptor(
        label="cold pane", surface_temperature_factor=0.3
    )
    result = surface_relative_humidity_pct(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    )
    # Sanity: surface T at fRsi=0.3 = 0 + 0.3 * (20 - 0) = 6 C, well
    # below the room's dew point (~14 C), so P_v > P_sat(6) and the
    # implied RH is > 100.
    assert result > 100.0


def test_illustrative_range_of_fRsi_produces_expected_ordering() -> None:
    """A 'warm surface', 'cold external wall', 'severe bridge' order correctly.

    Uses the three illustrative levels the module docstring names,
    at fixed indoor 20/60 and outdoor 5, and verifies:
        - all three see finite RH,
        - the ordering by fRsi maps to reverse ordering by RH.
    """
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=60.0)
    outdoor_t = 5.0
    rh_by_label = {}
    for label, factor in (
        ("warm surface", 0.90),
        ("cold external wall", 0.70),
        ("severe thermal bridge", 0.50),
    ):
        surface = SurfaceDescriptor(
            label=label, surface_temperature_factor=factor
        )
        rh_by_label[label] = surface_relative_humidity_pct(
            indoor_air_state=indoor_state,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        )
    # Colder -> higher RH.
    assert rh_by_label["warm surface"] < rh_by_label["cold external wall"]
    assert (
        rh_by_label["cold external wall"]
        < rh_by_label["severe thermal bridge"]
    )


# --- surface_relative_humidity alias --------------------------------------


def test_alias_is_same_function_as_the_long_name() -> None:
    """``surface_relative_humidity`` is a straight alias for the *_pct helper."""
    assert surface_relative_humidity is surface_relative_humidity_pct


def test_alias_returns_the_same_value() -> None:
    """Alias produces identical output on the same inputs."""
    indoor_state = AirState(temperature_c=20.0, relative_humidity_percent=60.0)
    surface = SurfaceDescriptor(label="wall", surface_temperature_factor=0.75)
    via_alias = surface_relative_humidity(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=5.0,
        surface=surface,
    )
    via_original = surface_relative_humidity_pct(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=5.0,
        surface=surface,
    )
    assert via_alias == via_original


# --- condensation_margin_c ------------------------------------------------


def test_condensation_margin_warm_surface_is_positive() -> None:
    """A warm interior surface (fRsi = 0.9) sits well above room dew point.

    Room 20 C / 60 %RH has a dew point around 12 C. Under a
    winter outdoor of 5 C, a fRsi-0.9 surface sits at
    5 + 0.9*(20-5) = 18.5 C, so the margin is roughly 6.5 K.
    """
    result = condensation_margin_c(
        indoor_air_state=AirState(20.0, 60.0),
        outdoor_temperature_c=5.0,
        surface=SurfaceDescriptor(
            label="warm wall", surface_temperature_factor=0.9
        ),
    )
    assert result > 0.0
    # Sanity band around the analytic value.
    assert 5.0 < result < 8.0


def test_condensation_margin_cold_surface_is_negative() -> None:
    """A severe thermal bridge (fRsi = 0.3) sits below the room dew point.

    Same room, same outdoor. T_surface = 5 + 0.3*15 = 9.5 C, well
    below the ~12 C dew point. Negative margin -> condensation
    predicted possible.
    """
    result = condensation_margin_c(
        indoor_air_state=AirState(20.0, 60.0),
        outdoor_temperature_c=5.0,
        surface=SurfaceDescriptor(
            label="severe bridge", surface_temperature_factor=0.3
        ),
    )
    assert result < 0.0


def test_condensation_margin_is_zero_at_the_dew_point_boundary() -> None:
    """Placing the surface exactly at the room's dew point -> margin = 0."""
    indoor_state = AirState(20.0, 60.0)
    outdoor_t = 5.0
    room_dew_point = dew_point_c(20.0, 60.0)
    # Pick fRsi so surface T == room dew point.
    critical_factor = (
        (room_dew_point - outdoor_t) / (20.0 - outdoor_t)
    )
    surface = SurfaceDescriptor(
        label="boundary", surface_temperature_factor=critical_factor
    )
    result = condensation_margin_c(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    )
    assert result == pytest.approx(0.0, abs=1e-9)


def test_condensation_margin_matches_surface_temp_minus_dew_point() -> None:
    """Definition check: margin equals (T_surface - dew point) exactly."""
    indoor_state = AirState(22.0, 55.0)
    outdoor_t = 3.0
    surface = SurfaceDescriptor(
        label="test", surface_temperature_factor=0.65
    )
    expected = surface_temperature_c(
        indoor_temperature_c=22.0,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    ) - dew_point_c(22.0, 55.0)
    result = condensation_margin_c(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    )
    assert result == pytest.approx(expected, rel=1e-12)


def test_condensation_margin_monotone_in_fRsi() -> None:
    """Warmer surfaces (higher fRsi) have larger margins.

    Sweeps fRsi from 0.3 to 0.9 under the same room and outdoor
    conditions; the margin must increase strictly.
    """
    indoor_state = AirState(20.0, 60.0)
    outdoor_t = 5.0
    margins = []
    for factor in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        surface = SurfaceDescriptor(
            label="test", surface_temperature_factor=factor
        )
        margins.append(
            condensation_margin_c(
                indoor_air_state=indoor_state,
                outdoor_temperature_c=outdoor_t,
                surface=surface,
            )
        )
    for earlier, later in zip(margins, margins[1:]):
        assert later > earlier


# --- SurfaceRiskResult and assess_surface ---------------------------------


def test_assess_surface_bundles_every_named_field() -> None:
    """The bundled result exposes every named quantity the docstring promises."""
    indoor_state = AirState(20.0, 60.0)
    outdoor_t = 5.0
    surface = SurfaceDescriptor(
        label="kitchen north wall behind fridge",
        surface_temperature_factor=0.7,
    )
    result = assess_surface(
        indoor_air_state=indoor_state,
        outdoor_temperature_c=outdoor_t,
        surface=surface,
    )
    assert isinstance(result, SurfaceRiskResult)
    assert result.surface_label == "kitchen north wall behind fridge"
    assert result.indoor_temperature_c == 20.0
    assert result.indoor_relative_humidity_pct == 60.0
    assert result.indoor_dew_point_c == pytest.approx(
        dew_point_c(20.0, 60.0), rel=1e-12
    )
    assert result.surface_temperature_c == pytest.approx(
        surface_temperature_c(
            indoor_temperature_c=20.0,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        ),
        rel=1e-12,
    )
    assert result.surface_relative_humidity_pct == pytest.approx(
        surface_relative_humidity(
            indoor_air_state=indoor_state,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        ),
        rel=1e-12,
    )
    assert result.condensation_margin_c == pytest.approx(
        condensation_margin_c(
            indoor_air_state=indoor_state,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        ),
        rel=1e-12,
    )


def test_assess_surface_warm_scenario_flags_no_condensation() -> None:
    """Warm surface -> positive margin, RH below 100 %."""
    result = assess_surface(
        indoor_air_state=AirState(20.0, 55.0),
        outdoor_temperature_c=8.0,
        surface=SurfaceDescriptor(
            label="warm wall", surface_temperature_factor=0.9
        ),
    )
    assert result.condensation_margin_c > 0.0
    assert result.surface_relative_humidity_pct < 100.0


def test_assess_surface_cold_scenario_flags_condensation() -> None:
    """Cold surface (fRsi 0.4) with humid room -> negative margin, RH > 100."""
    result = assess_surface(
        indoor_air_state=AirState(20.0, 70.0),
        outdoor_temperature_c=2.0,
        surface=SurfaceDescriptor(
            label="single-glazed pane", surface_temperature_factor=0.4
        ),
    )
    assert result.condensation_margin_c < 0.0
    assert result.surface_relative_humidity_pct > 100.0


def test_assess_surface_result_is_frozen() -> None:
    """SurfaceRiskResult is immutable."""
    result = assess_surface(
        indoor_air_state=AirState(20.0, 60.0),
        outdoor_temperature_c=5.0,
        surface=SurfaceDescriptor(
            label="test", surface_temperature_factor=0.75
        ),
    )
    with pytest.raises(FrozenInstanceError):
        result.condensation_margin_c = 0.0  # type: ignore[misc]


def test_assess_surface_equality_by_value() -> None:
    """Two assessments of the same inputs compare equal."""
    args = dict(
        indoor_air_state=AirState(20.0, 60.0),
        outdoor_temperature_c=5.0,
        surface=SurfaceDescriptor(
            label="test", surface_temperature_factor=0.75
        ),
    )
    assert assess_surface(**args) == assess_surface(**args)
