"""Tests for the shared ``experiments/_metrics.py`` helpers.

These helpers live under ``experiments/`` because they are
interpretation utilities, not physics. They still deserve test
coverage because every experiment that reports an engineering metric
(g/kWh, incremental g/kWh, relative-to-baseline percentages) routes
through them.
"""

import sys
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import pandas as pd
import pytest

from _metrics import NEAR_ZERO_TOLERANCE, safe_ratio


def test_scalar_ratio_returns_float_when_denominator_is_meaningful() -> None:
    """Ordinary case: 12 g / 0.5 kWh -> 24 g/kWh."""
    assert safe_ratio(12.0, 0.5) == pytest.approx(24.0, rel=1e-12)


def test_scalar_ratio_returns_nan_when_denominator_is_zero() -> None:
    """0.0 denominator -> NaN, not inf."""
    result = safe_ratio(5.0, 0.0)
    assert isnan(result)


def test_scalar_ratio_returns_nan_for_near_zero_denominator() -> None:
    """Denominators inside NEAR_ZERO_TOLERANCE are treated as zero.

    Catches the "T_in - T_out ~ 1e-6 K" trap the reviewer flagged:
    tiny non-zero energy denominators pass a strict != 0 check but
    yield ratios dominated by numerical noise.
    """
    near_zero = 0.5 * NEAR_ZERO_TOLERANCE
    assert isnan(safe_ratio(5.0, near_zero))


def test_scalar_ratio_preserves_sign() -> None:
    """No abs / no clamp; positive-over-negative gives a negative result."""
    assert safe_ratio(5.0, -2.5) == pytest.approx(-2.0, rel=1e-12)
    assert safe_ratio(-5.0, 2.5) == pytest.approx(-2.0, rel=1e-12)
    assert safe_ratio(-5.0, -2.5) == pytest.approx(2.0, rel=1e-12)


def test_series_ratio_returns_series_with_nan_in_zero_rows() -> None:
    """Series input -> Series output; zero-denominator rows are NaN."""
    numerator = pd.Series([10.0, 20.0, 30.0])
    denominator = pd.Series([2.0, 0.0, 5.0])
    result = safe_ratio(numerator, denominator)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == pytest.approx(5.0, rel=1e-12)
    assert isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(6.0, rel=1e-12)


def test_series_ratio_treats_near_zero_rows_as_undefined() -> None:
    """Series version applies the same tolerance as the scalar path."""
    numerator = pd.Series([10.0, 10.0])
    denominator = pd.Series([1.0, 0.5 * NEAR_ZERO_TOLERANCE])
    result = safe_ratio(numerator, denominator)
    assert result.iloc[0] == pytest.approx(10.0, rel=1e-12)
    assert isnan(result.iloc[1])


def test_ratio_survives_realistic_residential_values() -> None:
    """Sanity-check with the canonical example.

    Reference case: water_removed_g = 179.86, energy_removed_kwh =
    0.2374. Both are well above NEAR_ZERO_TOLERANCE, so the guard
    should let them through and produce ~757.5 g/kWh.
    """
    assert safe_ratio(179.86, 0.2374) == pytest.approx(757.6, rel=1e-3)
