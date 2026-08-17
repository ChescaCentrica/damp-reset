"""Shared metric helpers used across the experiments directory.

Kept in ``experiments/`` (not exported from the library) because these
helpers are reporting/interpretation utilities layered on top of the
physics, not physics themselves. Every experiment that computes an
engineering metric like ``water_removed / energy_removed`` should
route through ``safe_ratio`` here so the "when is the ratio
undefined?" rule is stated in exactly one place.

Design decision on the "near-zero denominator" case:
    A strict ``denominator != 0.0`` guard lets a denominator of, say,
    1e-9 kWh slip through and produce a ratio dominated by
    floating-point noise. This module rejects any denominator whose
    ABSOLUTE VALUE is below ``NEAR_ZERO_TOLERANCE`` and returns NaN
    for those rows. The chosen tolerance is small enough that a
    physically meaningful residential ventilation energy (>= 1e-6 kWh
    ~= 3.6 mJ; the reference 15-minute event is ~0.24 kWh, seven
    orders of magnitude larger) passes, but a numerical-noise value
    from a near-zero temperature or AH gap is caught.
"""

from math import nan
from typing import Union

import pandas as pd

# Denominator magnitudes below this threshold are treated as
# effectively zero. Chosen well above float64 ULP for typical
# residential-scale values (~0.01 to 1 kWh, ~1 to 300 g, ~0.1 to
# 10 K) but well below any physically meaningful reading.
NEAR_ZERO_TOLERANCE: float = 1e-12


def safe_ratio(
    numerator: Union[float, pd.Series],
    denominator: Union[float, pd.Series],
) -> Union[float, pd.Series]:
    """Return numerator / denominator, or NaN if the denominator is near zero.

    Works on scalars or pandas Series. For Series inputs, the returned
    Series has the same index and NaN in every row whose denominator
    absolute value is below ``NEAR_ZERO_TOLERANCE``. For scalar inputs,
    returns a bare float.

    NaN is used because efficiency metrics like ``water_removed /
    energy_removed`` are genuinely undefined when there is no thermal
    (or temperature) penalty to divide by. Silently returning +/-inf
    or a noise-dominated large number would mislead downstream
    reporting.
    """
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        num_series = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
        den_series = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
        result = pd.Series(nan, index=num_series.index, dtype="float64")
        far_from_zero = den_series.abs() > NEAR_ZERO_TOLERANCE
        result.loc[far_from_zero] = (
            num_series.loc[far_from_zero] / den_series.loc[far_from_zero]
        )
        return result
    if abs(denominator) <= NEAR_ZERO_TOLERANCE:
        return nan
    return numerator / denominator
