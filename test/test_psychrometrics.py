"""Reference-value and consistency tests for the psychrometrics module.

Reference numbers are taken from the ASHRAE Handbook of Fundamentals
psychrometric tables and cross-checked against a standard psychrometric chart.
Tolerances are loose enough to absorb the small residual error of the Magnus
saturation fit versus ASHRAE, and tight enough to catch a coding mistake.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from psychrometrics import (
    DEFAULT_ATM_PRESSURE_PA,
    DRYING_POTENTIAL_CATEGORY_HIGH,
    DRYING_POTENTIAL_CATEGORY_LOW,
    DRYING_POTENTIAL_CATEGORY_MODERATE,
    DRYING_POTENTIAL_CATEGORY_NONE,
    DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3,
    DRYING_POTENTIAL_LOW_THRESHOLD_G_M3,
    DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3,
    MAX_VALID_TEMPERATURE_C,
    MIN_VALID_TEMPERATURE_C,
    AirState,
    DryingPotential,
    absolute_humidity_g_per_m3,
    calculate_drying_potential,
    dew_point_c,
    humidity_ratio_kg_per_kg,
    relative_humidity_from_absolute_humidity,
    saturation_vapour_pressure,
    vapour_pressure,
)

# Reference saturation vapour pressures over a flat water surface, taken from
# the WMO / Wexler saturation table (WMO Guide to Meteorological Instruments
# and Methods of Observation, WMO-No. 8, Annex 4.B). These come from a
# different fit than the Magnus form in the module, so agreement is a genuine
# cross-check rather than a tautology.
WMO_P_SAT_PA = {
    0.0: 611.2,
    5.0: 872.5,
    20.0: 2338.8,
    30.0: 4245.5,
}


@pytest.mark.parametrize("temperature_c,expected_pa", sorted(WMO_P_SAT_PA.items()))
def test_saturation_vapour_pressure_matches_wmo_reference(
    temperature_c: float, expected_pa: float
) -> None:
    """P_sat agrees with the WMO / Wexler table to within the Magnus fit error.

    Alduchov & Eskridge (1996) quote <0.4 % maximum error over -40 to +50 C;
    a 0.5 % relative tolerance sits just above that so a coding bug (wrong
    constant, wrong sign, unit mix-up) would still fail the assertion.
    """
    assert saturation_vapour_pressure(temperature_c) == pytest.approx(
        expected_pa, rel=0.005
    )


def test_saturation_vapour_pressure_monotonic_in_temperature() -> None:
    """P_sat is a strictly increasing function of temperature."""
    values = [saturation_vapour_pressure(t) for t in (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_saturation_vapour_pressure_rejects_nan() -> None:
    """NaN input is rejected rather than silently propagated."""
    with pytest.raises(ValueError):
        saturation_vapour_pressure(float("nan"))


def test_saturation_vapour_pressure_rejects_infinity() -> None:
    """Infinite input is rejected rather than returning inf."""
    with pytest.raises(ValueError):
        saturation_vapour_pressure(float("inf"))


def test_saturation_vapour_pressure_rejects_temperature_below_range() -> None:
    """Sub-arctic temperatures are outside the residential-POC contract."""
    with pytest.raises(ValueError):
        saturation_vapour_pressure(MIN_VALID_TEMPERATURE_C - 1.0)


def test_saturation_vapour_pressure_rejects_temperature_above_range() -> None:
    """Oven-hot temperatures are outside the residential-POC contract."""
    with pytest.raises(ValueError):
        saturation_vapour_pressure(MAX_VALID_TEMPERATURE_C + 1.0)


def test_saturation_vapour_pressure_accepts_range_endpoints() -> None:
    """Endpoints of the accepted range must not raise."""
    saturation_vapour_pressure(MIN_VALID_TEMPERATURE_C)
    saturation_vapour_pressure(MAX_VALID_TEMPERATURE_C)


# Reference vapour pressures at 20 degC derived from the WMO / Wexler
# saturation value at 20 degC (2338.8 Pa) via the definition of RH. Using an
# external saturation reference keeps this from being a re-implementation
# check; the module-under-test uses the Magnus fit for P_sat, which agrees
# with the WMO value to within ~0.4 %, so a 0.5 % relative tolerance
# distinguishes correct behaviour from a coding mistake.
_WMO_P_SAT_20C_PA = 2338.8
VAPOUR_PRESSURE_REFERENCE_AT_20C = {
    0.0:   0.00 * _WMO_P_SAT_20C_PA / 100.0,
    50.0:  50.0 * _WMO_P_SAT_20C_PA / 100.0,
    70.0:  70.0 * _WMO_P_SAT_20C_PA / 100.0,
    85.0:  85.0 * _WMO_P_SAT_20C_PA / 100.0,
    100.0: 100.0 * _WMO_P_SAT_20C_PA / 100.0,
}


@pytest.mark.parametrize(
    "relative_humidity_pct,expected_pa",
    sorted(VAPOUR_PRESSURE_REFERENCE_AT_20C.items()),
)
def test_vapour_pressure_at_20c_matches_wmo_reference(
    relative_humidity_pct: float, expected_pa: float
) -> None:
    """P_v at 20 degC scales linearly with RH against the WMO P_sat value."""
    result = vapour_pressure(20.0, relative_humidity_pct)
    if relative_humidity_pct == 0.0:
        assert result == 0.0
    else:
        assert result == pytest.approx(expected_pa, rel=0.005)


def test_vapour_pressure_at_100pct_equals_saturation() -> None:
    """At 100 %RH the vapour pressure equals P_sat within floating point tolerance."""
    for t in (0.0, 5.0, 20.0, 30.0):
        assert vapour_pressure(t, 100.0) == pytest.approx(
            saturation_vapour_pressure(t), rel=1e-12, abs=1e-9
        )


def test_vapour_pressure_at_0pct_is_zero() -> None:
    """RH of exactly 0 % gives a vapour pressure of exactly zero."""
    assert vapour_pressure(20.0, 0.0) == 0.0


def test_vapour_pressure_rejects_negative_rh() -> None:
    """Negative RH is physically meaningless and rejected."""
    with pytest.raises(ValueError):
        vapour_pressure(20.0, -0.1)


def test_vapour_pressure_rejects_rh_above_100() -> None:
    """Supersaturated RH inputs are rejected."""
    with pytest.raises(ValueError):
        vapour_pressure(20.0, 100.1)


def test_vapour_pressure_rejects_nan_rh() -> None:
    """NaN RH is rejected rather than silently propagated."""
    with pytest.raises(ValueError):
        vapour_pressure(20.0, float("nan"))


def test_vapour_pressure_propagates_temperature_validation() -> None:
    """Out-of-range temperature bubbles up from saturation_vapour_pressure."""
    with pytest.raises(ValueError):
        vapour_pressure(MAX_VALID_TEMPERATURE_C + 1.0, 50.0)


def test_dew_point_equals_temperature_at_full_saturation() -> None:
    """At 100 %RH the dew point equals ambient temperature within tolerance."""
    for t in (5.0, 15.0, 22.0, 30.0):
        # The inversion is algebraic on the same Magnus constants, so the
        # round-trip should be exact up to floating-point round-off.
        assert dew_point_c(t, 100.0) == pytest.approx(t, abs=1e-9)


def test_dew_point_below_ambient_when_unsaturated() -> None:
    """Any RH < 100 % implies dew point strictly below ambient."""
    for rh in (10.0, 40.0, 80.0, 99.0):
        assert dew_point_c(22.0, rh) < 22.0


# Reference dew-point values for typical residential conditions, cross-checked
# against standard dew-point tables and independent psychrometric calculators.
# Not derived from our own Magnus fit, so agreement is a genuine check.
DEW_POINT_REFERENCE_POINTS = [
    (5.0, 80.0, 1.9),
    (10.0, 60.0, 2.6),
    (20.0, 50.0, 9.3),
    (20.0, 70.0, 14.4),
    (22.0, 55.0, 12.7),
    (25.0, 60.0, 16.7),
    (30.0, 40.0, 14.9),
]


@pytest.mark.parametrize(
    "temperature_c,relative_humidity_pct,expected_dew_point_c",
    DEW_POINT_REFERENCE_POINTS,
)
def test_dew_point_matches_reference_tables(
    temperature_c: float, relative_humidity_pct: float, expected_dew_point_c: float
) -> None:
    """Dew point agrees with independent reference tables within 0.3 degC."""
    assert dew_point_c(temperature_c, relative_humidity_pct) == pytest.approx(
        expected_dew_point_c, abs=0.3
    )


def test_dew_point_round_trips_through_saturation_vapour_pressure() -> None:
    """P_sat(dew_point(T, RH)) == P_v(T, RH) to floating-point precision.

    This is the exactness claim of the algebraic inversion: the same Magnus
    constants used for the forward curve are used for the inverse, so the
    round-trip must be tight.
    """
    for t, rh in ((5.0, 40.0), (12.0, 85.0), (20.0, 70.0), (30.0, 55.0)):
        assert saturation_vapour_pressure(dew_point_c(t, rh)) == pytest.approx(
            vapour_pressure(t, rh), rel=1e-12, abs=1e-9
        )


def test_dew_point_monotonic_in_rh() -> None:
    """At fixed temperature, dew point rises with relative humidity."""
    values = [dew_point_c(22.0, rh) for rh in (10.0, 30.0, 50.0, 70.0, 90.0, 99.0)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_dew_point_at_zero_rh_raises() -> None:
    """Dew point is undefined at 0 %RH; documented behaviour is to raise."""
    with pytest.raises(ValueError):
        dew_point_c(20.0, 0.0)


def test_dew_point_raises_if_result_falls_outside_module_validity_range() -> None:
    """Very low RH pushes T_d below MIN_VALID_TEMPERATURE_C -> ValueError.

    The water-phase Magnus branch we use is not appropriate that far from
    residential conditions; the module rejects rather than silently return
    a value it would refuse to accept on re-entry.
    """
    # At 20 C, RH = 0.01 % gives P_v ~= 0.23 Pa and T_d ~= -75 C, well below
    # the -50 C residential floor. This must raise, and the message should
    # cite the offending inputs.
    with pytest.raises(ValueError, match="residential validity range"):
        dew_point_c(20.0, 0.01)


def test_dew_point_propagates_rh_validation() -> None:
    """Out-of-range RH bubbles up from vapour_pressure."""
    with pytest.raises(ValueError):
        dew_point_c(20.0, -5.0)
    with pytest.raises(ValueError):
        dew_point_c(20.0, 150.0)


def test_humidity_ratio_monotonic_in_rh() -> None:
    """W is strictly increasing in RH at fixed temperature and pressure."""
    values = [humidity_ratio_kg_per_kg(22.0, rh) for rh in (10.0, 30.0, 50.0, 70.0, 90.0)]
    assert all(a < b for a, b in zip(values, values[1:]))


# Reference humidity ratios read from a standard ASHRAE psychrometric chart
# at 101325 Pa. These are published tabulations of W(T, RH) that do not use
# our Magnus fit for P_sat, so agreement is a genuine cross-check.
HUMIDITY_RATIO_ASHRAE_REFERENCE_POINTS = [
    (20.0, 60.0, 0.00872),
    (25.0, 50.0, 0.00988),
    (30.0, 80.0, 0.02159),
]


@pytest.mark.parametrize(
    "temperature_c,relative_humidity_pct,expected_kg_per_kg",
    HUMIDITY_RATIO_ASHRAE_REFERENCE_POINTS,
)
def test_humidity_ratio_matches_ashrae_chart(
    temperature_c: float, relative_humidity_pct: float, expected_kg_per_kg: float
) -> None:
    """W matches ASHRAE chart values to within ~1 % of reading."""
    assert humidity_ratio_kg_per_kg(
        temperature_c, relative_humidity_pct
    ) == pytest.approx(expected_kg_per_kg, rel=0.01)


def test_humidity_ratio_default_pressure_matches_explicit_sea_level() -> None:
    """Omitting pressure_pa uses the 101325 Pa sea-level standard."""
    assert humidity_ratio_kg_per_kg(22.0, 55.0) == humidity_ratio_kg_per_kg(
        22.0, 55.0, DEFAULT_ATM_PRESSURE_PA
    )


def test_humidity_ratio_decreases_with_higher_atmospheric_pressure() -> None:
    """At fixed T and RH, raising total pressure lowers the humidity ratio.

    W = 0.621945 * P_v / (P - P_v). At fixed T and RH, P_v is unchanged, so
    increasing P grows the denominator and shrinks W. This is the classic
    altitude correction and is the key reason humidity ratio depends on
    atmospheric pressure while relative humidity does not.
    """
    sea_level_pa = 101325.0
    high_altitude_pa = 85000.0  # roughly 1500 m
    higher_than_sea_pa = 105000.0

    w_sea = humidity_ratio_kg_per_kg(22.0, 60.0, sea_level_pa)
    w_high_altitude = humidity_ratio_kg_per_kg(22.0, 60.0, high_altitude_pa)
    w_high_pressure = humidity_ratio_kg_per_kg(22.0, 60.0, higher_than_sea_pa)

    assert w_high_altitude > w_sea > w_high_pressure


def test_humidity_ratio_rejects_pressure_at_or_below_vapour_pressure() -> None:
    """Total pressure must strictly exceed the water-vapour partial pressure."""
    p_v = vapour_pressure(20.0, 50.0)
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, p_v)
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, p_v * 0.5)


def test_humidity_ratio_rejects_non_positive_pressure() -> None:
    """Zero or negative pressure is unphysical and rejected."""
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, 0.0)
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, -101325.0)


def test_humidity_ratio_rejects_non_finite_pressure() -> None:
    """NaN / infinite pressure is rejected rather than silently propagated."""
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, float("nan"))
    with pytest.raises(ValueError):
        humidity_ratio_kg_per_kg(20.0, 50.0, float("inf"))


def test_humidity_ratio_is_zero_at_zero_rh() -> None:
    """No water vapour -> zero humidity ratio at any temperature or pressure."""
    for t, p in ((0.0, 101325.0), (20.0, 85000.0), (30.0, 105000.0)):
        assert humidity_ratio_kg_per_kg(t, 0.0, p) == 0.0


def test_absolute_humidity_reference_point() -> None:
    """AH(20 C, 50 %RH) ~= 8.65 g/m^3 from the ASHRAE chart."""
    assert absolute_humidity_g_per_m3(20.0, 50.0) == pytest.approx(8.65, abs=0.1)


# Independent reference: saturation absolute humidity (100 %RH) values from
# standard meteorological / psychrometric tables. These are widely tabulated
# separately from the Magnus fit used inside the module, so agreement is a
# genuine cross-check. At fixed T the ideal-gas relation makes AH linear in
# RH, so we scale each row by (RH / 100) to produce residential-condition
# targets.
_SATURATION_AH_G_PER_M3 = {
    5.0: 6.80,
    10.0: 9.40,
    20.0: 17.30,
    30.0: 30.37,
}

# Representative residential (T_c, RH_pct) points paired with their expected
# absolute humidity in g/m^3.
ABSOLUTE_HUMIDITY_REFERENCE_POINTS = [
    (5.0, 40.0, _SATURATION_AH_G_PER_M3[5.0] * 0.40),
    (5.0, 85.0, _SATURATION_AH_G_PER_M3[5.0] * 0.85),
    (10.0, 60.0, _SATURATION_AH_G_PER_M3[10.0] * 0.60),
    (20.0, 50.0, _SATURATION_AH_G_PER_M3[20.0] * 0.50),
    (20.0, 70.0, _SATURATION_AH_G_PER_M3[20.0] * 0.70),
    (30.0, 60.0, _SATURATION_AH_G_PER_M3[30.0] * 0.60),
]


@pytest.mark.parametrize(
    "temperature_c,relative_humidity_pct,expected_g_per_m3",
    ABSOLUTE_HUMIDITY_REFERENCE_POINTS,
)
def test_absolute_humidity_matches_standard_table(
    temperature_c: float, relative_humidity_pct: float, expected_g_per_m3: float
) -> None:
    """AH matches the standard saturation table (scaled by RH) within ~1.5 %.

    The tolerance covers the small differences between the Magnus fit used
    internally and the tabulated saturation values, without being loose
    enough to hide a coding error (wrong molar mass, wrong gas constant,
    missing Kelvin conversion, missing kg->g factor).
    """
    assert absolute_humidity_g_per_m3(
        temperature_c, relative_humidity_pct
    ) == pytest.approx(expected_g_per_m3, rel=0.015)


def test_absolute_humidity_indoor_carries_more_water_than_higher_rh_outdoor() -> None:
    """20 C / 70 %RH indoors holds substantially more water than 5 C / 85 %RH out.

    The higher outdoor RH is misleading: cool air simply cannot carry as much
    water in absolute terms. This is the whole point of comparing absolute
    humidity (or humidity ratio) rather than RH when deciding whether to
    ventilate.
    """
    indoor_ah = absolute_humidity_g_per_m3(20.0, 70.0)
    outdoor_ah = absolute_humidity_g_per_m3(5.0, 85.0)
    assert indoor_ah > outdoor_ah
    # Indoor should carry roughly twice as much water; require at least 1.5x
    # so a coding regression that flattens the temperature dependence would
    # be caught, but tabulated saturation-value scatter cannot fail the test.
    assert indoor_ah / outdoor_ah > 1.5
    # And sanity-check the absolute magnitudes against the standard table.
    assert indoor_ah == pytest.approx(_SATURATION_AH_G_PER_M3[20.0] * 0.70, rel=0.02)
    assert outdoor_ah == pytest.approx(_SATURATION_AH_G_PER_M3[5.0] * 0.85, rel=0.02)


def test_absolute_humidity_at_zero_rh_is_zero() -> None:
    """No water vapour -> zero absolute humidity, at any temperature."""
    for t in (0.0, 20.0, 30.0):
        assert absolute_humidity_g_per_m3(t, 0.0) == 0.0


def test_absolute_humidity_scales_linearly_with_rh() -> None:
    """At fixed T, AH is proportional to RH (P_v is linear in RH)."""
    ah_50 = absolute_humidity_g_per_m3(20.0, 50.0)
    ah_100 = absolute_humidity_g_per_m3(20.0, 100.0)
    assert ah_100 == pytest.approx(2.0 * ah_50, rel=1e-12)


# --- AirState dataclass -----------------------------------------------------
# The class must not duplicate math: every property should equal the
# corresponding standalone function called on the same inputs. Tests below
# verify that equivalence rather than re-testing the underlying physics,
# which is already covered above.

_SAMPLE_STATES = [
    (20.0, 70.0, DEFAULT_ATM_PRESSURE_PA),
    (5.0, 85.0, DEFAULT_ATM_PRESSURE_PA),
    (24.0, 60.0, DEFAULT_ATM_PRESSURE_PA),
    (30.0, 40.0, 95000.0),
    (10.0, 55.0, 87000.0),
]


@pytest.mark.parametrize("temperature_c,relative_humidity_percent,pressure_pa", _SAMPLE_STATES)
def test_airstate_properties_delegate_to_standalone_functions(
    temperature_c: float, relative_humidity_percent: float, pressure_pa: float
) -> None:
    """Each AirState property equals the standalone-function value exactly."""
    state = AirState(
        temperature_c=temperature_c,
        relative_humidity_percent=relative_humidity_percent,
        atmospheric_pressure_pa=pressure_pa,
    )
    assert state.saturation_vapour_pressure == saturation_vapour_pressure(temperature_c)
    assert state.vapour_pressure == vapour_pressure(temperature_c, relative_humidity_percent)
    assert state.absolute_humidity == absolute_humidity_g_per_m3(
        temperature_c, relative_humidity_percent
    )
    assert state.humidity_ratio == humidity_ratio_kg_per_kg(
        temperature_c, relative_humidity_percent, pressure_pa
    )
    assert state.dew_point == dew_point_c(temperature_c, relative_humidity_percent)


def test_airstate_pressure_defaults_to_sea_level() -> None:
    """Omitting pressure gives the module's sea-level default."""
    state = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
    assert state.atmospheric_pressure_pa == DEFAULT_ATM_PRESSURE_PA


def test_airstate_humidity_ratio_uses_own_pressure() -> None:
    """Non-default pressure on the state propagates into humidity_ratio."""
    sea_level = AirState(20.0, 70.0)
    high_altitude = AirState(20.0, 70.0, 85000.0)
    assert high_altitude.humidity_ratio > sea_level.humidity_ratio


def test_airstate_is_frozen() -> None:
    """Frozen dataclass: mutating a field must raise FrozenInstanceError."""
    state = AirState(20.0, 70.0)
    with pytest.raises(FrozenInstanceError):
        state.temperature_c = 25.0  # type: ignore[misc]


def test_airstate_equality_by_value() -> None:
    """Two AirState values with equal fields compare equal."""
    a = AirState(20.0, 70.0)
    b = AirState(20.0, 70.0)
    c = AirState(21.0, 70.0)
    assert a == b
    assert a != c


def test_airstate_matches_user_example_usage() -> None:
    """The docstring example must actually work as advertised.

    >>> indoor = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
    >>> indoor.absolute_humidity
    >>> indoor.dew_point
    """
    indoor = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
    # Physical sanity anchors: at 20 C, 70 %RH the standard chart gives
    # AH ~= 12.1 g/m^3 and T_d ~= 14.4 degC.
    assert indoor.absolute_humidity == pytest.approx(12.1, abs=0.3)
    assert indoor.dew_point == pytest.approx(14.4, abs=0.3)


def test_airstate_propagates_dew_point_zero_rh_error() -> None:
    """dew_point on a 0 %RH state raises, matching the standalone function."""
    state = AirState(20.0, 0.0)
    with pytest.raises(ValueError):
        state.dew_point


# --- Drying potential -------------------------------------------------------

def test_drying_potential_result_bundles_both_absolute_humidities() -> None:
    """The result exposes both inputs' absolute humidities alongside the diff."""
    indoor = AirState(20.0, 60.0)
    outdoor = AirState(10.0, 50.0)
    result = calculate_drying_potential(indoor, outdoor)
    assert result.indoor_absolute_humidity_g_m3 == indoor.absolute_humidity
    assert result.outdoor_absolute_humidity_g_m3 == outdoor.absolute_humidity
    assert result.difference_g_m3 == pytest.approx(
        indoor.absolute_humidity - outdoor.absolute_humidity, rel=1e-12
    )


def test_drying_potential_result_is_frozen_dataclass() -> None:
    """DryingPotential is a frozen dataclass so results can't be mutated."""
    result = calculate_drying_potential(AirState(20.0, 60.0), AirState(10.0, 50.0))
    assert isinstance(result, DryingPotential)
    with pytest.raises(FrozenInstanceError):
        result.difference_g_m3 = 0.0  # type: ignore[misc]


def test_drying_potential_identical_conditions_gives_zero_and_none() -> None:
    """Identical indoor/outdoor states -> zero difference -> NONE category."""
    state = AirState(20.0, 60.0)
    result = calculate_drying_potential(state, state)
    assert result.difference_g_m3 == pytest.approx(0.0, abs=1e-12)
    assert result.category == DRYING_POTENTIAL_CATEGORY_NONE


def test_drying_potential_outdoor_wetter_than_indoor_is_negative_and_none() -> None:
    """When outdoor air carries more water per m^3 the difference is negative."""
    indoor = AirState(10.0, 40.0)
    outdoor = AirState(25.0, 80.0)
    result = calculate_drying_potential(indoor, outdoor)
    assert result.difference_g_m3 < 0.0
    assert result.category == DRYING_POTENTIAL_CATEGORY_NONE


def test_drying_potential_slightly_drier_outdoor_gives_low_category() -> None:
    """A small positive difference (~0.5 g/m^3) lands in the LOW band."""
    # 20 C / 55 %RH ~= 9.51 g/m^3 vs 20 C / 50 %RH ~= 8.65 g/m^3  -> ~0.86 g/m^3
    indoor = AirState(20.0, 55.0)
    outdoor = AirState(20.0, 50.0)
    result = calculate_drying_potential(indoor, outdoor)
    assert result.difference_g_m3 > DRYING_POTENTIAL_LOW_THRESHOLD_G_M3
    assert result.difference_g_m3 <= DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3
    assert result.category == DRYING_POTENTIAL_CATEGORY_LOW


def test_drying_potential_moderately_drier_outdoor_gives_moderate_category() -> None:
    """A ~1-3 g/m^3 positive difference lands in the MODERATE band."""
    # 22 C / 65 %RH indoors vs 12 C / 60 %RH outdoors ~= ~12.6 g/m^3 - ~6.4 g/m^3
    # sits comfortably in the MODERATE-to-HIGH region; choose gentler values
    # for MODERATE.
    indoor = AirState(20.0, 65.0)   # ~= 11.24 g/m^3
    outdoor = AirState(15.0, 60.0)  # ~=  7.66 g/m^3   -> diff ~= 3.58 -> HIGH
    # Actually the above pair falls in HIGH; use a closer pair for MODERATE:
    indoor = AirState(20.0, 55.0)   # ~= 9.51
    outdoor = AirState(15.0, 60.0)  # ~= 7.66          -> diff ~= 1.85 g/m^3
    result = calculate_drying_potential(indoor, outdoor)
    assert result.difference_g_m3 > DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3
    assert result.difference_g_m3 <= DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3
    assert result.category == DRYING_POTENTIAL_CATEGORY_MODERATE


def test_drying_potential_substantially_drier_outdoor_gives_high_category() -> None:
    """A large positive difference (>3 g/m^3) lands in the HIGH band."""
    # 24 C / 70 %RH ~= 15.3 g/m^3 vs 5 C / 60 %RH ~= 4.1 g/m^3 -> diff ~= 11 g/m^3
    indoor = AirState(24.0, 70.0)
    outdoor = AirState(5.0, 60.0)
    result = calculate_drying_potential(indoor, outdoor)
    assert result.difference_g_m3 > DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3
    assert result.category == DRYING_POTENTIAL_CATEGORY_HIGH


def test_drying_potential_category_bands_across_the_full_range() -> None:
    """Each category is reached for a difference safely inside its band.

    Uses a synthetic outdoor state at 0 %RH (absolute humidity == 0) so the
    difference equals the indoor absolute humidity directly; then walks a
    ladder of RH values that produce differences well inside each band's
    interior, away from the thresholds themselves (boundary behaviour is
    covered by the two dedicated tests below).
    """
    outdoor = AirState(20.0, 0.0)
    assert outdoor.absolute_humidity == 0.0

    ah_saturated = AirState(20.0, 100.0).absolute_humidity
    cases = [
        (0.5, DRYING_POTENTIAL_CATEGORY_LOW),        # inside 0-1
        (2.0, DRYING_POTENTIAL_CATEGORY_MODERATE),   # inside 1-3
        (5.0, DRYING_POTENTIAL_CATEGORY_HIGH),       # above 3
    ]
    for target_diff_g_m3, expected_category in cases:
        rh = 100.0 * target_diff_g_m3 / ah_saturated
        indoor = AirState(20.0, rh)
        result = calculate_drying_potential(indoor, outdoor)
        assert result.category == expected_category, (
            f"diff={result.difference_g_m3:.4f} g/m^3 classified as "
            f"{result.category}, expected {expected_category}"
        )


def test_drying_potential_just_below_and_above_low_threshold() -> None:
    """Differences straddling the LOW threshold produce NONE and LOW."""
    outdoor = AirState(20.0, 0.0)
    ah_saturated = AirState(20.0, 100.0).absolute_humidity
    # Just above 0 g/m^3 -> LOW; a small negative difference -> NONE.
    indoor_positive = AirState(20.0, 100.0 * 0.05 / ah_saturated)
    result_positive = calculate_drying_potential(indoor_positive, outdoor)
    assert result_positive.difference_g_m3 > 0.0
    assert result_positive.category == DRYING_POTENTIAL_CATEGORY_LOW

    # Outdoor drier than indoor's 0 %RH isn't possible (can't go below 0),
    # so realise a negative difference by swapping which state is drier.
    indoor_zero = AirState(20.0, 0.0)
    outdoor_wet = AirState(20.0, 30.0)
    result_negative = calculate_drying_potential(indoor_zero, outdoor_wet)
    assert result_negative.difference_g_m3 < 0.0
    assert result_negative.category == DRYING_POTENTIAL_CATEGORY_NONE


def test_drying_potential_just_below_and_above_moderate_threshold() -> None:
    """Differences straddling the MODERATE threshold produce LOW and MODERATE."""
    outdoor = AirState(20.0, 0.0)
    ah_saturated = AirState(20.0, 100.0).absolute_humidity
    low_side = AirState(
        20.0,
        100.0 * (DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3 - 0.1) / ah_saturated,
    )
    high_side = AirState(
        20.0,
        100.0 * (DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3 + 0.1) / ah_saturated,
    )
    assert calculate_drying_potential(low_side, outdoor).category == DRYING_POTENTIAL_CATEGORY_LOW
    assert (
        calculate_drying_potential(high_side, outdoor).category
        == DRYING_POTENTIAL_CATEGORY_MODERATE
    )


def test_drying_potential_just_below_and_above_high_threshold() -> None:
    """Differences straddling the HIGH threshold produce MODERATE and HIGH."""
    outdoor = AirState(20.0, 0.0)
    ah_saturated = AirState(20.0, 100.0).absolute_humidity
    low_side = AirState(
        20.0,
        100.0 * (DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3 - 0.1) / ah_saturated,
    )
    high_side = AirState(
        20.0,
        100.0 * (DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3 + 0.1) / ah_saturated,
    )
    assert (
        calculate_drying_potential(low_side, outdoor).category
        == DRYING_POTENTIAL_CATEGORY_MODERATE
    )
    assert (
        calculate_drying_potential(high_side, outdoor).category
        == DRYING_POTENTIAL_CATEGORY_HIGH
    )


def test_drying_potential_thresholds_are_ordered_and_named() -> None:
    """The three named thresholds must be strictly ordered."""
    assert (
        DRYING_POTENTIAL_LOW_THRESHOLD_G_M3
        < DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3
        < DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3
    )


def test_drying_potential_categories_are_the_documented_labels() -> None:
    """Category labels are exactly the four strings the docstring promises."""
    labels = {
        DRYING_POTENTIAL_CATEGORY_NONE,
        DRYING_POTENTIAL_CATEGORY_LOW,
        DRYING_POTENTIAL_CATEGORY_MODERATE,
        DRYING_POTENTIAL_CATEGORY_HIGH,
    }
    assert labels == {"NONE", "LOW", "MODERATE", "HIGH"}


def test_drying_potential_sign_agrees_with_absolute_humidity_ordering() -> None:
    """The sign of difference_g_m3 mirrors indoor vs outdoor AH ordering."""
    cases = [
        (AirState(22.0, 60.0), AirState(10.0, 50.0)),  # outdoor drier
        (AirState(10.0, 40.0), AirState(25.0, 80.0)),  # outdoor wetter
        (AirState(20.0, 60.0), AirState(20.0, 60.0)),  # tied
    ]
    for indoor, outdoor in cases:
        result = calculate_drying_potential(indoor, outdoor)
        expected_sign_positive = indoor.absolute_humidity > outdoor.absolute_humidity
        assert (result.difference_g_m3 > 0) == expected_sign_positive


# --- relative_humidity_from_absolute_humidity (inverse of AH) --------------


@pytest.mark.parametrize(
    "temperature_c,relative_humidity_pct",
    [
        (0.0, 30.0),
        (5.0, 85.0),
        (10.0, 50.0),
        (20.0, 30.0),
        (20.0, 70.0),
        (20.0, 100.0),
        (25.0, 45.0),
        (30.0, 60.0),
        (30.0, 80.0),
    ],
)
def test_ah_rh_round_trip_recovers_input_rh(
    temperature_c: float, relative_humidity_pct: float
) -> None:
    """Forward-then-inverse recovers the original RH to floating-point precision.

    This is the load-bearing test for the new function: any drift here
    means the inverse is not the mathematical inverse of the forward
    calculation. Uses a spread of residential temperatures and RH
    values including the 100 %RH boundary.
    """
    absolute_humidity = absolute_humidity_g_per_m3(temperature_c, relative_humidity_pct)
    recovered_rh = relative_humidity_from_absolute_humidity(
        temperature_c, absolute_humidity
    )
    assert recovered_rh == pytest.approx(
        relative_humidity_pct, rel=1e-12, abs=1e-12
    )


def test_rh_from_ah_zero_absolute_humidity_gives_zero_rh() -> None:
    """No water vapour -> RH = 0 % at any temperature."""
    for t in (0.0, 10.0, 20.0, 30.0):
        assert relative_humidity_from_absolute_humidity(t, 0.0) == 0.0


def test_rh_from_ah_at_saturation_returns_100_percent() -> None:
    """AH at the saturation value for T -> RH = 100 %.

    Independent construction: build the saturation AH from the module's
    saturation vapour pressure and the ideal-gas relation directly (not
    via absolute_humidity_g_per_m3), then invert. This double-checks
    both the forward and inverse paths against the underlying physics.
    """
    from psychrometrics import (  # local import to keep the top of the file tidy
        G_PER_KG,
        M_WATER,
        R_UNIVERSAL,
        ZERO_CELSIUS_IN_KELVIN,
    )

    for temperature_c in (0.0, 10.0, 20.0, 30.0):
        p_sat_pa = saturation_vapour_pressure(temperature_c)
        temperature_k = temperature_c + ZERO_CELSIUS_IN_KELVIN
        saturation_ah = p_sat_pa * M_WATER / (R_UNIVERSAL * temperature_k) * G_PER_KG
        recovered_rh = relative_humidity_from_absolute_humidity(
            temperature_c, saturation_ah
        )
        assert recovered_rh == pytest.approx(100.0, rel=1e-12, abs=1e-12)


def test_rh_from_ah_returns_value_above_100_for_supersaturated_input() -> None:
    """Documented behaviour: no clamping, no exception, raw arithmetic returned.

    Take a warm-room AH (30 C, 90 %RH ~= 27.3 g/m^3) and ask what RH
    that would be at 5 C, where the saturation AH is only ~6.8 g/m^3.
    The result should be well above 100 % and should not raise.
    """
    warm_ah = absolute_humidity_g_per_m3(30.0, 90.0)
    inferred_rh_at_cold = relative_humidity_from_absolute_humidity(5.0, warm_ah)
    assert inferred_rh_at_cold > 100.0


def test_rh_from_ah_monotonic_in_absolute_humidity_at_fixed_t() -> None:
    """At fixed T, RH is a strictly increasing function of AH."""
    rhs = [
        relative_humidity_from_absolute_humidity(22.0, ah)
        for ah in (0.5, 2.0, 5.0, 10.0, 15.0)
    ]
    assert all(a < b for a, b in zip(rhs, rhs[1:]))


def test_rh_from_ah_matches_definition_p_v_over_p_sat() -> None:
    """RH = 100 * P_v / P_sat(T) reconstructed from independent primitives.

    For a specific (T, RH), compute P_v via ``vapour_pressure`` and
    P_sat via ``saturation_vapour_pressure`` (both are already
    validated elsewhere in this suite), take their ratio, and confirm
    it equals the round-tripped RH. Independent of the AH pathway, so
    a bug in the inverse-AH arithmetic would fail this even if the
    round-trip test still passed by coincidence.
    """
    temperature_c, rh_pct = 20.0, 65.0
    absolute_humidity = absolute_humidity_g_per_m3(temperature_c, rh_pct)
    recovered_rh = relative_humidity_from_absolute_humidity(
        temperature_c, absolute_humidity
    )
    expected_rh = 100.0 * vapour_pressure(temperature_c, rh_pct) / (
        saturation_vapour_pressure(temperature_c)
    )
    assert recovered_rh == pytest.approx(expected_rh, rel=1e-12)


def test_rh_from_ah_rejects_negative_absolute_humidity() -> None:
    """Negative AH is unphysical."""
    with pytest.raises(ValueError, match="absolute_humidity_g_m3"):
        relative_humidity_from_absolute_humidity(20.0, -0.5)


def test_rh_from_ah_rejects_non_finite_absolute_humidity() -> None:
    """NaN / inf AH rejected."""
    with pytest.raises(ValueError, match="absolute_humidity_g_m3"):
        relative_humidity_from_absolute_humidity(20.0, float("nan"))
    with pytest.raises(ValueError, match="absolute_humidity_g_m3"):
        relative_humidity_from_absolute_humidity(20.0, float("inf"))


def test_rh_from_ah_propagates_temperature_validation() -> None:
    """Out-of-range temperature bubbles up from saturation_vapour_pressure."""
    with pytest.raises(ValueError):
        relative_humidity_from_absolute_humidity(
            MAX_VALID_TEMPERATURE_C + 1.0, 5.0
        )
    with pytest.raises(ValueError):
        relative_humidity_from_absolute_humidity(
            MIN_VALID_TEMPERATURE_C - 1.0, 5.0
        )
