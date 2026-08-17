"""Tests for the thermal module.

At this slice the module only owns SI constants and unit-conversion
helpers. Tests verify:
    - the two physical constants land in the accepted engineering range,
      and can be independently reproduced by an alternative derivation
      (ideal-gas law for air density);
    - the two time / energy unit conversions round-trip exactly and
      match hand-computed reference values.

Deliberately NOT tested here: any temperature-drop, heat-loss, or
energy-balance behaviour - that math is not in the module yet, per the
scope agreed for this slice.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from dataclasses import FrozenInstanceError
from math import exp

from thermal import (
    AIR_DENSITY_KG_PER_M3,
    AIR_SPECIFIC_HEAT_J_PER_KG_K,
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    JOULES_PER_KWH,
    SECONDS_PER_HOUR,
    ThermalPrediction,
    ThermalProperties,
    airflow_rate_from_ach,
    joules_to_kwh,
    kwh_to_joules,
    predict_indoor_temperature,
    predict_thermal_response,
    ventilation_energy_loss_constant_temperature,
    ventilation_heat_loss_coefficient,
    ventilation_heat_loss_power,
)


# --- Physical constants ----------------------------------------------------


def test_air_density_matches_ideal_gas_at_reference_conditions() -> None:
    """rho = P*M/(R*T) at 20 degC, 101325 Pa reproduces AIR_DENSITY_KG_PER_M3.

    Independent derivation from the ideal gas law rather than a repeat of
    the module's own literal value. Uses the same molar mass of dry air
    (0.028966 kg/mol) and universal gas constant (8.31446 J/(mol.K)) as
    the psychrometric module; those values are cross-checked against
    ASHRAE and NIST in psychrometrics tests already.
    """
    pressure_pa = 101_325.0
    molar_mass_dry_air = 0.028966
    r_universal = 8.31446
    temperature_k = 20.0 + 273.15
    ideal_gas_density = pressure_pa * molar_mass_dry_air / (r_universal * temperature_k)
    assert AIR_DENSITY_KG_PER_M3 == pytest.approx(ideal_gas_density, rel=1e-3)


def test_air_density_is_in_residential_engineering_range() -> None:
    """Sanity check: residential air density sits ~1.15-1.25 kg/m^3.

    Catches gross unit / scale bugs (mixing kg/m^3 with g/m^3, say) that
    would slip past the ideal-gas check if someone rewrote the constant
    with a different set of reference conditions.
    """
    assert 1.15 <= AIR_DENSITY_KG_PER_M3 <= 1.25


def test_air_specific_heat_is_near_ashrae_reference() -> None:
    """cp for air near 20 degC is ~1005 J/(kg.K) (ASHRAE ch. 1).

    Loose tolerance since cp varies slightly with humidity and
    temperature; the value should not have drifted by more than a
    percent from the standard residential reference.
    """
    assert AIR_SPECIFIC_HEAT_J_PER_KG_K == pytest.approx(1005.0, rel=0.01)


# --- Time and energy unit conversions -------------------------------------


def test_seconds_per_hour_is_exactly_3600() -> None:
    """60 minutes * 60 seconds = 3600."""
    assert SECONDS_PER_HOUR == 60.0 * 60.0


def test_joules_per_kwh_is_exactly_3_6_million() -> None:
    """1 kWh = 1000 W * 3600 s = 3.6e6 J."""
    assert JOULES_PER_KWH == 1000.0 * SECONDS_PER_HOUR


def test_joules_to_kwh_matches_hand_computation() -> None:
    """3.6 MJ = 1 kWh; 7.2 MJ = 2 kWh."""
    assert joules_to_kwh(3_600_000.0) == pytest.approx(1.0, rel=1e-12)
    assert joules_to_kwh(7_200_000.0) == pytest.approx(2.0, rel=1e-12)


def test_kwh_to_joules_matches_hand_computation() -> None:
    """1 kWh = 3.6 MJ; 0.5 kWh = 1.8 MJ."""
    assert kwh_to_joules(1.0) == pytest.approx(3_600_000.0, rel=1e-12)
    assert kwh_to_joules(0.5) == pytest.approx(1_800_000.0, rel=1e-12)


@pytest.mark.parametrize("energy_j", [0.0, 1.0, 1_234.5, 3_600_000.0, 1.5e9])
def test_energy_conversions_round_trip(energy_j: float) -> None:
    """kwh_to_joules(joules_to_kwh(x)) == x to floating-point precision."""
    assert kwh_to_joules(joules_to_kwh(energy_j)) == pytest.approx(
        energy_j, rel=1e-12, abs=1e-9
    )


def test_zero_energy_converts_to_zero_in_both_directions() -> None:
    """Zero should be exactly zero after conversion, not a rounding artefact."""
    assert joules_to_kwh(0.0) == 0.0
    assert kwh_to_joules(0.0) == 0.0


# --- airflow_rate_from_ach -------------------------------------------------


def test_airflow_rate_40m3_5ach_matches_hand_computation() -> None:
    """V=40 m^3, ACH=5 h^-1 -> V_dot = 5*40/3600 = 200/3600 = 0.05555... m^3/s.

    The expected value is written as an explicit literal division so
    this test does not merely re-invoke the module's own constant.
    """
    result = airflow_rate_from_ach(room_volume_m3=40.0, ach=5.0)
    assert result == pytest.approx(200.0 / 3600.0, rel=1e-12)
    # Sanity-check the ballpark: ~55 mL/s per litre of room, or ~56 L/s.
    assert 0.055 < result < 0.056


def test_airflow_rate_hand_examples() -> None:
    """A couple of additional pen-and-paper values.

    30 m^3, 1 ACH  -> 30/3600 = 8.333e-3 m^3/s
    50 m^3, 6 ACH  -> 300/3600 = 0.08333 m^3/s
    100 m^3, 2 ACH -> 200/3600 = 0.05556 m^3/s
    """
    assert airflow_rate_from_ach(30.0, 1.0) == pytest.approx(30.0 / 3600.0, rel=1e-12)
    assert airflow_rate_from_ach(50.0, 6.0) == pytest.approx(300.0 / 3600.0, rel=1e-12)
    assert airflow_rate_from_ach(100.0, 2.0) == pytest.approx(200.0 / 3600.0, rel=1e-12)


def test_airflow_rate_scales_linearly_in_ach() -> None:
    """At fixed volume, doubling ACH doubles the flow rate."""
    single = airflow_rate_from_ach(40.0, 3.0)
    double = airflow_rate_from_ach(40.0, 6.0)
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_airflow_rate_scales_linearly_in_volume() -> None:
    """At fixed ACH, doubling the room volume doubles the flow rate."""
    small = airflow_rate_from_ach(40.0, 5.0)
    large = airflow_rate_from_ach(80.0, 5.0)
    assert large == pytest.approx(2.0 * small, rel=1e-12)


def test_airflow_rate_zero_ach_is_zero() -> None:
    """ACH = 0 -> no exchange -> flow rate is exactly zero."""
    assert airflow_rate_from_ach(40.0, 0.0) == 0.0


def test_airflow_rate_rejects_zero_volume() -> None:
    """Room volume must be strictly positive."""
    with pytest.raises(ValueError, match="room_volume_m3"):
        airflow_rate_from_ach(0.0, 5.0)


def test_airflow_rate_rejects_negative_volume() -> None:
    """Negative room volume is unphysical -> rejected."""
    with pytest.raises(ValueError, match="room_volume_m3"):
        airflow_rate_from_ach(-40.0, 5.0)


def test_airflow_rate_rejects_negative_ach() -> None:
    """Negative ACH is unphysical -> rejected."""
    with pytest.raises(ValueError, match="ach"):
        airflow_rate_from_ach(40.0, -1.0)


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("room_volume_m3", float("nan")),
        ("room_volume_m3", float("inf")),
        ("ach", float("nan")),
        ("ach", float("inf")),
    ],
)
def test_airflow_rate_rejects_non_finite_arguments(arg_name: str, bad_value: float) -> None:
    """NaN / inf on either argument is rejected."""
    kwargs = dict(room_volume_m3=40.0, ach=5.0)
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        airflow_rate_from_ach(**kwargs)


# --- ventilation_heat_loss_coefficient -------------------------------------


def test_ventilation_heat_loss_coefficient_40m3_5ach_matches_hand_computation() -> None:
    """H = rho * cp * V_dot for the canonical 40 m^3, 5 ACH case.

    V_dot = 5 * 40 / 3600 = 200/3600 = 0.05555... m^3/s
    H     = 1.204 * 1005 * 0.05555... = 67.223 W/K
    Expected value written as an explicit literal expression, not by
    re-invoking the module's own product.
    """
    result = ventilation_heat_loss_coefficient(room_volume_m3=40.0, ach=5.0)
    expected = 1.204 * 1005.0 * (200.0 / 3600.0)
    assert result == pytest.approx(expected, rel=1e-12)
    # Physical ballpark for a residential room being aggressively vented.
    assert 60.0 < result < 75.0


def test_ventilation_heat_loss_coefficient_scales_linearly() -> None:
    """H is linear in ACH, in volume, in density, and in cp separately."""
    base = ventilation_heat_loss_coefficient(40.0, 5.0)
    assert ventilation_heat_loss_coefficient(40.0, 10.0) == pytest.approx(2.0 * base, rel=1e-12)
    assert ventilation_heat_loss_coefficient(80.0, 5.0) == pytest.approx(2.0 * base, rel=1e-12)
    assert ventilation_heat_loss_coefficient(
        40.0, 5.0, air_density_kg_m3=2.0 * AIR_DENSITY_KG_PER_M3
    ) == pytest.approx(2.0 * base, rel=1e-12)
    assert ventilation_heat_loss_coefficient(
        40.0, 5.0, air_specific_heat_j_kg_k=2.0 * AIR_SPECIFIC_HEAT_J_PER_KG_K
    ) == pytest.approx(2.0 * base, rel=1e-12)


def test_ventilation_heat_loss_coefficient_zero_ach_is_zero() -> None:
    """No exchange -> no ventilation heat loss."""
    assert ventilation_heat_loss_coefficient(40.0, 0.0) == 0.0


def test_ventilation_heat_loss_coefficient_agrees_with_rule_of_thumb() -> None:
    """The 0.33 * ACH * V shortcut should be within a few percent.

    With rho=1.204 and cp=1005 the exact factor is 1.204*1005/3600 = 0.336,
    so the physical formula sits ~2 % above the 0.33 rule and ~1 % below
    the 0.34 rule. Test tolerance covers both common rounded forms.
    """
    for volume, ach in ((40.0, 5.0), (30.0, 1.0), (60.0, 3.0), (25.0, 8.0)):
        physical = ventilation_heat_loss_coefficient(volume, ach)
        rule_of_thumb_033 = 0.33 * ach * volume
        assert physical == pytest.approx(rule_of_thumb_033, rel=0.025)


def test_ventilation_heat_loss_coefficient_rule_of_thumb_prefactor() -> None:
    """Independent derivation of the rule-of-thumb prefactor.

    Divide H by (ACH * V) and confirm the result equals rho*cp/3600 to
    floating-point precision, i.e. the physical implementation exposes
    the same 0.336 constant the rule of thumb rounds down to 0.33.
    """
    prefactor = ventilation_heat_loss_coefficient(40.0, 5.0) / (5.0 * 40.0)
    assert prefactor == pytest.approx(1.204 * 1005.0 / 3600.0, rel=1e-12)
    # And it sits between the two commonly-quoted rounded values.
    assert 0.33 < prefactor < 0.34


def test_ventilation_heat_loss_coefficient_uses_supplied_air_properties() -> None:
    """Overridden rho or cp changes the result predictably."""
    default = ventilation_heat_loss_coefficient(40.0, 5.0)
    denser = ventilation_heat_loss_coefficient(40.0, 5.0, air_density_kg_m3=1.27)
    assert denser == pytest.approx(default * (1.27 / AIR_DENSITY_KG_PER_M3), rel=1e-12)


def test_ventilation_heat_loss_coefficient_units_are_watts_per_kelvin() -> None:
    """Dimensional sanity: H * dT * t must give an energy in joules.

    Compute H at 40 m^3 / 5 ACH, then simulate a 15 minute event with
    an indoor-outdoor gap of 15 K. Convert the resulting energy to kWh
    and cross-check the order of magnitude - well under 1 kWh for a
    short residential event, but noticeably above 0.1 kWh.
    """
    h_vent = ventilation_heat_loss_coefficient(40.0, 5.0)
    dt_kelvin = 15.0
    duration_seconds = 15.0 * 60.0
    energy_joules = h_vent * dt_kelvin * duration_seconds
    energy_kwh = joules_to_kwh(energy_joules)
    # H ~= 67 W/K, dT = 15 K -> ~1010 W, * 900 s -> ~908 kJ -> ~0.25 kWh.
    assert 0.2 < energy_kwh < 0.3


def test_ventilation_heat_loss_coefficient_rejects_negative_density() -> None:
    """Negative density is unphysical."""
    with pytest.raises(ValueError, match="air_density_kg_m3"):
        ventilation_heat_loss_coefficient(40.0, 5.0, air_density_kg_m3=-1.0)


def test_ventilation_heat_loss_coefficient_rejects_negative_specific_heat() -> None:
    """Negative specific heat is unphysical."""
    with pytest.raises(ValueError, match="air_specific_heat_j_kg_k"):
        ventilation_heat_loss_coefficient(40.0, 5.0, air_specific_heat_j_kg_k=-1.0)


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("room_volume_m3", -1.0),
        ("room_volume_m3", 0.0),
        ("ach", -1.0),
        ("room_volume_m3", float("nan")),
        ("ach", float("inf")),
        ("air_density_kg_m3", float("nan")),
        ("air_specific_heat_j_kg_k", float("inf")),
    ],
)
def test_ventilation_heat_loss_coefficient_propagates_validation(
    arg_name: str, bad_value: float
) -> None:
    """Invalid arguments are rejected with an error message naming the field."""
    kwargs = dict(
        room_volume_m3=40.0,
        ach=5.0,
        air_density_kg_m3=AIR_DENSITY_KG_PER_M3,
        air_specific_heat_j_kg_k=AIR_SPECIFIC_HEAT_J_PER_KG_K,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        ventilation_heat_loss_coefficient(**kwargs)


# --- ventilation_heat_loss_power -------------------------------------------


def test_ventilation_heat_loss_power_indoor_warmer_than_outdoor_is_positive() -> None:
    """Canonical winter case: indoor 20 C, outdoor 5 C, 40 m^3, 5 ACH.

    H_vent = 1.204 * 1005 * (5 * 40 / 3600) = 67.223 W/K
    dT     = 20 - 5 = 15 K
    Q_dot  = 67.223 * 15 = 1008.34 W (positive: heat leaving the room).
    Expected value assembled from literal SI components, not from the
    module's own product.
    """
    result = ventilation_heat_loss_power(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
    )
    expected = 1.204 * 1005.0 * (5.0 * 40.0 / 3600.0) * (20.0 - 5.0)
    assert result == pytest.approx(expected, rel=1e-12)
    assert result > 0.0
    # Ballpark: about a 1 kW loss for this aggressive vent.
    assert 900.0 < result < 1100.0


def test_ventilation_heat_loss_power_equal_temperatures_is_zero() -> None:
    """No driving gradient -> exactly zero heat exchange, any ACH or volume."""
    for indoor_t, outdoor_t in ((20.0, 20.0), (5.0, 5.0), (-5.0, -5.0)):
        for ach in (0.5, 5.0, 10.0):
            assert (
                ventilation_heat_loss_power(indoor_t, outdoor_t, 40.0, ach) == 0.0
            )


def test_ventilation_heat_loss_power_outdoor_warmer_is_negative() -> None:
    """Summer case: outdoor 30 C, indoor 22 C -> negative Q_dot (heat gained).

    The negative sign is preserved deliberately - no abs, no clamp.
    """
    result = ventilation_heat_loss_power(
        indoor_temperature_c=22.0,
        outdoor_temperature_c=30.0,
        room_volume_m3=40.0,
        ach=5.0,
    )
    assert result < 0.0
    # Magnitude cross-check: symmetric swap of a 15 K winter gap and an 8 K
    # summer gap gives result = -H_vent * 8 = -67.223 * 8 ~= -537.8 W.
    expected = 1.204 * 1005.0 * (5.0 * 40.0 / 3600.0) * (22.0 - 30.0)
    assert result == pytest.approx(expected, rel=1e-12)


def test_ventilation_heat_loss_power_is_antisymmetric_in_temperatures() -> None:
    """Swapping indoor and outdoor flips the sign but preserves magnitude."""
    forward = ventilation_heat_loss_power(20.0, 5.0, 40.0, 5.0)
    backward = ventilation_heat_loss_power(5.0, 20.0, 40.0, 5.0)
    assert backward == pytest.approx(-forward, rel=1e-12)


def test_ventilation_heat_loss_power_zero_ach_is_zero_regardless_of_gap() -> None:
    """A sealed room exchanges no ventilation heat, even with a huge dT."""
    for indoor_t, outdoor_t in ((20.0, 5.0), (5.0, 20.0), (-10.0, 30.0)):
        assert (
            ventilation_heat_loss_power(indoor_t, outdoor_t, 40.0, 0.0) == 0.0
        )


def test_ventilation_heat_loss_power_scales_linearly_with_temperature_gap() -> None:
    """Doubling the indoor-outdoor gap doubles the heat-loss power."""
    small_gap = ventilation_heat_loss_power(15.0, 10.0, 40.0, 5.0)
    large_gap = ventilation_heat_loss_power(20.0, 10.0, 40.0, 5.0)
    assert large_gap == pytest.approx(2.0 * small_gap, rel=1e-12)


def test_ventilation_heat_loss_power_scales_linearly_with_ach() -> None:
    """At fixed dT and V, doubling ACH doubles the heat-loss power."""
    single = ventilation_heat_loss_power(20.0, 5.0, 40.0, 2.5)
    double = ventilation_heat_loss_power(20.0, 5.0, 40.0, 5.0)
    assert double == pytest.approx(2.0 * single, rel=1e-12)


def test_ventilation_heat_loss_power_rejects_non_finite_temperatures() -> None:
    """NaN / inf temperatures are rejected."""
    with pytest.raises(ValueError, match="indoor_temperature_c"):
        ventilation_heat_loss_power(float("nan"), 5.0, 40.0, 5.0)
    with pytest.raises(ValueError, match="indoor_temperature_c"):
        ventilation_heat_loss_power(float("inf"), 5.0, 40.0, 5.0)
    with pytest.raises(ValueError, match="outdoor_temperature_c"):
        ventilation_heat_loss_power(20.0, float("nan"), 40.0, 5.0)


def test_ventilation_heat_loss_power_propagates_volume_and_ach_validation() -> None:
    """Bad V or ACH surfaces via the H_vent helper."""
    with pytest.raises(ValueError, match="room_volume_m3"):
        ventilation_heat_loss_power(20.0, 5.0, 0.0, 5.0)
    with pytest.raises(ValueError, match="ach"):
        ventilation_heat_loss_power(20.0, 5.0, 40.0, -1.0)


# --- ventilation_energy_loss_constant_temperature --------------------------
# First-order estimate: energy = initial power * duration. Real events
# cool the room during the event and lose LESS than this. Tests here
# check the constant-T arithmetic, not the eventual dynamic model.


def test_energy_loss_zero_duration_is_zero() -> None:
    """A 0-minute event costs nothing."""
    assert (
        ventilation_energy_loss_constant_temperature(
            indoor_temperature_c=20.0,
            outdoor_temperature_c=5.0,
            room_volume_m3=40.0,
            ach=5.0,
            duration_minutes=0.0,
        )
        == 0.0
    )


def test_energy_loss_zero_ach_is_zero() -> None:
    """ACH = 0 -> no exchange -> no energy leaves, any duration or gap."""
    for indoor_t, outdoor_t, minutes in (
        (20.0, 5.0, 30.0),
        (5.0, 20.0, 60.0),
        (0.0, 0.0, 0.0),
    ):
        assert (
            ventilation_energy_loss_constant_temperature(
                indoor_t, outdoor_t, 40.0, 0.0, minutes
            )
            == 0.0
        )


def test_energy_loss_equal_temperatures_is_zero() -> None:
    """No indoor-outdoor gap -> no temperature-driven energy exchange."""
    for indoor_t in (5.0, 20.0, 30.0):
        assert (
            ventilation_energy_loss_constant_temperature(
                indoor_temperature_c=indoor_t,
                outdoor_temperature_c=indoor_t,
                room_volume_m3=40.0,
                ach=5.0,
                duration_minutes=30.0,
            )
            == 0.0
        )


def test_energy_loss_five_minute_worked_example() -> None:
    """Hand-computed anchor: 40 m^3, ACH=5, indoor 20 C, outdoor 5 C, 5 min.

    H_vent = 1.204 * 1005 * (5 * 40 / 3600) = 67.223 W/K
    P_loss = H_vent * 15 K = 1008.35 W
    E      = 1008.35 W * 300 s = 302 504 J = 302 504 / 3.6e6 kWh
           ~= 0.08403 kWh
    """
    result = ventilation_energy_loss_constant_temperature(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        duration_minutes=5.0,
    )
    expected_kwh = (
        1.204 * 1005.0 * (5.0 * 40.0 / 3600.0) * (20.0 - 5.0) * (5.0 * 60.0) / 3_600_000.0
    )
    assert result == pytest.approx(expected_kwh, rel=1e-12)
    assert result == pytest.approx(0.08403, abs=1e-4)


def test_energy_loss_scales_linearly_with_duration() -> None:
    """Constant-T assumption: 10 min costs exactly twice what 5 min does."""
    five_min = ventilation_energy_loss_constant_temperature(20.0, 5.0, 40.0, 5.0, 5.0)
    ten_min = ventilation_energy_loss_constant_temperature(20.0, 5.0, 40.0, 5.0, 10.0)
    assert ten_min == pytest.approx(2.0 * five_min, rel=1e-12)


def test_energy_loss_signs_follow_temperature_gradient() -> None:
    """Positive when indoor warmer, negative when outdoor warmer, both preserved."""
    warm_out = ventilation_energy_loss_constant_temperature(20.0, 5.0, 40.0, 5.0, 15.0)
    warm_in = ventilation_energy_loss_constant_temperature(5.0, 20.0, 40.0, 5.0, 15.0)
    assert warm_out > 0.0
    assert warm_in < 0.0
    assert warm_in == pytest.approx(-warm_out, rel=1e-12)


def test_energy_loss_rejects_negative_duration() -> None:
    """Negative durations are unphysical."""
    with pytest.raises(ValueError, match="duration_minutes"):
        ventilation_energy_loss_constant_temperature(20.0, 5.0, 40.0, 5.0, -1.0)


def test_energy_loss_rejects_non_finite_duration() -> None:
    """NaN / inf duration rejected."""
    with pytest.raises(ValueError, match="duration_minutes"):
        ventilation_energy_loss_constant_temperature(
            20.0, 5.0, 40.0, 5.0, float("nan")
        )
    with pytest.raises(ValueError, match="duration_minutes"):
        ventilation_energy_loss_constant_temperature(
            20.0, 5.0, 40.0, 5.0, float("inf")
        )


def test_energy_loss_propagates_temperature_and_ach_validation() -> None:
    """Bad inputs surface via the underlying power / coefficient helpers."""
    with pytest.raises(ValueError, match="indoor_temperature_c"):
        ventilation_energy_loss_constant_temperature(
            float("nan"), 5.0, 40.0, 5.0, 15.0
        )
    with pytest.raises(ValueError, match="ach"):
        ventilation_energy_loss_constant_temperature(20.0, 5.0, 40.0, -1.0, 15.0)
    with pytest.raises(ValueError, match="room_volume_m3"):
        ventilation_energy_loss_constant_temperature(20.0, 5.0, 0.0, 5.0, 15.0)


# --- ThermalProperties -----------------------------------------------------


def test_thermal_properties_accepts_positive_capacity() -> None:
    """A plausible residential capacity constructs without raising."""
    props = ThermalProperties(effective_thermal_capacity_j_per_k=500_000.0)
    assert props.effective_thermal_capacity_j_per_k == 500_000.0


def test_thermal_properties_accepts_illustrative_default() -> None:
    """The illustrative value in the module can be constructed unmodified."""
    props = ThermalProperties(
        effective_thermal_capacity_j_per_k=ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
    )
    assert (
        props.effective_thermal_capacity_j_per_k
        == ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
    )


def test_illustrative_capacity_exceeds_naive_air_only_capacity() -> None:
    """Documentation invariant: the illustrative value must not degrade to air-only.

    If someone ever "corrects" the illustrative constant back down to
    C_air = rho * cp * V for a small room, this test fails. A room's
    real C is dominated by walls, floor, ceiling, and furniture, not by
    the ~48 kJ/K held in ~40 m^3 of air, and the illustrative value has
    to reflect that.
    """
    naive_air_capacity_40m3 = AIR_DENSITY_KG_PER_M3 * AIR_SPECIFIC_HEAT_J_PER_KG_K * 40.0
    assert (
        ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        >= 5.0 * naive_air_capacity_40m3
    )


def test_thermal_properties_rejects_zero_capacity() -> None:
    """Zero effective heat capacity is unphysical (nothing to store heat)."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        ThermalProperties(effective_thermal_capacity_j_per_k=0.0)


def test_thermal_properties_rejects_negative_capacity() -> None:
    """Negative effective heat capacity is unphysical."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        ThermalProperties(effective_thermal_capacity_j_per_k=-1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_thermal_properties_rejects_non_finite_capacity(bad_value: float) -> None:
    """NaN / inf capacity rejected."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        ThermalProperties(effective_thermal_capacity_j_per_k=bad_value)


def test_thermal_properties_is_frozen() -> None:
    """ThermalProperties is a frozen dataclass; mutation must raise."""
    props = ThermalProperties(effective_thermal_capacity_j_per_k=500_000.0)
    with pytest.raises(FrozenInstanceError):
        props.effective_thermal_capacity_j_per_k = 1_000_000.0  # type: ignore[misc]


def test_thermal_properties_equality_is_by_value() -> None:
    """Two ThermalProperties with identical fields compare equal."""
    a = ThermalProperties(effective_thermal_capacity_j_per_k=500_000.0)
    b = ThermalProperties(effective_thermal_capacity_j_per_k=500_000.0)
    c = ThermalProperties(effective_thermal_capacity_j_per_k=750_000.0)
    assert a == b
    assert a != c


def test_thermal_properties_and_room_are_independent() -> None:
    """ThermalProperties does not depend on Room from moisture.py.

    Import guard: the moisture module must NOT be required to build a
    ThermalProperties. This locks in the split of responsibilities so a
    future contributor cannot silently couple the two dataclasses.
    """
    # If ThermalProperties dragged in Room, this construction would
    # transitively execute moisture.py's imports. It's a soft guard
    # (Python caches modules), but combined with the fact that thermal.py
    # never `import`s from moisture, it documents the boundary.
    ThermalProperties(effective_thermal_capacity_j_per_k=123_456.0)


# --- predict_indoor_temperature --------------------------------------------
# Analytic solution T(t) = T_out + (T_0 - T_out) * exp(-H_vent/C_eff * t).
# Tests cover the two degenerate limits (ACH=0, t=0), the three sign
# scenarios for (T_0 - T_out), the long-time asymptote to T_out, and an
# independent one-time-constant check using tau = C_eff / H_vent that is
# physically anchored rather than a re-run of the module expression.


def _h_vent_w_per_k(room_volume_m3: float, ach: float) -> float:
    """Local re-derivation of H_vent from primitives, used to build test
    anchors independently of the module's helper."""
    return AIR_DENSITY_KG_PER_M3 * AIR_SPECIFIC_HEAT_J_PER_KG_K * (
        ach * room_volume_m3 / SECONDS_PER_HOUR
    )


def test_predict_indoor_temperature_zero_ach_leaves_temperature_unchanged() -> None:
    """No air exchange -> no ventilation heat transfer -> T is invariant."""
    assert (
        predict_indoor_temperature(
            initial_indoor_temperature_c=20.0,
            outdoor_temperature_c=5.0,
            room_volume_m3=40.0,
            ach=0.0,
            effective_thermal_capacity_j_per_k=500_000.0,
            duration_minutes=45.0,
        )
        == 20.0
    )


def test_predict_indoor_temperature_zero_duration_leaves_temperature_unchanged() -> None:
    """No time elapsed -> T(0) = T_0 by construction."""
    assert (
        predict_indoor_temperature(
            initial_indoor_temperature_c=20.0,
            outdoor_temperature_c=5.0,
            room_volume_m3=40.0,
            ach=5.0,
            effective_thermal_capacity_j_per_k=500_000.0,
            duration_minutes=0.0,
        )
        == 20.0
    )


def test_predict_indoor_temperature_indoor_warmer_than_outdoor_cools_toward_outdoor() -> None:
    """Winter vent: T falls from T_0 toward T_out, never below it."""
    result = predict_indoor_temperature(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    assert 5.0 < result < 20.0


def test_predict_indoor_temperature_indoor_cooler_than_outdoor_warms_toward_outdoor() -> None:
    """Summer vent: T rises from T_0 toward T_out, never above it."""
    result = predict_indoor_temperature(
        initial_indoor_temperature_c=22.0,
        outdoor_temperature_c=30.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=30.0,
    )
    assert 22.0 < result < 30.0


def test_predict_indoor_temperature_equal_temperatures_stays_constant() -> None:
    """No driving gradient -> T stays put for any ACH, duration, and C_eff."""
    for ach in (0.0, 5.0, 10.0):
        for minutes in (0.0, 15.0, 120.0):
            assert (
                predict_indoor_temperature(
                    initial_indoor_temperature_c=15.0,
                    outdoor_temperature_c=15.0,
                    room_volume_m3=40.0,
                    ach=ach,
                    effective_thermal_capacity_j_per_k=500_000.0,
                    duration_minutes=minutes,
                )
                == 15.0
            )


def test_predict_indoor_temperature_long_duration_asymptotes_to_outdoor() -> None:
    """As t grows large, T -> T_out.

    Pick a small C_eff and a large ACH so many time constants fit into a
    reasonable duration: tau = C_eff / H_vent = 10 000 / (rho*cp*ACH*V/3600)
    with rho*cp/3600 ~= 0.336 and ACH=10, V=40 gives H ~= 134 W/K, so
    tau ~= 75 s. 10 h = 480 tau -> exp(-480) is astronomically small.
    """
    result = predict_indoor_temperature(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=10.0,
        effective_thermal_capacity_j_per_k=10_000.0,
        duration_minutes=600.0,
    )
    assert result == pytest.approx(5.0, abs=1e-12)


def test_predict_indoor_temperature_hand_computed_anchor_at_15_minutes() -> None:
    """Reference case: 20 C indoor, 5 C outdoor, 40 m^3, 5 ACH, C=500 kJ/K, 15 min.

    Expected T assembled from literal SI components; the module MUST NOT
    just be re-invoking its own product.
    """
    result = predict_indoor_temperature(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    h_vent = 1.204 * 1005.0 * (5.0 * 40.0 / 3600.0)  # ~67.223 W/K
    tau_seconds = 500_000.0 / h_vent                 # ~7438 s ~= 124 min
    duration_seconds = 15.0 * 60.0
    expected = 5.0 + (20.0 - 5.0) * exp(-duration_seconds / tau_seconds)
    assert result == pytest.approx(expected, rel=1e-12)
    # Physical ballpark: 15 min is well under one tau (~124 min), so the
    # room has cooled a little (< 2 K) - definitely not "collapsed to
    # outdoor".
    assert 18.0 < result < 19.0


def test_predict_indoor_temperature_one_time_constant_closes_gap_by_factor_of_e() -> None:
    """After t = C_eff / H_vent seconds, the gap shrinks by exactly 1/e.

    Independent physical check anchored on the standard first-order-decay
    time constant. Would fail on a sign-of-exponent flip, or if the
    seconds/hours conversion drifted.
    """
    indoor_c, outdoor_c = 20.0, 5.0
    volume_m3, ach, c_eff = 40.0, 5.0, 500_000.0
    h_vent = _h_vent_w_per_k(volume_m3, ach)
    tau_seconds = c_eff / h_vent
    duration_minutes = tau_seconds / 60.0

    result = predict_indoor_temperature(
        initial_indoor_temperature_c=indoor_c,
        outdoor_temperature_c=outdoor_c,
        room_volume_m3=volume_m3,
        ach=ach,
        effective_thermal_capacity_j_per_k=c_eff,
        duration_minutes=duration_minutes,
    )
    expected_gap = (indoor_c - outdoor_c) / exp(1.0)
    assert (result - outdoor_c) == pytest.approx(expected_gap, rel=1e-12)


def test_predict_indoor_temperature_never_overshoots_outdoor() -> None:
    """Analytic solution is monotone toward T_out; result must stay between them."""
    for indoor_c, outdoor_c in (
        (20.0, 5.0),
        (5.0, 20.0),
        (-5.0, 15.0),
        (30.0, 22.0),
    ):
        for ach in (0.5, 5.0, 12.0):
            for c_eff in (100_000.0, 500_000.0, 2_000_000.0):
                for minutes in (1.0, 15.0, 60.0, 300.0):
                    result = predict_indoor_temperature(
                        initial_indoor_temperature_c=indoor_c,
                        outdoor_temperature_c=outdoor_c,
                        room_volume_m3=40.0,
                        ach=ach,
                        effective_thermal_capacity_j_per_k=c_eff,
                        duration_minutes=minutes,
                    )
                    assert min(indoor_c, outdoor_c) <= result <= max(indoor_c, outdoor_c)


def test_predict_indoor_temperature_larger_capacity_slows_the_response() -> None:
    """Higher C_eff -> longer time constant -> smaller change in the same interval."""
    small_c = predict_indoor_temperature(
        20.0, 5.0, 40.0, 5.0, 100_000.0, 15.0
    )
    large_c = predict_indoor_temperature(
        20.0, 5.0, 40.0, 5.0, 2_000_000.0, 15.0
    )
    # Both cool, but the higher-C room cools less.
    assert 5.0 < small_c < large_c < 20.0


def test_predict_indoor_temperature_rejects_zero_effective_capacity() -> None:
    """C_eff = 0 would make the rate H/C diverge."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        predict_indoor_temperature(20.0, 5.0, 40.0, 5.0, 0.0, 15.0)


def test_predict_indoor_temperature_rejects_negative_effective_capacity() -> None:
    """C_eff < 0 is unphysical."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        predict_indoor_temperature(20.0, 5.0, 40.0, 5.0, -1.0, 15.0)


def test_predict_indoor_temperature_rejects_negative_duration() -> None:
    """No going back in time."""
    with pytest.raises(ValueError, match="duration_minutes"):
        predict_indoor_temperature(20.0, 5.0, 40.0, 5.0, 500_000.0, -1.0)


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("initial_indoor_temperature_c", float("nan")),
        ("initial_indoor_temperature_c", float("inf")),
        ("outdoor_temperature_c", float("nan")),
        ("effective_thermal_capacity_j_per_k", float("nan")),
        ("effective_thermal_capacity_j_per_k", float("inf")),
        ("duration_minutes", float("nan")),
        ("duration_minutes", float("inf")),
    ],
)
def test_predict_indoor_temperature_rejects_non_finite_arguments(
    arg_name: str, bad_value: float
) -> None:
    """NaN / inf on any temperature, capacity, or duration is rejected."""
    kwargs = dict(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        predict_indoor_temperature(**kwargs)


def test_predict_indoor_temperature_propagates_volume_and_ach_validation() -> None:
    """Volume / ACH errors surface via ventilation_heat_loss_coefficient."""
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_indoor_temperature(20.0, 5.0, 0.0, 5.0, 500_000.0, 15.0)
    with pytest.raises(ValueError, match="ach"):
        predict_indoor_temperature(20.0, 5.0, 40.0, -1.0, 500_000.0, 15.0)


# --- predict_thermal_response (integration) --------------------------------
# The wrapper must not reimplement any physics. Every field on
# ThermalPrediction must equal what the corresponding standalone
# function returns for the same inputs.


def test_thermal_prediction_fields_delegate_to_standalone_functions() -> None:
    """Each field equals what its underlying helper returns for the same inputs."""
    kwargs = dict(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    result = predict_thermal_response(**kwargs)

    assert isinstance(result, ThermalPrediction)
    assert result.initial_temperature_c == 20.0
    assert result.outdoor_temperature_c == 5.0
    assert result.ach == 5.0
    assert result.duration_minutes == 15.0
    assert result.ventilation_heat_loss_coefficient_w_per_k == (
        ventilation_heat_loss_coefficient(room_volume_m3=40.0, ach=5.0)
    )
    assert result.initial_heat_loss_power_w == ventilation_heat_loss_power(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
    )
    assert result.final_temperature_c == predict_indoor_temperature(**kwargs)


def test_thermal_prediction_temperature_change_signs() -> None:
    """change = final - initial. Cooling event -> negative; warming -> positive."""
    cooling = predict_thermal_response(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    assert cooling.final_temperature_c < cooling.initial_temperature_c
    assert cooling.temperature_change_c < 0.0
    assert cooling.temperature_change_c == pytest.approx(
        cooling.final_temperature_c - cooling.initial_temperature_c, rel=1e-12, abs=1e-15
    )
    assert cooling.initial_heat_loss_power_w > 0.0

    warming = predict_thermal_response(
        initial_indoor_temperature_c=22.0,
        outdoor_temperature_c=30.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=30.0,
    )
    assert warming.final_temperature_c > warming.initial_temperature_c
    assert warming.temperature_change_c > 0.0
    assert warming.initial_heat_loss_power_w < 0.0


def test_thermal_prediction_zero_gradient_gives_zero_change() -> None:
    """T_in == T_out -> no change, zero H*dT power, but non-zero H_vent."""
    result = predict_thermal_response(
        initial_indoor_temperature_c=15.0,
        outdoor_temperature_c=15.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    assert result.final_temperature_c == 15.0
    assert result.temperature_change_c == 0.0
    assert result.initial_heat_loss_power_w == 0.0
    assert result.ventilation_heat_loss_coefficient_w_per_k > 0.0


def test_thermal_prediction_zero_ach_leaves_room_unchanged() -> None:
    """ACH = 0 -> H_vent = 0 -> no change and no initial power."""
    result = predict_thermal_response(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=0.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    assert result.final_temperature_c == 20.0
    assert result.temperature_change_c == 0.0
    assert result.ventilation_heat_loss_coefficient_w_per_k == 0.0
    assert result.initial_heat_loss_power_w == 0.0


def test_thermal_prediction_zero_duration_leaves_room_unchanged() -> None:
    """duration = 0 -> final = initial, but H_vent and initial power still reported."""
    result = predict_thermal_response(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=0.0,
    )
    assert result.final_temperature_c == 20.0
    assert result.temperature_change_c == 0.0
    assert result.ventilation_heat_loss_coefficient_w_per_k > 0.0
    assert result.initial_heat_loss_power_w > 0.0


def test_thermal_prediction_hand_computed_anchor_15_minutes() -> None:
    """Reference: 40 m^3, ACH=5, C=500 kJ/K, 20 C indoor, 5 C outdoor, 15 min.

    Expected values assembled from literal SI components so this test
    exercises the whole wiring end-to-end, not the module's own products.
    """
    result = predict_thermal_response(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    h_vent = 1.204 * 1005.0 * (5.0 * 40.0 / 3600.0)          # ~67.223 W/K
    initial_power = h_vent * (20.0 - 5.0)                     # ~1008.35 W
    duration_seconds = 15.0 * 60.0
    tau_seconds = 500_000.0 / h_vent
    expected_final = 5.0 + (20.0 - 5.0) * exp(-duration_seconds / tau_seconds)
    assert result.ventilation_heat_loss_coefficient_w_per_k == pytest.approx(
        h_vent, rel=1e-12
    )
    assert result.initial_heat_loss_power_w == pytest.approx(initial_power, rel=1e-12)
    assert result.final_temperature_c == pytest.approx(expected_final, rel=1e-12)
    assert result.temperature_change_c == pytest.approx(
        expected_final - 20.0, rel=1e-12
    )


def test_thermal_prediction_initial_power_matches_h_vent_times_gap() -> None:
    """Cross-consistency: initial_power = H_vent * (T_in - T_out)."""
    result = predict_thermal_response(
        initial_indoor_temperature_c=22.0,
        outdoor_temperature_c=8.0,
        room_volume_m3=30.0,
        ach=3.5,
        effective_thermal_capacity_j_per_k=750_000.0,
        duration_minutes=25.0,
    )
    reconstructed_power = result.ventilation_heat_loss_coefficient_w_per_k * (
        result.initial_temperature_c - result.outdoor_temperature_c
    )
    assert result.initial_heat_loss_power_w == pytest.approx(
        reconstructed_power, rel=1e-12
    )


def test_thermal_prediction_is_frozen() -> None:
    """ThermalPrediction is a frozen dataclass; mutation raises."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    with pytest.raises(FrozenInstanceError):
        result.final_temperature_c = 0.0  # type: ignore[misc]


def test_thermal_prediction_equality_is_by_value() -> None:
    """Two predictions with identical inputs compare equal."""
    a = predict_thermal_response(20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0)
    b = predict_thermal_response(20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0)
    c = predict_thermal_response(21.0, 5.0, 40.0, 5.0, 500_000.0, 15.0)
    assert a == b
    assert a != c


@pytest.mark.parametrize(
    "arg_name,bad_value,error_fragment",
    [
        # The wrapper renames initial_indoor_temperature_c -> indoor_temperature_c
        # when calling ventilation_heat_loss_power, so the raised message
        # names the helper's parameter.
        ("initial_indoor_temperature_c", float("nan"), "indoor_temperature_c"),
        ("outdoor_temperature_c", float("inf"), "outdoor_temperature_c"),
        ("room_volume_m3", 0.0, "room_volume_m3"),
        ("ach", -1.0, "ach"),
        ("effective_thermal_capacity_j_per_k", 0.0, "effective_thermal_capacity_j_per_k"),
        ("effective_thermal_capacity_j_per_k", float("inf"), "effective_thermal_capacity_j_per_k"),
        ("duration_minutes", -1.0, "duration_minutes"),
        ("duration_minutes", float("nan"), "duration_minutes"),
    ],
)
def test_predict_thermal_response_propagates_validation(
    arg_name: str, bad_value: float, error_fragment: str
) -> None:
    """Invalid arguments bubble up from the underlying physics helpers."""
    kwargs = dict(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=error_fragment):
        predict_thermal_response(**kwargs)


# --- Dynamic energy removed ------------------------------------------------


def test_energy_removed_j_equals_c_eff_times_temperature_drop() -> None:
    """energy_removed_j = C_eff * (T_0 - T_f) by definition."""
    result = predict_thermal_response(
        initial_indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=500_000.0,
        duration_minutes=15.0,
    )
    assert result.energy_removed_j == pytest.approx(
        500_000.0
        * (result.initial_temperature_c - result.final_temperature_c),
        rel=1e-12,
        abs=1e-9,
    )


def test_energy_removed_kwh_matches_joules_via_helper() -> None:
    """The kWh field must equal joules_to_kwh(energy_removed_j) exactly."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    assert result.energy_removed_kwh == pytest.approx(
        joules_to_kwh(result.energy_removed_j), rel=1e-12
    )


def test_energy_removed_hand_computed_15_minutes() -> None:
    """Anchor: T_0=20, T_out=5, V=40, ACH=5, C=500k, 15 min.

    From the earlier temperature test: T_f ~= 18.290 C, drop ~= 1.710 K.
    Expected energy = 500_000 * 1.710 = 855 000 J = 0.2375 kWh.
    """
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    h_vent = 1.204 * 1005.0 * (5.0 * 40.0 / 3600.0)
    tau_s = 500_000.0 / h_vent
    expected_final_c = 5.0 + 15.0 * exp(-(15.0 * 60.0) / tau_s)
    expected_energy_j = 500_000.0 * (20.0 - expected_final_c)
    assert result.energy_removed_j == pytest.approx(expected_energy_j, rel=1e-12)
    # Ballpark: ~855 kJ = ~0.24 kWh for this canonical winter vent.
    assert 800_000.0 < result.energy_removed_j < 900_000.0
    assert 0.22 < result.energy_removed_kwh < 0.25


def test_energy_removed_positive_when_cooling() -> None:
    """Cooling event -> positive energy removed."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    assert result.temperature_change_c < 0.0
    assert result.energy_removed_j > 0.0
    assert result.energy_removed_kwh > 0.0


def test_energy_removed_negative_when_warming() -> None:
    """Warm outdoor air heats the room -> negative energy_removed (energy added)."""
    result = predict_thermal_response(
        22.0, 30.0, 40.0, 5.0, 500_000.0, 30.0
    )
    assert result.temperature_change_c > 0.0
    assert result.energy_removed_j < 0.0
    assert result.energy_removed_kwh < 0.0


def test_energy_removed_zero_at_zero_gradient_or_zero_ach_or_zero_duration() -> None:
    """Any of the three degenerate conditions gives zero dynamic energy."""
    equal_temps = predict_thermal_response(15.0, 15.0, 40.0, 5.0, 500_000.0, 15.0)
    zero_ach = predict_thermal_response(20.0, 5.0, 40.0, 0.0, 500_000.0, 15.0)
    zero_duration = predict_thermal_response(20.0, 5.0, 40.0, 5.0, 500_000.0, 0.0)
    for r in (equal_temps, zero_ach, zero_duration):
        assert r.energy_removed_j == 0.0
        assert r.energy_removed_kwh == 0.0


def test_energy_removed_conservation_by_numerical_integration() -> None:
    """Cross-check: C_eff * (T_0 - T_f) equals the time-integral of H*dT.

    Under the simplified model where ventilation is the only heat-transfer
    mechanism, energy conservation on the lumped room mass gives:

        C_eff * (T_0 - T_f) = integral_0^t H_vent * (T(tau) - T_out) d(tau)

    The left-hand side is what ``energy_removed_j`` computes. The right-hand
    side is what the "keep integrating instantaneous power" mental model
    would produce. This test numerically integrates the RHS using the
    analytic T(tau) via ``predict_indoor_temperature`` and confirms the
    two agree to a tight tolerance. The derivations are algebraically
    different, so agreement is a genuine energy-conservation check, not
    a tautology.
    """
    initial_c, outdoor_c = 20.0, 5.0
    volume_m3, ach, c_eff = 40.0, 5.0, 500_000.0
    duration_minutes = 30.0

    result = predict_thermal_response(
        initial_c, outdoor_c, volume_m3, ach, c_eff, duration_minutes
    )

    # Simpson's rule with 1000 sub-intervals across [0, duration_seconds].
    duration_s = duration_minutes * 60.0
    n = 1000  # must be even
    h = duration_s / n
    h_vent = result.ventilation_heat_loss_coefficient_w_per_k

    def instantaneous_power_w(t_seconds: float) -> float:
        # T at time t in minutes -> instantaneous power H*(T - T_out).
        t_minutes = t_seconds / 60.0
        temperature_at_t_c = predict_indoor_temperature(
            initial_indoor_temperature_c=initial_c,
            outdoor_temperature_c=outdoor_c,
            room_volume_m3=volume_m3,
            ach=ach,
            effective_thermal_capacity_j_per_k=c_eff,
            duration_minutes=t_minutes,
        )
        return h_vent * (temperature_at_t_c - outdoor_c)

    total = instantaneous_power_w(0.0) + instantaneous_power_w(duration_s)
    for i in range(1, n):
        t = i * h
        coefficient = 4.0 if i % 2 == 1 else 2.0
        total += coefficient * instantaneous_power_w(t)
    integrated_energy_j = (h / 3.0) * total

    # Simpson's rule with n=1000 hits the analytic exponential decay to
    # far more precision than this test needs; require 1 part in 10^6.
    assert result.energy_removed_j == pytest.approx(integrated_energy_j, rel=1e-6)


# --- Constant-T comparison retained ---------------------------------------


def test_constant_temperature_energy_matches_standalone_helper() -> None:
    """The bundled constant-T fields must equal the standalone-function output."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    standalone_kwh = ventilation_energy_loss_constant_temperature(
        indoor_temperature_c=20.0,
        outdoor_temperature_c=5.0,
        room_volume_m3=40.0,
        ach=5.0,
        duration_minutes=15.0,
    )
    assert result.energy_removed_constant_temperature_kwh == standalone_kwh
    assert result.energy_removed_constant_temperature_j == pytest.approx(
        kwh_to_joules(standalone_kwh), rel=1e-12
    )


def test_dynamic_energy_below_constant_temperature_for_cooling_event() -> None:
    """Constant-T over-estimates a cooling event because it ignores the
    shrinking gap: dynamic energy must be strictly smaller."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 15.0
    )
    assert 0.0 < result.energy_removed_kwh
    assert result.energy_removed_kwh < result.energy_removed_constant_temperature_kwh


def test_dynamic_energy_above_constant_temperature_magnitude_for_warming_event() -> None:
    """Symmetric flip: for a warming event both quantities are negative,
    the dynamic value is closer to zero (smaller |gap growth| assumption
    -> constant-T over-estimates the magnitude of the energy added)."""
    result = predict_thermal_response(
        22.0, 30.0, 40.0, 5.0, 500_000.0, 30.0
    )
    assert result.energy_removed_kwh < 0.0
    assert result.energy_removed_constant_temperature_kwh < 0.0
    # Dynamic magnitude is smaller -> dynamic > constant_T (less negative).
    assert result.energy_removed_kwh > result.energy_removed_constant_temperature_kwh


def test_short_event_dynamic_and_constant_temperature_are_close() -> None:
    """For events short compared with tau, the gap barely shrinks and the
    constant-T approximation is a good one; the two energies should agree
    to within a few percent."""
    result = predict_thermal_response(
        20.0, 5.0, 40.0, 5.0, 500_000.0, 5.0
    )
    # tau ~= 124 min; 5 min is well under one time constant.
    assert result.energy_removed_kwh == pytest.approx(
        result.energy_removed_constant_temperature_kwh, rel=0.05
    )
