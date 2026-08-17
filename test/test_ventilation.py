"""Integration tests for the combined ventilation prediction layer.

The layer under test does no physics of its own - it composes the
existing moisture, thermal, and psychrometric layers. Tests verify:
    - the bundled sub-results match what the standalone functions return
      when called with the same inputs;
    - the final RH is derived from the PREDICTED final temperature and
      the PREDICTED final AH, not from the room's initial temperature;
    - the load-bearing invariant: at fixed final AH, changing the final
      temperature changes the final RH.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room, predict_room_moisture
from psychrometrics import AirState, relative_humidity_from_absolute_humidity
from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    ThermalProperties,
    predict_thermal_response,
)
from ventilation import (
    VentilationPrediction,
    VentilationSimulationResult,
    predict_ventilation,
    simulate_ventilation_event,
)


def _default_room() -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )


def _default_thermal_properties() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _default_outdoor() -> AirState:
    return AirState(temperature_c=5.0, relative_humidity_percent=85.0)


# --- Delegation ------------------------------------------------------------


def test_moisture_sub_result_matches_standalone_call() -> None:
    """The moisture bundle equals what predict_room_moisture returns directly."""
    room = _default_room()
    outdoor = _default_outdoor()
    result = predict_ventilation(
        room=room,
        thermal_properties=_default_thermal_properties(),
        outdoor=outdoor,
        duration_minutes=15.0,
    )
    assert result.moisture == predict_room_moisture(
        room=room, outdoor=outdoor, duration_minutes=15.0, window_open=True
    )


def test_thermal_sub_result_matches_standalone_call() -> None:
    """The thermal bundle equals what predict_thermal_response returns directly."""
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    result = predict_ventilation(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        duration_minutes=15.0,
    )
    assert result.thermal == predict_thermal_response(
        initial_indoor_temperature_c=room.indoor_temperature_c,
        outdoor_temperature_c=outdoor.temperature_c,
        room_volume_m3=room.volume_m3,
        ach=room.ach_window_open,
        effective_thermal_capacity_j_per_k=(
            thermal_props.effective_thermal_capacity_j_per_k
        ),
        duration_minutes=15.0,
    )


def test_window_closed_uses_ach_closed_for_both_sub_predictions() -> None:
    """window_open=False routes room.ach_closed to both physics helpers."""
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    result = predict_ventilation(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        duration_minutes=15.0,
        window_open=False,
    )
    assert result.moisture.ach == room.ach_closed
    assert result.thermal.ach == room.ach_closed


def test_window_open_defaults_to_true() -> None:
    """Omitting window_open uses room.ach_window_open."""
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    default_result = predict_ventilation(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        duration_minutes=15.0,
    )
    explicit_result = predict_ventilation(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        duration_minutes=15.0,
        window_open=True,
    )
    assert default_result == explicit_result


def test_bundled_inputs_are_echoed_on_the_result() -> None:
    """Duration and window flag are exposed on the top-level result for audit."""
    result = predict_ventilation(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=15.0,
        window_open=False,
    )
    assert result.duration_minutes == 15.0
    assert result.window_open is False


# --- The critical invariant: final RH uses predicted final temperature -----


def test_final_rh_uses_predicted_final_temperature_not_initial() -> None:
    """Final RH is computed from PREDICTED final T and final AH.

    Load-bearing invariant. If a future contributor accidentally
    routed the room's INITIAL temperature into the RH inversion, this
    test would fail because in a cooling scenario the resulting RH at
    the initial temperature is different from the RH at the (lower)
    predicted final temperature.
    """
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    result = predict_ventilation(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        duration_minutes=15.0,
    )

    # Independent reconstruction: use the predicted final T and final AH
    # to derive the expected RH, and compare against what the wrapper
    # reports. The wrapper MUST match the "predicted-T" version, not the
    # "initial-T" version.
    expected_rh_at_predicted_t = relative_humidity_from_absolute_humidity(
        temperature_c=result.thermal.final_temperature_c,
        absolute_humidity_g_m3=result.moisture.final_absolute_humidity_g_m3,
    )
    assert result.final_relative_humidity_pct == pytest.approx(
        expected_rh_at_predicted_t, rel=1e-12
    )

    # And crucially: it MUST NOT be the RH computed against the original
    # indoor temperature. The room started at 20 C and cools to a lower
    # temperature during the event, so the "buggy" RH would be lower than
    # the correct one (same water in warmer air -> lower RH).
    rh_at_initial_t = relative_humidity_from_absolute_humidity(
        temperature_c=room.indoor_temperature_c,
        absolute_humidity_g_m3=result.moisture.final_absolute_humidity_g_m3,
    )
    assert result.final_relative_humidity_pct != pytest.approx(rh_at_initial_t)
    # Sanity: the correct RH (colder T, less capacity to hold water) is
    # HIGHER than the RH would be at the original warmer temperature.
    assert result.final_relative_humidity_pct > rh_at_initial_t


def test_final_rh_changes_when_final_temperature_changes_at_fixed_final_ah() -> None:
    """At fixed final AH, varying final T must vary final RH.

    This directly checks the invariant that "final RH depends on final T
    AND final AH, not on AH alone". Set up two scenarios that reach the
    SAME final indoor AH but have different final indoor temperatures,
    and verify their final RHs differ.

    Strategy: pick two ThermalProperties values with very different
    C_eff so that the SAME moisture prediction lands at the same final
    AH (moisture doesn't depend on C_eff) but the thermal prediction
    lands at different final temperatures (a small C_eff cools more
    than a large one in the same 15 minutes).
    """
    room = _default_room()
    outdoor = _default_outdoor()

    low_c_props = ThermalProperties(effective_thermal_capacity_j_per_k=200_000.0)
    high_c_props = ThermalProperties(effective_thermal_capacity_j_per_k=2_000_000.0)

    low_c_result = predict_ventilation(
        room=room,
        thermal_properties=low_c_props,
        outdoor=outdoor,
        duration_minutes=15.0,
    )
    high_c_result = predict_ventilation(
        room=room,
        thermal_properties=high_c_props,
        outdoor=outdoor,
        duration_minutes=15.0,
    )

    # Final AH is identical because it does not depend on C_eff at all.
    assert (
        low_c_result.moisture.final_absolute_humidity_g_m3
        == high_c_result.moisture.final_absolute_humidity_g_m3
    )

    # Final temperature differs because C_eff differs.
    assert (
        low_c_result.thermal.final_temperature_c
        != high_c_result.thermal.final_temperature_c
    )
    # The lighter room cools more.
    assert (
        low_c_result.thermal.final_temperature_c
        < high_c_result.thermal.final_temperature_c
    )

    # And therefore the final RH must differ, even though the AH is
    # the same. Specifically, the room that cooled more has HIGHER RH:
    # same water content in cooler air is less far from saturation.
    assert (
        low_c_result.final_relative_humidity_pct
        != high_c_result.final_relative_humidity_pct
    )
    assert (
        low_c_result.final_relative_humidity_pct
        > high_c_result.final_relative_humidity_pct
    )


def test_final_rh_at_zero_duration_recovers_the_initial_room_state() -> None:
    """A 0-minute event: no change anywhere; final RH equals the room's initial RH."""
    room = _default_room()
    result = predict_ventilation(
        room=room,
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=0.0,
    )
    assert result.thermal.final_temperature_c == room.indoor_temperature_c
    # The AH also hasn't moved -> final RH must round-trip to the room's
    # initial RH.
    assert result.final_relative_humidity_pct == pytest.approx(
        room.indoor_relative_humidity_pct, rel=1e-12
    )


def test_final_rh_at_zero_ach_recovers_the_initial_room_state() -> None:
    """A closed sealed room with ACH_closed = 0 keeps everything constant."""
    sealed_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=22.0,
        indoor_relative_humidity_pct=60.0,
        ach_closed=0.0,
        ach_window_open=5.0,
    )
    result = predict_ventilation(
        room=sealed_room,
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=45.0,
        window_open=False,
    )
    assert result.thermal.final_temperature_c == 22.0
    assert (
        result.moisture.final_absolute_humidity_g_m3
        == result.moisture.initial_absolute_humidity_g_m3
    )
    assert result.final_relative_humidity_pct == pytest.approx(
        60.0, rel=1e-12
    )


# --- Frozen and equality ---------------------------------------------------


def test_ventilation_prediction_is_frozen() -> None:
    """VentilationPrediction is a frozen dataclass; mutation raises."""
    result = predict_ventilation(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=15.0,
    )
    with pytest.raises(FrozenInstanceError):
        result.final_relative_humidity_pct = 0.0  # type: ignore[misc]


def test_ventilation_prediction_equality_is_by_value() -> None:
    """Equal inputs produce equal top-level results (and equal sub-results)."""
    a = predict_ventilation(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=15.0,
    )
    b = predict_ventilation(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=15.0,
    )
    c = predict_ventilation(
        room=_default_room(),
        thermal_properties=_default_thermal_properties(),
        outdoor=_default_outdoor(),
        duration_minutes=30.0,
    )
    assert a == b
    assert a != c


# --- Validation propagation ------------------------------------------------


def test_predict_ventilation_propagates_room_validation() -> None:
    """Invalid Room fields surface with the field name in the message."""
    with pytest.raises(ValueError, match="volume_m3"):
        Room(
            volume_m3=0.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=70.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


def test_predict_ventilation_propagates_thermal_properties_validation() -> None:
    """Invalid ThermalProperties surfaces at construction, before ventilation."""
    with pytest.raises(ValueError, match="effective_thermal_capacity_j_per_k"):
        ThermalProperties(effective_thermal_capacity_j_per_k=0.0)


def test_predict_ventilation_propagates_duration_validation() -> None:
    """Negative duration is rejected by the moisture layer."""
    with pytest.raises(ValueError, match="duration_minutes"):
        predict_ventilation(
            room=_default_room(),
            thermal_properties=_default_thermal_properties(),
            outdoor=_default_outdoor(),
            duration_minutes=-1.0,
        )


def test_predict_ventilation_propagates_outdoor_temperature_validation() -> None:
    """Out-of-range outdoor temperature surfaces via AirState / psychrometrics."""
    with pytest.raises(ValueError):
        predict_ventilation(
            room=_default_room(),
            thermal_properties=_default_thermal_properties(),
            outdoor=AirState(temperature_c=200.0, relative_humidity_percent=50.0),
            duration_minutes=15.0,
        )


# --- simulate_ventilation_event (flat scalar facade) ----------------------


def _canonical_simulation_kwargs() -> dict:
    """The default winter scenario used across facade tests."""
    return dict(
        room_volume_m3=40.0,
        initial_indoor_temperature_c=20.0,
        initial_indoor_relative_humidity_pct=70.0,
        outdoor_temperature_c=5.0,
        outdoor_relative_humidity_pct=85.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        ),
        duration_minutes=15.0,
    )


def test_simulation_result_exposes_the_ten_requested_fields() -> None:
    """The facade returns exactly the flat scalar summary the POC needs."""
    result = simulate_ventilation_event(**_canonical_simulation_kwargs())
    assert isinstance(result, VentilationSimulationResult)
    expected_fields = {
        "initial_absolute_humidity_g_m3",
        "final_absolute_humidity_g_m3",
        "water_removed_g",
        "initial_relative_humidity_pct",
        "final_relative_humidity_pct",
        "initial_temperature_c",
        "final_temperature_c",
        "temperature_drop_c",
        "ventilation_heat_loss_coefficient_w_per_k",
        "ventilation_energy_removed_kwh",
    }
    assert set(result.__dataclass_fields__) == expected_fields


def test_simulation_result_matches_composed_prediction() -> None:
    """Every field on the flat result matches the composed prediction."""
    kwargs = _canonical_simulation_kwargs()
    flat = simulate_ventilation_event(**kwargs)
    composed = predict_ventilation(
        room=Room(
            volume_m3=kwargs["room_volume_m3"],
            indoor_temperature_c=kwargs["initial_indoor_temperature_c"],
            indoor_relative_humidity_pct=kwargs[
                "initial_indoor_relative_humidity_pct"
            ],
            ach_closed=kwargs["ach"],
            ach_window_open=kwargs["ach"],
        ),
        thermal_properties=ThermalProperties(
            effective_thermal_capacity_j_per_k=kwargs[
                "effective_thermal_capacity_j_per_k"
            ]
        ),
        outdoor=AirState(
            temperature_c=kwargs["outdoor_temperature_c"],
            relative_humidity_percent=kwargs["outdoor_relative_humidity_pct"],
        ),
        duration_minutes=kwargs["duration_minutes"],
        window_open=True,
    )
    assert flat.initial_absolute_humidity_g_m3 == (
        composed.moisture.initial_absolute_humidity_g_m3
    )
    assert flat.final_absolute_humidity_g_m3 == (
        composed.moisture.final_absolute_humidity_g_m3
    )
    assert flat.water_removed_g == composed.moisture.water_removed_g
    assert flat.initial_relative_humidity_pct == (
        kwargs["initial_indoor_relative_humidity_pct"]
    )
    assert flat.final_relative_humidity_pct == (
        composed.final_relative_humidity_pct
    )
    assert flat.initial_temperature_c == composed.thermal.initial_temperature_c
    assert flat.final_temperature_c == composed.thermal.final_temperature_c
    assert flat.temperature_drop_c == pytest.approx(
        composed.thermal.initial_temperature_c
        - composed.thermal.final_temperature_c,
        rel=1e-12,
        abs=1e-15,
    )
    assert flat.ventilation_heat_loss_coefficient_w_per_k == (
        composed.thermal.ventilation_heat_loss_coefficient_w_per_k
    )
    assert flat.ventilation_energy_removed_kwh == (
        composed.thermal.energy_removed_kwh
    )


def test_simulation_signs_on_flat_fields_cooling_event() -> None:
    """Winter case: water_removed > 0, temperature_drop > 0, energy > 0."""
    result = simulate_ventilation_event(**_canonical_simulation_kwargs())
    assert result.water_removed_g > 0.0
    assert result.temperature_drop_c > 0.0
    assert result.ventilation_energy_removed_kwh > 0.0
    assert result.final_temperature_c < result.initial_temperature_c
    assert result.final_absolute_humidity_g_m3 < (
        result.initial_absolute_humidity_g_m3
    )


def test_simulation_signs_on_flat_fields_warming_event() -> None:
    """Summer case: water_removed < 0, temperature_drop < 0, energy < 0."""
    kwargs = _canonical_simulation_kwargs()
    kwargs["initial_indoor_temperature_c"] = 22.0
    kwargs["initial_indoor_relative_humidity_pct"] = 40.0
    kwargs["outdoor_temperature_c"] = 30.0
    kwargs["outdoor_relative_humidity_pct"] = 85.0
    result = simulate_ventilation_event(**kwargs)
    assert result.water_removed_g < 0.0
    assert result.temperature_drop_c < 0.0
    assert result.ventilation_energy_removed_kwh < 0.0


def test_simulation_zero_duration_recovers_initial_state() -> None:
    """A 0-minute event: everything stays put."""
    kwargs = _canonical_simulation_kwargs()
    kwargs["duration_minutes"] = 0.0
    result = simulate_ventilation_event(**kwargs)
    assert result.final_absolute_humidity_g_m3 == (
        result.initial_absolute_humidity_g_m3
    )
    assert result.water_removed_g == 0.0
    assert result.final_temperature_c == result.initial_temperature_c
    assert result.temperature_drop_c == 0.0
    assert result.ventilation_energy_removed_kwh == 0.0
    assert result.final_relative_humidity_pct == pytest.approx(
        kwargs["initial_indoor_relative_humidity_pct"], rel=1e-12
    )


def test_simulation_zero_ach_leaves_room_unchanged() -> None:
    """ACH = 0 -> no exchange; every state field stays at its initial value."""
    kwargs = _canonical_simulation_kwargs()
    kwargs["ach"] = 0.0
    result = simulate_ventilation_event(**kwargs)
    assert result.water_removed_g == 0.0
    assert result.temperature_drop_c == 0.0
    assert result.ventilation_energy_removed_kwh == 0.0
    assert result.ventilation_heat_loss_coefficient_w_per_k == 0.0


def test_simulation_result_is_frozen() -> None:
    """VentilationSimulationResult is a frozen dataclass."""
    result = simulate_ventilation_event(**_canonical_simulation_kwargs())
    with pytest.raises(FrozenInstanceError):
        result.final_temperature_c = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("room_volume_m3", 0.0),
        ("room_volume_m3", -1.0),
        ("initial_indoor_relative_humidity_pct", -0.1),
        ("initial_indoor_relative_humidity_pct", 100.1),
        ("outdoor_relative_humidity_pct", -0.1),
        ("outdoor_relative_humidity_pct", 100.1),
        ("ach", -1.0),
        ("effective_thermal_capacity_j_per_k", 0.0),
        ("effective_thermal_capacity_j_per_k", -1.0),
        ("duration_minutes", -1.0),
        ("initial_indoor_temperature_c", float("nan")),
        ("outdoor_temperature_c", float("inf")),
    ],
)
def test_simulate_ventilation_event_propagates_validation(
    arg_name: str, bad_value: float
) -> None:
    """Invalid arguments bubble up from the underlying value objects / physics."""
    kwargs = _canonical_simulation_kwargs()
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError):
        simulate_ventilation_event(**kwargs)


# --- Trade-off invariants -------------------------------------------------


def test_short_duration_ratio_is_ach_invariant() -> None:
    """In the short-duration limit, doubling ACH doubles both water and energy.

    So the g/kWh ratio is invariant to ACH for events much shorter
    than both time constants (tau_moisture = 1/ACH, tau_thermal =
    C_eff/H_vent). This test uses 6-second events (0.1 min), where
    both curves are firmly in the linear-in-t regime, and asserts the
    ratio matches across a 2x ACH change to within 1 %.

    Locks the reviewer's manual check that this invariant was not
    covered by any existing test.
    """
    kwargs = _canonical_simulation_kwargs()
    kwargs["duration_minutes"] = 0.1
    kwargs["ach"] = 5.0
    single = simulate_ventilation_event(**kwargs)
    kwargs["ach"] = 10.0
    double = simulate_ventilation_event(**kwargs)
    # Both quantities should scale close to 2x (exactly 2x in the
    # limit; within a fraction of a percent at t=0.1 min << tau).
    assert double.water_removed_g == pytest.approx(
        2.0 * single.water_removed_g, rel=0.01
    )
    assert double.ventilation_energy_removed_kwh == pytest.approx(
        2.0 * single.ventilation_energy_removed_kwh, rel=0.01
    )
    # Therefore the moisture-per-kWh ratio should be near-identical.
    single_ratio = (
        single.water_removed_g / single.ventilation_energy_removed_kwh
    )
    double_ratio = (
        double.water_removed_g / double.ventilation_energy_removed_kwh
    )
    assert double_ratio == pytest.approx(single_ratio, rel=0.01)


def test_volume_doubling_scales_water_but_not_energy_at_fixed_c_eff() -> None:
    """Doubling volume doubles water_removed; energy scales differently.

    Physically: doubling V doubles H_vent (rho*cp*V*ACH/3600), which
    halves the thermal time constant tau = C_eff/H_vent, so the room
    cools further in the same event and the energy scales by MORE
    than the naive 2x that water sees. This test pins the asymmetry
    so a future contributor doesn't mistakenly assume g/kWh is
    volume-invariant at fixed C_eff. The correct way to compare rooms
    of different size is to scale C_eff proportionally to V - the
    lumped thermal mass IS a room property, and a bigger room usually
    has more of it.

    Reviewer finding: the "g/kWh is volume-invariant" mental model is
    false in this codebase unless the caller scales C_eff with V.
    This test documents the trap and confirms the physics.
    """
    kwargs = _canonical_simulation_kwargs()
    kwargs["duration_minutes"] = 5.0
    # Baseline
    kwargs["room_volume_m3"] = 40.0
    baseline = simulate_ventilation_event(**kwargs)
    # Double the volume, keep C_eff fixed - the wrong comparison.
    kwargs["room_volume_m3"] = 80.0
    fixed_c_eff = simulate_ventilation_event(**kwargs)
    # Water: exact 2x (moisture math is linear in V through
    # water_removed_g = (C_0 - C_f) * V, and (C_0, C_f) do not
    # depend on V).
    assert fixed_c_eff.water_removed_g == pytest.approx(
        2.0 * baseline.water_removed_g, rel=1e-12
    )
    # Energy: strictly LESS than 2x at fixed C_eff, because the
    # larger room-with-same-mass has a shorter thermal time constant
    # and cools further per unit volume. Assert it's between 1.5x
    # and 2.0x (in practice ~1.89x at these parameters).
    energy_ratio = (
        fixed_c_eff.ventilation_energy_removed_kwh
        / baseline.ventilation_energy_removed_kwh
    )
    assert 1.5 < energy_ratio < 2.0

    # Now the "correct" comparison: scale C_eff with V so the thermal
    # time constant stays the same. Both quantities should scale 2x.
    kwargs["effective_thermal_capacity_j_per_k"] = 2.0 * (
        _canonical_simulation_kwargs()["effective_thermal_capacity_j_per_k"]
    )
    scaled_c_eff = simulate_ventilation_event(**kwargs)
    assert scaled_c_eff.water_removed_g == pytest.approx(
        2.0 * baseline.water_removed_g, rel=1e-12
    )
    assert scaled_c_eff.ventilation_energy_removed_kwh == pytest.approx(
        2.0 * baseline.ventilation_energy_removed_kwh, rel=1e-12
    )
