"""Tests for the configurable heating model and the heating-aware simulator.

Covers:
    - NoHeating: always returns zero power, preserves off state.
    - ThermostaticHeating: bang-bang switching around the setpoint,
      hysteresis dead-band respected, thermal / input power ratio
      matches the efficiency/COP.
    - Validation: rejects non-positive max power, non-positive
      efficiency, negative hysteresis, non-finite inputs.
    - simulate_room_period_with_heating: distinguishes ventilation
      heat removed, heating thermal energy supplied, and heating
      input energy purchased. Same thermal supply -> different
      input for different efficiency/COP values. No heating trace
      equal to a forecast-only simulation.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room
from moisture_sources import MoistureSourceSchedule
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from time_simulation import (
    RoomHeatingTrajectory,
    VentilationEvent,
    simulate_room_period_with_forecast,
    simulate_room_period_with_heating,
)
from weather_forecast import ForecastPoint, WeatherForecast

from heating import (
    HeatingModel,
    HeatingResponse,
    NoHeating,
    ThermostaticHeating,
)


def _room() -> Room:
    return Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.3,
        ach_window_open=5.0,
    )


def _thermal() -> ThermalProperties:
    return ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )


def _cold_flat_forecast() -> WeatherForecast:
    return WeatherForecast(
        points=(
            ForecastPoint(0.0, -2.0, 70.0),
            ForecastPoint(6.0, -2.0, 70.0),
        )
    )


# --- NoHeating ------------------------------------------------------------


def test_no_heating_returns_zero_power_and_off_state() -> None:
    response = NoHeating().respond_to_indoor_temperature(
        indoor_temperature_c=15.0, currently_on=True
    )
    assert response.thermal_power_w == 0.0
    assert response.input_power_w == 0.0
    assert response.next_on is False


def test_no_heating_rejects_non_finite_input() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="indoor_temperature_c"):
            NoHeating().respond_to_indoor_temperature(bad, False)


# --- ThermostaticHeating: switching behaviour ----------------------------


def test_thermostat_turns_on_below_lower_bound() -> None:
    """T strictly below setpoint - hysteresis/2 -> heater ON."""
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1000.0,
        efficiency_or_cop=1.0,
        hysteresis_c=0.5,
    )
    response = model.respond_to_indoor_temperature(
        indoor_temperature_c=19.5, currently_on=False
    )
    assert response.next_on is True
    assert response.thermal_power_w == 1000.0
    assert response.input_power_w == 1000.0


def test_thermostat_turns_off_above_upper_bound() -> None:
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1000.0,
        efficiency_or_cop=1.0,
        hysteresis_c=0.5,
    )
    response = model.respond_to_indoor_temperature(
        indoor_temperature_c=20.5, currently_on=True
    )
    assert response.next_on is False
    assert response.thermal_power_w == 0.0


def test_thermostat_dead_band_preserves_state() -> None:
    """Inside the dead-band the previous state carries forward."""
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1000.0,
        efficiency_or_cop=1.0,
        hysteresis_c=1.0,
    )
    # T = 20.2 sits inside [19.5, 20.5]; state is preserved.
    on_response = model.respond_to_indoor_temperature(20.2, currently_on=True)
    off_response = model.respond_to_indoor_temperature(20.2, currently_on=False)
    assert on_response.next_on is True
    assert off_response.next_on is False


def test_thermostat_zero_hysteresis_chatters_at_setpoint() -> None:
    """With hysteresis 0, setpoint sits at both boundaries.

    At setpoint the ``T <= setpoint`` branch fires and the heater
    turns ON, regardless of previous state. Callers using zero
    hysteresis in a real controller are asking for chatter; the
    model reflects that faithfully.
    """
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1000.0,
        efficiency_or_cop=1.0,
        hysteresis_c=0.0,
    )
    at_setpoint_off = model.respond_to_indoor_temperature(20.0, False)
    at_setpoint_on = model.respond_to_indoor_temperature(20.0, True)
    assert at_setpoint_off.next_on is True
    assert at_setpoint_on.next_on is True


# --- ThermostaticHeating: input vs thermal power ------------------------


def test_resistive_heater_input_equals_thermal() -> None:
    """efficiency = 1.0 -> input power == thermal power."""
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1500.0,
        efficiency_or_cop=1.0,
    )
    response = model.respond_to_indoor_temperature(15.0, False)
    assert response.thermal_power_w == 1500.0
    assert response.input_power_w == pytest.approx(1500.0)


def test_heat_pump_input_below_thermal_by_cop() -> None:
    """COP = 3 -> input power is one third of thermal power."""
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1500.0,
        efficiency_or_cop=3.0,
    )
    response = model.respond_to_indoor_temperature(15.0, False)
    assert response.thermal_power_w == 1500.0
    assert response.input_power_w == pytest.approx(500.0)


def test_gas_boiler_input_above_thermal_by_efficiency() -> None:
    """Efficiency 0.9 -> input power slightly exceeds thermal."""
    model = ThermostaticHeating(
        setpoint_temperature_c=20.0,
        max_thermal_power_w=1000.0,
        efficiency_or_cop=0.9,
    )
    response = model.respond_to_indoor_temperature(15.0, False)
    assert response.thermal_power_w == 1000.0
    # 1000 / 0.9 == 1111.11...
    assert response.input_power_w == pytest.approx(1111.111, rel=1e-3)


# --- Validation ----------------------------------------------------------


def test_thermostat_is_frozen() -> None:
    model = ThermostaticHeating(20.0, 1000.0, 1.0)
    with pytest.raises(FrozenInstanceError):
        model.setpoint_temperature_c = 21.0  # type: ignore[misc]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_thermostat_rejects_non_positive_max_power(bad: float) -> None:
    if bad != bad or bad in (float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="max_thermal_power_w"):
            ThermostaticHeating(20.0, bad, 1.0)
    else:
        with pytest.raises(ValueError, match="max_thermal_power_w"):
            ThermostaticHeating(20.0, bad, 1.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_thermostat_rejects_non_positive_efficiency(bad: float) -> None:
    with pytest.raises(ValueError, match="efficiency_or_cop"):
        ThermostaticHeating(20.0, 1000.0, bad)


def test_thermostat_rejects_negative_hysteresis() -> None:
    with pytest.raises(ValueError, match="hysteresis_c"):
        ThermostaticHeating(20.0, 1000.0, 1.0, hysteresis_c=-0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_thermostat_rejects_non_finite_setpoint(bad: float) -> None:
    with pytest.raises(ValueError, match="setpoint_temperature_c"):
        ThermostaticHeating(bad, 1000.0, 1.0)


# --- HeatingResponse shape -----------------------------------------------


def test_heating_response_is_frozen_dataclass() -> None:
    response = HeatingResponse(next_on=True, thermal_power_w=500.0, input_power_w=500.0)
    with pytest.raises(FrozenInstanceError):
        response.thermal_power_w = 0.0  # type: ignore[misc]


# --- Integration with simulate_room_period_with_heating -----------------


def test_no_heating_matches_forecast_variant() -> None:
    """simulate_room_period_with_heating(NoHeating) reproduces the forecast-only run.

    Both should produce sample-for-sample equal indoor T / AH.
    """
    room = _room()
    thermal = _thermal()
    forecast = _cold_flat_forecast()
    schedule = MoistureSourceSchedule(constant_background_rate_g_per_hour=50.0)
    events = (VentilationEvent(0.5, 0.7),)
    duration_hours = 3.0
    timestep_minutes = 5.0
    with_no_heat = simulate_room_period_with_heating(
        room=room,
        thermal_properties=thermal,
        forecast=forecast,
        moisture_schedule=schedule,
        ventilation_events=events,
        heating_model=NoHeating(),
        duration_hours=duration_hours,
        timestep_minutes=timestep_minutes,
    )
    without_heat = simulate_room_period_with_forecast(
        room=room,
        thermal_properties=thermal,
        forecast=forecast,
        moisture_schedule=schedule,
        ventilation_events=events,
        duration_hours=duration_hours,
        timestep_minutes=timestep_minutes,
    )
    for i in range(len(without_heat.times_hours)):
        assert (
            with_no_heat.trajectory.indoor_temperature_c[i]
            == pytest.approx(
                without_heat.indoor_temperature_c[i], rel=1e-9
            )
        )
        assert (
            with_no_heat.trajectory.indoor_absolute_humidity_g_m3[i]
            == pytest.approx(
                without_heat.indoor_absolute_humidity_g_m3[i], rel=1e-9
            )
        )
    assert with_no_heat.heating_thermal_energy_supplied_kwh == 0.0
    assert with_no_heat.heating_input_energy_purchased_kwh == 0.0


def test_thermostat_holds_room_near_setpoint() -> None:
    """A running thermostat keeps the room hovering around the setpoint."""
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        heating_model=ThermostaticHeating(
            setpoint_temperature_c=20.0,
            max_thermal_power_w=2000.0,
            efficiency_or_cop=1.0,
            hysteresis_c=0.5,
        ),
        duration_hours=6.0,
        timestep_minutes=5.0,
    )
    # Every recorded indoor T after the first sample should sit
    # inside a small band around the setpoint.
    for t in result.trajectory.indoor_temperature_c[1:]:
        assert 19.0 <= t <= 21.0


def test_heating_supply_equals_input_when_efficiency_is_one() -> None:
    """On a resistive heater the two totals are equal."""
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        heating_model=ThermostaticHeating(
            setpoint_temperature_c=20.0,
            max_thermal_power_w=1500.0,
            efficiency_or_cop=1.0,
            hysteresis_c=0.5,
        ),
        duration_hours=6.0,
        timestep_minutes=5.0,
    )
    assert result.heating_thermal_energy_supplied_kwh == pytest.approx(
        result.heating_input_energy_purchased_kwh, rel=1e-9
    )


def test_heat_pump_input_below_thermal_supply_by_cop() -> None:
    """On a heat pump input < thermal supply by roughly the COP."""
    cop = 3.0
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        heating_model=ThermostaticHeating(
            setpoint_temperature_c=20.0,
            max_thermal_power_w=1500.0,
            efficiency_or_cop=cop,
            hysteresis_c=0.5,
        ),
        duration_hours=6.0,
        timestep_minutes=5.0,
    )
    assert (
        result.heating_input_energy_purchased_kwh
        == pytest.approx(
            result.heating_thermal_energy_supplied_kwh / cop, rel=1e-9
        )
    )


def test_ventilation_heat_removed_is_positive_when_indoor_warmer() -> None:
    """A cold outdoor + open window drives ventilation heat OUT of the room."""
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(VentilationEvent(0.5, 1.0),),
        heating_model=NoHeating(),
        duration_hours=2.0,
        timestep_minutes=5.0,
    )
    assert result.ventilation_heat_removed_kwh > 0.0


def test_heating_totals_non_negative() -> None:
    """Heating supplied / purchased are book-kept as non-negative sums."""
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(VentilationEvent(0.5, 1.0),),
        heating_model=ThermostaticHeating(20.0, 1500.0, 1.0, 0.5),
        duration_hours=2.0,
        timestep_minutes=5.0,
    )
    assert result.heating_thermal_energy_supplied_kwh >= 0.0
    assert result.heating_input_energy_purchased_kwh >= 0.0


def test_heating_reduces_temperature_drop_from_ventilation() -> None:
    """A vent event under heating drops less than the same event under no heating."""
    events = (VentilationEvent(0.5, 0.7),)
    no_heat = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=events,
        heating_model=NoHeating(),
        duration_hours=1.5,
        timestep_minutes=5.0,
    )
    with_heat = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=events,
        heating_model=ThermostaticHeating(20.0, 2000.0, 1.0, 0.5),
        duration_hours=1.5,
        timestep_minutes=5.0,
    )
    # Final indoor T under heating stays higher.
    assert (
        with_heat.trajectory.indoor_temperature_c[-1]
        > no_heat.trajectory.indoor_temperature_c[-1]
    )


def test_room_heating_trajectory_is_frozen() -> None:
    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        heating_model=NoHeating(),
        duration_hours=1.0,
        timestep_minutes=15.0,
    )
    assert isinstance(result, RoomHeatingTrajectory)
    with pytest.raises(FrozenInstanceError):
        result.heating_thermal_energy_supplied_kwh = 0.0  # type: ignore[misc]


def test_custom_subclass_of_heating_model_works() -> None:
    """A caller can subclass HeatingModel with their own logic."""

    class AlwaysOnResistive(HeatingModel):
        def respond_to_indoor_temperature(
            self, indoor_temperature_c: float, currently_on: bool
        ) -> HeatingResponse:
            return HeatingResponse(
                next_on=True, thermal_power_w=800.0, input_power_w=800.0
            )

    result = simulate_room_period_with_heating(
        room=_room(),
        thermal_properties=_thermal(),
        forecast=_cold_flat_forecast(),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_events=(),
        heating_model=AlwaysOnResistive(),
        duration_hours=1.0,
        timestep_minutes=15.0,
    )
    assert result.heating_thermal_energy_supplied_kwh > 0.0
    # Every step's heating_thermal_power_w should be 800 after the
    # initial marker.
    for p in result.heating_thermal_power_w[1:]:
        assert p == pytest.approx(800.0)
