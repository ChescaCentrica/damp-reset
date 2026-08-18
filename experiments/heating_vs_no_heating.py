"""Compare identical ventilation events under two heating regimes.

Runs the SAME ventilation event, room, forecast, and moisture
schedule twice:

    * heating OFF (``NoHeating``)
    * heating maintaining a setpoint
      (``ThermostaticHeating(setpoint=20 C, ...)``)

and reports for each run:

    - final room temperature at the end of the horizon
    - room temperature at the moment the window closes
    - total ventilation heat removed from the room
    - total heating THERMAL energy supplied to the room
    - total heating INPUT energy the occupant purchases
    - a coarse temperature-trajectory sketch (four sampled points)

Includes a resistive-heater case (efficiency 1.0) and a heat-pump
case (COP 3.0) so the "supplied vs purchased" distinction is
visible.

Rationale printed at the bottom explains why the ventilation heat
loss and the heating-system energy consumption are related but not
identical quantities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass

from heating import HeatingModel, NoHeating, ThermostaticHeating
from moisture import Room
from moisture_sources import MoistureSourceSchedule
from thermal import (
    ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K,
    ThermalProperties,
)
from time_simulation import (
    RoomHeatingTrajectory,
    VentilationEvent,
    simulate_room_period_with_heating,
)
from weather_forecast import ForecastPoint, WeatherForecast


@dataclass(frozen=True)
class Scenario:
    room: Room
    thermal_properties: ThermalProperties
    forecast: WeatherForecast
    moisture_schedule: MoistureSourceSchedule
    ventilation_event: VentilationEvent
    duration_hours: float
    timestep_minutes: float


def _scenario() -> Scenario:
    """Illustrative POC scenario, identical across both regimes.

    Indoor 20 C / 55 %RH, outdoor -2 C / 70 %RH held constant over
    the horizon (a flat forecast; the demonstration is about the
    heating regime, not the outdoor evolution). A 30-minute window
    open at t = 0.5 h is the ventilation event.
    """
    return Scenario(
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=55.0,
            ach_closed=0.3,
            ach_window_open=5.0,
        ),
        thermal_properties=ThermalProperties(
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            )
        ),
        forecast=WeatherForecast(
            points=(
                ForecastPoint(0.0, -2.0, 70.0),
                ForecastPoint(6.0, -2.0, 70.0),
            )
        ),
        moisture_schedule=MoistureSourceSchedule(),
        ventilation_event=VentilationEvent(
            start_time_hours=0.5, end_time_hours=1.0
        ),
        duration_hours=3.0,
        timestep_minutes=5.0,
    )


def _run(
    scenario: Scenario, heating_model: HeatingModel
) -> RoomHeatingTrajectory:
    return simulate_room_period_with_heating(
        room=scenario.room,
        thermal_properties=scenario.thermal_properties,
        forecast=scenario.forecast,
        moisture_schedule=scenario.moisture_schedule,
        ventilation_events=(scenario.ventilation_event,),
        heating_model=heating_model,
        duration_hours=scenario.duration_hours,
        timestep_minutes=scenario.timestep_minutes,
    )


def _sample_temperatures_c(
    result: RoomHeatingTrajectory, query_times_hours: tuple
) -> list:
    """Return the recorded indoor temperature at each query time."""
    times = result.trajectory.times_hours
    temps = result.trajectory.indoor_temperature_c
    samples = []
    for query in query_times_hours:
        best_i = 0
        for i, t in enumerate(times):
            if t <= query + 1e-9:
                best_i = i
        samples.append(temps[best_i])
    return samples


def _print_run(label: str, result: RoomHeatingTrajectory) -> None:
    sample_times = (0.0, 0.5, 1.0, 3.0)
    temps = _sample_temperatures_c(result, sample_times)
    print(f"\n{label}")
    print("-" * len(label))
    print("Indoor temperature trajectory (degC):")
    for t, temp in zip(sample_times, temps):
        marker = "  window opens" if t == 0.5 else (
            "  window closes" if t == 1.0 else ""
        )
        print(f"    t = {t:>4.1f} h   T = {temp:6.2f}{marker}")
    print(
        f"Final indoor temperature at t = "
        f"{result.trajectory.times_hours[-1]:g} h: "
        f"{result.trajectory.indoor_temperature_c[-1]:.2f} C"
    )
    print(
        f"Ventilation heat REMOVED from the room: "
        f"{result.ventilation_heat_removed_kwh:.3f} kWh"
    )
    print(
        f"Heating THERMAL energy SUPPLIED to the room: "
        f"{result.heating_thermal_energy_supplied_kwh:.3f} kWh"
    )
    print(
        f"Heating INPUT energy PURCHASED by the occupant: "
        f"{result.heating_input_energy_purchased_kwh:.3f} kWh"
    )


def main() -> None:
    scenario = _scenario()
    print("Compare identical ventilation events under three heating regimes.")
    print(
        "Room 20 C / 55 %RH, volume 40 m^3, outdoor -2 C / 70 %RH held\n"
        "constant. Ventilation event: window open t = 0.5 to 1.0 h "
        "(30 min).\n"
        "Every threshold below is illustrative POC. Efficiency and COP\n"
        "are caller-supplied numbers, not appliance-validated figures."
    )

    off = _run(scenario, NoHeating())
    _print_run("[1] Heating OFF", off)

    resistive = _run(
        scenario,
        ThermostaticHeating(
            setpoint_temperature_c=20.0,
            max_thermal_power_w=2000.0,
            efficiency_or_cop=1.0,
            hysteresis_c=0.5,
        ),
    )
    _print_run(
        "[2] Heating ON - resistive electric (efficiency 1.0), "
        "setpoint 20 C, 2 kW",
        resistive,
    )

    heat_pump = _run(
        scenario,
        ThermostaticHeating(
            setpoint_temperature_c=20.0,
            max_thermal_power_w=2000.0,
            efficiency_or_cop=3.0,
            hysteresis_c=0.5,
        ),
    )
    _print_run(
        "[3] Heating ON - heat pump (COP 3.0), setpoint 20 C, 2 kW thermal",
        heat_pump,
    )

    print(
        "\nWhy ventilation heat loss and heating energy consumption "
        "are related but NOT identical:\n"
    )
    print(
        "  1. Ventilation heat REMOVED is the sensible heat that\n"
        "     physically left the room through the window during the\n"
        "     event, C_eff * (T_before - T_after_ventilation)\n"
        "     integrated across the horizon. It is a property of the\n"
        "     ROOM's thermal balance and does not depend on whether\n"
        "     the heater is on. A running heater KEEPS the indoor\n"
        "     air warmer while the window is open, which INCREASES\n"
        "     the indoor-outdoor temperature gap, which INCREASES\n"
        "     the instantaneous heat-loss power - so ventilation\n"
        "     heat removed is slightly larger under heating than\n"
        "     under no heating. Look at case [1] vs [2] above: same\n"
        "     window, higher ventilation heat removed when the\n"
        "     room stays warm.\n"
    )
    print(
        "  2. Heating THERMAL energy SUPPLIED is the heat the\n"
        "     heating system dumps INTO the room over the horizon,\n"
        "     as thermal energy (watts of heat, integrated). It is\n"
        "     zero under case [1]. Under case [2] and [3] it covers\n"
        "     both the loss during the vent event AND the ongoing\n"
        "     background loss at the closed-window air-change rate.\n"
    )
    print(
        "  3. Heating INPUT energy PURCHASED is the electricity or\n"
        "     gas the occupant actually pays for. It equals the\n"
        "     supplied thermal energy divided by the appliance's\n"
        "     efficiency / COP:\n"
        "         resistive heater  (eff = 1.0):  input == supplied\n"
        "         gas boiler        (eff ~ 0.9):  input slightly ABOVE supplied\n"
        "         heat pump         (COP ~ 3.0):  input roughly a THIRD of supplied\n"
        "     Compare cases [2] and [3]: identical supplied thermal\n"
        "     energy, very different purchased input energy.\n"
    )
    print(
        "  4. The bill the occupant pays is therefore neither the\n"
        "     ventilation heat lost nor the thermal energy the\n"
        "     heater delivered - it is (supplied thermal) / (COP or\n"
        "     efficiency). Ventilation heat loss drives the heater's\n"
        "     workload; the appliance efficiency / COP then decides\n"
        "     what that workload costs at the meter.\n"
    )
    print(
        "Bookkeeping cross-check (informal):\n"
        "  For [2] with heating maintaining setpoint over the full\n"
        "  horizon, the room ends near the initial temperature, so\n"
        "  the total heat delivered by the heater roughly equals the\n"
        "  total heat that left the room (ventilation + background\n"
        "  losses through ach_closed). Under no heating [1] the room\n"
        "  cools far below the initial temperature: heat left the\n"
        "  room and nothing replaced it."
    )


if __name__ == "__main__":
    main()
