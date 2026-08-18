"""Simple configurable heating models for the time-domain simulator.

Distinguishes two energy quantities the caller cares about:

    * ventilation heat REMOVED from the room
        Owned by the ventilation / thermal layer; this module does
        not touch it. It is the sensible heat sitting in the air
        that just left through the window.
    * heating energy SUPPLIED to the room (thermal side, W_thermal
      integrated over time)
        Delivered by the room's heating system to keep the room
        warm. This IS the physical thermal energy that goes into
        the room, NOT the input energy the occupant purchases.
    * input energy the occupant purchases (electricity or gas)
        Equal to supplied thermal energy divided by
        ``efficiency_or_cop``. For a resistive heater
        efficiency == 1.0 and the two are equal. For a heat pump
        the input is smaller than the thermal supply by the COP;
        for a condensing gas boiler the input is roughly the
        supply divided by ~0.9. This module DOES NOT model a
        specific boiler or heat pump; the COP / efficiency is a
        caller-supplied number, POC only.

Public shape:

    HeatingModel
        Protocol-style ABC that answers, per step:
            respond_to_indoor_temperature(
                indoor_temperature_c: float,
                currently_on: bool,
            ) -> HeatingResponse

        The response carries the new on/off state and the delivered
        thermal power. ``currently_on`` is caller-tracked state so
        the model itself stays a frozen value; a stateful hysteresis
        loop is expressed by feeding ``response.next_on`` into the
        next call.

    NoHeating
        Always returns zero power and preserves the caller's
        on/off state (trivially "always off").

    ThermostaticHeating(
        setpoint_temperature_c,
        max_thermal_power_w,
        efficiency_or_cop,
        hysteresis_c=0.0,
    )
        Turns the heater ON when the room is below
        ``setpoint - hysteresis / 2`` and OFF when above
        ``setpoint + hysteresis / 2``. Delivers
        ``max_thermal_power_w`` while on, zero while off. This is a
        simple bang-bang thermostat, not a modulating controller;
        real thermostats do fancier things and a later slice can
        add them.

Explicitly NOT in this module:
    - Boiler start-up / shut-down losses.
    - Heat pump defrost cycles or COP curves versus outdoor T.
    - Building fabric heat balance (already ignored by ``thermal.py``).
    - Tariffs, cost, carbon: the caller who wants those combines
      the reported "input energy" number with their own tariff /
      carbon-intensity data.

Every value ``efficiency_or_cop`` is a POC caller input, NOT a
validated performance figure for any specific appliance. It is a
knob the caller sets to distinguish "heat put into the room" from
"energy the occupant pays for". A future component can identify a
real device's COP or seasonal efficiency from measurements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class HeatingResponse:
    """Per-step output of a heating model.

    Fields:
        next_on: whether the heater is ON after this step's decision.
            Callers feed this back into the next call as
            ``currently_on`` to carry the hysteresis state forward.
        thermal_power_w: heat delivered TO the room during the step,
            in watts (thermal side). Always non-negative.
        input_power_w: energy the OCCUPANT purchases per unit time,
            in watts. Equal to ``thermal_power_w / efficiency_or_cop``
            for the currently active model. Non-negative.
    """

    next_on: bool
    thermal_power_w: float
    input_power_w: float


class HeatingModel(ABC):
    """Base class for pluggable heating models.

    A model is a pure function from (indoor_temperature_c,
    currently_on) to a ``HeatingResponse``. State is carried by the
    caller; the model itself is a frozen configuration.
    """

    @abstractmethod
    def respond_to_indoor_temperature(
        self,
        indoor_temperature_c: float,
        currently_on: bool,
    ) -> HeatingResponse:
        """Return the heater's response at one time step.

        Args:
            indoor_temperature_c: current room temperature.
            currently_on: whether the heater was on at the end of
                the previous step.
        """


@dataclass(frozen=True)
class NoHeating(HeatingModel):
    """Heating disabled. Always returns zero power.

    Use this to run the room as if there were no heating system at
    all (or if the heating is simply off during the modelling
    window). Compatible with every downstream consumer that would
    otherwise integrate heating power.
    """

    def respond_to_indoor_temperature(
        self,
        indoor_temperature_c: float,
        currently_on: bool,
    ) -> HeatingResponse:
        if not isfinite(indoor_temperature_c):
            raise ValueError(
                "indoor_temperature_c must be finite, got "
                f"{indoor_temperature_c!r}"
            )
        return HeatingResponse(
            next_on=False,
            thermal_power_w=0.0,
            input_power_w=0.0,
        )


@dataclass(frozen=True)
class ThermostaticHeating(HeatingModel):
    """Bang-bang thermostatic heater with optional hysteresis.

    Fields:
        setpoint_temperature_c: target indoor temperature. The
            hysteresis band is centred on this value.
        max_thermal_power_w: peak thermal output while ON, in watts.
            The heater is either delivering this power or delivering
            zero; no modulation. Must be strictly positive.
        efficiency_or_cop: ratio of delivered thermal power to input
            power the OCCUPANT purchases. Interpretation depends on
            the appliance:
              * resistive electric heater: 1.0 (input == thermal)
              * gas boiler: seasonal efficiency, roughly 0.85 - 0.95
              * heat pump: coefficient of performance, roughly
                2.5 - 4.5 depending on source temperature
            Must be strictly positive. THIS IS A POC INPUT, not a
            validated performance figure for any specific device.
        hysteresis_c: half-width of the on/off dead-band, in
            kelvin. The heater switches ON when
            ``T < setpoint - hysteresis / 2`` and OFF when
            ``T > setpoint + hysteresis / 2``. Inside the dead-band
            the heater PRESERVES its current state (which is why the
            caller has to feed the state forward). Must be
            non-negative. Zero (no dead-band) reduces to an ideal
            bang-bang controller that chatters at each step; a
            small positive value (e.g. 0.5 K) mimics a real
            thermostat.

    Switching rule:
        A single indoor-temperature reading and the caller-supplied
        ``currently_on`` decide the next state.
            T <= setpoint - hysteresis / 2 -> next_on = True
            T >= setpoint + hysteresis / 2 -> next_on = False
            otherwise                       -> next_on = currently_on

    Delivered power:
        thermal_power_w = max_thermal_power_w if next_on else 0.0
        input_power_w   = thermal_power_w / efficiency_or_cop

    Real-world caveats:
        - True heating systems have thermal inertia, ramp-up losses,
          and modulating controllers; a bang-bang model catches only
          the coarsest behaviour.
        - The ``efficiency_or_cop`` is treated as a single number,
          not a function of outdoor T. A heat pump's real COP drops
          in the cold; this POC does not model that.
        - The heater is assumed to have unlimited authority to hit
          ``max_thermal_power_w`` on demand; startup transients are
          not modelled.
    """

    setpoint_temperature_c: float
    max_thermal_power_w: float
    efficiency_or_cop: float
    hysteresis_c: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("setpoint_temperature_c", self.setpoint_temperature_c),
            ("max_thermal_power_w", self.max_thermal_power_w),
            ("efficiency_or_cop", self.efficiency_or_cop),
            ("hysteresis_c", self.hysteresis_c),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.max_thermal_power_w <= 0.0:
            raise ValueError(
                "max_thermal_power_w must be strictly positive, got "
                f"{self.max_thermal_power_w}"
            )
        if self.efficiency_or_cop <= 0.0:
            raise ValueError(
                "efficiency_or_cop must be strictly positive, got "
                f"{self.efficiency_or_cop}"
            )
        if self.hysteresis_c < 0.0:
            raise ValueError(
                f"hysteresis_c must be non-negative, got {self.hysteresis_c}"
            )

    def respond_to_indoor_temperature(
        self,
        indoor_temperature_c: float,
        currently_on: bool,
    ) -> HeatingResponse:
        if not isfinite(indoor_temperature_c):
            raise ValueError(
                "indoor_temperature_c must be finite, got "
                f"{indoor_temperature_c!r}"
            )
        lower_bound = (
            self.setpoint_temperature_c - self.hysteresis_c / 2.0
        )
        upper_bound = (
            self.setpoint_temperature_c + self.hysteresis_c / 2.0
        )
        if indoor_temperature_c <= lower_bound:
            next_on = True
        elif indoor_temperature_c >= upper_bound:
            next_on = False
        else:
            next_on = currently_on
        thermal_power_w = self.max_thermal_power_w if next_on else 0.0
        input_power_w = thermal_power_w / self.efficiency_or_cop
        return HeatingResponse(
            next_on=next_on,
            thermal_power_w=thermal_power_w,
            input_power_w=input_power_w,
        )
