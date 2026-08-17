# Proposed architecture — extending the damp-reset POC

## What we have today

A validated, tested pipeline that answers ONE question:

> for a single ventilation event under fixed indoor and outdoor
> conditions, which duration achieves a caller-set moisture
> target with the least heating energy loss?

That is a **moisture-reduction and heat-loss trade-off tool** running
on the following modules (physics equations owned in exactly one place
each, 492 passing tests):

- `psychrometrics.py` — Magnus + ideal gas + AH↔RH inverse.
- `moisture.py` — well-mixed moisture ODE (single event).
- `thermal.py` — well-mixed thermal ODE (single event) + lumped C_eff.
- `ventilation.py` — composed simulator; exposes the flat facade
  `simulate_ventilation_event(...)` and its ten-field result.
- `optimiser.py` — five strategies + `recommend_ventilation_action`
  as the POC default.
- `presentation.py` — human-readable format of a recommendation.

## What we want to extend to

Same physical layers, plus **time-domain simulation over hours to days
with realistic sources, surface risk, weather forecasts, heating
control, and calibrated inputs**. The single-event simulator remains
the low-level primitive — it is not deprecated, and it is not
rewritten. Every new module either wraps it, feeds it, or reads its
outputs.

Seven new modules, each with a single well-scoped responsibility.
None of them owns any physics equation the existing modules already
own.

## Proposed module set

| # | Module                | One-line responsibility                                        |
|---|-----------------------|----------------------------------------------------------------|
| 1 | `moisture_sources.py` | Compute a time-varying moisture source term for the room air.  |
| 2 | `surface_risk.py`     | Compute cold-surface RH and integrate a mould-risk indicator.  |
| 3 | `heating.py`          | Model heating-system response (setpoint / power / hysteresis). |
| 4 | `weather.py`          | Provide time-varying outdoor conditions (measured or forecast).|
| 5 | `time_simulation.py`  | Step the moisture + thermal state through a schedule of events.|
| 6 | `calibration.py`      | Identify ACH_open, C_eff, and other room parameters from data. |
| 7 | `validation.py`       | Compare model predictions against measured room trajectories.  |

### 1. `moisture_sources.py`

**Purpose.** Return a moisture-generation rate (g/hour or g/s) as a
function of time, occupancy, activity, and room type. Turns "there
are two adults asleep in a 40 m³ bedroom" into a source term the
moisture step of `time_simulation.py` can consume.

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class MoistureSourceSchedule:
    """Piecewise-constant source term in g/hour over a schedule."""
    intervals: tuple[MoistureSourceInterval, ...]

def moisture_source_rate_g_per_hour_at(
    schedule: MoistureSourceSchedule,
    time_hours: float,
) -> float: ...
```

**Data.** Common residential rates (occupant respiration, cooking,
laundry, showers, plants) are widely tabulated (ASHRAE HoF, CIBSE
Guide A). Values that go into the module are **POC catalog entries,
NOT validated for a specific building**; every value must carry a
citation string in its docstring and be overridable by the caller.

**Dependencies.** None outward (standard library only). Consumed by
`time_simulation.py`.

**Reuses today.** Nothing. This is a genuinely new physics layer.

### 2. `surface_risk.py`

**Purpose.** Answer "is any surface in the room close enough to its
dew point for long enough that mould-growth conditions accumulate?"

Two computations:

- **Instantaneous surface RH** at a caller-defined surface (thermal
  bridge, single-glazed window, cold external wall). Requires a
  surface-temperature model — for the POC that means a fixed
  temperature-offset from indoor air, i.e. `T_surface = T_air −
  ΔT_bridge`, where `ΔT_bridge` is a caller-set POC parameter.
- **Integrated mould-risk indicator** — for the POC something like a
  cumulative hours-above-80 %-surface-RH counter, escalating to a
  proper VTT / isopleth / ASHRAE 160 model in a later slice.

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class SurfaceDescriptor:
    label: str
    temperature_offset_below_air_c: float  # POC surrogate for U*ΔT

def surface_temperature_c(air_state, surface) -> float: ...
def surface_relative_humidity_pct(air_state, surface) -> float: ...

@dataclass(frozen=True)
class MouldRiskSummary:
    hours_above_80_pct_rh: float
    hours_above_saturation: float
    peak_surface_rh_pct: float

def accumulate_mould_risk(
    trajectory: RoomTrajectory,
    surface: SurfaceDescriptor,
) -> MouldRiskSummary: ...
```

**Dependencies.**

- **Reads from** `psychrometrics.py`:
  - `saturation_vapour_pressure` (needed for surface P_sat at the
    surface temperature).
  - `vapour_pressure` (indoor P_v is the same as room P_v).
  - `relative_humidity_from_absolute_humidity` (already exists;
    surface RH = P_v / P_sat(T_surface); reuse the inverse we
    already validated).
- **Consumes** trajectories produced by `time_simulation.py`.

**Reuses today.**

- `psychrometrics.saturation_vapour_pressure` — for the surface's
  saturation curve.
- `psychrometrics.relative_humidity_from_absolute_humidity` —
  computed against the SURFACE temperature and the ROOM AH. This is
  the same "final RH derived from predicted T and AH" invariant the
  existing pipeline already respects; the surface risk module just
  substitutes the surface temperature for indoor air temperature.
- `psychrometrics.AirState` — as the input container.

### 3. `heating.py`

**Purpose.** Represent an idealised heating system. Answer "if the
room drops to T while the setpoint is T_set, how much heat does the
heater deliver, and how quickly does the room recover?"

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class HeatingSystem:
    setpoint_c: float
    max_power_w: float
    hysteresis_k: float
    controller: Callable[[float, float], float]
    # returns delivered power (W) given current T and setpoint

def heat_delivered_w(system, current_air_temperature_c) -> float: ...
```

**Dependencies.** None outward. Consumed by `time_simulation.py`,
which adds the heating power term to the thermal ODE step.

**Reuses today.** Nothing directly. It's a caller-set energy input
that the time step needs to integrate alongside ventilation heat
loss.

**Note.** The single-event optimiser today assumes NO heating. Once
heating is present, "the energy leaving the room via ventilation" no
longer equals "the energy the caller pays for" — that becomes a
fuel-tariff × heater-COP calculation. `heating.py` provides the raw
power input; a future cost module composes it with tariff data.

### 4. `weather.py`

**Purpose.** Provide time-varying outdoor conditions. Two shapes:

- **Historic / measured** — a pandas.DataFrame or a small dataclass
  containing timestamps + (T, RH) pairs, replayed against the
  simulator.
- **Forecast** — an interface into an external service (Met Office /
  ECMWF / OpenWeatherMap), gated behind an adapter so the rest of
  the code never depends on the provider.

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class OutdoorConditionsTimeseries:
    """Piecewise-constant or piecewise-linear outdoor state."""
    timestamps_hours: tuple[float, ...]
    temperatures_c: tuple[float, ...]
    relative_humidities_pct: tuple[float, ...]

def outdoor_air_state_at(
    series: OutdoorConditionsTimeseries,
    time_hours: float,
) -> AirState: ...
```

**Dependencies.**

- **Reads from** `psychrometrics.AirState` — every returned outdoor
  condition is packaged as the existing value object so downstream
  modules use the same interface they use today.

**Reuses today.**

- `psychrometrics.AirState` — as the return type of the "outdoor at
  time t" accessor.

### 5. `time_simulation.py`

**The keystone module.** Steps the room forward through a schedule
of ventilation events (or a control decision function), applying at
each step:

- outdoor conditions from `weather.py`;
- moisture source term from `moisture_sources.py`;
- heating power from `heating.py`;
- ventilation exchange via the existing
  `simulate_ventilation_event(...)` between event boundaries;
- an accumulated `RoomTrajectory` for downstream risk / validation
  analysis.

**Design choice.** Between two ventilation events, the room evolves
under:

- moisture source S(t) driving the AH up;
- ambient ACH_closed driving the AH toward outdoor;
- heating power P_heat(T, T_set) driving the temperature toward the
  setpoint;
- envelope conduction (out of POC scope for now — deferred).

The existing simulator already models the moisture and thermal ODEs
under constant ACH; a **short-interval discretisation** re-uses
`simulate_ventilation_event(...)` per small time step, updating the
indoor state at each step. No new ODE is written.

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class ScheduledEvent:
    start_time_hours: float
    duration_minutes: float
    window_open: bool

@dataclass(frozen=True)
class RoomTrajectory:
    times_hours: tuple[float, ...]
    indoor_temperature_c: tuple[float, ...]
    indoor_absolute_humidity_g_m3: tuple[float, ...]
    indoor_relative_humidity_pct: tuple[float, ...]
    outdoor_temperature_c: tuple[float, ...]
    outdoor_absolute_humidity_g_m3: tuple[float, ...]
    heating_power_w: tuple[float, ...]
    ventilation_events: tuple[ScheduledEvent, ...]

def simulate_schedule(
    initial_room: Room,
    thermal_properties: ThermalProperties,
    weather: OutdoorConditionsTimeseries,
    sources: MoistureSourceSchedule,
    heating: HeatingSystem | None,
    events: Sequence[ScheduledEvent],
    total_duration_hours: float,
    step_minutes: float,
) -> RoomTrajectory: ...
```

**Dependencies.**

- **Reads from** `ventilation.simulate_ventilation_event(...)` — the
  low-level primitive stays untouched. Every step of the schedule
  reads its indoor T and AH, calls the simulator over the small
  interval, receives the new state, updates the trajectory.
- **Reads from** `weather.outdoor_air_state_at(...)`.
- **Reads from** `moisture_sources.moisture_source_rate_g_per_hour_at(...)`.
- **Reads from** `heating.heat_delivered_w(...)`.
- **Emits** a `RoomTrajectory` for `surface_risk.py` and
  `validation.py` to consume.

**Reuses today.**

- `ventilation.simulate_ventilation_event(...)` — the entire single-
  event physics stack, unchanged.
- `moisture.Room` — describes the room the trajectory runs against.
- `thermal.ThermalProperties` — supplies `C_eff`.
- `psychrometrics.AirState` — every indoor / outdoor state on the
  trajectory is a validated psychrometric object.
- `psychrometrics.relative_humidity_from_absolute_humidity` —
  computes indoor RH at every step from the STEP-END temperature
  and AH, preserving the same invariant the existing pipeline
  respects.

**Note on moisture source integration.** The existing moisture ODE
`dC/dt = n · (C_out − C)` has no source term. The time-simulation
module adds one by DISCRETISING outside the simulator: over a step
of Δt, the moisture-source contribution is applied as
`ΔC_source = S · Δt / V` and superposed on the simulator's output.
Physically that's a straightforward operator-split (source updates
between ventilation calls); no change to `moisture.py`.

### 6. `calibration.py`

**Purpose.** Estimate the caller-set POC inputs (`ach_window_open`,
`ach_closed`, `effective_thermal_capacity_j_per_k`,
`temperature_offset_below_air_c` for cold surfaces) from measured
room trajectories.

**Approach.** For each parameter, run the time-simulation module
against measured T / RH data and fit by non-linear least squares
using scipy.optimize. The fit's residuals are the difference between
the trajectory the simulator predicts and the sensor data.

**Public shape (proposed).**

```python
@dataclass(frozen=True)
class CalibrationResult:
    fitted_parameters: dict[str, float]
    residual_rmse: float
    fit_diagnostics: dict[str, float]

def calibrate_from_trajectory(
    measured_times_hours,
    measured_temperature_c,
    measured_relative_humidity_pct,
    room: Room,
    initial_estimates: dict[str, float],
) -> CalibrationResult: ...
```

**Dependencies.**

- **Reads from** `time_simulation.simulate_schedule(...)` — the
  candidate parameter set drives a full trajectory, which is scored
  against the measured data.
- **Reads from** `psychrometrics.AirState` — indoor RH from measured
  T + AH goes through the same inverse the rest of the system uses.

**Reuses today.**

- Everything the time-simulation module uses, plus scipy for the
  optimisation.

### 7. `validation.py`

**Purpose.** Given a `RoomTrajectory` from the model and a measured
sensor time series, report how well they agree.

**Metrics (proposed).**

- RMSE and mean bias on indoor temperature.
- RMSE and mean bias on indoor RH.
- Peak-difference statistics.
- Number of surface-RH>80% hours predicted vs measured, for a named
  surface.

**Dependencies.**

- **Reads from** `time_simulation.RoomTrajectory` — model output.
- **Reads from** `surface_risk.MouldRiskSummary` — for surface-RH
  comparison.
- **Reads from** `weather.OutdoorConditionsTimeseries` — same
  outdoor conditions drive the model and are observed by the
  sensor.

**Reuses today.** Nothing directly (comparison logic is new); but
every model output being compared comes from the existing physics
stack.

## Dependency graph

```
                    +----------------------+
                    |  psychrometrics.py   |  (unchanged)
                    +----------+-----------+
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
      +------------+   +--------------+   +--------------+
      | moisture.py|   |  thermal.py  |   | surface_risk |  <-- reads P_sat, RH_from_AH
      +------+-----+   +------+-------+   +--------------+
             |                |                    ^
             +--------+-------+                    |
                      v                            |
              +-----------------+                  |
              |  ventilation.py |  <-- unchanged   |
              |  simulate_...() |                  |
              +--------+--------+                  |
                       |                           |
                       |  reused as low-level      |
                       |  primitive               |
                       v                           |
              +---------------------+              |
              | time_simulation.py  |--- emits ----+
              |  simulate_schedule  |
              +---------+-----------+
                        ^          ^          ^
                        |          |          |
              +---------+   +------+     +----+------+
              | weather |   |heating|    | moisture_ |
              |   .py   |   |  .py  |    | sources.py|
              +---------+   +-------+    +-----------+
                        |
                        v
              +---------------------+
              |    optimiser.py     |  (unchanged for single event;
              |    (existing)       |   later slice adds a
              +---------------------+   time-horizon variant)
                        |
                        v
              +---------------------+
              | presentation.py     |  (unchanged; later slice adds
              +---------------------+   trajectory-shaped reasons)

                +--------------+       +---------------+
                | calibration  | ----> | validation.py |
                |     .py      |       +---------------+
                +--------------+
```

**Rules for the boundary.**

- Nothing in the seven new modules is allowed to import into
  `psychrometrics.py`, `moisture.py`, or `thermal.py`. The physics
  stack stays one-way; new modules read from it.
- `ventilation.simulate_ventilation_event(...)` stays the low-level
  primitive. `time_simulation.py` calls it per step; nothing else
  reimplements moisture or thermal ODEs.
- `optimiser.py` and `presentation.py` are UNCHANGED in the first
  extension slice. A LATER slice will add a "time-horizon
  optimiser" that consumes `RoomTrajectory` outputs, but that lives
  in a new module (proposal name `time_optimiser.py`); the existing
  event-scale optimiser stays where it is.

## Existing functions reused (checklist)

| new module         | reuses (existing function / class)                                             |
|--------------------|--------------------------------------------------------------------------------|
| moisture_sources   | none                                                                           |
| surface_risk       | `saturation_vapour_pressure`, `relative_humidity_from_absolute_humidity`, `AirState`|
| heating            | none                                                                           |
| weather            | `AirState`                                                                     |
| time_simulation    | `simulate_ventilation_event`, `Room`, `ThermalProperties`, `AirState`, `relative_humidity_from_absolute_humidity` |
| calibration        | everything time_simulation reuses, plus scipy                                  |
| validation         | consumes trajectories; no physics imports                                      |

## What NOT to do in this proposal

- **Do not rewrite the single-event simulator.** It stays as the
  low-level primitive, callable directly, and the extension modules
  compose it rather than replace it.
- **Do not touch the physics modules.** All bugs there have been
  independently reviewed and closed.
- **Do not add a mould-risk model that claims to be validated.** The
  first slice of `surface_risk.py` is a POC hours-above-threshold
  counter; a proper VTT / isopleth / ASHRAE 160 implementation is a
  separate slice with its own literature-review and validation
  step.
- **Do not couple the modules through global state.** Every module
  is invoked with explicit dataclass inputs and returns explicit
  dataclass outputs, same discipline as the existing modules.

## Rough implementation order

Each of these is a separate slice; each should be reviewed and
tested before the next.

1. `moisture_sources.py` — smallest, no reads from physics; unit
   tests only.
2. `heating.py` — same shape, standalone.
3. `weather.py` — `AirState` reuse; unit tests around a small
   fixture time series.
4. `time_simulation.py` — the biggest slice. Composes the existing
   simulator over a schedule. Requires operator-split verification
   (source term applied outside the ODE) and a regression test
   that reproduces the existing single-event simulator's output
   when sources / heating / weather are zeroed.
5. `surface_risk.py` — reads trajectories, calls the psychrometric
   inverse.
6. `validation.py` — compares model vs measured; no new physics.
7. `calibration.py` — fits the POC's caller-set inputs against
   measured trajectories. Requires scipy.
8. **(Future slice)** `time_optimiser.py` — a time-horizon optimiser
   that consumes trajectory-shaped predictions and picks a
   SCHEDULE of ventilation events. Requires all of the above to
   exist and to be validated.

## Testing invariants that must hold

- Nothing in the seven new modules imports from any other new module
  in a cycle. The proposed graph above is acyclic; the AST guards
  in the test suite are extended to enforce it.
- Physics equations still live in exactly one place. Every new
  module reads simulator outputs or `AirState` values; none writes
  a Magnus, ideal-gas, or ODE expression.
- The single-event simulator produces IDENTICAL outputs to what it
  produces today. A regression test in `test_time_simulation.py`
  wires an empty schedule / zero sources / zero heating and
  confirms the trajectory equals the existing simulator's output at
  every step within floating-point precision.
- Sensor-data-shaped inputs are validated at the boundary of every
  new module in the same way `Room` and `ThermalProperties` validate
  their inputs today (`isfinite`, non-negative where applicable,
  range checks documented).

## Not proposed here

None of the following are in this architecture proposal, and any of
them requires its own separate design step:

- Machine learning (of any kind).
- Automatic actuation of a physical window or a physical heater.
- Cost / tariff modelling.
- Multi-zone / multi-room airflow.
- Dashboards, notification integrations, web APIs.
- Cloud integration.

Each of those adds trust and safety obligations that this repository
is not equipped to meet today, and they should wait until the seven
proposed modules are implemented, validated, and calibrated against
real measurements.
