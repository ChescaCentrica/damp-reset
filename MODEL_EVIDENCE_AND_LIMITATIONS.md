# MODEL_EVIDENCE_AND_LIMITATIONS

**Purpose.** This document is a stocktake of every parameter and threshold the
damp-reset POC currently uses, so that no reader mistakes an illustrative
placeholder for a validated value. Every entry says:

- what the parameter is,
- the current value in the code and its unit,
- where it lives (file / symbol),
- the **source** of that value,
- whether it is **measured**, **fitted**, from **literature**, or an **arbitrary POC
  assumption**,
- a coarse **sensitivity** rating (how much downstream behaviour changes if the
  value is off),
- whether it **needs validation** before any real deployment.

Nothing in this repo is a validated risk threshold, energy budget, or comfort
guideline. Every "POC assumption" below is a knob the caller supplies, chosen so
the pipeline runs end-to-end and the experiments show visibly different
behaviour — not because we know the right number.

Provenance abbreviations (used in the "Source type" column):

- **PHYSICS** — established residential-HVAC physics constant, tabulated in
  standard references (ASHRAE Fundamentals, WMO). No POC judgement in the
  value; changing it means the physics is wrong.
- **LITERATURE** — number sourced from published building-services / mould /
  psychrometric literature, but the exact appropriate value for a specific
  room / occupant / study is not universal. Cite and defend before deployment.
- **POC PLACEHOLDER** — arbitrary value chosen so the pipeline runs and
  experiments produce visibly different behaviour. Not defended by any
  measurement or study. Must be replaced with a defended value in any real
  deployment.
- **CALLER INPUT** — no default in the code; the value is entirely on the
  caller. Listed here because a caller who does not defend their choice
  produces an equally unvalidated result.
- **FITTED** — can be derived from measurements via one of the calibration
  modules. If the fit was not run on the caller's specific room, the value
  falls back to POC PLACEHOLDER territory.

Sensitivity ratings are intentionally coarse: **LOW / MED / HIGH**. They
describe how much the ventilation controller's recommendation shifts across a
plausible range of the parameter, based on the experiments already in this repo.
None of them are formal sensitivity analyses.

---

## 1. Physics constants (do not treat as POC assumptions)

These are conventional residential-HVAC physical constants. Changing them
would mean the model no longer represents air, water vapour, or the ideal-gas
law; they are listed here for completeness, not because they are up for
review.

| Parameter | Value | Unit | File / symbol | Source type | Notes |
|---|---|---|---|---|---|
| Magnus coefficient A | 17.625 | — | `psychrometrics.MAGNUS_A` | PHYSICS | Alduchov & Eskridge (1996), Magnus form over liquid water. |
| Magnus coefficient B | 243.04 | °C | `psychrometrics.MAGNUS_B` | PHYSICS | As above. |
| Saturation vapour pressure at 0 °C | 610.94 | Pa | `psychrometrics.P_SAT_0` | PHYSICS | Same fit. |
| Molar mass of water | 0.018015 | kg / mol | `psychrometrics.M_WATER` | PHYSICS | Standard IUPAC. |
| Universal gas constant | 8.31446 | J / (mol·K) | `psychrometrics.R_UNIVERSAL` | PHYSICS | CODATA. |
| Water–dry-air molar mass ratio | 0.621945 | — | `psychrometrics.MW_RATIO` | PHYSICS | ASHRAE Handbook of Fundamentals. |
| Default atmospheric pressure | 101 325 | Pa | `psychrometrics.DEFAULT_ATM_PRESSURE_PA` | PHYSICS | Standard sea level. |
| Air density (at 20 °C, 1 atm) | 1.204 | kg / m³ | `thermal.AIR_DENSITY_KG_PER_M3` | PHYSICS | Ideal-gas density; ±5 % across the residential range. |
| Air specific heat at constant pressure | 1005 | J / (kg·K) | `thermal.AIR_SPECIFIC_HEAT_J_PER_KG_K` | PHYSICS | ASHRAE Handbook of Fundamentals; <1 % variation across residential range. |

**Validation needed?** No — these are physics constants, not POC decisions.

---

## 2. Model parameters the caller must supply (or fit)

These are the two knobs that change from room to room. The POC does **not**
ship defaults for either; the caller supplies them, and the calibration
modules can fit both from a controlled ventilation event.

### 2.1 Effective air-change rate — **ach_window_open** and **ach_closed**

| Attribute | Value |
|---|---|
| Parameter | ACH (air changes per hour) for window-open events and for the closed-window background |
| Current value | **caller-supplied** (`Room.ach_window_open`, `Room.ach_closed`) |
| Unit | h⁻¹ |
| Where used | Everything downstream of `moisture.predict_final_absolute_humidity` and `thermal.predict_indoor_temperature`; the time-domain simulator (`time_simulation`) and every optimiser strategy consume it. |
| Source type | CALLER INPUT (or FITTED via `calibration.estimate_ach_from_observations`) |
| Illustrative values seen in experiments | `ach_closed` typically 0.3–0.5, `ach_window_open` typically 5.0. |
| Sensitivity | **HIGH.** The ACH sets the moisture and thermal time constants of the room. A factor-of-two error propagates directly into the predicted final indoor AH and the predicted energy penalty of every candidate action. |
| Validation needed? | **YES** for any real deployment. The `calibration` module can recover ACH from a controlled window-open event; run it on the caller's room. Bounds: default search range 0.05–50 h⁻¹ (`calibration.DEFAULT_ACH_SEARCH_MIN / _MAX`). |
| Failure modes if wrong | Optimiser recommends actions that either under- or over-ventilate; every energy penalty is off by the same factor; risk indicator is biased. |

### 2.2 Effective thermal capacitance — **effective_thermal_capacity_j_per_k**

| Attribute | Value |
|---|---|
| Parameter | Lumped effective heat capacity of the room and its coupled contents |
| Current value | **caller-supplied** (`ThermalProperties.effective_thermal_capacity_j_per_k`); an **illustrative** 500 000 J/K is exported at `thermal.ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K` for demos only |
| Unit | J / K |
| Where used | Thermal ODE (`thermal.predict_indoor_temperature`), heating trajectory, energy-penalty calculations, thermal calibration. |
| Source type | CALLER INPUT (or FITTED via `thermal_calibration.estimate_effective_thermal_capacity_from_observations`) |
| Notes | Air-only capacity for a 40 m³ room is ~48 kJ/K. Real rooms include the fast-responding fabric (surface layers of walls, floor, furniture, textiles) and are typically ~10× larger. The 500 000 J/K illustrative value is roughly that; NOT calibrated against any specific building. |
| Sensitivity | **HIGH.** Sets the thermal time constant τ = C_eff / H_vent. A factor-of-two error changes how far the room cools per minute of window-open time and therefore every energy-penalty prediction. |
| Validation needed? | **YES** for any real deployment. `thermal_calibration` can recover C_eff from the same controlled event used for the ACH fit (default search bracket 10 kJ/K – 50 MJ/K). |
| Failure modes if wrong | Predicted temperature drops are wrong, energy penalties are wrong, and the heating model over- or under-supplies. Fitted ACH ends up entangled with the C_eff estimate because both control the observed decay time. |

---

## 3. Surface / risk parameters

### 3.1 Surface temperature factor — **fRsi**

| Attribute | Value |
|---|---|
| Parameter | Ratio (T_surface − T_outdoor) / (T_indoor − T_outdoor); zero for a completely uninsulated surface, one for a surface at indoor air temperature |
| Current value | **caller-supplied** (`SurfaceDescriptor.surface_temperature_factor`); no default exported |
| Unit | dimensionless (in [0, 1]) |
| Where used | `surface_risk.surface_temperature_c` → `mould_risk.evaluate_moisture_risk` → every risk-constrained optimiser strategy. |
| Source type | LITERATURE for the *definition* (BS EN ISO 13788 defines fRsi); CALLER INPUT for the specific value |
| Illustrative values used in experiments | 0.65 (severe thermal bridge / cold corner), 0.72 (cold external wall), 0.75, 0.80, 0.90 (well-insulated wall). None calibrated to a real surface. |
| Sensitivity | **HIGH** on the risk indicator: fRsi below 0.7 pushes the surface RH sharply upward under any realistic winter case, changing when the optimiser will act. Comfort / energy predictions are unaffected. |
| Validation needed? | **YES.** ISO 13788 gives methods for computing fRsi from wall U-values and surface resistances; alternatively an infrared surface-temperature measurement at known indoor/outdoor conditions yields it directly. Neither is done in this repo. |
| Failure modes if wrong | Predicted surface RH — and therefore the cumulative risk score — is systematically over- or under-stated. The optimiser will act too eagerly or too late. |

### 3.2 Elevated surface RH threshold

| Attribute | Value |
|---|---|
| Parameter | Surface RH above which a surface counts as "exposed" for the risk indicator |
| Current value | **default 80.0** (`RiskConfig.elevated_surface_rh_threshold_percent`) |
| Unit | % |
| Where used | `mould_risk.evaluate_moisture_risk` (contributes to `time_above_surface_rh_threshold_hours`). |
| Source type | LITERATURE-inspired POC PLACEHOLDER. 80 % is commonly cited (Sedlbauer thesis, ISO 13788 informative notes) as a broad "elevated risk" indicator, but real growth thresholds are species-specific and temperature-dependent. |
| Sensitivity | **MED–HIGH.** Choice of 70 % vs 80 % vs 85 % changes which candidate actions clear the "risk ceiling" test and therefore what the optimiser recommends. |
| Validation needed? | **YES** before any deployment claim. Even the illustrative 80 % is not a claim about mould growth. See `mould_risk.py` docstring: the score is an *indicator*, not a prediction. |
| Failure modes if wrong | Comparison across scenarios remains internally consistent but the absolute number cannot be interpreted as risk of biological growth. |

### 3.3 Condensation surface RH threshold

| Attribute | Value |
|---|---|
| Parameter | Surface RH at or above which a surface counts as "condensing" |
| Current value | **default 100.0** (`RiskConfig.condensation_surface_rh_threshold_percent`) |
| Unit | % |
| Where used | `mould_risk.evaluate_moisture_risk` (contributes to `time_in_condensation_hours`). |
| Source type | PHYSICS — 100 % RH is the physical dew point boundary, no POC judgement needed. Callers who want to include *near*-saturated conditions may lower it to e.g. 95 %. |
| Sensitivity | **LOW** for the physical case; **MED** if the caller chooses to lower it. |
| Validation needed? | Not for the physics. If the caller lowers it, they should defend the chosen value. |
| Failure modes if wrong | Time-in-condensation is over- or under-counted at the margin. |

### 3.4 Allowed exposure duration (implicit — via risk-ceiling weights)

The POC does **not** ship an "allowed hours above threshold" number
directly. It expresses exposure as a weighted sum on `RiskConfig`:

| Attribute | Value |
|---|---|
| Parameter | Weighting between accumulated exposure hours and the "cumulative risk score" ceiling |
| Current values | `elevated_time_weight = 1.0`, `condensation_time_weight = 1.0`, `peak_rh_excess_weight_hours_per_percent = 0.0` (all defaults on `RiskConfig`) |
| Unit | dimensionless (weights) |
| Where used | `mould_risk.evaluate_moisture_risk`; the resulting `cumulative_risk_score` is compared against `VentilationConstraints.max_cumulative_risk_score` in the optimiser. |
| Source type | POC PLACEHOLDER. There is no established residential-scale mapping from "hours above 80 % surface RH" to a risk score. |
| Illustrative ceilings used in experiments | 1.0–5.0 across the risk-constrained experiments (see `forecasting_matters.py`, `risk_metric_vs_ventilation.py`). |
| Sensitivity | **HIGH.** Cross-scenario comparisons of the cumulative score are only meaningful under a fixed set of weights; different weights re-order which controller wins. |
| Validation needed? | **YES.** The `mould_risk` docstring names this explicitly: a full published mould-growth model (VTT, Sedlbauer isopleth, ASHRAE 160, WUFI-Bio) would replace the weighted-sum indicator with a species- and temperature-specific integral. Not in scope for this POC. |
| Failure modes if wrong | The indicator is a valid *comparison* tool across two strategies in the same scenario; using its absolute magnitude to claim biological growth is not justified. |

---

## 4. Moisture generation rates

| Attribute | Value |
|---|---|
| Parameter | Background moisture-generation rate + per-event rates (showers, cooking, laundry, occupancy) |
| Current value | **caller-supplied** (`MoistureSourceSchedule.constant_background_rate_g_per_hour` and `MoistureSourceEvent.generation_rate_g_per_hour`); no defaults exported |
| Unit | g / hour |
| Where used | `moisture_sources`; consumed by every time-domain simulation and by the optimisers that route through it. |
| Illustrative values used in experiments | 40–100 g/h background; 400 g/h cooking; 1500 g/h shower; 2000 g/h in the more extreme demo cases. |
| Source type | POC PLACEHOLDER. The `moisture_sources` docstring says explicitly: this repo does **not** ship authoritative moisture-generation values. Real numbers depend on kitchen ventilation, shower duration, laundry load, and occupant metabolic rate. |
| Reference literature | ASHRAE Handbook of Fundamentals and CIBSE Guide A tabulate residential moisture generation rates; those numbers still have factor-of-two spreads across occupancy assumptions. |
| Sensitivity | **HIGH.** The whole reason for opening the window in this model is to remove moisture; a factor-of-two error in the source term propagates directly into the predicted final indoor AH and into how often the optimiser recommends ventilating. |
| Validation needed? | **YES.** For a specific room + occupant, use measured indoor AH with the window closed and no active occupancy events to back out the effective steady-state background rate. Per-event rates can be estimated from the AH transient during a known-duration event. Neither is done in this repo. |
| Failure modes if wrong | Predicted risk trajectory drifts; controller under- or over-ventilates. |

---

## 5. Comfort and energy caps (optimiser constraints)

### 5.1 Maximum temperature drop — **max_temperature_drop_c**

| Attribute | Value |
|---|---|
| Parameter | Ceiling on the indoor temperature drop the room may experience during a single ventilation event |
| Current value | **caller-supplied** on `VentilationConstraints`; every experiment sets its own POC value |
| Unit | K (or equivalently °C, since it is a temperature *difference*) |
| Where used | Optimiser feasibility check (`optimiser._check_feasibility`); every risk-constrained strategy. |
| Illustrative values used in experiments | 1.0, 2.0, 3.0, 5.0, 8.0 K depending on the experiment. |
| Source type | POC PLACEHOLDER. ISO 7730 and EN 16798 discuss thermal comfort in terms of PMV / PPD and operative temperature, not a single "K drop" budget. |
| Sensitivity | **HIGH** on the recommended action. Setting the cap at 1 K blocks most winter ventilation actions; setting it at 5 K unblocks them. |
| Validation needed? | **YES** if a caller intends to claim comfort compliance. A defended comfort criterion would probably be operative-temperature-based, not raw-air-temperature-drop-based, and would depend on occupant clothing / activity. |
| Failure modes if wrong | Either the optimiser refuses actions that a real occupant would accept, or it recommends actions the occupant finds uncomfortable. |

### 5.2 Maximum energy loss — **max_energy_loss_kwh**

| Attribute | Value |
|---|---|
| Parameter | Ceiling on the ventilation-event thermal energy loss |
| Current value | **caller-supplied** on `VentilationConstraints` |
| Unit | kWh (thermal) |
| Where used | Optimiser feasibility check; used by `optimise_max_moisture_under_energy_budget` and as an optional constraint on the risk-constrained strategies. |
| Illustrative values used in experiments | 0.15 kWh in the Pareto / budget experiments. |
| Source type | POC PLACEHOLDER. There is no built-in "how many kWh may an event cost". |
| Sensitivity | **HIGH.** Below ~0.05 kWh, no meaningful winter ventilation event is feasible; above ~1 kWh the budget is effectively unbounded. |
| Validation needed? | **YES.** A real occupant's tolerance depends on tariff, comfort priorities, and whether the heating system compensates. This POC does not translate purchased energy into cost. |
| Failure modes if wrong | Over- or under-restrictive; no direct physical harm because the constraint is preference-shaped. |

### 5.3 Maximum cumulative risk score — **max_cumulative_risk_score**

| Attribute | Value |
|---|---|
| Parameter | Ceiling on the horizon-wide cumulative surface-risk indicator |
| Current value | **caller-supplied** on `VentilationConstraints` |
| Unit | dimensionless (indicator, see §3.4) |
| Where used | `optimise_min_energy_under_risk_limit`, `optimise_scheduled_action_under_risk_limit`. |
| Illustrative values used in experiments | 1.0–5.0 across the risk-constrained experiments. |
| Source type | POC PLACEHOLDER, layered on top of §3.4 (the caller-configured indicator itself is a POC composition). |
| Sensitivity | **HIGH.** Choice of ceiling is what the whole optimiser output pivots on. |
| Validation needed? | **YES.** Requires the underlying indicator to be defended first (§3.4). |
| Failure modes if wrong | Same as §3.4: cross-scenario comparisons are internally consistent, absolute magnitudes are not interpretable as biological risk. |

---

## 6. Control-loop and forecast parameters

### 6.1 Control horizon — **control_horizon_hours**

| Attribute | Value |
|---|---|
| Parameter | How far ahead the optimiser simulates when evaluating candidate actions |
| Current value | **caller-supplied** (typically 4–6 h in the experiments) |
| Unit | h |
| Where used | `optimise_min_energy_under_risk_limit`, `optimise_scheduled_action_under_risk_limit`, every experiment that runs a controller. |
| Source type | POC PLACEHOLDER, informed by two constraints: (i) the forecast module treats outdoor conditions as piecewise-constant and does not extrapolate beyond its last point, so the horizon should not exceed the forecast length; (ii) longer horizons integrate more moisture drift, so risk-ceiling and background-generation choices interact with it. |
| Sensitivity | **MED.** A 4 h vs 6 h vs 12 h horizon changes which "wait for the milder weather" actions become feasible; the demonstration in `forecasting_matters.py` uses 6 h. |
| Validation needed? | **YES** in the sense that the caller should pick a horizon appropriate to their forecast reliability. There is no universally correct value. |
| Failure modes if wrong | Too-short horizon means the optimiser cannot see the milder weather it should wait for; too-long horizon means it plans against forecast points that no longer accurately represent conditions. |

### 6.2 Trajectory timestep — **trajectory_timestep_minutes**

| Attribute | Value |
|---|---|
| Parameter | Time step of the operator-split trajectory simulator |
| Current value | **caller-supplied** (typically 2–5 min in the experiments) |
| Unit | min |
| Where used | `time_simulation.simulate_room_period` and its forecast / heating variants; consumed by every optimiser strategy that runs a trajectory. |
| Source type | POC PLACEHOLDER, chosen by the "at least ten steps per shortest time constant" rule of thumb noted in the `time_simulation` docstring. |
| Sensitivity | **LOW** provided the timestep is at least ~5× smaller than the ACH time constant. Larger timesteps introduce operator-splitting error; smaller timesteps only cost CPU. |
| Validation needed? | Not physically, but should be sanity-checked against the fitted ACH. At ACH = 5 h⁻¹ the time constant is 12 min; a 5-min step is fine. At ACH = 30 h⁻¹ the time constant is 2 min and 5-min steps would drift noticeably. |
| Failure modes if wrong | Operator-split error on the moisture and thermal ODEs; grows super-linearly as the timestep approaches the time constant. |

### 6.3 Forecast sampling — piecewise-constant, START-of-interval

| Attribute | Value |
|---|---|
| Parameter | Semantics of the outdoor forecast between explicit `ForecastPoint`s |
| Current value | Hard-coded piecewise-constant, START-of-interval, no extrapolation beyond the last point |
| Where used | `weather_forecast.WeatherForecast.sample_at`; the risk-constrained scheduled optimiser and every heating / trajectory function that consumes it. |
| Source type | POC PLACEHOLDER (choice of interpolation scheme). Matches every other piecewise-constant convention in the pipeline (window state, moisture rate). |
| Sensitivity | **LOW** for short horizons (~6 h) with hourly forecast points; **MED** for longer horizons where the outdoor T ramp between points is significant. |
| Validation needed? | A production system with real hourly-forecast data would probably want linear interpolation for temperature; the piecewise-constant choice deliberately keeps the semantics simple for the POC. |
| Failure modes if wrong | Predicted energy penalty for a delayed vent event uses the T at the segment start, not the average across the event; small bias in edge cases. |

---

## 7. Heating model parameters

### 7.1 Setpoint temperature — **setpoint_temperature_c**

| Attribute | Value |
|---|---|
| Parameter | Target indoor temperature for the thermostat |
| Current value | **caller-supplied** on `ThermostaticHeating`; every experiment sets its own POC value (typically 20 °C). |
| Unit | °C |
| Source type | CALLER INPUT (personal preference). |
| Sensitivity | **HIGH** on purchased energy — a 1 °C setpoint shift changes heating-supplied energy meaningfully. |
| Validation needed? | No; this is a user preference, not a modelling assumption. |
| Failure modes if wrong | Simulator reproduces whatever setpoint the caller entered; there is nothing to validate. |

### 7.2 Maximum thermal power — **max_thermal_power_w**

| Attribute | Value |
|---|---|
| Parameter | Peak thermal output while the heater is ON |
| Current value | **caller-supplied** (e.g. 1500–2000 W in experiments) |
| Unit | W (thermal) |
| Source type | CALLER INPUT (nameplate / rated). |
| Sensitivity | **LOW** as long as it exceeds the room's steady-state loss; **HIGH** when it doesn't, because the room drifts below setpoint. |
| Validation needed? | No; a nameplate figure from the appliance is enough. |

### 7.3 Efficiency or COP — **efficiency_or_cop**

| Attribute | Value |
|---|---|
| Parameter | Ratio of delivered thermal power to input power |
| Current value | **caller-supplied**; 1.0 for resistive, ~0.9 for gas boilers, 2.5–4.5 for heat pumps |
| Unit | dimensionless |
| Source type | POC PLACEHOLDER. The `heating` module docstring says explicitly: this is not a validated performance figure for any specific appliance. Real heat pumps have COP that varies with outdoor T and part-load ratio; real gas boilers have flue-loss-dependent seasonal efficiency. |
| Sensitivity | **HIGH** on purchased energy. Choice of COP = 3 vs 4 changes the energy bill by 25 %. |
| Validation needed? | **YES** for anything downstream that claims a purchased-energy figure. Deriving a room-specific seasonal COP from measured input-energy vs delivered-thermal-energy data would replace this. |
| Failure modes if wrong | Purchased-energy estimate is off by the same factor; thermal delivery is unaffected. |

### 7.4 Hysteresis — **hysteresis_c**

| Attribute | Value |
|---|---|
| Parameter | Half-width of the thermostat dead-band |
| Current value | **caller-supplied**; typically 0.5 K in the experiments |
| Unit | K |
| Source type | POC PLACEHOLDER. Real thermostats have both hysteresis and time-based cycling; this POC only models the temperature dead-band. |
| Sensitivity | **LOW** on total energy; **MED** on the count of on/off cycles. |
| Validation needed? | Not critical for the POC; would need refinement if the goal were appliance-level control. |

---

## 8. Calibration search brackets

These are only relevant when a caller runs the fitters. They bound the
search interval for the fitted parameter.

| Parameter | Value | Unit | File / symbol | Source type | Sensitivity | Validation needed? |
|---|---|---|---|---|---|---|
| ACH search min | 0.05 | h⁻¹ | `calibration.DEFAULT_ACH_SEARCH_MIN` | POC PLACEHOLDER (informed by blower-door "tight" range) | LOW while inside the residential range; the fitter should not saturate here | The result reports the bracket used; check for saturation. |
| ACH search max | 50.0 | h⁻¹ | `calibration.DEFAULT_ACH_SEARCH_MAX` | POC PLACEHOLDER (headroom above realistic wide-open opening) | LOW; used as a cap | Same. |
| C_eff search min | 10 000 | J / K | `thermal_calibration.DEFAULT_C_EFF_SEARCH_MIN_J_PER_K` | POC PLACEHOLDER; well below air-only capacity for a modest room | LOW while inside residential range | Check for saturation in the result. |
| C_eff search max | 50 000 000 | J / K | `thermal_calibration.DEFAULT_C_EFF_SEARCH_MAX_J_PER_K` | POC PLACEHOLDER | LOW | Same. |
| Golden-section tolerance (ACH) | 1e-5 | h⁻¹ | `calibration.DEFAULT_GOLDEN_SECTION_TOLERANCE` | POC PLACEHOLDER, well below sensor resolution | LOW | — |
| Golden-section tolerance (C_eff) | 1.0 | J / K | `thermal_calibration.DEFAULT_GOLDEN_SECTION_TOLERANCE_J_PER_K` | POC PLACEHOLDER | LOW | — |

---

## 9. Not modelled (out of scope for this POC)

Listed here so a reader does not infer they exist:

- **Wall conduction, radiation, infiltration losses when the window is closed.**
  The thermal ODE uses ventilation as the only heat-transfer mechanism. Real
  buildings lose heat through walls too; this repo does not.
- **Solar gains.** A sunny winter day can offset ventilation loss substantially.
  Not modelled.
- **Occupant and equipment gains.** ~100 W per adult, more for cooking or high-
  power electronics. Not modelled.
- **Multi-zone airflow.** Every model in this repo is single-zone / well-mixed.
- **Variable-speed / modulating heating.** The heating model is bang-bang only.
- **Species-specific mould-growth models** (VTT, Sedlbauer isopleth, ASHRAE 160,
  WUFI-Bio). The risk indicator is a caller-composed weighted sum, not a growth
  prediction.
- **Weather forecast uncertainty.** Forecast values are treated as exact.
- **Actuators, sensors, hardware, dashboards, tariff / cost models, ML.**
  Explicitly excluded from every module.

---

## 10. How this document should be used

1. Before running the optimiser on a real room, read the "Validation needed?"
   column and check every entry that answers **YES**. That is the list of
   parameters that must be defended, not accepted from the code's defaults.
2. When comparing controllers in an experiment, note which parameters are
   held fixed and which are swept. Cross-scenario comparisons are only
   meaningful under a fixed calibration.
3. When reporting results outside this repo, quote the specific numeric
   values used and cite this file — so a reader can see which ones were
   POC placeholders rather than measurements.

This is a POC. Its purpose is to make the *shape* of a residential
ventilation controller visible end-to-end. It is not calibrated against any
specific building and does not claim to be.
