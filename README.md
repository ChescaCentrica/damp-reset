# damp-reset

Proof-of-concept for an intelligent residential ventilation system. Two
components exist so far:

- `psychrometrics.py` — moisture math for moist air (saturation curve,
  absolute humidity, humidity ratio, dew point, indoor-vs-outdoor drying
  potential).
- `moisture.py` — well-mixed single-zone ventilation moisture model.
  Predicts how indoor absolute humidity evolves when a window is open
  for X minutes, and reports the mass of water added to or removed from
  the room air.

Not in this repo: sensors, API, dashboard, database, machine learning,
weather forecasts, optimiser, thermal / energy model, mould-risk model,
moisture generation, multi-zone airflow, control logic.

## Purpose

Two questions this repo answers today:

1. *Given indoor and outdoor T and RH, how much water does each air
   mass carry, and does replacing indoor air with outdoor air remove or
   add moisture?* (`psychrometrics.py`)
2. *If a window is open for X minutes with an assumed ACH, how much
   moisture remains in the room, and how many grams of water leave the
   room air?* (`moisture.py`)

## Inputs

- Temperature — degrees Celsius (`_c`), residential range roughly
  −50 to +60 °C
- Relative humidity — percent on a 0–100 scale (`_percent` / `_pct`)
- Atmospheric pressure — pascals (`_pa`), defaults to 101 325 Pa
- Room volume — cubic metres (`_m3`)
- Air-change rate — hours⁻¹ (ACH)
- Ventilation duration — minutes

## Outputs

Psychrometric layer:
- Saturation vapour pressure — Pa
- Vapour pressure — Pa
- Absolute humidity — g/m³
- Humidity ratio — kg water / kg dry air
- Dew point — °C
- `DryingPotential` — indoor and outdoor absolute humidities, their
  difference in g/m³, and a POC heuristic category (`NONE` / `LOW` /
  `MODERATE` / `HIGH`)

Ventilation moisture layer:
- Final indoor absolute humidity — g/m³
- `MoisturePrediction` — initial / outdoor / final AH, signed change and
  reduction in g/m³, percentage reduction, signed water mass removed
  from the room air in grams, plus the inputs echoed for audit

Units are encoded in every parameter and return name; nothing is
inferred.

## Equations

Psychrometrics:

1. **Saturation vapour pressure** — Magnus form, Alduchov & Eskridge (1996):
   `P_sat(T) = 610.94 * exp(17.625 * T / (T + 243.04))  [Pa]`
2. **Vapour pressure** from RH:  `P_v = (RH/100) * P_sat(T)`
3. **Humidity ratio** — ASHRAE Handbook of Fundamentals, ch. 1:
   `W = 0.621945 * P_v / (P − P_v)`  (Dalton + ideal gas;
   0.621945 = M_water / M_dry_air)
4. **Absolute humidity** — ideal-gas on the vapour partial pressure:
   `AH = P_v * M_water / (R * T_K)`  (×1000 for g/m³)
5. **Dew point** — algebraic inverse of Magnus (same constants):
   `α = ln(P_v / 610.94);   T_d = 243.04 * α / (17.625 − α)`
6. **Drying potential** — `Δ = AH_indoor − AH_outdoor`; POC heuristic
   bands at 0 / 1 / 3 g/m³.

Ventilation moisture:

7. **Well-mixed moisture balance** (no internal sources):
   `dC/dt = n · (C_out − C)`
   with C = indoor absolute humidity, C_out = outdoor absolute humidity,
   n = ACH in hours⁻¹, t in hours.
8. **Analytic solution:**
   `C(t) = C_out + (C_0 − C_out) · exp(−n · t)`
   The room asymptotically equilibrates with outdoor air; time constant
   `τ = 1 / n`.
9. **Water mass removed from the room air:**
   `water_removed_g = (C_0 − C_final) · V_room`  (positive = removed)

## Example

```python
from psychrometrics import AirState
from moisture import Room, predict_room_moisture

room = Room(
    volume_m3=40.0,
    indoor_temperature_c=20.0,
    indoor_relative_humidity_pct=70.0,
    ach_closed=0.4,
    ach_window_open=5.0,
)
outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

result = predict_room_moisture(room, outdoor, duration_minutes=15.0)
print(result.final_absolute_humidity_g_m3)   # ~7.58 g/m^3
print(result.water_removed_g)                # ~179.9 g
```

Runnable walk-throughs:

- `examples/basic_example.py` — indoor vs outdoor psychrometrics, drying
  potential verdict
- `examples/moisture_duration_comparison.py` — table of AH and water
  removed at 0 / 2 / 5 / 10 / 15 minutes
- `experiments/indoor_ah_vs_time.py` — indoor AH vs duration curve
- `experiments/indoor_ah_vs_time_ach_sweep.py` — same, one curve per ACH
- `experiments/window_open_vs_closed.py` — infiltration vs active
  ventilation comparison

## Assumptions

Psychrometrics:

- Ideal-gas behaviour for dry air and water vapour
- Dalton's law: total pressure = dry-air partial pressure + P_v
- Saturation is over **liquid water**, not ice (standard meteorological
  convention below 0 °C, i.e. supercooled water)
- Near-1-atm regime; humidity ratio takes an explicit pressure argument
  for altitude corrections, absolute humidity is pressure-invariant in
  this formulation
- Category thresholds in `DryingPotential` are POC heuristics, not
  validated safety limits

Ventilation moisture (documented in full in `moisture.py`):

- Room air is perfectly mixed at every instant
- ACH is constant across a single event
- Outdoor absolute humidity is constant across the event
- No internal moisture generation (occupants, cooking, laundry)
- No moisture buffering, condensation, or evaporation at surfaces
- No inter-room airflow
- No temperature change during the event (AH is only strictly conserved
  under isothermal ventilation)

## Limitations

- Psychrometric range roughly −50 to +60 °C, 0–100 %RH, near sea level.
  Every entry point rejects out-of-range inputs.
- Water phase only — no ice/frost point, no sublimation.
- Magnus fit error ≈ 0.4 % at the range edges vs WMO / Wexler.
- `AirState` validates on property access, not at construction.
- `MoisturePrediction.water_removed_g` counts water leaving the AIR, not
  water leaving the building fabric — walls, textiles, and furniture
  hold orders of magnitude more water than the air and can re-wet the
  room over subsequent hours.
- Category thresholds (0 / 1 / 3 g/m³) are illustrative only; downstream
  logic should key off `difference_g_m3`, not `category`.
- ACH values quoted in the experiments (0.4, 1, 2, 5, 10 h⁻¹) are
  illustrative inputs, not calibrated measurements of specific windows.

## Layout

```
damp-reset/
  psychrometrics.py               air-state math + AirState + DryingPotential
  moisture.py                     Room + MoisturePrediction + ventilation model
  examples/
    basic_example.py                psychrometrics demo
    moisture_duration_comparison.py table of AH and water removed vs time
  experiments/
    indoor_ah_vs_time.py            AH-vs-time plot for one ACH
    indoor_ah_vs_time_ach_sweep.py  AH-vs-time plot for several ACH
    window_open_vs_closed.py        infiltration vs active ventilation
  test/
    test_psychrometrics.py          reference-value + consistency tests
    test_moisture.py                Room, physics, wiring, and integration tests
  outputs/                          saved plots (created on first run)
  requirements.txt                  matplotlib, pytest
  README.md
```

## Run

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest test/
python examples/basic_example.py
python examples/moisture_duration_comparison.py
python experiments/indoor_ah_vs_time.py
python experiments/indoor_ah_vs_time_ach_sweep.py
python experiments/window_open_vs_closed.py
```
