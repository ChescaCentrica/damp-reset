# HYGROSCOPIC_BUFFERING_DESIGN_ASSESSMENT

**Purpose.** This document is a design assessment, not an implementation
proposal. It asks: does the damp-reset POC's current well-mixed
air-only moisture model materially mis-predict the quantities the
optimiser cares about — RH peaks, drying rate, post-ventilation
rebound, required ventilation duration — because it neglects moisture
exchange with walls, furniture, and textiles? If yes, what is the
simplest possible extension, and what data would identify its
parameters? If no, don't build it.

**Current model.** Every module in this repo — moisture, thermal,
time-domain simulator, risk indicator, all four optimiser strategies —
treats the room as a single well-mixed air node with one state
variable (indoor absolute humidity), governed by

    dC/dt = ach * (C_out - C) + G / V

with `G` the moisture-generation rate and `V` the room volume. No
storage in walls, curtains, plasterboard, timber, textiles, or
furnishings. See `moisture.py` module docstring, assumption 1
("well-mixed room"), which names the omission but does not quantify
its effect.

**Standing project rule.** Do not add complexity beyond what the task
requires. A buffering model is a real cost in code, calibration
burden, and interpretability. The rest of this document is the
evidence trail we would need before spending that cost.

---

## 1. What hygroscopic buffering actually does

Building fabric and contents adsorb moisture when the room air is
humid and release it when the air is dry. Empirically:

- The **effective moisture capacity** of a furnished residential
  room is typically **several times** larger than the moisture held
  in the air itself. Values in the published sorption-isotherm
  literature (Sedlbauer; Kunzel; ASHRAE 160 informative annex; the
  UK BRE Digest 245 range) suggest a room-air moisture inventory of
  ~0.5 kg water in a 40 m³ room at ~12 g/m³ AH, while the
  fabric-and-contents adsorbed inventory can be 2–10× that at
  equilibrium — dominated by soft furnishings, exposed timber, and
  unpainted plasterboard.
- Adsorption / desorption is **not instantaneous**. Timber and thick
  plasterboard release moisture on a timescale of hours to days;
  textiles and paper on the order of ~15 minutes to a few hours.
  Painted, sealed, or metallic surfaces are effectively inert.
- The exchange direction depends on whether **surface material RH**
  exceeds or trails **air RH**, not on absolute humidity levels.

The upshot is that in a real furnished room, the moisture the caller
would like to remove during a ventilation event is a mix of:

1. Water currently in the air (fast to remove).
2. Water in the fast-responding surface layers of textiles,
   plasterboard, timber (releases over the event, partially).
3. Water bound in the deep fabric (releases over days).

Only (1) is currently modelled. (2) and (3) are what the well-mixed
air-only assumption discards.

---

## 2. Predicted qualitative effects of the omission

The bulleted claims below are the *direction* of the error under
each quantity. Magnitudes are experimental questions (see §5); the
signs are what physics alone tells us.

### 2.1 Predicted RH peaks (during a moisture event)

- **Direction of error: model OVER-PREDICTS peak indoor RH.**
- **Reason:** When a shower or cooking event dumps water vapour into
  the air, real walls and textiles start adsorbing it within
  minutes; the fabric acts as a first-order sink that flattens the
  transient. The air-only model has no sink, so the whole moisture
  spike goes into the single air node.
- **Downstream implication:** The `mould_risk` indicator, which is
  driven by the surface-RH computation and therefore by the indoor
  AH transient, overstates the peak surface RH the room actually
  experiences. Under a tight risk ceiling the optimiser may act too
  eagerly.

### 2.2 Drying rate (while the window is open)

- **Direction of error: model OVER-PREDICTS how fast the room dries
  in the SHORT term.**
- **Reason:** During the vent event, adsorbed moisture in the fabric
  desorbs *back into the air* as the air dries — the fabric slows
  down the observed decrease in indoor AH. The air-only model
  predicts a clean first-order exponential decay; the real trace is
  a decay with a "long tail" driven by the fabric release.
- **Downstream implication:** For a fixed target final indoor AH,
  the optimiser predicts a shorter ventilation duration than
  reality requires. Post-event indoor AH sits *higher* than the
  air-only model expected.

### 2.3 Post-ventilation rebound (after the window closes)

- **Direction of error: model UNDER-PREDICTS the rebound.**
- **Reason:** After the window closes, the fabric has partially
  desorbed and now sits at a moisture content below equilibrium
  with the (drier) room air. As the room re-equilibrates with the
  moisture source AND with the fabric, the air AH rebounds faster
  than pure background-source physics would predict. The air-only
  model has no reservoir left to catch up; it predicts a slow drift
  driven only by the constant background rate.
- **Downstream implication:** The `predict_final_absolute_humidity`
  used in the optimiser's post-event trajectory underestimates how
  quickly the room re-humidifies. A candidate action that looks
  "safe for the full 6 h horizon" against the air-only model may in
  reality re-cross the risk ceiling within an hour or two.

### 2.4 Required ventilation duration

- **Direction of error: model UNDER-PREDICTS the duration needed to
  reach a given target final indoor AH.**
- **Reason:** Combined effect of 2.2 (slower observed drying) and
  2.3 (faster rebound). To hit the same 6-hour-average risk score,
  a real event needs to be longer, and/or repeated. This is exactly
  the case where the risk-constrained optimiser's recommendation is
  systematically off in the direction of "not enough ventilation".

### 2.5 Summary table

| Quantity                     | Sign of error (air-only vs reality) | Mechanism                            |
|------------------------------|-------------------------------------|--------------------------------------|
| Peak indoor RH during spike  | OVER                                | Fabric adsorbs during the spike      |
| Short-term drying rate       | OVER                                | Fabric desorbs into drying air       |
| Post-event rebound           | UNDER                               | Fabric desorbs into low-RH air       |
| Required vent duration       | UNDER                               | Combination of 2.2 and 2.3           |

---

## 3. When these errors matter for THIS POC

Not every use of the model is equally affected.

- **Short vent events under mild moisture background.** The fabric
  reservoir barely moves. Air-only model is fine.
- **Sharp moisture spikes (shower / cooking) followed by short
  vents.** All four effects above kick in. This is the regime where
  the air-only model is most misleading.
- **Comparing two candidate ventilation strategies against each
  other in the same room.** Ordering is largely preserved: both
  strategies get the same systematic bias. The `forecasting_matters`
  and `risk_metric_vs_ventilation` experiments are internally
  consistent even with the biased air-only model.
- **Absolute predictions of final indoor AH / post-event RH /
  purchased energy against a real room.** Air-only model is
  quantitatively wrong; the ACH calibrator will absorb some of the
  bias into a wrong-but-fit ACH.
- **The `calibration.estimate_ach_from_observations` fit itself.**
  The fitter assumes the air-only ODE. When run on a real event
  with active fabric buffering, the residual sits on the fabric-
  release tail and the fitted ACH is a *biased-toward-slower* number.
  See §5.2.

The design decision therefore hinges on whether the caller intends to
use the model for **strategy comparison** (buffering probably doesn't
change conclusions) or for **absolute prediction against a real
sensor** (buffering probably changes conclusions).

---

## 4. Proposed simplest possible extension (design only — DO NOT
implement yet)

The simplest defensible extension is a **two-node moisture model**
with a single buffer reservoir. This adds one state variable, one
parameter, and one coupling coefficient — nothing else.

### 4.1 Model shape

Two mass balances:

    dC_air / dt = ach * (C_out - C_air) + G / V + k * (C_buf - C_air)      [air node]
    dC_buf / dt =                              - (V / M_buf) * k * (C_buf - C_air)   [buffer]

where:

- `C_air`  [g/m³]   air-node absolute humidity (what all downstream
                     code already consumes).
- `C_buf`  [g/m³]   equivalent-air absolute humidity representing the
                     buffer's current moisture state (a fictitious
                     concentration; the equivalence is defined by
                     `M_buf` below).
- `k`      [h⁻¹]    caller-configurable **exchange coefficient**
                     between air and buffer. Sets the buffer's
                     response time constant τ_buf = 1 / k. Illustrative
                     residential range from the literature: k in
                     ~0.5–5 h⁻¹ (fast textiles at the top; timber /
                     plasterboard at the bottom).
- `M_buf`  [m³ air-equivalent]  the buffer's **moisture capacity**
                     expressed as the volume of air-at-current-AH that
                     would hold the same water content. Multiplying by
                     the room volume gives the buffer mass ratio: a
                     `M_buf / V` of 4 means the buffer holds four
                     times as much water as the room air at
                     equilibrium.
- Everything else exactly as in the existing `moisture.py`.

The coupling term `k * (C_buf - C_air)` says: when the buffer is
"wetter" than the air, moisture flows out of the buffer into the air
(as during ventilation drying, causing the long tail); when the air
is wetter, moisture flows into the buffer (as during a shower spike,
flattening the peak). The `V / M_buf` factor on the buffer equation
enforces the mass conservation that the air balance implies.

### 4.2 Why this shape and not something more elaborate

- **Two nodes, not three or four.** A single exponential is the
  simplest way to capture *both* the flattened peak and the slow
  rebound; a caller who wants to model separately-responding
  textiles-and-plasterboard-vs-timber can always split it later.
- **No sorption isotherm.** A real isotherm is nonlinear
  (Sedlbauer / Kunzel curves). The linear buffer above is a first-
  order approximation valid within a narrow RH band; a fuller
  isotherm-based model (e.g. Kunzel's WUFI framework) is
  substantially more complex and requires per-material identification
  data. The linear buffer is the *smallest* useful extension.
- **Single scalar coefficient.** `k` and `M_buf` are two numbers the
  caller sets; both have direct physical interpretations and both
  can be fitted from a single controlled event (see §5).

### 4.3 What this would touch in the codebase

- `moisture.py` grows a second ODE. The existing
  `predict_final_absolute_humidity` function stays as the "air-only
  fast path"; a new `predict_final_absolute_humidity_with_buffer`
  function integrates the 2-D linear system analytically (both
  eigenvalues real, both negative for `k > 0` and `M_buf > 0`).
- `time_simulation.py` grows an optional `BufferProperties` argument
  and threads `C_buf` through the trajectory alongside `C_air`. The
  existing single-air-node function stays as the default.
- `moisture.Room` grows an optional `buffer_properties`; when unset,
  the model is exactly today's model.
- `calibration.py` gains a companion fitter
  `estimate_ach_and_buffer_from_observations` that fits three
  scalars — ACH, `k`, `M_buf` — jointly from a controlled event.
  See §5 for the identifiability question.

**Nothing in `optimiser.py`, `mould_risk.py`, `surface_risk.py`, or
the heating model needs to change** — those layers consume indoor AH
from the trajectory; the trajectory just becomes more accurate.

### 4.4 Why NOT to build this now

- The POC does **not** yet have a validated air-only model on a real
  room. `validation.py` and `experiments/validate_from_csv.py` were
  added exactly to make this measurement possible, but they have not
  been run against real sensor data. Until the air-only model has
  been evaluated on a measured event with the calibration/validation
  split, we don't know whether buffering is the dominant unmodelled
  effect or a second-order correction hidden behind bigger issues
  (sensor drift, wrong effective volume, unmeasured moisture
  sources, ACH varying with wind).
- Building the two-node model without that evidence bakes in a
  specific structural choice (one buffer, linear coupling) that we
  can't defend from data. A future measurement might show a
  bimodal response (fast textile + slow deep timber) that a single
  buffer cannot fit.
- Every extra parameter is a knob a caller can get wrong. Two extra
  knobs (`k`, `M_buf`), fitted from short single events, are prone
  to running to bracket edges when the event is uninformative
  about them (§5.2).

### 4.5 The rule for when to build this

Build the two-node buffer model when at least one of the following
is true and documented with real-sensor data:

1. **The RMS validation error of the air-only model on a real event
   exceeds ~0.5 g/m³** (about 5 % of a typical residential indoor
   AH) AND the fitted ACH from `calibration` sits at a
   suspiciously low value (i.e. the fitter absorbed the buffer's
   long tail into a wrong-but-fit ACH).
2. **The residual on the AH validation plot has visible structure**
   (a distinct long-time-tail during window-open, or a fast rebound
   after window-close) — not white noise. Structure in the residual
   is the signature that a physical effect is missing.
3. **A caller explicitly reports that the model's post-event rebound
   prediction diverges from a real sensor** within 1–2 h of the
   window closing (§2.3 above), AND that divergence changes the
   controller's next-action recommendation.

None of these have been checked yet. Doing them requires only running
`experiments/validate_from_csv.py` on a real CSV; that is cheaper
than building the buffer model.

---

## 5. Experimental data required to identify the buffering parameters

If §4.5 fires and the two-node model does need to be built, here is
what would need to be measured. Numbers are illustrative for a
single-room domestic case.

### 5.1 Single-event protocol (identifies ACH, k, M_buf jointly)

Recommended experiment (mirrors the ACH calibration protocol
already documented in `calibration.py`, extended with a rebound
phase):

1. **Pre-event conditioning: at least 4 hours undisturbed.** Room
   closed, no occupants doing thermally or moisture-significant
   things, heating off or thermostatically stable. This gives the
   air and the buffer time to sit near equilibrium so the fit does
   not have to guess the initial buffer state.
2. **Baseline logging: 30 minutes of closed-window observations
   before the event.** Sample indoor T, indoor RH, outdoor T,
   outdoor RH at least every 2 minutes. Compute indoor and outdoor
   AH via `psychrometrics.AirState`.
3. **Controlled moisture pulse (optional but strongly informative).**
   A known 5-minute moisture release — for example a measured
   volume of boiled water evaporated at a known rate — spikes the
   air AH and gives the fit the "shower-like transient" needed to
   identify `k` and `M_buf`. Without a spike the fit sees only the
   decay half of the response and `M_buf` becomes weakly
   identifiable.
4. **Window-open event: 20–40 minutes.** Same protocol as the
   existing single-ACH-event calibration. Log throughout.
5. **Rebound phase: 60+ minutes of closed-window observations after
   the window closes.** This is the *new* required phase — the air-
   only calibrator does not need it, but the buffer's `k` and
   `M_buf` are identified primarily from the rebound curve.
6. **All along:** log window state, log any occupancy events, log
   heating state, log outdoor T / RH from a nearby weather station.

### 5.2 Identifiability caveat

- With only the window-open decay (steps 1, 2, 4), the fit can
  identify ACH cleanly but the two buffer parameters are only weakly
  identifiable: the "long tail" is a small correction to a
  first-order decay that ACH can partially absorb.
- Adding the rebound phase (5) separates the timescales: rebound is
  buffer-driven only (window closed, ACH ≈ ach_closed, and if the
  moisture source is zero the air only re-humidifies from the
  buffer). This makes `k` identifiable from the rebound time
  constant.
- Adding the controlled pulse (3) gives an additional transient on a
  different timescale, which separates `M_buf` from `k` — the pulse
  height is set by `k`, and the pulse relaxation time by
  `1 / k * (1 + V/M_buf)`.

An informal identifiability rule of thumb: without a controlled
pulse, expect ACH to be fit within ~5 % but `k` and `M_buf` to sit
in fairly wide confidence intervals. With a pulse, all three
parameters should be recoverable within ~10–20 %.

### 5.3 Cross-check protocol (validates the fitted model on a held-out event)

- Repeat the single-event protocol on a **second event with different
  conditions** (different outdoor T, different pre-event indoor RH,
  different moisture pulse) held back for validation, not fitting.
- Run the fitted three-parameter model forward on the second event's
  observed initial state and outdoor trace. Compare predicted vs
  measured indoor AH.
- Acceptance criterion (illustrative): MAE on validation event <
  0.3 g/m³ AH, and no visible structure in the residual. This
  matches the metrics `validation.py` already computes for the
  air-only model.

### 5.4 Sensor and instrumentation requirements

- Indoor and outdoor **T + RH sensor pairs** with resolution ≤ 0.1 K
  and ≤ 2 % RH. Sampling ≤ 2 minutes. Real sensors' calibration
  drift is the biggest source of RMS error at this scale.
- **Window state** — magnetic reed switch or manual log.
- **Room volume** — measured (not paced-out) to within a few
  percent, since it enters every mass balance linearly.
- **Optional but valuable:** a **CO₂ sensor** — CO₂ decay during a
  window-open event provides a *separate* independent estimate of
  ACH (CO₂ is a passive tracer with no fabric buffering), which
  cross-checks the moisture-derived ACH and helps pin down the
  buffer parameters. Without CO₂, ACH and `k` are somewhat
  entangled.

### 5.5 Data quantity

- A single well-conducted event with all four phases (pre / pulse /
  vent / rebound) should be enough to identify (ACH, k, M_buf).
- A second held-out event is needed for validation, not fitting.
- **Two events total** is the minimum viable dataset. More events
  across a range of outdoor T and pre-event indoor RH would let
  the caller check whether `k` and `M_buf` are actually constant or
  varying with conditions (they will drift — real sorption
  isotherms are nonlinear — but the drift may be small enough to
  ignore for the residential-scale POC).

---

## 6. Decision so far

Given:

- The POC has just built `validation.py` for measuring the current
  air-only model's error against real sensor data.
- No such measurement has been run yet.
- The two-node buffer model adds two parameters that require a
  richer experimental protocol (§5) to identify honestly.
- The existing risk-constrained optimiser is designed for
  *comparison* between strategies, where the systematic bias from
  neglecting buffering largely cancels.

**Recommendation: do not implement the buffer model at this stage.**

Next step is to run `experiments/validate_from_csv.py` on the first
piece of real sensor data the project acquires and re-open this
document with the actual RMS errors and residual plots. If the
air-only model's error is below the "structured residual"
threshold in §4.5, the buffer model stays parked. If it is above,
§4.1 becomes the specification to build against, and §5 becomes the
data collection plan.

This document should be revisited whenever real data becomes
available and BEFORE any decision to add buffering to the code.
