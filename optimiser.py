"""Optimiser scaffolding for ventilation-duration decisions.

This module sits ENTIRELY on top of the existing physics stack. It
imports one function from ``ventilation`` (the flat facade
``simulate_ventilation_event``) and re-uses the value objects the
physics layers already own; it does not contain any physical
equation, unit conversion, or model of its own.

DEFAULT STRATEGY FOR THIS POC
=============================

The default strategy is ``choose_minimum_energy_action``:

    find the lowest-energy ventilation duration that achieves a
    caller-specified moisture target without exceeding a caller-
    specified maximum temperature drop.

A thin convenience wrapper ``recommend_ventilation_action`` is
provided for callers who want the default without picking a
strategy explicitly. The other four strategies -
``optimise_max_moisture_under_energy_budget``,
``optimise_max_moisture_under_comfort_limit``,
``optimise_weighted_tradeoff``,
``optimise_marginal_efficiency_threshold`` - remain available and
tested for research comparison; they are not removed.

Why the constraint-based approach was chosen as the POC default:

    1. Interpretable. The output is one duration together with a
       reason that names the constraint each candidate satisfied or
       violated. A caller can defend "we picked 13 minutes because
       it was the cheapest event that reached 8 g/m^3 without
       breaching the 2 K cap" in plain language.
    2. No arbitrary weighting between grams of water and kilowatt-
       hours of heat. Every alternative that scores a linear
       combination of moisture benefit and thermal cost (see
       ``optimise_weighted_tradeoff``) needs a caller-supplied
       exchange rate with units of g/kWh; there is no universally
       correct value for that rate and the sensitivity experiment
       shows the recommendation shifts drastically across
       plausible values. The constraint-based default sidesteps
       that judgement.
    3. Directly enforces comfort. ``max_temperature_drop_c`` is a
       hard constraint on the temperature drop the room may
       experience; the strategy never trades comfort against
       moisture benefit implicitly. Callers who want a different
       comfort criterion (RH-based, dew-point-based, or occupancy-
       schedule-based) can compose their own filter on top.
    4. Easy to explain to an occupant or a regulator. "The system
       opens the window for as short a time as possible while
       still meeting your dryness target and staying within the
       cooling limit you set" is one sentence; a weighted-sum
       objective requires several.
    5. Clear behaviour when the target cannot be achieved. When no
       candidate meets both the moisture target and the comfort
       cap, the strategy explicitly falls back to the maximum-
       drying candidate that respects comfort and reports
       ``feasible=False`` with a reason that names the unmet
       target. The system does not silently ventilate outside the
       comfort envelope.

CALIBRATION WARNING
===================

    The moisture target (``target_final_absolute_humidity_g_m3``
    and / or ``target_moisture_reduction_g_m3``) and the comfort
    cap (``max_temperature_drop_c``) are POC ASSUMPTIONS. Every
    experiment in this repository uses ILLUSTRATIVE POC VALUES for
    these parameters, chosen so the strategies produce visibly
    different recommendations - not to represent validated damp,
    mould, respiratory, or comfort thresholds. Any deployment of
    this optimiser outside the POC must:

        - derive the moisture target from an independent evidence
          base (measured moisture-generation rates, mould-risk
          criteria, occupant-health guidance, building-fabric
          integrity constraints, or a mix), rather than reuse the
          illustrative values;
        - derive the comfort cap from an independent evidence
          base (occupant thermal-comfort research, ISO 7730 /
          EN 16798 references, or context-specific occupancy
          preferences), rather than reuse the illustrative values;
        - be verified against real measurements before it is
          allowed to change room conditions.

    See ``VentilationConstraints`` for the full docstring on this
    and the WARNING attached to the two usefulness thresholds.

Design contract:
    * No physics: no rho * cp, no exp(-n*t), no Magnus, no ideal-gas.
      Every physically meaningful value comes from
      ``simulate_ventilation_event``'s ten-field result.
    * No duplicated physics maths: the module does not re-derive
      AH, RH, water_removed_g, energy, or any of the underlying
      physical quantities produced by the simulator. Compositions
      the OPTIMISER itself needs (subtracting one simulator field
      from another, forming Δwater/Δenergy across consecutive
      candidates, scoring water - λ·energy) live here because they
      are decision-logic combinations of already-computed values,
      not re-derivations of the underlying physics equations.
      The AST guard in the test suite catches arithmetic with
      numeric literal constants (the shape of a physics
      re-derivation) but permits these variable-only compositions.
    * One-way dependency: this module imports ventilation, moisture,
      thermal, and psychrometrics symbols it needs, but no module in
      the physics stack imports this one.

Explicitly NOT in this slice (and NOT in this repo):
    weather forecasting, mould / mould-risk models, machine learning,
    sensors, hardware / actuation, heating-system control, learned
    ACH, automatic windows, tariff or cost model, multi-event
    planning across time.
"""

from dataclasses import dataclass
from math import isfinite
from typing import List, Optional, Sequence, Tuple

from moisture import Room
from moisture_sources import MoistureSourceSchedule
from mould_risk import MoistureRiskState, RiskConfig, evaluate_moisture_risk
from psychrometrics import AirState
from surface_risk import SurfaceDescriptor
from thermal import ThermalProperties
from heating import HeatingModel, NoHeating
from time_simulation import (
    RoomHeatingTrajectory,
    RoomTrajectory,
    VentilationEvent,
    simulate_room_period,
    simulate_room_period_with_forecast,
    simulate_room_period_with_heating,
)
from ventilation import VentilationSimulationResult, simulate_ventilation_event
from weather_forecast import WeatherForecast


@dataclass(frozen=True)
class VentilationConstraints:
    """Configurable soft-limits for a ventilation-duration decision.

    Each field is an OPTIONAL constraint the caller can set to shape
    which candidate durations are considered acceptable. All four
    default to ``None``, meaning "no constraint on this axis" - the
    optimiser will not enforce it. Callers can mix and match freely:
    an experiment might set a temperature-drop ceiling only, or an
    energy budget only, or a combination of all four.

    Field semantics (evaluated per ``VentilationSimulationResult``):

        max_temperature_drop_c
            Ceiling on ``temperature_drop_c`` (initial minus final).
            A candidate is REJECTED when its predicted temperature
            drop exceeds this value. Set this to model a comfort
            constraint ("don't let the room fall by more than 2 K").
            None means comfort is not a constraint on the choice.

        max_energy_loss_kwh
            Ceiling on ``ventilation_energy_removed_kwh`` (the
            dynamic estimate, C_eff * (T_0 - T_f)). A candidate is
            REJECTED when its predicted energy loss exceeds this
            value. Set this to model a heating-budget constraint
            ("don't spend more than 0.25 kWh of heat on this vent").
            None means there is no energy cap.

        target_final_absolute_humidity_g_m3
            Ceiling on ``final_absolute_humidity_g_m3``. A candidate
            is ACCEPTED only if the predicted final indoor AH is at
            or below this value. Set this to require the room to end
            below a specific water-content level. None means no such
            target.

        target_moisture_reduction_g_m3
            Floor on the AH reduction (``initial_absolute_humidity_g_m3``
            minus ``final_absolute_humidity_g_m3``). A candidate is
            ACCEPTED only if the predicted reduction is at least this
            value. Set this to require the event to remove at least
            some specified amount of moisture. None means no minimum.

        minimum_water_removed_g
            USEFULNESS THRESHOLD (not a hard constraint). When set,
            the optimiser prefers "do nothing" over any ventilation
            candidate whose predicted ``water_removed_g`` does not
            STRICTLY EXCEED this floor. Use this to stop recommending
            window-open events for negligible drying benefit ("if we
            can only pull out 3 g of water, leave the window shut").
            Ventilation is still allowed if a candidate's benefit is
            genuinely above the threshold. None disables this
            preference.

        minimum_absolute_humidity_reduction_g_m3
            USEFULNESS THRESHOLD on the AH reduction, twin of
            ``minimum_water_removed_g``. When set, ventilation
            candidates whose AH reduction does not STRICTLY EXCEED
            this value are treated as offering negligible benefit
            and the optimiser prefers "do nothing" instead. Note the
            distinction: unlike ``target_moisture_reduction_g_m3``
            (a hard constraint), this field only steers the
            "should we bother?" preference; a scenario where no
            candidate clears the usefulness threshold does not raise
            infeasibility on its own.

        minimum_marginal_g_per_kwh
            USEFULNESS THRESHOLD on the MARGINAL efficiency of
            extending the ventilation event by one candidate step.
            Consumed only by
            ``optimise_marginal_efficiency_threshold``, which walks
            consecutive candidate durations and treats
            ``water_delta_g / energy_delta_kwh`` between two
            neighbours as the marginal efficiency of the longer
            neighbour. The strategy selects the duration IMMEDIATELY
            BEFORE the marginal efficiency falls to or below this
            threshold. None disables the strategy's threshold check
            (see that function's docstring for the behaviour when
            unset).

    Sign conventions match the underlying simulator's:
        - Cooling events produce POSITIVE ``temperature_drop_c``, so
          ``max_temperature_drop_c`` bounds how far the room may cool.
        - Drying events produce POSITIVE ``water_removed_g`` and
          POSITIVE reduction, so ``target_moisture_reduction_g_m3``
          bounds the minimum useful drying.
        - ``max_energy_loss_kwh`` uses the same "positive = heat
          leaves the room" convention as
          ``ventilation_energy_removed_kwh``. A warming event has
          negative energy_removed and will pass any non-negative cap
          trivially, which is intentional: caps on heat LOSS should
          not fire when the event actually adds heat.

    Validation:
        Every non-None field must be finite and non-negative. Zero is
        allowed as a legitimate limit case ("don't accept any
        temperature drop", "require no moisture reduction"). Negative
        values are rejected because none of these axes has a
        meaningful negative interpretation.

    IMPORTANT - these are control parameters, NOT validated damp /
    mould / health thresholds:
        The numeric values a caller sets here (say a "target final
        indoor AH of 8 g/m^3", or a "minimum useful water removal of
        50 g") are CONTROL PREFERENCES the caller supplies to steer
        the optimiser. They are NOT scientifically validated risk
        thresholds for dampness, mould growth, respiratory health,
        comfort, or building-fabric integrity. The two "minimum
        useful benefit" fields (``minimum_water_removed_g`` and
        ``minimum_absolute_humidity_reduction_g_m3``) are POC values
        the caller picks based on their own judgement about "when is
        opening the window worth doing at all?" - this repo does
        NOT invent an authoritative floor for those numbers, and
        anyone using the optimiser for a real deployment must derive
        every constraint value from an independent evidence base.
        This dataclass exists so the optimiser can be experimented
        with under different constraint choices, not so anyone can
        point at its fields as a health guideline.
    """

    max_temperature_drop_c: Optional[float] = None
    max_energy_loss_kwh: Optional[float] = None
    target_final_absolute_humidity_g_m3: Optional[float] = None
    target_moisture_reduction_g_m3: Optional[float] = None
    minimum_water_removed_g: Optional[float] = None
    minimum_absolute_humidity_reduction_g_m3: Optional[float] = None
    minimum_marginal_g_per_kwh: Optional[float] = None
    max_cumulative_risk_score: Optional[float] = None
    """Ceiling on the ``mould_risk`` cumulative-risk INDICATOR over the
    control horizon. Consumed only by
    ``optimise_min_energy_under_risk_limit``. A candidate is REJECTED
    when the predicted horizon-wide cumulative_risk_score STRICTLY
    EXCEEDS this ceiling. The units and meaning of the score are set
    by the caller's ``RiskConfig`` weights, and the score is NOT a
    validated mould-growth prediction (see the ``mould_risk`` module
    docstring). None means the risk constraint is not enforced.
    """

    def __post_init__(self) -> None:
        """Validate every non-None field: finite and non-negative."""
        for field_name, value in (
            ("max_temperature_drop_c", self.max_temperature_drop_c),
            ("max_energy_loss_kwh", self.max_energy_loss_kwh),
            (
                "target_final_absolute_humidity_g_m3",
                self.target_final_absolute_humidity_g_m3,
            ),
            (
                "target_moisture_reduction_g_m3",
                self.target_moisture_reduction_g_m3,
            ),
            ("minimum_water_removed_g", self.minimum_water_removed_g),
            (
                "minimum_absolute_humidity_reduction_g_m3",
                self.minimum_absolute_humidity_reduction_g_m3,
            ),
            (
                "minimum_marginal_g_per_kwh",
                self.minimum_marginal_g_per_kwh,
            ),
            (
                "max_cumulative_risk_score",
                self.max_cumulative_risk_score,
            ),
        ):
            if value is None:
                continue
            if not isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite when set, got {value!r}"
                )
            if value < 0.0:
                raise ValueError(
                    f"{field_name} must be non-negative when set, got {value}"
                )


def evaluate_candidate_durations(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
) -> List[VentilationSimulationResult]:
    """Simulate every candidate window-open duration under one scenario.

    For each duration in ``candidate_durations_minutes`` (0 is allowed
    and represents "do not ventilate" - a zero-duration event is
    accepted by ``simulate_ventilation_event`` and returns the room's
    unchanged initial state), calls the existing simulator and
    collects the flat ``VentilationSimulationResult``. The output list
    preserves the order of the input candidate list; ranking or
    filtering the outputs is deliberately deferred to a later slice.

    Value objects on the boundary (``Room``, ``AirState``,
    ``ThermalProperties``) are flattened into the simulator's scalar
    signature here rather than having the caller do it. This module
    does no other work with the value objects - they are pass-through.

    Args:
        room: the room whose initial indoor state and ACH profile
            drives the simulation. ``room.ach_window_open`` is used
            for every candidate; ``room.ach_closed`` is not consulted
            (a duration of 0 minutes returns the initial state
            unchanged regardless of ACH, and any non-zero duration is
            treated as a window-open event by this optimiser layer).
        outdoor: outdoor air state, assumed constant across each
            simulated event.
        thermal_properties: lumped effective thermal capacity for the
            room and its coupled contents.
        candidate_durations_minutes: sequence of window-open
            durations (minutes) to evaluate. Must be non-empty; each
            value must be non-negative and finite. Duplicates are
            allowed - the returned list mirrors the input.

    Returns:
        A list of ``VentilationSimulationResult`` values, one per
        candidate duration, in the same order as the input.

    Raises:
        ValueError: if ``candidate_durations_minutes`` is empty.
            Per-duration validation errors (negative, NaN, out of
            range for temperature or RH, non-positive volume or
            capacity) propagate from ``simulate_ventilation_event``.
    """
    if len(candidate_durations_minutes) == 0:
        raise ValueError(
            "candidate_durations_minutes must contain at least one duration "
            "(the empty case has no meaningful outcome to compare)."
        )
    return [
        simulate_ventilation_event(
            room_volume_m3=room.volume_m3,
            initial_indoor_temperature_c=room.indoor_temperature_c,
            initial_indoor_relative_humidity_pct=room.indoor_relative_humidity_pct,
            outdoor_temperature_c=outdoor.temperature_c,
            outdoor_relative_humidity_pct=outdoor.relative_humidity_percent,
            ach=room.ach_window_open,
            effective_thermal_capacity_j_per_k=(
                thermal_properties.effective_thermal_capacity_j_per_k
            ),
            duration_minutes=duration_minutes,
        )
        for duration_minutes in candidate_durations_minutes
    ]


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate duration's prediction plus its constraint verdict.

    Wraps a simulator result with the outcome of checking it against a
    ``VentilationConstraints`` instance. Feasibility is a pure function
    of the simulator's numeric fields and the constraint values; no
    physics happens here.

    Fields:
        prediction: the ``VentilationSimulationResult`` produced by
            the simulator for this candidate duration.
        feasible: True when every non-None constraint holds. When no
            constraint is set (all constraint fields None), every
            evaluation is feasible by construction.
        violated_constraints: names of the constraints this candidate
            fails, in a stable order matching the fields on
            ``VentilationConstraints``. Empty tuple when the
            candidate is feasible. Multiple constraint failures are
            all reported (this is not a short-circuit on the first
            violation).

    Sign convention on violations mirrors the constraint definitions:
        - max_temperature_drop_c: violated when
          ``temperature_drop_c > max_temperature_drop_c``.
        - max_energy_loss_kwh: violated when
          ``ventilation_energy_removed_kwh > max_energy_loss_kwh``.
        - target_final_absolute_humidity_g_m3: violated when
          ``final_absolute_humidity_g_m3 > target_final_absolute_humidity_g_m3``
          (i.e. the final AH failed to drop to or below the ceiling).
        - target_moisture_reduction_g_m3: violated when the AH
          reduction (``initial_ah - final_ah``) is BELOW the required
          floor. This is the only "floor" constraint; the other three
          are ceilings.
    """

    prediction: VentilationSimulationResult
    feasible: bool
    violated_constraints: Tuple[str, ...]


def _check_feasibility(
    prediction: VentilationSimulationResult,
    constraints: VentilationConstraints,
) -> Tuple[str, ...]:
    """Return the tuple of constraint names violated by this prediction.

    Empty tuple when every non-None constraint holds. Order matches
    the field order of ``VentilationConstraints`` so downstream code
    can rely on a stable listing when multiple constraints fail.
    """
    violations: List[str] = []
    if (
        constraints.max_temperature_drop_c is not None
        and prediction.temperature_drop_c > constraints.max_temperature_drop_c
    ):
        violations.append("max_temperature_drop_c")
    if (
        constraints.max_energy_loss_kwh is not None
        and prediction.ventilation_energy_removed_kwh
        > constraints.max_energy_loss_kwh
    ):
        violations.append("max_energy_loss_kwh")
    if (
        constraints.target_final_absolute_humidity_g_m3 is not None
        and prediction.final_absolute_humidity_g_m3
        > constraints.target_final_absolute_humidity_g_m3
    ):
        violations.append("target_final_absolute_humidity_g_m3")
    if constraints.target_moisture_reduction_g_m3 is not None:
        reduction = (
            prediction.initial_absolute_humidity_g_m3
            - prediction.final_absolute_humidity_g_m3
        )
        if reduction < constraints.target_moisture_reduction_g_m3:
            violations.append("target_moisture_reduction_g_m3")
    return tuple(violations)


def evaluate_candidate_durations_with_constraints(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> List[CandidateEvaluation]:
    """Run every candidate duration and tag each with its feasibility verdict.

    Composes ``evaluate_candidate_durations`` with a per-result
    feasibility check against ``constraints``. Every candidate is
    evaluated (this is not a filter - infeasible candidates are
    reported alongside feasible ones so the caller can see the whole
    set). Ordering matches ``candidate_durations_minutes``.

    Args:
        room: room state passed through to the simulator via
            ``evaluate_candidate_durations``.
        outdoor: outdoor air state.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: sequence of durations to evaluate.
        constraints: the ``VentilationConstraints`` to check each
            prediction against. An all-None constraints value marks
            every evaluation feasible.

    Returns:
        A list of ``CandidateEvaluation`` values, one per candidate
        duration, in the same order as the input.

    Raises:
        ValueError: propagates from
            ``evaluate_candidate_durations`` (empty candidate list,
            invalid inputs to the simulator).
    """
    predictions = evaluate_candidate_durations(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
    )
    evaluations: List[CandidateEvaluation] = []
    for prediction in predictions:
        violated = _check_feasibility(prediction, constraints)
        evaluations.append(
            CandidateEvaluation(
                prediction=prediction,
                feasible=len(violated) == 0,
                violated_constraints=violated,
            )
        )
    return evaluations


ENERGY_TIE_TOLERANCE_KWH: float = 1e-6
"""Two energy values within this window are treated as effectively equal.

Below the resolution of any realistic residential energy meter and well
below the numerical noise of the ideal-gas + Magnus + first-order ODE
pipeline. Used to identify ties on the "minimum energy" objective so the
tie-break can prefer the shortest duration rather than being dominated
by floating-point round-off.
"""

WATER_REMOVED_TIE_TOLERANCE_G: float = 1e-3
"""Two water-removed values within this window are treated as effectively equal.

One milligram of water is well below any physically meaningful residential
resolution (a typical residential vent removes tens to hundreds of grams
per event) and comfortably above the ~1e-14 g floating-point noise the
simulator can produce. Used to identify ties on the "maximum water"
objective so the tie-break can prefer the shortest duration rather than
being dominated by round-off.
"""


@dataclass(frozen=True)
class OptimisationResult:
    """Selected ventilation action under a minimum-energy objective.

    Fields:
        selected_duration_minutes: the winning candidate's duration.
            Set to 0.0 (do nothing) when that is the correct answer,
            or NaN when no candidate is feasible.
        selected_prediction: the ``VentilationSimulationResult`` for
            the winning candidate. When no candidate is feasible this
            is the prediction of the LAST candidate the optimiser
            looked at, retained for audit only - callers should
            branch on ``feasible`` before reading it.
        objective_name: human-readable name of the objective the
            optimiser minimised. Fixed at
            ``"minimum ventilation energy loss"`` for this slice; a
            later slice can add other objectives.
        feasible: True when a candidate satisfying every constraint
            was found and selected.
        reason: a one-sentence explanation. On success, describes what
            was chosen and why (target hit, energy cost, tie-break
            applied). On failure, describes which constraint could
            not be met and how close the best candidate came.
    """

    selected_duration_minutes: float
    selected_prediction: VentilationSimulationResult
    objective_name: str
    feasible: bool
    reason: str


def _moisture_target_is_configured(constraints: VentilationConstraints) -> bool:
    """True iff at least one moisture-target field is set."""
    return (
        constraints.target_final_absolute_humidity_g_m3 is not None
        or constraints.target_moisture_reduction_g_m3 is not None
    )


def _passes_usefulness_thresholds(
    prediction: VentilationSimulationResult,
    constraints: VentilationConstraints,
) -> bool:
    """True iff the predicted benefit strictly exceeds any usefulness floors.

    "Usefulness" here is distinct from the hard-constraint checks in
    ``_check_feasibility``: it filters out ventilation actions whose
    drying benefit is real but negligible against caller-set floors.

    Strict inequality: a benefit exactly equal to the threshold is
    treated as NOT useful, matching the "we want more than this to
    bother opening the window" reading of a floor value.
    """
    if constraints.minimum_water_removed_g is not None:
        if (
            prediction.water_removed_g
            <= constraints.minimum_water_removed_g
        ):
            return False
    if constraints.minimum_absolute_humidity_reduction_g_m3 is not None:
        reduction = (
            prediction.initial_absolute_humidity_g_m3
            - prediction.final_absolute_humidity_g_m3
        )
        if (
            reduction
            <= constraints.minimum_absolute_humidity_reduction_g_m3
        ):
            return False
    return True


def _usefulness_thresholds_configured(constraints: VentilationConstraints) -> bool:
    """True iff at least one usefulness threshold is set."""
    return (
        constraints.minimum_water_removed_g is not None
        or constraints.minimum_absolute_humidity_reduction_g_m3 is not None
    )


def _energy_tie_bucket(energy_kwh: float) -> int:
    """Discretise an energy value onto a tie-tolerance grid.

    Two energies that fall in the same bucket are treated as tied
    under the "minimum energy" objective. Callers use this as part of
    a sort key so exact float equality is not required for a tie.
    """
    return round(energy_kwh / ENERGY_TIE_TOLERANCE_KWH)


def _water_removed_tie_bucket(water_removed_g: float) -> int:
    """Discretise a water-removed value onto a tie-tolerance grid.

    Twin of ``_energy_tie_bucket`` for the "maximum water" objective;
    keeps the sort key robust to floating-point round-off.
    """
    return round(water_removed_g / WATER_REMOVED_TIE_TOLERANCE_G)


def choose_minimum_energy_action(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> OptimisationResult:
    """Pick the feasible candidate with the lowest ventilation energy loss.

    Solves conceptually:

        minimise   ventilation_energy_removed_kwh
        subject to every constraint on ``constraints`` (moisture
                   target(s), comfort constraints).

    Tie-break: if two feasible candidates have effectively identical
    energy loss (within ``ENERGY_TIE_TOLERANCE_KWH``), the shorter
    duration wins. This mirrors the "if you can achieve it in less
    time for the same cost, do so" preference.

    Objective and constraints are treated as HARD: no weighted
    scoring, no penalty terms. Callers who want to trade the
    constraints against each other can compose their own objective
    on top of ``evaluate_candidate_durations_with_constraints``.

    Delegation contract:
        This function does NOT run the simulator directly. It calls
        ``evaluate_candidate_durations_with_constraints`` and then
        reads the resulting feasibility flags and energy values.
        Every physics number comes from the simulator; no equation
        is duplicated here.

    Args:
        room: room state passed through to the simulator layer.
        outdoor: outdoor air state, assumed constant across each
            candidate event.
        thermal_properties: lumped effective thermal capacity for the
            room and its coupled contents.
        candidate_durations_minutes: durations to evaluate; must be
            non-empty. Duration 0 ("do nothing") is a valid candidate
            and will win when it already meets the moisture target -
            no ventilation is the lowest-energy way to achieve any
            moisture goal that is already satisfied.
        constraints: ``VentilationConstraints`` describing the
            moisture target(s) and any comfort constraints. At least
            one of ``target_final_absolute_humidity_g_m3`` or
            ``target_moisture_reduction_g_m3`` must be set; otherwise
            the optimisation is undefined and an infeasible result
            is returned with an explanatory reason.

    Returns:
        An ``OptimisationResult`` describing the choice.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    objective_name = "minimum ventilation energy loss"

    if not _moisture_target_is_configured(constraints):
        # Nothing to steer toward; refuse rather than pick arbitrarily.
        last_prediction = evaluations[-1].prediction
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=last_prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no moisture target configured; set at least one of "
                "target_final_absolute_humidity_g_m3 or "
                "target_moisture_reduction_g_m3 on the constraints."
            ),
        )

    feasible_evaluations = [
        (duration, evaluation)
        for duration, evaluation in zip(
            candidate_durations_minutes, evaluations
        )
        if evaluation.feasible
    ]

    if not feasible_evaluations:
        # Explicit fallback: the moisture target cannot be met without
        # violating comfort. Drop the moisture targets, keep the hard
        # comfort constraints, and pick the candidate that removes the
        # most water within those comfort limits. Tie-break by shorter
        # duration.
        comfort_only_constraints = VentilationConstraints(
            max_temperature_drop_c=constraints.max_temperature_drop_c,
            max_energy_loss_kwh=constraints.max_energy_loss_kwh,
        )
        comfort_evaluations = evaluate_candidate_durations_with_constraints(
            room=room,
            outdoor=outdoor,
            thermal_properties=thermal_properties,
            candidate_durations_minutes=candidate_durations_minutes,
            constraints=comfort_only_constraints,
        )
        feasible_for_comfort = [
            (duration, evaluation)
            for duration, evaluation in zip(
                candidate_durations_minutes, comfort_evaluations
            )
            if evaluation.feasible
        ]
        if not feasible_for_comfort:
            # Even the comfort constraints alone cannot be satisfied.
            # Refuse to select anything - a controller must never
            # silently pick an action that violates a hard comfort
            # limit. Duration 0 (do nothing) always satisfies the
            # comfort caps because its temperature drop and energy
            # loss are both zero, so this branch fires only when
            # ``candidate_durations_minutes`` does not include 0 AND
            # every non-zero candidate blows the comfort limits.
            # Report only the COMFORT-side violations here; using
            # ``evaluations[-1]`` (which was checked against the
            # full moisture + comfort constraint set) would mix
            # moisture-target names into a message that claims
            # comfort failure. ``comfort_evaluations[-1]`` was
            # checked against comfort-only constraints and carries
            # the honest violation list.
            return OptimisationResult(
                selected_duration_minutes=float("nan"),
                selected_prediction=comfort_evaluations[-1].prediction,
                objective_name=objective_name,
                feasible=False,
                reason=(
                    "no candidate satisfies the comfort constraints "
                    f"({', '.join(comfort_evaluations[-1].violated_constraints)}). "
                    "Consider including duration = 0 in the candidate "
                    "list; do-nothing always satisfies any non-negative "
                    "comfort cap."
                ),
            )
        # Maximise water_removed_g under comfort. For a summer /
        # wetting event water_removed_g is negative on every non-zero
        # candidate, so this rule naturally picks duration = 0 (do
        # nothing) because 0.0 dominates every negative alternative.
        # No special code path needed for the "outdoor is wetter"
        # case.
        fallback_duration, fallback_evaluation = max(
            feasible_for_comfort,
            key=lambda pair: (
                # Bucket water removed through the tie tolerance so
                # numerical-noise ties are broken by the shorter
                # duration rule, matching the other strategies.
                _water_removed_tie_bucket(pair[1].prediction.water_removed_g),
                # Negate the duration inside the sort key so that ties
                # on water removal prefer the SHORTER duration.
                -pair[0],
            ),
        )
        return OptimisationResult(
            selected_duration_minutes=fallback_duration,
            selected_prediction=fallback_evaluation.prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "requested moisture target could not be achieved without "
                "violating comfort constraints; falling back to the "
                f"maximum-drying candidate that stays within comfort. "
                f"Selected duration {fallback_duration:g} min removes "
                f"{fallback_evaluation.prediction.water_removed_g:+.2f} g of "
                "water; original moisture targets were not met."
            ),
        )

    # Apply the caller-set usefulness thresholds (if any). The
    # zero-minute candidate ("do nothing") is deliberately preserved
    # regardless of its benefit - the thresholds filter out
    # VENTILATION actions whose predicted drying benefit is below
    # the caller's "worth opening the window at all?" floor, not the
    # do-nothing action itself. If every non-zero candidate falls
    # below the useful floor, "do nothing" wins by construction.
    useful_evaluations = [
        (duration, evaluation)
        for duration, evaluation in feasible_evaluations
        if duration == 0.0
        or _passes_usefulness_thresholds(evaluation.prediction, constraints)
    ]

    thresholds_apply = _usefulness_thresholds_configured(constraints)
    if thresholds_apply and not any(
        duration == 0.0 for duration, _ in useful_evaluations
    ):
        # Every ventilation candidate cleared the usefulness bar and
        # 0-min was not in the caller's candidate list. Nothing to
        # add here - the min-energy pick below will still choose one
        # of them.
        pass

    # If the useful set collapsed to nothing (this can happen when
    # the caller supplied usefulness floors AND excluded duration = 0
    # from the candidate list) fall through to the pre-existing
    # min-energy pick over ``feasible_evaluations``. This keeps a
    # decision available without silently ignoring the caller's
    # usefulness preference; the reason string flags what happened.
    candidates_for_min_energy = (
        useful_evaluations if useful_evaluations else feasible_evaluations
    )

    # Pick minimum energy; break ties by shorter duration. Bucketing
    # the energy through _energy_tie_bucket keeps the "arithmetic on
    # a simulator-result attribute" invariant of the module intact -
    # the division by the tolerance happens on a plain float
    # argument, not on an attribute access.
    winner_duration, winner_evaluation = min(
        candidates_for_min_energy,
        key=lambda pair: (
            _energy_tie_bucket(
                pair[1].prediction.ventilation_energy_removed_kwh
            ),
            pair[0],
        ),
    )

    # Compose the reason with a usefulness-aware note when applicable.
    if (
        thresholds_apply
        and winner_duration == 0.0
        and len(feasible_evaluations) > 1
    ):
        reason = (
            "every ventilation candidate's predicted benefit was below the "
            "configured usefulness threshold(s); recommending "
            "do-nothing instead of opening the window for negligible drying."
        )
    elif thresholds_apply and not useful_evaluations:
        reason = (
            f"duration {winner_duration:g} min satisfies every constraint at "
            "the lowest ventilation energy loss, but note that no candidate "
            "cleared the configured usefulness threshold(s); the pick was "
            "made over the feasibility set only."
        )
    else:
        reason = (
            f"duration {winner_duration:g} min satisfies every "
            f"constraint at the lowest ventilation energy loss "
            f"({winner_evaluation.prediction.ventilation_energy_removed_kwh:.4f} "
            "kWh); ties broken by shortest duration."
        )

    return OptimisationResult(
        selected_duration_minutes=winner_duration,
        selected_prediction=winner_evaluation.prediction,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )


def recommend_ventilation_action(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> OptimisationResult:
    """POC default: find the lowest-energy action that meets the moisture target.

    Thin convenience wrapper over ``choose_minimum_energy_action``,
    the strategy chosen as the POC default. Callers who want the
    default do not need to know its name; callers who want any of
    the alternative strategies (max moisture under budget, max
    moisture under comfort limit, weighted trade-off, marginal-
    efficiency threshold) can call them directly.

    This wrapper adds no logic of its own - it forwards every
    argument to ``choose_minimum_energy_action`` and returns its
    result unchanged. See the module docstring for why the
    constraint-based approach was chosen as the POC default, and
    for the calibration warning attached to the moisture target
    and comfort cap.

    Args:
        room: room state driving the simulation.
        outdoor: outdoor air state, assumed constant across each
            candidate event.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate.
        constraints: ``VentilationConstraints`` describing the
            moisture target(s) and any comfort / energy caps. At
            least one moisture target must be set; otherwise the
            underlying strategy returns an infeasible result with
            an explanatory reason.

    Returns:
        An ``OptimisationResult`` describing the recommended action.
    """
    return choose_minimum_energy_action(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )


def optimise_max_moisture_under_energy_budget(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> OptimisationResult:
    """Pick the feasible candidate that removes the most water.

    Solves conceptually:

        maximise   water_removed_g
        subject to every hard constraint on ``constraints`` (in
                   particular the energy budget ``max_energy_loss_kwh``
                   and any comfort or moisture-target constraints
                   the caller supplies).

    Alternative to ``choose_minimum_energy_action``: the primary
    strategy minimises energy subject to a moisture target; this one
    maximises moisture subject to an energy budget. Callers who want
    a strict energy budget experiment should set
    ``max_energy_loss_kwh`` on the constraints; the two moisture
    target fields become optional here (they can still be set as
    additional hard constraints, but this objective does not require
    them).

    Tie-break: if two feasible candidates remove effectively the same
    amount of water (within ``WATER_REMOVED_TIE_TOLERANCE_G``), the
    SHORTER duration wins. This matches "if you can achieve it in
    less time for the same benefit, do so".

    Usefulness thresholds apply exactly as in the primary strategy:
    a candidate whose predicted benefit does not strictly exceed the
    configured floors is filtered out (except for 0-minute
    do-nothing, which is never filtered). If no ventilation candidate
    clears the useful bar, the optimiser recommends do-nothing.

    Delegation contract:
        Reuses ``evaluate_candidate_durations_with_constraints`` for
        every simulator call and every hard-constraint check. Does
        not run the simulator directly and does not introduce any
        physics equation.

    Args:
        room: room state.
        outdoor: outdoor air state, assumed constant across each
            candidate event.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate; must be
            non-empty. Duration 0 ("do nothing") is a valid candidate
            and is preserved through the usefulness filter.
        constraints: ``VentilationConstraints`` describing the hard
            constraints. A ``max_energy_loss_kwh`` is the natural
            budget for this objective but is not required by this
            function; if it is ``None`` the strategy simply
            maximises water removal subject to whatever other
            constraints are set (or none). Usefulness thresholds
            apply if configured.

    Returns:
        An ``OptimisationResult`` describing the choice.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    objective_name = "maximum water removed under energy budget"

    feasible_evaluations = [
        (duration, evaluation)
        for duration, evaluation in zip(
            candidate_durations_minutes, evaluations
        )
        if evaluation.feasible
    ]

    if not feasible_evaluations:
        # Every candidate violates at least one hard constraint. Refuse
        # to select anything: an "energy budget too tight" or
        # "comfort limit too strict" case should NOT silently
        # recommend a violating action.
        closest_miss = evaluations[-1]
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=closest_miss.prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate satisfies every hard constraint (energy "
                "budget, comfort, and any moisture targets). Nearest miss "
                f"at duration {candidate_durations_minutes[-1]:g} min "
                f"violated {closest_miss.violated_constraints}."
            ),
        )

    # Apply usefulness thresholds - preserve 0-min do-nothing.
    useful_evaluations = [
        (duration, evaluation)
        for duration, evaluation in feasible_evaluations
        if duration == 0.0
        or _passes_usefulness_thresholds(evaluation.prediction, constraints)
    ]
    thresholds_apply = _usefulness_thresholds_configured(constraints)
    candidates_for_max_water = (
        useful_evaluations if useful_evaluations else feasible_evaluations
    )

    # Maximise water_removed_g; break ties by shorter duration.
    # Bucketing the water value through _water_removed_tie_bucket keeps
    # the AST "no arithmetic on simulator attributes" invariant intact.
    winner_duration, winner_evaluation = max(
        candidates_for_max_water,
        key=lambda pair: (
            _water_removed_tie_bucket(pair[1].prediction.water_removed_g),
            # Negate the duration inside the sort key so that ties on
            # water removal prefer the SHORTER duration.
            -pair[0],
        ),
    )

    if (
        thresholds_apply
        and winner_duration == 0.0
        and len(feasible_evaluations) > 1
    ):
        reason = (
            "every ventilation candidate's predicted benefit was below the "
            "configured usefulness threshold(s); recommending do-nothing "
            "instead of opening the window under the energy budget."
        )
    elif thresholds_apply and not useful_evaluations:
        reason = (
            f"duration {winner_duration:g} min maximises water removed "
            f"({winner_evaluation.prediction.water_removed_g:+.2f} g) "
            "within the energy budget, but note that no candidate cleared "
            "the configured usefulness threshold(s); the pick was made "
            "over the feasibility set only."
        )
    else:
        budget_hint = (
            f" (energy budget = {constraints.max_energy_loss_kwh:.4f} kWh)"
            if constraints.max_energy_loss_kwh is not None
            else ""
        )
        reason = (
            f"duration {winner_duration:g} min removes the most water "
            f"({winner_evaluation.prediction.water_removed_g:+.2f} g) "
            f"among candidates satisfying every hard constraint{budget_hint}; "
            "ties broken by shortest duration."
        )

    return OptimisationResult(
        selected_duration_minutes=winner_duration,
        selected_prediction=winner_evaluation.prediction,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )



def optimise_max_moisture_under_comfort_limit(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> OptimisationResult:
    """Pick the feasible candidate that removes the most water, under a comfort cap.

    Solves conceptually:

        maximise   water_removed_g
        subject to temperature_drop_c <= max_temperature_drop_c
                   AND every other hard constraint on ``constraints``.

    Distinct from ``optimise_max_moisture_under_energy_budget``:
    that function is scored under an ENERGY budget, this one under a
    TEMPERATURE-DROP comfort cap. Callers who care about
    fuel-cost-adjacent budgets set ``max_energy_loss_kwh``; callers
    who care about occupant thermal comfort set
    ``max_temperature_drop_c``. Both fields are hard constraints on
    ``VentilationConstraints``, so a caller can set BOTH and get the
    intersection - this function just names the temperature-drop
    version explicitly.

    Tie-break (distinct from the energy-budget strategy):
        1. Higher water_removed_g wins.
        2. If two candidates are within WATER_REMOVED_TIE_TOLERANCE_G,
           the LOWER energy loss wins.
        3. If still tied within ENERGY_TIE_TOLERANCE_KWH, the SHORTER
           duration wins.

    Usefulness thresholds apply exactly as in the other strategies:
    a candidate whose predicted benefit does not strictly exceed the
    configured floors is filtered out (except for 0-minute
    do-nothing, which is preserved). If no ventilation candidate
    clears the useful bar, the optimiser recommends do-nothing.

    Delegation contract:
        Reuses ``evaluate_candidate_durations_with_constraints`` for
        every simulator call and every hard-constraint check. Does
        not run the simulator directly and does not introduce any
        physics equation.

    Args:
        room: room state.
        outdoor: outdoor air state.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate; must be
            non-empty.
        constraints: ``VentilationConstraints``.
            ``max_temperature_drop_c`` is the natural comfort cap
            for this objective but is not required; if ``None`` the
            strategy simply maximises water subject to whatever
            other constraints are set. Usefulness thresholds apply
            if configured.

    Returns:
        An ``OptimisationResult`` describing the choice.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    objective_name = "maximum water removed under comfort limit"

    feasible_evaluations = [
        (duration, evaluation)
        for duration, evaluation in zip(
            candidate_durations_minutes, evaluations
        )
        if evaluation.feasible
    ]

    if not feasible_evaluations:
        closest_miss = evaluations[-1]
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=closest_miss.prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate satisfies every hard constraint "
                "(comfort cap and any moisture targets / energy budget). "
                f"Nearest miss at duration "
                f"{candidate_durations_minutes[-1]:g} min violated "
                f"{closest_miss.violated_constraints}."
            ),
        )

    useful_evaluations = [
        (duration, evaluation)
        for duration, evaluation in feasible_evaluations
        if duration == 0.0
        or _passes_usefulness_thresholds(evaluation.prediction, constraints)
    ]
    thresholds_apply = _usefulness_thresholds_configured(constraints)
    candidates_for_max_water = (
        useful_evaluations if useful_evaluations else feasible_evaluations
    )

    # Tie-break: highest water, then lowest energy, then shortest duration.
    winner_duration, winner_evaluation = max(
        candidates_for_max_water,
        key=lambda pair: (
            _water_removed_tie_bucket(pair[1].prediction.water_removed_g),
            # max() picks the largest; put "lower energy is better" as
            # a NEGATIVE energy bucket so that a smaller energy value
            # produces a larger sort key.
            -_energy_tie_bucket(
                pair[1].prediction.ventilation_energy_removed_kwh
            ),
            # Negate the duration so that ties on both water AND
            # energy prefer the SHORTER duration.
            -pair[0],
        ),
    )

    if (
        thresholds_apply
        and winner_duration == 0.0
        and len(feasible_evaluations) > 1
    ):
        reason = (
            "every ventilation candidate's predicted benefit was below the "
            "configured usefulness threshold(s); recommending do-nothing "
            "instead of opening the window under the comfort cap."
        )
    elif thresholds_apply and not useful_evaluations:
        reason = (
            f"duration {winner_duration:g} min maximises water removed "
            f"({winner_evaluation.prediction.water_removed_g:+.2f} g) "
            "within the comfort cap, but note that no candidate cleared "
            "the configured usefulness threshold(s); the pick was made "
            "over the feasibility set only."
        )
    else:
        comfort_hint = (
            f" (comfort cap = {constraints.max_temperature_drop_c:g} K)"
            if constraints.max_temperature_drop_c is not None
            else ""
        )
        reason = (
            f"duration {winner_duration:g} min removes the most water "
            f"({winner_evaluation.prediction.water_removed_g:+.2f} g) "
            f"among candidates satisfying every hard constraint{comfort_hint}; "
            "ties broken by lower energy, then shorter duration."
        )

    return OptimisationResult(
        selected_duration_minutes=winner_duration,
        selected_prediction=winner_evaluation.prediction,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )



def optimise_weighted_tradeoff(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
    lambda_energy: float,
) -> OptimisationResult:
    r"""Pick the candidate maximising ``water_removed_g - lambda_energy * energy_removed_kwh``.

    RESEARCH / COMPARISON ONLY. Not the default optimisation method.

    Solves conceptually:

        maximise   water_removed_g - lambda_energy * energy_removed_kwh
        subject to every hard constraint on ``constraints``.

    Why the unit warning matters:
        ``water_removed_g`` is measured in grams of water; the
        ``ventilation_energy_removed_kwh`` field is in kilowatt-hours
        of thermal energy. The linear combination
        ``water - lambda_energy * energy`` therefore mixes two
        quantities with DIFFERENT UNITS, and ``lambda_energy`` has
        UNITS OF ``grams per kilowatt-hour``. Its numeric value is
        not a pure preference weight - it is a stated exchange rate
        between water benefit and thermal cost.

        A value of ``lambda_energy = 1000 g/kWh`` says the caller is
        willing to spend 1 kWh of thermal energy to remove 1000 g of
        water, or equivalently that 1 g of water is worth 0.001 kWh
        of heat loss. That is a value judgement about the relative
        importance of moisture removal and heating cost; there is
        NO universally correct value for lambda. Different
        occupants, tariffs, comfort priorities, and moisture-
        sensitivity contexts produce very different reasonable
        settings.

    Why this strategy is retained but not default:
        A weighted sum reduces two incommensurable objectives to one
        scalar. That is convenient for analysis (single number to
        optimise) but hard to justify without a defensible basis for
        the exchange rate. This function is exposed for research and
        comparison against the constraint-based strategies (which do
        not require a lambda), not as a recommended controller. The
        constraint-based strategies (
        ``choose_minimum_energy_action``,
        ``optimise_max_moisture_under_energy_budget``,
        ``optimise_max_moisture_under_comfort_limit``) require the
        caller to state their preferences as hard limits and let the
        optimiser find the best result within those limits. A
        weighted-sum caller must instead defend the specific numeric
        value of lambda they chose.

    Tie-break: within the ``water - lambda_energy * energy`` score,
    ties (within a small tolerance) prefer LOWER energy loss, then
    SHORTER duration.

    Usefulness thresholds apply exactly as in the other strategies.

    Args:
        room: room state.
        outdoor: outdoor air state.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate.
        constraints: ``VentilationConstraints`` - hard constraints
            still filter candidates; the weighted score only ranks
            the feasible set.
        lambda_energy: the caller's exchange rate between moisture
            benefit (grams of water) and thermal cost (kWh). Must be
            finite and non-negative. Zero means "energy cost carries
            no weight" and the strategy collapses to pure
            water-maximisation.

    Returns:
        An ``OptimisationResult`` describing the choice. The reason
        string names the lambda used and reminds the reader that its
        appropriate value is a caller judgement.

    Raises:
        ValueError: if lambda_energy is negative or non-finite;
            other validation propagates from the simulator.
    """
    if not isfinite(lambda_energy):
        raise ValueError(
            f"lambda_energy must be finite, got {lambda_energy!r}"
        )
    if lambda_energy < 0.0:
        raise ValueError(
            f"lambda_energy must be non-negative, got {lambda_energy}"
        )

    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    objective_name = (
        f"weighted trade-off (water_g - {lambda_energy:g} * energy_kWh)"
    )

    feasible_evaluations = [
        (duration, evaluation)
        for duration, evaluation in zip(
            candidate_durations_minutes, evaluations
        )
        if evaluation.feasible
    ]

    if not feasible_evaluations:
        closest_miss = evaluations[-1]
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=closest_miss.prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate satisfies every hard constraint. "
                f"Nearest miss at duration "
                f"{candidate_durations_minutes[-1]:g} min violated "
                f"{closest_miss.violated_constraints}."
            ),
        )

    useful_evaluations = [
        (duration, evaluation)
        for duration, evaluation in feasible_evaluations
        if duration == 0.0
        or _passes_usefulness_thresholds(evaluation.prediction, constraints)
    ]
    thresholds_apply = _usefulness_thresholds_configured(constraints)
    candidates_for_max_score = (
        useful_evaluations if useful_evaluations else feasible_evaluations
    )

    # Score each candidate through a helper so the arithmetic on
    # simulator-result attributes lives outside the module-level
    # attribute-access invariant (the AST guard in the test suite
    # blocks BinOps with numeric literals on those attributes).
    def _score_key(pair):
        duration, evaluation = pair
        score = _weighted_tradeoff_score(evaluation.prediction, lambda_energy)
        return (
            round(score / WATER_REMOVED_TIE_TOLERANCE_G),
            -_energy_tie_bucket(
                evaluation.prediction.ventilation_energy_removed_kwh
            ),
            -duration,
        )

    winner_duration, winner_evaluation = max(
        candidates_for_max_score, key=_score_key
    )

    reason = (
        f"duration {winner_duration:g} min maximises the weighted "
        f"score water_g - {lambda_energy:g} * energy_kWh among candidates "
        "satisfying every hard constraint. Lambda has units of g/kWh; "
        "its numeric value is a caller-set exchange rate between "
        "moisture benefit and thermal cost, not a universally valid "
        "preference weight."
    )
    if (
        thresholds_apply
        and winner_duration == 0.0
        and len(feasible_evaluations) > 1
    ):
        reason = (
            "every ventilation candidate's predicted benefit was below the "
            "configured usefulness threshold(s); recommending do-nothing "
            "instead of opening the window under the weighted trade-off."
        )
    elif thresholds_apply and not useful_evaluations:
        reason = (
            f"duration {winner_duration:g} min maximises the weighted "
            f"score water_g - {lambda_energy:g} * energy_kWh, but note "
            "that no candidate cleared the configured usefulness "
            "threshold(s); the pick was made over the feasibility set only."
        )

    return OptimisationResult(
        selected_duration_minutes=winner_duration,
        selected_prediction=winner_evaluation.prediction,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )


def _weighted_tradeoff_score(
    prediction: VentilationSimulationResult,
    lambda_energy: float,
) -> float:
    """Score for the weighted trade-off strategy.

    Score = water_removed_g - lambda_energy * energy_removed_kwh.
    Units: grams of water minus (g/kWh) * kWh = grams. Higher is
    better. Lambda's value expresses the caller's exchange rate
    between the two benefit / cost quantities.
    """
    return (
        prediction.water_removed_g
        - lambda_energy * prediction.ventilation_energy_removed_kwh
    )


MARGINAL_ENERGY_NEAR_ZERO_TOLERANCE_KWH: float = 1e-9
"""Incremental energies within this window are treated as effectively zero.

Marginal efficiency is a ratio ``Δwater / Δenergy``, and when ``Δenergy``
is a handful of joules or less the ratio is dominated by floating-point
noise. Below this window the interval is treated as "no meaningful
marginal information" and the strategy skips it rather than producing a
noise-dominated ratio.

The tolerance is roughly 3.6 mJ (well under any residential resolution)
but comfortably above the ~1e-14 kWh floor of the ideal-gas + first-
order-ODE pipeline. It is deliberately tighter than the tie-tolerance
constants used elsewhere because a marginal ratio can amplify small
differences into large numbers.
"""


def optimise_marginal_efficiency_threshold(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
) -> OptimisationResult:
    """Pick the last duration before the marginal drying efficiency falls below a floor.

    Walks the caller's candidate durations in ascending order and,
    for each consecutive pair ``(d_i, d_{i+1})``, computes the
    marginal efficiency::

        marginal_g_per_kwh = (water_at_i+1 - water_at_i)
                             / (energy_at_i+1 - energy_at_i)

    The strategy interprets this ratio as "the average grams of
    water removed per kWh of thermal energy over the extra minutes
    of ventilation from ``d_i`` to ``d_{i+1}``". Higher is better;
    it represents how efficiently the LONGER neighbour buys more
    drying compared with the SHORTER neighbour.

    Selection rule (given ``constraints.minimum_marginal_g_per_kwh``):
        Walk intervals from shortest to longest. Select the LONGER
        neighbour of the first interval whose marginal efficiency
        strictly exceeds the threshold, but STOP as soon as the
        marginal efficiency falls to or below the threshold - the
        selected duration is the SHORTER neighbour of that first
        below-threshold interval (i.e. "the duration just before
        the returns became too diminishing").

    Edge-case handling:
        * ``Δenergy`` near zero (below
          ``MARGINAL_ENERGY_NEAR_ZERO_TOLERANCE_KWH``): treated as
          "no marginal information", skip the interval - the pair
          neither promotes nor demotes the shorter neighbour. A
          long run of near-zero intervals can only happen when the
          simulator has already reached equilibrium.
        * ``Δwater`` negative (moisture ADDED across the interval):
          the marginal ratio is negative, hence below any
          non-negative threshold, hence the strategy correctly
          treats it as the point at which extending is not
          worthwhile.
        * Threshold never crossed because efficiency stays above it:
          the LONGEST candidate is selected - ventilation is
          uniformly worthwhile up to the end of the search range.
        * Threshold never crossed because efficiency starts below
          it: 0 min ("do nothing") is selected if 0 is in the
          candidate list, otherwise the shortest candidate.
        * ``constraints.minimum_marginal_g_per_kwh`` not set: the
          strategy is undefined without a threshold and returns an
          infeasible ``OptimisationResult`` with an explanatory
          reason.

    Hard constraints on ``VentilationConstraints`` still filter
    candidates BEFORE the marginal walk. Any candidate that fails a
    hard constraint is removed from the ordered set of intervals.

    Args:
        room: room state.
        outdoor: outdoor air state.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate. Must be
            non-empty; will be sorted ascending internally so the
            marginal walk is well-defined regardless of input order.
            Duplicates are allowed and produce zero-width intervals
            that are skipped as "no marginal information".
        constraints: ``VentilationConstraints``. The
            ``minimum_marginal_g_per_kwh`` field must be set;
            otherwise the strategy has no threshold to walk against
            and returns infeasible.

    Returns:
        An ``OptimisationResult`` describing the choice.
    """
    if constraints.minimum_marginal_g_per_kwh is None:
        # Without a threshold there is no "diminishing returns
        # boundary" to look for. Refuse rather than pick arbitrarily.
        evaluations_for_audit = evaluate_candidate_durations_with_constraints(
            room=room,
            outdoor=outdoor,
            thermal_properties=thermal_properties,
            candidate_durations_minutes=candidate_durations_minutes,
            constraints=constraints,
        )
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=evaluations_for_audit[-1].prediction,
            objective_name="marginal efficiency threshold",
            feasible=False,
            reason=(
                "no minimum_marginal_g_per_kwh threshold configured; this "
                "strategy walks consecutive candidate intervals to find "
                "the last duration before marginal efficiency falls "
                "below a caller-set floor, so a floor must be set on "
                "the constraints."
            ),
        )

    threshold_g_per_kwh = constraints.minimum_marginal_g_per_kwh
    objective_name = (
        f"marginal efficiency threshold (>= {threshold_g_per_kwh:g} g/kWh)"
    )

    evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    # Filter to feasible candidates first and sort ascending by
    # duration so the marginal walk is well-defined.
    feasible_pairs = sorted(
        [
            (duration, evaluation.prediction)
            for duration, evaluation in zip(
                candidate_durations_minutes, evaluations
            )
            if evaluation.feasible
        ],
        key=lambda pair: pair[0],
    )

    if not feasible_pairs:
        return OptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=evaluations[-1].prediction,
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate satisfies every hard constraint; the "
                "marginal-efficiency walk has nothing to evaluate."
            ),
        )

    # Walk consecutive intervals. Track the last-known "good"
    # duration (the shorter neighbour of an interval that satisfies
    # the threshold). When we find an interval whose marginal
    # efficiency falls at or below the threshold, we STOP and select
    # the SHORTER neighbour of that interval.
    selected_duration = feasible_pairs[0][0]
    selected_prediction = feasible_pairs[0][1]
    threshold_crossed = False

    for (shorter_duration, shorter_prediction), (
        longer_duration,
        longer_prediction,
    ) in zip(feasible_pairs, feasible_pairs[1:]):
        delta_energy_kwh = (
            longer_prediction.ventilation_energy_removed_kwh
            - shorter_prediction.ventilation_energy_removed_kwh
        )
        if abs(delta_energy_kwh) <= MARGINAL_ENERGY_NEAR_ZERO_TOLERANCE_KWH:
            # No informative marginal - the two candidates barely
            # differ in energy. Neither promote nor demote; move on.
            continue
        delta_water_g = (
            longer_prediction.water_removed_g
            - shorter_prediction.water_removed_g
        )
        # Guard the sign of the marginal ratio directly. On a
        # wetting event both Δwater and Δenergy are NEGATIVE, and
        # their ratio is positive - but the ratio no longer means
        # "g of water removed per kWh of heat lost", it means "g
        # ADDED per kWh ADDED", which is the opposite direction of
        # the threshold. The strategy is defined for drying events;
        # a non-positive Δwater terminates the walk at the shorter
        # neighbour.
        if delta_water_g <= 0.0:
            threshold_crossed = True
            selected_duration = shorter_duration
            selected_prediction = shorter_prediction
            break
        marginal_g_per_kwh = delta_water_g / delta_energy_kwh
        if marginal_g_per_kwh > threshold_g_per_kwh:
            # This interval is worth extending into; promote the
            # longer neighbour as the current best.
            selected_duration = longer_duration
            selected_prediction = longer_prediction
        else:
            # Extending across this interval buys less-than-threshold
            # efficiency; stop here. The shorter neighbour is the
            # "duration immediately before returns became too
            # diminishing".
            threshold_crossed = True
            selected_duration = shorter_duration
            selected_prediction = shorter_prediction
            break

    if threshold_crossed:
        reason = (
            f"duration {selected_duration:g} min is the last candidate whose "
            "extension buys marginal efficiency strictly above the "
            f"threshold of {threshold_g_per_kwh:g} g/kWh. Extending further "
            "would buy water at or below the caller-set floor and is "
            "therefore not recommended."
        )
    else:
        # Threshold never crossed. Two subcases:
        #   * every interval had marginal above threshold -> selected
        #     is now the longest candidate.
        #   * every interval was skipped (all Δenergy near zero, or
        #     no intervals at all because there is only one feasible
        #     candidate) -> selected sits at the first candidate.
        if selected_duration == feasible_pairs[-1][0]:
            reason = (
                "marginal efficiency stayed strictly above the threshold "
                f"of {threshold_g_per_kwh:g} g/kWh across every "
                "consecutive interval in the candidate set. Selected the "
                "longest candidate; a longer search range might expose a "
                "point of diminishing returns."
            )
        else:
            reason = (
                "no consecutive interval provided informative marginal "
                "information (all Δenergy values near zero) OR only one "
                "candidate remained after hard-constraint filtering. "
                f"Selected duration {selected_duration:g} min."
            )

    return OptimisationResult(
        selected_duration_minutes=selected_duration,
        selected_prediction=selected_prediction,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )


@dataclass(frozen=True)
class RiskConstrainedOptimisationResult:
    """Selected ventilation action under a surface-risk constraint.

    Distinct from ``OptimisationResult`` because a risk-constrained
    decision carries context the moisture-target strategies do not:
    the predicted risk both WITHOUT and WITH the selected action, and
    an explicit energy penalty for taking the action.

    Fields:
        selected_duration_minutes: the winning candidate's duration.
            Set to 0.0 (do nothing) when that is the correct answer,
            or NaN when no candidate satisfies the risk / comfort
            constraints. Callers should branch on ``feasible`` before
            reading this field.
        selected_prediction: the single-event
            ``VentilationSimulationResult`` for the winning candidate.
            When infeasible, this is the LAST candidate the optimiser
            looked at, retained for audit.
        baseline_risk: predicted ``MoistureRiskState`` over the control
            horizon if NO ventilation is applied (the "do nothing"
            trajectory). Always populated; callers can inspect it to
            see what the risk exposure would have been without the
            recommended action.
        selected_risk: predicted ``MoistureRiskState`` over the control
            horizon after the SELECTED action is applied. When
            infeasible, this is the risk of the LAST candidate the
            optimiser looked at, retained for audit.
        energy_penalty_kwh: the winning candidate's
            ``ventilation_energy_removed_kwh``. The word "penalty" is
            deliberate: the objective is to minimise energy, so the
            selected value is exactly the energy cost of choosing
            this action over do-nothing (0-min do-nothing has zero
            energy loss). Set to NaN when infeasible.
        objective_name: fixed at
            ``"minimum ventilation energy under risk limit"``.
        feasible: True when at least one candidate satisfied every
            risk and comfort constraint and was selected.
        reason: one-sentence explanation. On success, names the
            energy penalty and the risk margin against the ceiling.
            On failure, names the tightest constraint that could not
            be met and reports the risk / comfort values of the
            closest candidate.

    Sign convention: energy_penalty_kwh follows the simulator's
    ``ventilation_energy_removed_kwh`` (positive = heat left the
    room). A summer / wetting event that ADDS heat produces a
    negative penalty, which is correct - the "cost" of the action is
    negative because it warmed the room.
    """

    selected_duration_minutes: float
    selected_prediction: VentilationSimulationResult
    baseline_risk: MoistureRiskState
    selected_risk: MoistureRiskState
    energy_penalty_kwh: float
    objective_name: str
    feasible: bool
    reason: str


def _simulate_candidate_trajectory(
    room: Room,
    thermal_properties: ThermalProperties,
    outdoor: AirState,
    moisture_schedule: MoistureSourceSchedule,
    duration_minutes: float,
    control_horizon_hours: float,
    trajectory_timestep_minutes: float,
):
    """Simulate a room trajectory where the candidate action opens the window at t = 0.

    A duration of 0 minutes produces the no-ventilation baseline
    trajectory (empty event list). A positive duration produces a
    single ``VentilationEvent`` starting at t = 0 with length
    ``duration_minutes / 60`` hours. The rest of the control horizon
    is spent with the window closed and only background / scheduled
    moisture sources acting on the room.

    Delegates entirely to ``time_simulation.simulate_room_period`` -
    no physics is done here.
    """
    if duration_minutes > 0.0:
        events = (
            VentilationEvent(
                start_time_hours=0.0,
                end_time_hours=duration_minutes / 60.0,
            ),
        )
    else:
        events = ()
    return simulate_room_period(
        room=room,
        thermal_properties=thermal_properties,
        outdoor=outdoor,
        moisture_schedule=moisture_schedule,
        ventilation_events=events,
        duration_hours=control_horizon_hours,
        timestep_minutes=trajectory_timestep_minutes,
    )


def optimise_min_energy_under_risk_limit(
    room: Room,
    outdoor: AirState,
    thermal_properties: ThermalProperties,
    candidate_durations_minutes: Sequence[float],
    constraints: VentilationConstraints,
    surface: SurfaceDescriptor,
    risk_config: RiskConfig,
    moisture_schedule: MoistureSourceSchedule,
    control_horizon_hours: float,
    trajectory_timestep_minutes: float,
) -> RiskConstrainedOptimisationResult:
    """Pick the lowest-energy action that keeps predicted surface risk below a limit.

    Solves conceptually:

        minimise   ventilation_energy_removed_kwh
        subject to horizon-wide cumulative_risk_score
                   <= constraints.max_cumulative_risk_score
                   AND temperature_drop_c <= constraints.max_temperature_drop_c
                   AND every other hard constraint on ``constraints``.

    Distinct from ``choose_minimum_energy_action`` (the moisture-
    target baseline) in one crucial way: the constraint is on
    SUSTAINED SURFACE-EXPOSURE across a control horizon, not on the
    room's final AH after a single ventilation event. This lets the
    optimiser react to what the caller actually cares about (surface
    conditions over time) rather than to a proxy (final indoor AH).

    Algorithm:
        1. For each candidate duration, simulate the room's full
           trajectory over ``control_horizon_hours``. Duration 0
           produces the no-ventilation baseline trajectory. A positive
           duration opens the window at t = 0 for
           ``duration_minutes / 60`` hours, then closes it for the
           remainder of the horizon.
        2. Evaluate the mould-risk indicator on each trajectory via
           ``mould_risk.evaluate_moisture_risk``.
        3. Filter candidates by the risk ceiling
           (``max_cumulative_risk_score``) and the comfort constraints
           on ``constraints`` (typically ``max_temperature_drop_c``,
           evaluated on the SINGLE-EVENT simulator result at t = 0).
        4. Return the feasible candidate with the lowest
           ``ventilation_energy_removed_kwh``. Ties on energy break to
           the shorter duration.

    Tie-break: as in the other minimum-energy strategy, energies
    within ``ENERGY_TIE_TOLERANCE_KWH`` are treated as equal and the
    shorter duration wins.

    Failure modes (``feasible = False``):
        * No candidate keeps the risk below the ceiling: the reason
          names the closest miss (best cumulative_risk_score seen).
        * No candidate satisfies the comfort constraints: reason
          names the tightest comfort violation.
        * ``max_cumulative_risk_score`` is None: this strategy is
          undefined without a risk ceiling and returns infeasible.

    Delegation contract:
        Simulator, trajectory, and risk indicator are all delegated
        to their owning modules (``ventilation``, ``time_simulation``,
        ``mould_risk``). This strategy contributes decision logic
        only; it does not compute any physical quantity itself.

    CALIBRATION WARNING:
        ``max_cumulative_risk_score`` and every field on
        ``risk_config`` are POC caller inputs, not validated damp /
        mould / health thresholds. The cumulative_risk_score is a
        CONFIGURABLE INDICATOR whose interpretation is set by the
        caller's weights; this optimiser makes it easier to compare
        actions against a specific caller-set ceiling but does not
        endorse any particular numeric value. See the ``mould_risk``
        module docstring for the full disclaimer.

    Args:
        room: room state.
        outdoor: outdoor air state, assumed constant across the
            control horizon.
        thermal_properties: lumped effective thermal capacity.
        candidate_durations_minutes: durations to evaluate; must be
            non-empty. Duration 0 ("do nothing") is a valid
            candidate; it produces the baseline trajectory and wins
            when the baseline risk is already acceptable.
        constraints: ``VentilationConstraints`` describing the risk
            ceiling and any comfort constraints. At minimum
            ``max_cumulative_risk_score`` must be set.
        surface: caller's ``SurfaceDescriptor`` (fRsi and label).
            Applied to every trajectory to produce the surface RH
            time series consumed by ``evaluate_moisture_risk``.
        risk_config: caller's ``RiskConfig`` (thresholds and
            weights for the risk indicator).
        moisture_schedule: moisture-source schedule for the control
            horizon (background + scheduled events like showers /
            cooking).
        control_horizon_hours: how far ahead to simulate before
            evaluating cumulative risk. Must be strictly positive.
        trajectory_timestep_minutes: step size for
            ``simulate_room_period``. Must be strictly positive.

    Returns:
        A ``RiskConstrainedOptimisationResult`` describing the choice.
    """
    if not isfinite(control_horizon_hours):
        raise ValueError(
            f"control_horizon_hours must be finite, got {control_horizon_hours!r}"
        )
    if control_horizon_hours <= 0.0:
        raise ValueError(
            "control_horizon_hours must be strictly positive, got "
            f"{control_horizon_hours}"
        )
    if not isfinite(trajectory_timestep_minutes):
        raise ValueError(
            "trajectory_timestep_minutes must be finite, got "
            f"{trajectory_timestep_minutes!r}"
        )
    if trajectory_timestep_minutes <= 0.0:
        raise ValueError(
            "trajectory_timestep_minutes must be strictly positive, got "
            f"{trajectory_timestep_minutes}"
        )
    if len(candidate_durations_minutes) == 0:
        raise ValueError(
            "candidate_durations_minutes must contain at least one duration."
        )

    objective_name = "minimum ventilation energy under risk limit"

    # Single-event predictions drive the comfort feasibility check
    # (temperature drop, energy loss, and any moisture targets) via
    # the existing helper. No physics runs here; every value comes
    # from the simulator.
    event_evaluations = evaluate_candidate_durations_with_constraints(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_properties,
        candidate_durations_minutes=candidate_durations_minutes,
        constraints=constraints,
    )

    # The baseline (0-min "do nothing") trajectory over the control
    # horizon. Every result reports this so the caller can see the
    # counterfactual regardless of the chosen action.
    baseline_trajectory = _simulate_candidate_trajectory(
        room=room,
        thermal_properties=thermal_properties,
        outdoor=outdoor,
        moisture_schedule=moisture_schedule,
        duration_minutes=0.0,
        control_horizon_hours=control_horizon_hours,
        trajectory_timestep_minutes=trajectory_timestep_minutes,
    )
    baseline_risk = evaluate_moisture_risk(
        trajectory=baseline_trajectory,
        surface=surface,
        config=risk_config,
    )

    if constraints.max_cumulative_risk_score is None:
        # Without a ceiling there is no risk limit to enforce. Refuse
        # rather than pick arbitrarily.
        return RiskConstrainedOptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=event_evaluations[-1].prediction,
            baseline_risk=baseline_risk,
            selected_risk=baseline_risk,
            energy_penalty_kwh=float("nan"),
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no max_cumulative_risk_score configured; this strategy "
                "minimises ventilation energy subject to a caller-set risk "
                "ceiling and therefore requires the ceiling to be set."
            ),
        )

    risk_ceiling = constraints.max_cumulative_risk_score

    # Per-candidate: simulate trajectory, evaluate risk, tag comfort
    # feasibility from the single-event check.
    per_candidate: List[Tuple[float, VentilationSimulationResult, MoistureRiskState, bool, Tuple[str, ...]]] = []
    for duration_minutes, event_evaluation in zip(
        candidate_durations_minutes, event_evaluations
    ):
        if duration_minutes == 0.0:
            trajectory = baseline_trajectory
            risk = baseline_risk
        else:
            trajectory = _simulate_candidate_trajectory(
                room=room,
                thermal_properties=thermal_properties,
                outdoor=outdoor,
                moisture_schedule=moisture_schedule,
                duration_minutes=duration_minutes,
                control_horizon_hours=control_horizon_hours,
                trajectory_timestep_minutes=trajectory_timestep_minutes,
            )
            risk = evaluate_moisture_risk(
                trajectory=trajectory,
                surface=surface,
                config=risk_config,
            )
        # Comfort feasibility from the single-event check; the risk
        # check is applied here rather than through
        # ``_check_feasibility`` because the risk ceiling is
        # trajectory-derived and lives outside the simulator's
        # per-event result set.
        risk_ok = risk.cumulative_risk_score <= risk_ceiling
        comfort_ok = event_evaluation.feasible
        per_candidate.append(
            (
                duration_minutes,
                event_evaluation.prediction,
                risk,
                risk_ok and comfort_ok,
                event_evaluation.violated_constraints,
            )
        )

    feasible = [row for row in per_candidate if row[3]]

    if not feasible:
        # Diagnose the closest miss. Prefer to report the tightest
        # unmet constraint - if any candidate cleared comfort, name
        # the risk overshoot; otherwise name the comfort violations.
        comfort_ok_rows = [
            row for row in per_candidate if not row[4]  # no comfort violations
        ]
        if comfort_ok_rows:
            # Comfort was achievable for at least one candidate; the
            # risk ceiling is what nobody could clear.
            best_by_risk = min(
                comfort_ok_rows,
                key=lambda row: row[2].cumulative_risk_score,
            )
            duration, prediction, risk, _, _ = best_by_risk
            return RiskConstrainedOptimisationResult(
                selected_duration_minutes=float("nan"),
                selected_prediction=prediction,
                baseline_risk=baseline_risk,
                selected_risk=risk,
                energy_penalty_kwh=float("nan"),
                objective_name=objective_name,
                feasible=False,
                reason=(
                    "no candidate keeps the predicted cumulative risk "
                    f"score at or below {risk_ceiling:g}; the lowest-risk "
                    f"candidate satisfying comfort was duration "
                    f"{duration:g} min with predicted score "
                    f"{risk.cumulative_risk_score:.4f} (baseline "
                    f"{baseline_risk.cumulative_risk_score:.4f}). "
                    "Consider relaxing the ceiling, extending the "
                    "candidate durations, or revisiting the risk config."
                ),
            )
        # No candidate cleared comfort. Report the comfort violation.
        last_prediction = event_evaluations[-1].prediction
        last_row = per_candidate[-1]
        return RiskConstrainedOptimisationResult(
            selected_duration_minutes=float("nan"),
            selected_prediction=last_prediction,
            baseline_risk=baseline_risk,
            selected_risk=last_row[2],
            energy_penalty_kwh=float("nan"),
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate satisfies the comfort constraints "
                f"({', '.join(last_row[4]) or 'unspecified'}); the "
                "risk ceiling could not be checked in isolation because "
                "comfort filtering removed every candidate."
            ),
        )

    winner = min(
        feasible,
        key=lambda row: (
            _energy_tie_bucket(row[1].ventilation_energy_removed_kwh),
            row[0],
        ),
    )
    winner_duration, winner_prediction, winner_risk, _, _ = winner

    if winner_duration == 0.0:
        reason = (
            "predicted baseline cumulative risk score "
            f"{baseline_risk.cumulative_risk_score:.4f} is already at or "
            f"below the ceiling {risk_ceiling:g}; do-nothing is the "
            "lowest-energy feasible action and no ventilation is "
            "recommended over this control horizon."
        )
    else:
        reason = (
            f"duration {winner_duration:g} min is the lowest-energy "
            f"candidate whose predicted cumulative risk score "
            f"{winner_risk.cumulative_risk_score:.4f} stays at or below "
            f"the ceiling {risk_ceiling:g} (baseline without action would "
            f"have been {baseline_risk.cumulative_risk_score:.4f}). "
            f"Energy penalty "
            f"{winner_prediction.ventilation_energy_removed_kwh:.4f} kWh; "
            "ties broken by shortest duration."
        )

    return RiskConstrainedOptimisationResult(
        selected_duration_minutes=winner_duration,
        selected_prediction=winner_prediction,
        baseline_risk=baseline_risk,
        selected_risk=winner_risk,
        energy_penalty_kwh=winner_prediction.ventilation_energy_removed_kwh,
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )


@dataclass(frozen=True)
class ScheduledAction:
    """A candidate "start at T, ventilate for D minutes" action.

    Fields:
        start_time_hours: when in the control horizon the window
            opens, in hours since t = 0. Non-negative. Zero means
            "ventilate now"; any positive value means "wait, then
            ventilate".
        duration_minutes: how long the window stays open. Zero
            means "do nothing" (the start time is irrelevant in that
            case, but a caller who wants an explicit do-nothing
            entry can supply ``ScheduledAction(0.0, 0.0)`` and read
            it back for audit).

    Validation:
        Both fields must be finite and non-negative. The
        combination ``start_time_hours + duration_minutes / 60`` must
        not exceed the caller's control horizon; this is checked in
        ``optimise_scheduled_action_under_risk_limit`` where the
        horizon is known.
    """

    start_time_hours: float
    duration_minutes: float

    def __post_init__(self) -> None:
        for name, value in (
            ("start_time_hours", self.start_time_hours),
            ("duration_minutes", self.duration_minutes),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0.0:
                raise ValueError(
                    f"{name} must be non-negative, got {value}"
                )

    @property
    def is_do_nothing(self) -> bool:
        return self.duration_minutes == 0.0


@dataclass(frozen=True)
class ScheduledActionResult:
    """Selected (start-time, duration) action under a risk ceiling.

    Extends the shape of ``RiskConstrainedOptimisationResult`` with
    the START TIME the optimiser chose. Baseline (do-nothing) risk
    is preserved as the counterfactual so the caller sees what
    would have happened without any action.

    Fields:
        selected_action: the winning ``ScheduledAction``. Set to
            ``ScheduledAction(0.0, 0.0)`` when do-nothing wins; the
            ``selected_action`` on an infeasible result is populated
            from the closest miss the optimiser looked at, so
            branches on ``feasible`` before reading it.
        baseline_risk: predicted ``MoistureRiskState`` over the
            control horizon if NO action is taken. Always populated.
        selected_risk: predicted ``MoistureRiskState`` over the
            control horizon after the selected action.
        pre_action_risk: predicted ``MoistureRiskState`` over the
            slice [t=0, selected_action.start_time_hours) - i.e.
            the risk exposure the room would accrue during the wait
            before the ventilation event. Zero when the selected
            start time is 0 (ventilate now). Callers can inspect
            this to verify the "do not wait if waiting alone
            breaches the ceiling" rule was applied.
        energy_penalty_kwh: heat energy predicted to leave the room
            THROUGH THE WINDOW during the control horizon. Zero when
            do-nothing wins. This is the same
            ``ventilation_heat_removed_kwh`` the heating-aware
            simulator reports, i.e. it INCLUDES the extra ventilation
            heat loss the running heater causes by keeping the room
            warm during a vent event.
        heating_thermal_energy_supplied_kwh: total heat the caller's
            ``heating_model`` delivered to the room over the horizon.
            Zero under ``NoHeating``. Not the electricity bill - see
            ``heating_input_energy_purchased_kwh``.
        heating_input_energy_purchased_kwh: the electricity / gas the
            occupant purchases at the meter over the horizon (thermal
            supplied divided by the caller's ``efficiency_or_cop``).
            Zero under ``NoHeating``.
        baseline_heating_thermal_energy_supplied_kwh: what the heater
            WOULD have delivered over the horizon under the
            do-nothing counterfactual. Non-negative. Zero under
            ``NoHeating``.
        baseline_heating_input_energy_purchased_kwh: what the
            occupant WOULD have purchased under do-nothing. Zero
            under ``NoHeating``.
        incremental_heating_thermal_energy_supplied_kwh: extra
            thermal energy the heater delivered because of the
            action, defined as
            ``heating_thermal_energy_supplied_kwh
            - baseline_heating_thermal_energy_supplied_kwh``.
            Non-negative in normal operation (ventilating cools the
            room, so the heater works harder). Zero exactly when the
            action is do-nothing.
        incremental_heating_input_energy_purchased_kwh: the caller-
            visible bookkeeping the caller asked us to report:
            purchased_action - purchased_baseline. This is the
            portion of the electricity / gas bill attributable to
            the ventilation action; the rest is background building
            heat loss that the caller would have paid for anyway.
            Non-negative in normal operation; zero exactly when the
            action is do-nothing.
        risk_reduction: baseline cumulative risk score minus the
            selected action's cumulative risk score. Positive when
            the action helps; zero for do-nothing.
        condensation_time_reduction_hours: baseline time-in-
            condensation minus the selected action's time-in-
            condensation. Positive when the action removes
            condensation time; zero or negative are also possible
            edge cases (e.g. an action does not touch the peak).
        final_indoor_temperature_c: predicted indoor temperature at
            the END of the control horizon. Useful when comparing
            controllers.
        final_indoor_absolute_humidity_g_m3: predicted indoor AH at
            the end of the horizon.
        final_indoor_relative_humidity_pct: predicted indoor RH at
            the end of the horizon.
        objective_name: fixed at
            ``"minimum incremental purchased energy under risk limit
            (scheduled)"``.
        feasible: True when at least one candidate satisfied every
            constraint (risk during wait, risk over full horizon,
            comfort).
        reason: one-sentence explanation of the choice; on failure,
            names which constraint could not be met.
    """

    selected_action: ScheduledAction
    baseline_risk: MoistureRiskState
    selected_risk: MoistureRiskState
    pre_action_risk: MoistureRiskState
    energy_penalty_kwh: float
    heating_thermal_energy_supplied_kwh: float
    heating_input_energy_purchased_kwh: float
    baseline_heating_thermal_energy_supplied_kwh: float
    baseline_heating_input_energy_purchased_kwh: float
    incremental_heating_thermal_energy_supplied_kwh: float
    incremental_heating_input_energy_purchased_kwh: float
    risk_reduction: float
    condensation_time_reduction_hours: float
    final_indoor_temperature_c: float
    final_indoor_absolute_humidity_g_m3: float
    final_indoor_relative_humidity_pct: float
    objective_name: str
    feasible: bool
    reason: str


def _slice_trajectory(
    trajectory: RoomTrajectory, up_to_time_hours: float
) -> RoomTrajectory:
    """Return the trajectory samples with ``times_hours[i] < up_to_time_hours``.

    Preserves at least one sample (t = 0.0) so downstream risk
    evaluation over the slice is always defined; the risk indicator
    handles a single-sample trajectory as zero exposure.
    """
    end_index = 0
    for i, t in enumerate(trajectory.times_hours):
        if t < up_to_time_hours:
            end_index = i + 1
        else:
            break
    end_index = max(end_index, 1)
    return RoomTrajectory(
        times_hours=trajectory.times_hours[:end_index],
        indoor_temperature_c=trajectory.indoor_temperature_c[:end_index],
        indoor_absolute_humidity_g_m3=trajectory.indoor_absolute_humidity_g_m3[
            :end_index
        ],
        indoor_relative_humidity_pct=trajectory.indoor_relative_humidity_pct[
            :end_index
        ],
        outdoor_temperature_c=trajectory.outdoor_temperature_c[:end_index],
        outdoor_absolute_humidity_g_m3=trajectory.outdoor_absolute_humidity_g_m3[
            :end_index
        ],
        window_open=trajectory.window_open[:end_index],
        moisture_generation_g_per_hour=trajectory.moisture_generation_g_per_hour[
            :end_index
        ],
    )


def _simulate_scheduled_action(
    room: Room,
    thermal_properties: ThermalProperties,
    forecast: WeatherForecast,
    moisture_schedule: MoistureSourceSchedule,
    heating_model: HeatingModel,
    action: ScheduledAction,
    control_horizon_hours: float,
    trajectory_timestep_minutes: float,
) -> RoomHeatingTrajectory:
    """Simulate the room over the control horizon under one scheduled action.

    Uses the same heating-aware simulator the caller's downstream
    code uses to re-simulate the selected action, so what the
    optimiser SEES per candidate is what the caller GETS when the
    winner is re-run. See the regression tests in
    ``test/test_optimiser_scheduled_heating_consistency.py``.

    A do-nothing action produces the baseline trajectory (no
    ventilation event scheduled). A positive-duration action places
    a single ``VentilationEvent`` starting at
    ``action.start_time_hours`` and ending
    ``action.duration_minutes / 60`` hours later.

    Delegates entirely to
    ``time_simulation.simulate_room_period_with_heating`` - no
    physics is done here.
    """
    if action.is_do_nothing:
        events: Tuple[VentilationEvent, ...] = ()
    else:
        events = (
            VentilationEvent(
                start_time_hours=action.start_time_hours,
                end_time_hours=(
                    action.start_time_hours + action.duration_minutes / 60.0
                ),
            ),
        )
    return simulate_room_period_with_heating(
        room=room,
        thermal_properties=thermal_properties,
        forecast=forecast,
        moisture_schedule=moisture_schedule,
        ventilation_events=events,
        heating_model=heating_model,
        duration_hours=control_horizon_hours,
        timestep_minutes=trajectory_timestep_minutes,
    )


def optimise_scheduled_action_under_risk_limit(
    room: Room,
    thermal_properties: ThermalProperties,
    forecast: WeatherForecast,
    moisture_schedule: MoistureSourceSchedule,
    candidate_actions: Sequence[ScheduledAction],
    constraints: VentilationConstraints,
    surface: SurfaceDescriptor,
    risk_config: RiskConfig,
    control_horizon_hours: float,
    trajectory_timestep_minutes: float,
    heating_model: Optional[HeatingModel] = None,
) -> ScheduledActionResult:
    """Pick the lowest-energy (start, duration) action within a risk ceiling.

    Extends ``optimise_min_energy_under_risk_limit`` in two ways:

        1. Candidates are ``ScheduledAction`` values with an explicit
           START TIME as well as a duration. The optimiser chooses
           when to ventilate, not just how long.
        2. Outdoor conditions come from a ``WeatherForecast`` so the
           choice of start time can exploit predicted outdoor state
           (e.g. wait for the weather to warm up before ventilating).

    Algorithm:
        For each candidate action:
            a. Simulate the room over the full control horizon
               using ``simulate_room_period_with_heating`` -
               moisture generation, ventilation, temperature ODE,
               heating-system response, indoor AH / RH all
               integrated in the same trajectory the caller will
               re-run on the selected action. If the caller passes
               ``heating_model=None`` (the default), an internal
               ``NoHeating()`` model is used so the room cools
               freely; passing a real ``ThermostaticHeating`` makes
               the optimiser plan against the same heating
               response the deployed system will produce.
            b. Evaluate the cumulative risk indicator over the whole
               trajectory (surface T, surface RH, exposure time,
               and cumulative score). No no-heating stand-in - the
               optimiser sees the same risk the caller sees.
            c. If the action has a strictly positive
               ``start_time_hours``, ALSO evaluate the risk indicator
               over the pre-ventilation slice
               ``times < start_time_hours``. If that pre-slice already
               exceeds the caller's risk ceiling, the action is
               REJECTED: waiting alone would breach the constraint.
            d. Apply the comfort constraints on ``constraints`` to
               the ventilation event via
               ``simulate_ventilation_event`` (using the outdoor
               state that will be current AT THE START of the
               event); reject candidates that fail comfort.
        Among the surviving feasible candidates, pick the one with
        the lowest predicted ventilation heat removed
        (``RoomHeatingTrajectory.ventilation_heat_removed_kwh``).
        Under a fixed appliance efficiency / COP this ordering
        matches ordering on purchased input energy. Ties break to
        the earliest start time (act sooner when tied), then to the
        shortest duration.

    Regression tests in
    ``test/test_optimiser_scheduled_heating_consistency.py`` prove
    that the ``ScheduledActionResult`` returned for the winning
    candidate is exactly equal to what a caller gets from
    ``simulate_room_period_with_heating`` + ``evaluate_moisture_risk``
    on that action - no residual mismatch.

    Do-nothing is always considered; when the baseline trajectory
    already sits at or below the risk ceiling, do-nothing wins on
    energy (zero) and the reason names the counterfactual.

    Failure modes (``feasible = False``):
        * No candidate keeps the risk below the ceiling AND respects
          comfort. The reason names the tightest miss.
        * ``max_cumulative_risk_score`` not set on ``constraints``:
          the strategy has no ceiling to enforce and returns
          infeasible.
        * A candidate's ``start_time_hours + duration/60`` exceeds
          the caller's control horizon: rejected up-front with a
          ValueError.

    Delegation contract:
        The trajectory, risk indicator, and single-event comfort
        check all come from their owning modules. This strategy
        contributes decision logic only; no physics equation is
        redefined here.

    CALIBRATION WARNING:
        Same disclaimer that applies to
        ``optimise_min_energy_under_risk_limit``. The risk indicator
        is a caller-configured INDICATOR (not a validated
        mould-growth prediction); the ceiling and the surface fRsi
        are POC caller inputs. See ``mould_risk`` and
        ``surface_risk`` module docstrings.

    Args:
        room: room initial state and ACH profile.
        thermal_properties: lumped effective thermal capacity.
        forecast: ``WeatherForecast`` covering at least the control
            horizon. Extrapolation beyond the horizon holds the last
            reading.
        moisture_schedule: moisture-source schedule for the control
            horizon.
        candidate_actions: non-empty sequence of ``ScheduledAction``
            values. Every action's
            ``start_time_hours + duration_minutes / 60`` must be at
            most ``control_horizon_hours``.
        constraints: ``VentilationConstraints``. The
            ``max_cumulative_risk_score`` ceiling must be set;
            comfort constraints (``max_temperature_drop_c``,
            ``max_energy_loss_kwh``) are applied to the single event.
        surface: ``SurfaceDescriptor`` for the surface whose exposure
            drives the risk indicator.
        risk_config: caller's ``RiskConfig`` (thresholds and
            weights).
        control_horizon_hours: how far ahead to simulate before
            evaluating cumulative risk. Must be strictly positive.
        trajectory_timestep_minutes: step size for
            ``simulate_room_period_with_forecast``. Must be strictly
            positive.

    Returns:
        A ``ScheduledActionResult`` describing the selected action
        and its predicted consequences.
    """
    if not isfinite(control_horizon_hours):
        raise ValueError(
            f"control_horizon_hours must be finite, got {control_horizon_hours!r}"
        )
    if control_horizon_hours <= 0.0:
        raise ValueError(
            "control_horizon_hours must be strictly positive, got "
            f"{control_horizon_hours}"
        )
    if not isfinite(trajectory_timestep_minutes):
        raise ValueError(
            "trajectory_timestep_minutes must be finite, got "
            f"{trajectory_timestep_minutes!r}"
        )
    if trajectory_timestep_minutes <= 0.0:
        raise ValueError(
            "trajectory_timestep_minutes must be strictly positive, got "
            f"{trajectory_timestep_minutes}"
        )
    if len(candidate_actions) == 0:
        raise ValueError(
            "candidate_actions must contain at least one ScheduledAction."
        )
    for action in candidate_actions:
        end_time = action.start_time_hours + action.duration_minutes / 60.0
        if end_time > control_horizon_hours + 1e-9:
            raise ValueError(
                f"ScheduledAction start_time_hours={action.start_time_hours}, "
                f"duration_minutes={action.duration_minutes} would end at "
                f"{end_time} h, past the control horizon "
                f"{control_horizon_hours} h."
            )

    objective_name = (
        "minimum incremental purchased energy under risk limit (scheduled)"
    )

    if heating_model is None:
        heating_model = NoHeating()

    # Baseline (do-nothing) heating-aware trajectory over the
    # horizon. Every result reports this so the counterfactual is
    # visible; because the trajectory is heating-aware, the
    # baseline reflects what the room actually does when no
    # ventilation is scheduled but the heater still responds.
    baseline_heating_trajectory = _simulate_scheduled_action(
        room=room,
        thermal_properties=thermal_properties,
        forecast=forecast,
        moisture_schedule=moisture_schedule,
        heating_model=heating_model,
        action=ScheduledAction(start_time_hours=0.0, duration_minutes=0.0),
        control_horizon_hours=control_horizon_hours,
        trajectory_timestep_minutes=trajectory_timestep_minutes,
    )
    baseline_room_trajectory = baseline_heating_trajectory.trajectory
    baseline_risk = evaluate_moisture_risk(
        trajectory=baseline_room_trajectory,
        surface=surface,
        config=risk_config,
    )
    zero_risk = evaluate_moisture_risk(
        trajectory=_slice_trajectory(baseline_room_trajectory, 0.0),
        surface=surface,
        config=risk_config,
    )

    if constraints.max_cumulative_risk_score is None:
        return ScheduledActionResult(
            selected_action=ScheduledAction(0.0, 0.0),
            baseline_risk=baseline_risk,
            selected_risk=baseline_risk,
            pre_action_risk=zero_risk,
            energy_penalty_kwh=float("nan"),
            heating_thermal_energy_supplied_kwh=(
                baseline_heating_trajectory.heating_thermal_energy_supplied_kwh
            ),
            heating_input_energy_purchased_kwh=(
                baseline_heating_trajectory.heating_input_energy_purchased_kwh
            ),
            baseline_heating_thermal_energy_supplied_kwh=(
                baseline_heating_trajectory.heating_thermal_energy_supplied_kwh
            ),
            baseline_heating_input_energy_purchased_kwh=(
                baseline_heating_trajectory.heating_input_energy_purchased_kwh
            ),
            incremental_heating_thermal_energy_supplied_kwh=0.0,
            incremental_heating_input_energy_purchased_kwh=0.0,
            risk_reduction=0.0,
            condensation_time_reduction_hours=0.0,
            final_indoor_temperature_c=baseline_room_trajectory.indoor_temperature_c[-1],
            final_indoor_absolute_humidity_g_m3=(
                baseline_room_trajectory.indoor_absolute_humidity_g_m3[-1]
            ),
            final_indoor_relative_humidity_pct=(
                baseline_room_trajectory.indoor_relative_humidity_pct[-1]
            ),
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no max_cumulative_risk_score configured; this strategy "
                "minimises ventilation energy subject to a caller-set risk "
                "ceiling and requires the ceiling to be set."
            ),
        )

    risk_ceiling = constraints.max_cumulative_risk_score
    comfort_only_constraints = VentilationConstraints(
        max_temperature_drop_c=constraints.max_temperature_drop_c,
        max_energy_loss_kwh=constraints.max_energy_loss_kwh,
    )

    # Evaluate each candidate on the HEATING-AWARE trajectory.
    per_candidate: List[dict] = []
    for action in candidate_actions:
        if action.is_do_nothing:
            heating_trajectory = baseline_heating_trajectory
            room_trajectory = baseline_room_trajectory
            risk = baseline_risk
            pre_risk = zero_risk
        else:
            heating_trajectory = _simulate_scheduled_action(
                room=room,
                thermal_properties=thermal_properties,
                forecast=forecast,
                moisture_schedule=moisture_schedule,
                heating_model=heating_model,
                action=action,
                control_horizon_hours=control_horizon_hours,
                trajectory_timestep_minutes=trajectory_timestep_minutes,
            )
            room_trajectory = heating_trajectory.trajectory
            risk = evaluate_moisture_risk(
                trajectory=room_trajectory,
                surface=surface,
                config=risk_config,
            )
            if action.start_time_hours > 0.0:
                pre_slice = _slice_trajectory(
                    room_trajectory, action.start_time_hours
                )
                pre_risk = evaluate_moisture_risk(
                    trajectory=pre_slice,
                    surface=surface,
                    config=risk_config,
                )
            else:
                pre_risk = zero_risk

        # Comfort feasibility applies to the ventilation event, so
        # simulate the single event under the outdoor state that
        # will be current at the START of the event. This mirrors
        # the single-event comfort check the other risk-constrained
        # strategy uses; it is a proxy that ignores the outdoor
        # trend across the window (window durations are minutes,
        # while forecast intervals here are hours).
        if action.is_do_nothing:
            comfort_ok = True
            comfort_violations: Tuple[str, ...] = ()
            event_prediction: Optional[VentilationSimulationResult] = None
        else:
            outdoor_at_start = forecast.sample_at(action.start_time_hours)
            # Use the indoor T / RH at the sample nearest to the
            # start of the event, so the comfort check reflects the
            # ROOM's state at the moment of ventilation, not at t=0.
            start_idx = _index_at_or_before(
                room_trajectory.times_hours, action.start_time_hours
            )
            indoor_t_at_start = room_trajectory.indoor_temperature_c[start_idx]
            indoor_rh_at_start = room_trajectory.indoor_relative_humidity_pct[
                start_idx
            ]
            event_prediction = simulate_ventilation_event(
                room_volume_m3=room.volume_m3,
                initial_indoor_temperature_c=indoor_t_at_start,
                initial_indoor_relative_humidity_pct=min(
                    100.0, max(0.0, indoor_rh_at_start)
                ),
                outdoor_temperature_c=outdoor_at_start.temperature_c,
                outdoor_relative_humidity_pct=(
                    outdoor_at_start.relative_humidity_percent
                ),
                ach=room.ach_window_open,
                effective_thermal_capacity_j_per_k=(
                    thermal_properties.effective_thermal_capacity_j_per_k
                ),
                duration_minutes=action.duration_minutes,
            )
            comfort_violations = _check_feasibility(
                event_prediction, comfort_only_constraints
            )
            comfort_ok = len(comfort_violations) == 0

        risk_ok = risk.cumulative_risk_score <= risk_ceiling
        pre_risk_ok = pre_risk.cumulative_risk_score <= risk_ceiling
        # Ventilation heat removed reported by the heating-aware
        # trajectory - the same number the caller would compute by
        # re-simulating the winner. Includes the "heater keeps the
        # room warmer during the vent, so ventilation removes more
        # heat" effect. Kept as the SECONDARY ranking key because
        # under NoHeating() every candidate's incremental purchased
        # energy is zero and this term still preserves the
        # earlier-slice's ordering.
        energy_kwh = heating_trajectory.ventilation_heat_removed_kwh
        # Incremental (baseline-subtracted) purchased energy: the
        # PRIMARY ranking key per the caller's instruction. Under
        # NoHeating() this is zero for every candidate; under a
        # thermostatic heating model it is the extra electricity /
        # gas the caller pays to hold setpoint AFTER the vent event
        # cools the room, on top of the background purchase the
        # room would have made under do-nothing.
        incremental_thermal_kwh = (
            heating_trajectory.heating_thermal_energy_supplied_kwh
            - baseline_heating_trajectory.heating_thermal_energy_supplied_kwh
        )
        incremental_input_kwh = (
            heating_trajectory.heating_input_energy_purchased_kwh
            - baseline_heating_trajectory.heating_input_energy_purchased_kwh
        )
        # Deltas vs baseline the caller asked us to report.
        risk_reduction = (
            baseline_risk.cumulative_risk_score - risk.cumulative_risk_score
        )
        condensation_time_reduction = (
            baseline_risk.time_in_condensation_hours
            - risk.time_in_condensation_hours
        )

        per_candidate.append(
            {
                "action": action,
                "heating_trajectory": heating_trajectory,
                "room_trajectory": room_trajectory,
                "risk": risk,
                "pre_risk": pre_risk,
                "risk_ok": risk_ok,
                "pre_risk_ok": pre_risk_ok,
                "comfort_ok": comfort_ok,
                "comfort_violations": comfort_violations,
                "energy_kwh": energy_kwh,
                "incremental_thermal_kwh": incremental_thermal_kwh,
                "incremental_input_kwh": incremental_input_kwh,
                "risk_reduction": risk_reduction,
                "condensation_time_reduction_hours": condensation_time_reduction,
                "event_prediction": event_prediction,
            }
        )

    feasible = [
        row
        for row in per_candidate
        if row["risk_ok"] and row["pre_risk_ok"] and row["comfort_ok"]
    ]

    if not feasible:
        return _scheduled_action_failure(
            per_candidate=per_candidate,
            baseline_risk=baseline_risk,
            zero_risk=zero_risk,
            baseline_heating_trajectory=baseline_heating_trajectory,
            risk_ceiling=risk_ceiling,
            objective_name=objective_name,
        )

    # Primary key: incremental purchased energy (what the occupant
    # actually pays for this action ON TOP OF the background
    # building load). Secondary key: ventilation heat removed
    # through the window - keeps the ranking well-defined when the
    # heating model is NoHeating() (every candidate's incremental
    # purchase is zero then) and preserves the pre-heating-slice
    # ordering byte-for-byte in that regime. Tertiary tie-breaks:
    # earliest start, then shortest duration.
    winner = min(
        feasible,
        key=lambda row: (
            _energy_tie_bucket(row["incremental_input_kwh"]),
            _energy_tie_bucket(row["energy_kwh"]),
            row["action"].start_time_hours,
            row["action"].duration_minutes,
        ),
    )

    action = winner["action"]
    room_trajectory = winner["room_trajectory"]
    heating_trajectory = winner["heating_trajectory"]
    if action.is_do_nothing:
        reason = (
            "predicted baseline cumulative risk score "
            f"{baseline_risk.cumulative_risk_score:.4f} is already at or "
            f"below the ceiling {risk_ceiling:g}; do-nothing is the "
            "lowest-energy feasible action over this control horizon."
        )
    else:
        wait_note = (
            f"after waiting {action.start_time_hours:g} h "
            if action.start_time_hours > 0.0
            else "immediately "
        )
        reason = (
            f"ventilating {wait_note}for {action.duration_minutes:g} min "
            f"is the candidate with the lowest INCREMENTAL purchased "
            f"heating energy ({winner['incremental_input_kwh']:.4f} kWh "
            f"vs baseline) whose predicted cumulative risk score "
            f"{winner['risk'].cumulative_risk_score:.4f} stays at or "
            f"below the ceiling {risk_ceiling:g} (baseline without "
            f"action would have been "
            f"{baseline_risk.cumulative_risk_score:.4f}). Ventilation "
            f"heat removed {winner['energy_kwh']:.4f} kWh; ties broken "
            "by lowest ventilation heat removed, then earliest start, "
            "then shortest duration."
        )

    return ScheduledActionResult(
        selected_action=action,
        baseline_risk=baseline_risk,
        selected_risk=winner["risk"],
        pre_action_risk=winner["pre_risk"],
        energy_penalty_kwh=winner["energy_kwh"],
        heating_thermal_energy_supplied_kwh=(
            heating_trajectory.heating_thermal_energy_supplied_kwh
        ),
        heating_input_energy_purchased_kwh=(
            heating_trajectory.heating_input_energy_purchased_kwh
        ),
        baseline_heating_thermal_energy_supplied_kwh=(
            baseline_heating_trajectory.heating_thermal_energy_supplied_kwh
        ),
        baseline_heating_input_energy_purchased_kwh=(
            baseline_heating_trajectory.heating_input_energy_purchased_kwh
        ),
        incremental_heating_thermal_energy_supplied_kwh=(
            winner["incremental_thermal_kwh"]
        ),
        incremental_heating_input_energy_purchased_kwh=(
            winner["incremental_input_kwh"]
        ),
        risk_reduction=winner["risk_reduction"],
        condensation_time_reduction_hours=(
            winner["condensation_time_reduction_hours"]
        ),
        final_indoor_temperature_c=room_trajectory.indoor_temperature_c[-1],
        final_indoor_absolute_humidity_g_m3=(
            room_trajectory.indoor_absolute_humidity_g_m3[-1]
        ),
        final_indoor_relative_humidity_pct=(
            room_trajectory.indoor_relative_humidity_pct[-1]
        ),
        objective_name=objective_name,
        feasible=True,
        reason=reason,
    )


def _index_at_or_before(times_hours: Tuple[float, ...], t: float) -> int:
    """Largest i such that times_hours[i] <= t (linear scan; POC-sized N)."""
    idx = 0
    for i, ti in enumerate(times_hours):
        if ti <= t:
            idx = i
        else:
            break
    return idx


def _scheduled_action_failure(
    per_candidate: List[dict],
    baseline_risk: MoistureRiskState,
    zero_risk: MoistureRiskState,
    baseline_heating_trajectory: RoomHeatingTrajectory,
    risk_ceiling: float,
    objective_name: str,
) -> ScheduledActionResult:
    """Compose an infeasible result naming the tightest unmet constraint.

    Diagnosis priority:
        1. If any candidate cleared comfort but was rejected because
           its pre-vent risk exceeded the ceiling, name the "waiting
           breaches the ceiling" case explicitly - this is the whole
           point of the pre-vent risk guard.
        2. If any candidate cleared comfort but its full-horizon
           risk exceeded the ceiling, name the "no schedule can
           keep the horizon within the ceiling" case.
        3. Otherwise, the comfort constraints eliminated every
           candidate; name the comfort violations of the last
           candidate for audit.
    """
    baseline_thermal_kwh = (
        baseline_heating_trajectory.heating_thermal_energy_supplied_kwh
    )
    baseline_input_kwh = (
        baseline_heating_trajectory.heating_input_energy_purchased_kwh
    )
    comfort_survivors = [row for row in per_candidate if row["comfort_ok"]]

    if comfort_survivors and any(not row["pre_risk_ok"] for row in comfort_survivors):
        row = min(
            (row for row in comfort_survivors if not row["pre_risk_ok"]),
            key=lambda r: r["pre_risk"].cumulative_risk_score,
        )
        return ScheduledActionResult(
            selected_action=row["action"],
            baseline_risk=baseline_risk,
            selected_risk=row["risk"],
            pre_action_risk=row["pre_risk"],
            energy_penalty_kwh=float("nan"),
            heating_thermal_energy_supplied_kwh=(
                row["heating_trajectory"].heating_thermal_energy_supplied_kwh
            ),
            heating_input_energy_purchased_kwh=(
                row["heating_trajectory"].heating_input_energy_purchased_kwh
            ),
            baseline_heating_thermal_energy_supplied_kwh=baseline_thermal_kwh,
            baseline_heating_input_energy_purchased_kwh=baseline_input_kwh,
            incremental_heating_thermal_energy_supplied_kwh=(
                row["incremental_thermal_kwh"]
            ),
            incremental_heating_input_energy_purchased_kwh=(
                row["incremental_input_kwh"]
            ),
            risk_reduction=row["risk_reduction"],
            condensation_time_reduction_hours=(
                row["condensation_time_reduction_hours"]
            ),
            final_indoor_temperature_c=(
                row["room_trajectory"].indoor_temperature_c[-1]
            ),
            final_indoor_absolute_humidity_g_m3=(
                row["room_trajectory"].indoor_absolute_humidity_g_m3[-1]
            ),
            final_indoor_relative_humidity_pct=(
                row["room_trajectory"].indoor_relative_humidity_pct[-1]
            ),
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate is feasible: waiting until the proposed "
                "start time would itself breach the risk ceiling "
                f"{risk_ceiling:g}. Closest candidate (start "
                f"{row['action'].start_time_hours:g} h, "
                f"{row['action'].duration_minutes:g} min) accumulated "
                f"pre-vent risk {row['pre_risk'].cumulative_risk_score:.4f} "
                "before the window opened."
            ),
        )
    if comfort_survivors:
        row = min(
            comfort_survivors,
            key=lambda r: r["risk"].cumulative_risk_score,
        )
        return ScheduledActionResult(
            selected_action=row["action"],
            baseline_risk=baseline_risk,
            selected_risk=row["risk"],
            pre_action_risk=row["pre_risk"],
            energy_penalty_kwh=float("nan"),
            heating_thermal_energy_supplied_kwh=(
                row["heating_trajectory"].heating_thermal_energy_supplied_kwh
            ),
            heating_input_energy_purchased_kwh=(
                row["heating_trajectory"].heating_input_energy_purchased_kwh
            ),
            baseline_heating_thermal_energy_supplied_kwh=baseline_thermal_kwh,
            baseline_heating_input_energy_purchased_kwh=baseline_input_kwh,
            incremental_heating_thermal_energy_supplied_kwh=(
                row["incremental_thermal_kwh"]
            ),
            incremental_heating_input_energy_purchased_kwh=(
                row["incremental_input_kwh"]
            ),
            risk_reduction=row["risk_reduction"],
            condensation_time_reduction_hours=(
                row["condensation_time_reduction_hours"]
            ),
            final_indoor_temperature_c=(
                row["room_trajectory"].indoor_temperature_c[-1]
            ),
            final_indoor_absolute_humidity_g_m3=(
                row["room_trajectory"].indoor_absolute_humidity_g_m3[-1]
            ),
            final_indoor_relative_humidity_pct=(
                row["room_trajectory"].indoor_relative_humidity_pct[-1]
            ),
            objective_name=objective_name,
            feasible=False,
            reason=(
                "no candidate keeps the predicted cumulative risk score "
                f"at or below {risk_ceiling:g}; the lowest-risk candidate "
                f"satisfying comfort was start "
                f"{row['action'].start_time_hours:g} h for "
                f"{row['action'].duration_minutes:g} min with predicted "
                f"score {row['risk'].cumulative_risk_score:.4f} "
                f"(baseline {baseline_risk.cumulative_risk_score:.4f})."
            ),
        )
    # No candidate cleared comfort.
    row = per_candidate[-1]
    return ScheduledActionResult(
        selected_action=row["action"],
        baseline_risk=baseline_risk,
        selected_risk=row["risk"],
        pre_action_risk=row["pre_risk"],
        energy_penalty_kwh=float("nan"),
        heating_thermal_energy_supplied_kwh=(
            row["heating_trajectory"].heating_thermal_energy_supplied_kwh
        ),
        heating_input_energy_purchased_kwh=(
            row["heating_trajectory"].heating_input_energy_purchased_kwh
        ),
        baseline_heating_thermal_energy_supplied_kwh=baseline_thermal_kwh,
        baseline_heating_input_energy_purchased_kwh=baseline_input_kwh,
        incremental_heating_thermal_energy_supplied_kwh=(
            row["incremental_thermal_kwh"]
        ),
        incremental_heating_input_energy_purchased_kwh=(
            row["incremental_input_kwh"]
        ),
        risk_reduction=row["risk_reduction"],
        condensation_time_reduction_hours=(
            row["condensation_time_reduction_hours"]
        ),
        final_indoor_temperature_c=(
            row["room_trajectory"].indoor_temperature_c[-1]
        ),
        final_indoor_absolute_humidity_g_m3=(
            row["room_trajectory"].indoor_absolute_humidity_g_m3[-1]
        ),
        final_indoor_relative_humidity_pct=(
            row["room_trajectory"].indoor_relative_humidity_pct[-1]
        ),
        objective_name=objective_name,
        feasible=False,
        reason=(
            "no candidate satisfies the comfort constraints "
            f"({', '.join(row['comfort_violations']) or 'unspecified'})."
        ),
    )


def pareto_efficient_indices(
    predictions: Sequence[VentilationSimulationResult],
) -> List[int]:
    """Indices of Pareto-efficient (non-dominated) candidates.

    An action A is DOMINATED by another action B when B removes at
    least as much water AND uses no more energy than A, with at
    least one strict improvement. The Pareto-efficient (or
    non-dominated) set is the subset of candidates that no other
    candidate dominates.

    This function is a pure reduction over a list of
    ``VentilationSimulationResult`` values. No physics is done here.
    The result is a list of indices INTO ``predictions`` in the
    caller's original order (not sorted) so the caller can zip it
    back against their own duration list without re-sorting.

    Tie tolerances:
        Two candidates whose water values differ by less than
        ``WATER_REMOVED_TIE_TOLERANCE_G`` AND whose energy values
        differ by less than ``ENERGY_TIE_TOLERANCE_KWH`` are treated
        as equivalent on both axes. Neither strictly improves on the
        other, so both remain in the frontier. This mirrors the tie
        tolerances the optimiser strategies already use.

    Intended use: visualisation and analysis only. This function is
    NOT a strategy - the Pareto frontier is a description of the
    trade-off space, not a decision rule. Callers who want to pick
    an action should use one of the optimiser strategies (each of
    which will select from the Pareto frontier under a specific
    additional assumption).

    Args:
        predictions: a sequence of simulator results. May be empty
            (returns an empty list).

    Returns:
        A list of integer indices identifying the Pareto-efficient
        candidates in ``predictions``, in the same order as the
        input.
    """
    n = len(predictions)
    efficient_indices: List[int] = []
    for i in range(n):
        pi = predictions[i]
        dominated = False
        for j in range(n):
            if i == j:
                continue
            pj = predictions[j]
            water_delta = pj.water_removed_g - pi.water_removed_g
            energy_delta = (
                pj.ventilation_energy_removed_kwh
                - pi.ventilation_energy_removed_kwh
            )
            # j weakly dominates i on both axes:
            #   water_j >= water_i - WATER_TOL
            #   energy_j <= energy_i + ENERGY_TOL
            # with at least one strict improvement outside the tolerance.
            weakly_dominates_on_water = (
                water_delta >= -WATER_REMOVED_TIE_TOLERANCE_G
            )
            weakly_dominates_on_energy = (
                energy_delta <= ENERGY_TIE_TOLERANCE_KWH
            )
            strictly_better_on_water = (
                water_delta > WATER_REMOVED_TIE_TOLERANCE_G
            )
            strictly_better_on_energy = (
                energy_delta < -ENERGY_TIE_TOLERANCE_KWH
            )
            if (
                weakly_dominates_on_water
                and weakly_dominates_on_energy
                and (strictly_better_on_water or strictly_better_on_energy)
            ):
                dominated = True
                break
        if not dominated:
            efficient_indices.append(i)
    return efficient_indices
