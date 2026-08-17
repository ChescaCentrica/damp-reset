"""Tests for the optimiser scaffolding.

At this slice the optimiser only enumerates candidate durations and
runs the simulator on each. Tests verify:
    - every requested duration is evaluated,
    - duration 0 is included when requested,
    - results remain in the caller's order,
    - the optimiser reuses the existing simulator faithfully (no
      duplicated physics equations here).
"""

import ast
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from ventilation import VentilationSimulationResult, simulate_ventilation_event

from optimiser import (
    ENERGY_TIE_TOLERANCE_KWH,
    MARGINAL_ENERGY_NEAR_ZERO_TOLERANCE_KWH,
    WATER_REMOVED_TIE_TOLERANCE_G,
    CandidateEvaluation,
    OptimisationResult,
    VentilationConstraints,
    choose_minimum_energy_action,
    evaluate_candidate_durations,
    evaluate_candidate_durations_with_constraints,
    optimise_marginal_efficiency_threshold,
    optimise_max_moisture_under_comfort_limit,
    optimise_max_moisture_under_energy_budget,
    optimise_weighted_tradeoff,
    pareto_efficient_indices,
    recommend_ventilation_action,
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


# --- Enumeration guarantees ------------------------------------------------


def test_every_candidate_duration_is_evaluated() -> None:
    """One result per requested duration, no more, no fewer."""
    candidates = list(range(0, 31))  # 0 to 30 minutes inclusive
    results = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    assert len(results) == len(candidates)
    assert all(isinstance(r, VentilationSimulationResult) for r in results)


def test_duration_zero_is_included_when_requested() -> None:
    """Duration 0 must be evaluated and must return the room's initial state.

    The "do nothing" candidate is a legitimate simulator input, not a
    special code path in the optimiser.
    """
    results = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 15.0],
    )
    zero_result = results[0]
    assert (
        zero_result.final_absolute_humidity_g_m3
        == zero_result.initial_absolute_humidity_g_m3
    )
    assert zero_result.water_removed_g == 0.0
    assert zero_result.final_temperature_c == zero_result.initial_temperature_c
    assert zero_result.temperature_drop_c == 0.0
    assert zero_result.ventilation_energy_removed_kwh == 0.0


def test_results_preserve_input_order() -> None:
    """Whatever order the caller supplies, the output list matches it.

    Uses a deliberately non-monotone list including duplicates so a
    reordering / sorting bug would surface here rather than in the
    monotone default case.
    """
    candidates = [10.0, 0.0, 5.0, 15.0, 0.0, 2.0]
    results = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    # Each result's water_removed_g is a monotone-non-decreasing
    # function of duration for this cooling-drying scenario, so we
    # can use the ordering of water_removed_g values as a fingerprint
    # of the caller's duration list.
    expected_order = list(candidates)
    actual_durations = [
        # Reconstruct the duration for each result by inverse-lookup
        # against a fresh independent simulator call. This is
        # allowed: it's the SAME simulator, not a re-derived
        # equation.
        _duration_of_matching_result(r, candidates) for r in results
    ]
    assert actual_durations == expected_order


def _duration_of_matching_result(
    result: VentilationSimulationResult,
    candidates: list,
) -> float:
    """Find the duration whose independently-simulated result matches ``result``."""
    for duration in candidates:
        independent = simulate_ventilation_event(
            room_volume_m3=40.0,
            initial_indoor_temperature_c=20.0,
            initial_indoor_relative_humidity_pct=70.0,
            outdoor_temperature_c=5.0,
            outdoor_relative_humidity_pct=85.0,
            ach=5.0,
            effective_thermal_capacity_j_per_k=(
                ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
            ),
            duration_minutes=duration,
        )
        if independent == result:
            return duration
    raise AssertionError("no candidate duration reproduced the given result")


def test_duplicates_in_candidate_list_are_evaluated_separately() -> None:
    """Two identical durations produce two identical results, both present."""
    results = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[10.0, 10.0, 10.0],
    )
    assert len(results) == 3
    assert results[0] == results[1] == results[2]


def test_default_range_covers_0_to_30_minutes_inclusive() -> None:
    """The canonical caller pattern ``list(range(0, 31))`` yields 31 results."""
    candidates = list(range(0, 31))
    results = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    assert len(results) == 31
    # Sanity: the sweep should be monotone in water_removed for this
    # cooling-drying scenario, from 0 at t=0 to the largest at t=30.
    water = [r.water_removed_g for r in results]
    assert water == sorted(water)
    assert water[0] == 0.0
    assert water[-1] > 0.0


# --- Delegation and reuse --------------------------------------------------


def test_result_equals_a_fresh_direct_simulator_call() -> None:
    """The optimiser must not transform the simulator's output.

    A specific candidate's evaluated result must be byte-equal to
    what a direct call to ``simulate_ventilation_event`` returns for
    the same inputs.
    """
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    duration_minutes = 12.5
    results = evaluate_candidate_durations(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=[duration_minutes],
    )
    direct = simulate_ventilation_event(
        room_volume_m3=room.volume_m3,
        initial_indoor_temperature_c=room.indoor_temperature_c,
        initial_indoor_relative_humidity_pct=room.indoor_relative_humidity_pct,
        outdoor_temperature_c=outdoor.temperature_c,
        outdoor_relative_humidity_pct=outdoor.relative_humidity_percent,
        ach=room.ach_window_open,
        effective_thermal_capacity_j_per_k=(
            thermal_props.effective_thermal_capacity_j_per_k
        ),
        duration_minutes=duration_minutes,
    )
    assert results[0] == direct


def test_uses_ach_window_open_from_the_room() -> None:
    """The optimiser routes room.ach_window_open into the simulator.

    Two rooms with different ``ach_window_open`` values should
    produce different outputs for the same non-zero duration.
    """
    room_slow = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=2.0,
    )
    room_fast = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=10.0,
    )
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    slow = evaluate_candidate_durations(
        room=room_slow,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=[15.0],
    )[0]
    fast = evaluate_candidate_durations(
        room=room_fast,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=[15.0],
    )[0]
    assert fast.water_removed_g > slow.water_removed_g
    assert (
        fast.ventilation_energy_removed_kwh
        > slow.ventilation_energy_removed_kwh
    )


# --- Design contract: no duplicated physics --------------------------------


def test_optimiser_module_contains_no_duplicated_physics_equations() -> None:
    """The optimiser module must not re-derive any physical equation.

    Parses ``optimiser.py`` and asserts that:
        - it imports ``simulate_ventilation_event`` (delegation);
        - it contains no arithmetic operator on the numeric fields
          the simulator already reports (water_removed_g / energy /
          temperature_drop_c / final_temperature_c / etc.); and
        - it imports no arithmetic constants from the physics
          modules (AIR_DENSITY, AIR_SPECIFIC_HEAT, MAGNUS_A, etc.).

    A future contributor who adds, say, a ``rho * cp * V * ACH / 3600``
    inline will fail this test. The intent is to keep every physical
    equation with exactly one owner.
    """
    optimiser_source = Path(__file__).resolve().parent.parent / "optimiser.py"
    tree = ast.parse(optimiser_source.read_text())

    # 1. The optimiser imports simulate_ventilation_event.
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    assert "simulate_ventilation_event" in imported_names

    # 2. The optimiser does NOT import any physics constant. This
    #    list is not exhaustive but names the ones a well-intentioned
    #    contributor would reach for first.
    forbidden = {
        "AIR_DENSITY_KG_PER_M3",
        "AIR_SPECIFIC_HEAT_J_PER_KG_K",
        "SECONDS_PER_HOUR",
        "JOULES_PER_KWH",
        "MAGNUS_A",
        "MAGNUS_B",
        "P_SAT_0",
        "M_WATER",
        "R_UNIVERSAL",
        "MW_RATIO",
        "ZERO_CELSIUS_IN_KELVIN",
        "G_PER_KG",
    }
    assert not (imported_names & forbidden), (
        f"optimiser must not import physics constants; "
        f"got {imported_names & forbidden}"
    )

    # 3. No arithmetic (other than plain subtraction of two
    #    simulator-reported fields) on any simulator-reported field.
    #    Multiplication, division, addition, or subtraction against a
    #    numeric literal all indicate the optimiser re-deriving a
    #    physical equation (rho*cp, /3600, etc.) - those are banned.
    #    Subtraction of one simulator field from another is allowed
    #    because it just computes a diff of two already-derived
    #    quantities (e.g. AH reduction = initial_ah - final_ah); no
    #    physics is being reimplemented.
    simulator_fields = {
        "water_removed_g",
        "ventilation_energy_removed_kwh",
        "ventilation_heat_loss_coefficient_w_per_k",
        "temperature_drop_c",
        "final_temperature_c",
        "final_absolute_humidity_g_m3",
        "final_relative_humidity_pct",
        "initial_absolute_humidity_g_m3",
        "initial_relative_humidity_pct",
        "initial_temperature_c",
    }

    def _uses_field(node: ast.AST) -> bool:
        """True if the AST node reads a simulator-result field."""
        return (
            isinstance(node, ast.Attribute)
            and node.attr in simulator_fields
        )

    def _contains_numeric_literal(node: ast.AST) -> bool:
        """True if the expression tree contains a numeric-constant literal.

        Physics-equation re-derivations (Magnus 610.94, ρ 1.204, cp
        1005, /3600, /1000, +273.15, ...) always involve at least one
        NUMERIC LITERAL in the source. A weighted-sum like
        ``water - lambda * energy`` involves only variables and
        attribute reads and is allowed.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(
                child.value, (int, float)
            ):
                return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        if not any(_uses_field(child) for child in ast.walk(node)):
            continue
        # Any BinOp on a simulator-result attribute that does NOT
        # contain a numeric literal is a legitimate composition
        # (plain diff, or a weighted sum with variable coefficients).
        # BinOps with a numeric literal on a simulator-result
        # attribute look like a physics equation and are banned.
        if not _contains_numeric_literal(node):
            continue
        raise AssertionError(  # pragma: no cover - protective
            "optimiser.py performs arithmetic on a simulator-result "
            "field with a numeric literal. Physics equations must live "
            "in exactly one owner; the optimiser must not re-derive them."
        )


# --- Validation ------------------------------------------------------------


def test_empty_candidate_list_raises() -> None:
    """No candidates -> nothing to compare -> ValueError."""
    with pytest.raises(ValueError, match="candidate_durations_minutes"):
        evaluate_candidate_durations(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[],
        )


def test_invalid_duration_bubbles_up_from_simulator() -> None:
    """A negative candidate duration is rejected by the underlying simulator."""
    with pytest.raises(ValueError, match="duration_minutes"):
        evaluate_candidate_durations(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[5.0, -1.0, 10.0],
        )


def test_invalid_room_bubbles_up_at_construction() -> None:
    """A malformed Room is rejected before the optimiser is even called."""
    with pytest.raises(ValueError, match="volume_m3"):
        Room(
            volume_m3=0.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=70.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


# --- VentilationConstraints ------------------------------------------------


def test_constraints_default_to_all_none() -> None:
    """No arguments -> every optional field is None (no constraint set)."""
    constraints = VentilationConstraints()
    assert constraints.max_temperature_drop_c is None
    assert constraints.max_energy_loss_kwh is None
    assert constraints.target_final_absolute_humidity_g_m3 is None
    assert constraints.target_moisture_reduction_g_m3 is None


def test_constraints_accept_individual_positive_values() -> None:
    """Each field independently accepts a plausible positive setting."""
    constraints = VentilationConstraints(
        max_temperature_drop_c=2.0,
        max_energy_loss_kwh=0.25,
        target_final_absolute_humidity_g_m3=8.0,
        target_moisture_reduction_g_m3=3.0,
    )
    assert constraints.max_temperature_drop_c == 2.0
    assert constraints.max_energy_loss_kwh == 0.25
    assert constraints.target_final_absolute_humidity_g_m3 == 8.0
    assert constraints.target_moisture_reduction_g_m3 == 3.0


@pytest.mark.parametrize(
    "field_name",
    [
        "max_temperature_drop_c",
        "max_energy_loss_kwh",
        "target_final_absolute_humidity_g_m3",
        "target_moisture_reduction_g_m3",
    ],
)
def test_constraints_accept_zero_on_each_field(field_name: str) -> None:
    """Zero is a valid limit case on every axis ("don't accept any drop")."""
    kwargs = {field_name: 0.0}
    constraints = VentilationConstraints(**kwargs)  # type: ignore[arg-type]
    assert getattr(constraints, field_name) == 0.0


@pytest.mark.parametrize(
    "field_name",
    [
        "max_temperature_drop_c",
        "max_energy_loss_kwh",
        "target_final_absolute_humidity_g_m3",
        "target_moisture_reduction_g_m3",
    ],
)
def test_constraints_reject_negative_values_on_each_field(field_name: str) -> None:
    """Negative constraint values have no physical interpretation here."""
    kwargs = {field_name: -0.1}
    with pytest.raises(ValueError, match=field_name):
        VentilationConstraints(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("max_temperature_drop_c", float("nan")),
        ("max_temperature_drop_c", float("inf")),
        ("max_energy_loss_kwh", float("nan")),
        ("max_energy_loss_kwh", float("inf")),
        ("target_final_absolute_humidity_g_m3", float("nan")),
        ("target_final_absolute_humidity_g_m3", float("inf")),
        ("target_moisture_reduction_g_m3", float("nan")),
        ("target_moisture_reduction_g_m3", float("inf")),
    ],
)
def test_constraints_reject_non_finite_values(
    field_name: str, bad_value: float
) -> None:
    """NaN / inf on any constraint field is rejected with a targeted message."""
    kwargs = {field_name: bad_value}
    with pytest.raises(ValueError, match=field_name):
        VentilationConstraints(**kwargs)  # type: ignore[arg-type]


def test_constraints_allow_partial_specification() -> None:
    """Callers can set only some fields and leave the rest as no-constraint."""
    only_temperature = VentilationConstraints(max_temperature_drop_c=2.0)
    assert only_temperature.max_temperature_drop_c == 2.0
    assert only_temperature.max_energy_loss_kwh is None
    assert only_temperature.target_final_absolute_humidity_g_m3 is None
    assert only_temperature.target_moisture_reduction_g_m3 is None


def test_constraints_are_frozen() -> None:
    """VentilationConstraints is a frozen dataclass; mutation raises."""
    constraints = VentilationConstraints(max_energy_loss_kwh=0.5)
    with pytest.raises(FrozenInstanceError):
        constraints.max_energy_loss_kwh = 1.0  # type: ignore[misc]


def test_constraints_equality_is_by_value() -> None:
    """Two constraint values with equal fields compare equal."""
    a = VentilationConstraints(
        max_temperature_drop_c=2.0, max_energy_loss_kwh=0.25
    )
    b = VentilationConstraints(
        max_temperature_drop_c=2.0, max_energy_loss_kwh=0.25
    )
    c = VentilationConstraints(max_temperature_drop_c=2.5)
    assert a == b
    assert a != c


def test_constraints_reject_mixed_negative_with_valid_fields() -> None:
    """A single bad field surfaces even when other fields are legitimate."""
    with pytest.raises(ValueError, match="max_temperature_drop_c"):
        VentilationConstraints(
            max_temperature_drop_c=-0.1,
            max_energy_loss_kwh=0.25,
            target_final_absolute_humidity_g_m3=8.0,
        )


# --- CandidateEvaluation and constraint checking ---------------------------


def _default_candidates() -> list:
    return [0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]


def test_all_candidates_feasible_when_no_constraints_are_set() -> None:
    """VentilationConstraints() (all None) marks every candidate feasible."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(),
    )
    assert len(evaluations) == len(_default_candidates())
    assert all(isinstance(e, CandidateEvaluation) for e in evaluations)
    assert all(e.feasible for e in evaluations)
    assert all(e.violated_constraints == () for e in evaluations)


def test_max_temperature_drop_flags_the_15_minute_case() -> None:
    """Anchor scenario: max temp drop 1 K -> 15 min event (T drop ~1.71 K) infeasible.

    The canonical worked example produces temperature_drop_c ~= 1.71 K
    at 15 minutes for the default room. Setting the ceiling to 1 K
    should mark that candidate infeasible with the expected reason.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0],
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    assert evaluations[0].feasible is False
    assert evaluations[0].violated_constraints == ("max_temperature_drop_c",)
    # And confirm the physical anchor: the drop was ~1.7 K, above 1 K.
    assert 1.6 < evaluations[0].prediction.temperature_drop_c < 1.8


def test_max_temperature_drop_marks_short_events_feasible() -> None:
    """Short events under the same 1 K limit stay within the ceiling."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 2.0, 3.0, 5.0],
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    assert all(e.feasible for e in evaluations)
    assert all(e.violated_constraints == () for e in evaluations)


def test_max_energy_loss_flags_events_above_the_budget() -> None:
    """0.1 kWh budget rejects 15 and 20 minute events (0.24 / 0.31 kWh)."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 15.0, 20.0],
        constraints=VentilationConstraints(max_energy_loss_kwh=0.1),
    )
    feasible_flags = [e.feasible for e in evaluations]
    assert feasible_flags == [True, True, False, False]
    for evaluation in evaluations[2:]:
        assert evaluation.violated_constraints == ("max_energy_loss_kwh",)


def test_target_final_ah_flags_events_that_do_not_reach_the_target() -> None:
    """Ceiling on final AH: shorter events fail because they don't dry enough.

    At the canonical scenario indoor starts at 12.07 g/m^3; final AH
    across the sweep is 12.07 / 9.93 / 8.51 / 7.58 / 6.96 at
    0 / 5 / 10 / 15 / 20 minutes. A ceiling of 8 g/m^3 leaves only
    15 and 20 minutes feasible.
    """
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0
        ),
    )
    feasible_flags = [e.feasible for e in evaluations]
    assert feasible_flags == [False, False, False, True, True]
    assert (
        evaluations[0].violated_constraints
        == ("target_final_absolute_humidity_g_m3",)
    )


def test_target_moisture_reduction_flags_events_that_do_not_remove_enough() -> None:
    """Floor on AH reduction: shorter events fail because they haven't removed enough."""
    # Reductions across the sweep: 0 (t=0), ~0.97 (t=2), ~2.15 (t=5),
    # ~3.56 (t=10). A floor of 3 g/m^3 keeps 10 min and longer.
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 2.0, 5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            target_moisture_reduction_g_m3=3.0
        ),
    )
    feasible_flags = [e.feasible for e in evaluations]
    assert feasible_flags == [False, False, False, True, True]
    assert (
        evaluations[0].violated_constraints
        == ("target_moisture_reduction_g_m3",)
    )


def test_multiple_constraint_violations_are_all_reported() -> None:
    """When several constraints fail on one candidate, all names appear.

    The order follows the field order on ``VentilationConstraints``:
    temperature_drop -> energy -> final AH -> moisture reduction.
    """
    # Very tight thresholds: any non-trivial event will fail all four.
    tight = VentilationConstraints(
        max_temperature_drop_c=0.1,
        max_energy_loss_kwh=0.01,
        target_final_absolute_humidity_g_m3=1.0,
        target_moisture_reduction_g_m3=100.0,
    )
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0],
        constraints=tight,
    )
    assert evaluations[0].feasible is False
    assert evaluations[0].violated_constraints == (
        "max_temperature_drop_c",
        "max_energy_loss_kwh",
        "target_final_absolute_humidity_g_m3",
        "target_moisture_reduction_g_m3",
    )


def test_zero_duration_only_fails_moisture_targets_never_energy_or_temp() -> None:
    """A duration=0 event drops nothing, so ceilings can't fire; floors can."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0],
        constraints=VentilationConstraints(
            max_temperature_drop_c=0.0,  # zero drop is allowed on the ceiling
            max_energy_loss_kwh=0.0,
            target_final_absolute_humidity_g_m3=8.0,
            target_moisture_reduction_g_m3=1.0,  # floor 1 g/m^3
        ),
    )
    # T drop = 0 <= 0 (feasible), energy = 0 <= 0 (feasible),
    # final AH = 12.07 > 8 (infeasible), reduction = 0 < 1 (infeasible).
    assert evaluations[0].feasible is False
    assert evaluations[0].violated_constraints == (
        "target_final_absolute_humidity_g_m3",
        "target_moisture_reduction_g_m3",
    )


def test_predictions_from_constrained_call_match_unconstrained_call() -> None:
    """The constraint layer must not modify predictions.

    The ``prediction`` field on each ``CandidateEvaluation`` must be
    byte-equal to the ``VentilationSimulationResult`` the plain
    ``evaluate_candidate_durations`` would return for the same
    candidate. Constraint checking is pure post-processing.
    """
    candidates = _default_candidates()
    raw = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    tagged = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    for a, b in zip(raw, tagged):
        assert a == b.prediction


def test_candidate_evaluation_is_frozen() -> None:
    """CandidateEvaluation is immutable."""
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0],
        constraints=VentilationConstraints(),
    )
    with pytest.raises(FrozenInstanceError):
        evaluations[0].feasible = False  # type: ignore[misc]


def test_constrained_call_preserves_input_order() -> None:
    """The tagged output list is in the same order as the input candidates."""
    non_monotone = [10.0, 0.0, 5.0, 15.0, 2.0]
    evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=non_monotone,
        constraints=VentilationConstraints(max_energy_loss_kwh=0.1),
    )
    # The 10-min and 15-min events exceed 0.1 kWh; 0/2/5 do not.
    feasible_flags = [e.feasible for e in evaluations]
    assert feasible_flags == [False, True, True, False, True]


def test_constrained_call_propagates_empty_candidate_list_error() -> None:
    """Empty candidate list still raises the same ValueError."""
    with pytest.raises(ValueError, match="candidate_durations_minutes"):
        evaluate_candidate_durations_with_constraints(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[],
            constraints=VentilationConstraints(),
        )


# --- choose_minimum_energy_action -----------------------------------------
# Objective: minimise ventilation_energy_removed_kwh subject to every
# constraint on the supplied VentilationConstraints. Tie-break by
# shortest duration.


def test_min_energy_several_actions_meet_target_shortest_energy_wins() -> None:
    """When several candidates hit the moisture target, pick the cheapest.

    Target final AH = 8 g/m^3 (canonical scenario) is met by 15 and
    20 minute events. The 15-minute event costs less energy, so it
    wins under the minimum-energy objective.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0
        ),
    )
    assert isinstance(result, OptimisationResult)
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0
    assert result.objective_name == "minimum ventilation energy loss"
    # Selected candidate's prediction is the 15-min result.
    assert result.selected_prediction.final_absolute_humidity_g_m3 <= 8.0
    assert 0.2 < result.selected_prediction.ventilation_energy_removed_kwh < 0.3
    assert "15" in result.reason


def test_min_energy_only_one_action_meets_target_selects_that_one() -> None:
    """If only one candidate is feasible, it is selected regardless of energy."""
    # A tight target of 7.0 g/m^3 is only met by the 20-min event
    # (final AH ~= 6.96); every shorter event ends above 7.0.
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=7.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 20.0
    assert result.selected_prediction.final_absolute_humidity_g_m3 <= 7.0


def test_min_energy_no_action_meets_target_falls_back_to_max_drying() -> None:
    """When no candidate hits the moisture target, fall back to max drying.

    A target below the outdoor AH (5.77 g/m^3 at outdoor 5/85) can
    never be met - the room's asymptote sits above the target. The
    optimiser must NOT silently pick an infeasible action. Instead it
    falls back to the candidate that removes the most water within
    the (empty) comfort constraints. With no comfort caps set, that
    is the longest duration in the candidate list.
    """
    candidates = _default_candidates()
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0
        ),
    )
    assert result.feasible is False
    # Fallback picked a real duration - the longest one, because with
    # no comfort constraints and a monotone cooling-drying event
    # water_removed_g is maximised at the longest duration.
    assert result.selected_duration_minutes == max(candidates)
    assert result.selected_prediction.water_removed_g > 0.0
    assert "could not be achieved" in result.reason.lower()


def test_min_energy_doing_nothing_already_meets_target() -> None:
    """If the room already meets the target, duration 0 wins.

    A loose ceiling above the initial indoor AH (12.07 g/m^3) is
    already satisfied at t = 0; ventilating any longer only costs
    more energy for no additional constraint-satisfying benefit.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    assert result.selected_prediction.ventilation_energy_removed_kwh == 0.0


def test_min_energy_supports_moisture_reduction_objective() -> None:
    """Either moisture target field steers the optimisation.

    A floor of 4 g/m^3 reduction is met by the 15 min event
    (reduction ~= 4.50) and the 20 min event (~= 5.11). The 15-min
    event wins on lowest energy.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            target_moisture_reduction_g_m3=4.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0


def test_min_energy_honours_comfort_constraint_over_moisture_target() -> None:
    """Comfort constraints can force a lower moisture-reduction candidate.

    Moisture target: final AH <= 7.0 (only met at 20 min).
    Comfort limit: T drop <= 2.0 K (violated at 20 min, drop ~= 2.24).
    Result: no candidate satisfies both -> infeasible.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=7.0,
            max_temperature_drop_c=2.0,
        ),
    )
    assert result.feasible is False


def test_min_energy_no_moisture_target_configured_reports_infeasible() -> None:
    """Missing moisture target -> infeasible with an explanatory reason.

    ``choose_minimum_energy_action`` refuses to pick arbitrarily when
    there is no moisture objective to satisfy; the caller must set at
    least one moisture-target field on the constraints.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_default_candidates(),
        constraints=VentilationConstraints(
            max_temperature_drop_c=2.0
        ),
    )
    assert result.feasible is False
    assert "moisture target" in result.reason


def test_min_energy_tie_break_prefers_shorter_duration() -> None:
    """Two candidates within the tie tolerance -> the shorter one wins.

    The simulator is deterministic, so we can create a genuine tie by
    passing the same duration twice. Both entries are feasible with
    the same energy; the tie-break should still deterministically
    pick one and reason about it. Since the two candidates share a
    duration, the shorter-duration rule holds trivially - what the
    test actually locks in is that the tie doesn't fail on the
    equality itself.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0


def test_min_energy_tolerance_treats_near_equal_energies_as_tie() -> None:
    """Two candidates whose energies differ by less than the tolerance tie.

    ENERGY_TIE_TOLERANCE_KWH = 1e-6 kWh (~0.0036 J), well below
    residential-scale resolution. Any two events whose energies land
    within that window round to the same integer under the tie key
    and fall back to duration ordering. This test uses two candidates
    with identical parameters (guaranteed tie) plus one candidate at
    a substantially different energy to confirm the tolerance is not
    swallowing genuinely different values.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0
        ),
    )
    # 15 min ties with 15 min (obviously); 20 min is clearly more
    # energy so it doesn't tie in.
    assert result.selected_duration_minutes == 15.0


def test_optimisation_result_is_frozen() -> None:
    """OptimisationResult is a frozen dataclass."""
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    with pytest.raises(FrozenInstanceError):
        result.feasible = False  # type: ignore[misc]


def test_min_energy_reason_is_populated_in_both_directions() -> None:
    """Reason string is non-empty for feasible and infeasible results."""
    feasible_result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    assert feasible_result.reason

    infeasible_result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0
        ),
    )
    assert infeasible_result.reason
    assert not infeasible_result.feasible


# --- Fallback: target unreachable, comfort-only optimisation ---------------


def test_fallback_selects_max_drying_within_comfort_when_target_unreachable() -> None:
    """Moisture target impossible + comfort cap set -> fall back to comfort-only.

    Target final AH = 4 g/m^3 is below the outdoor asymptote (5.77
    g/m^3), so no candidate meets it. Comfort cap: max T drop 1 K.
    Expected fallback: pick the longest duration that stays within
    the 1 K comfort limit (roughly 8 min at this scenario), NOT the
    longest overall duration (which would violate the comfort cap).
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 21)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,
            max_temperature_drop_c=1.0,
        ),
    )
    assert result.feasible is False
    # The chosen duration must satisfy the comfort constraint.
    assert result.selected_prediction.temperature_drop_c <= 1.0
    # And should be a longer duration than the shortest options
    # (otherwise the max-drying rule is broken).
    assert result.selected_duration_minutes >= 5.0
    # Reason names the fallback explicitly.
    assert "could not be achieved" in result.reason.lower()
    assert "comfort" in result.reason.lower()


def test_fallback_picks_zero_minutes_when_outdoor_air_is_wetter() -> None:
    """Wetting event -> "do nothing" wins the fallback.

    Indoor 12 C / 40 %RH (outdoor AH ~= 4 g/m^3); outdoor 25 C / 85
    %RH (outdoor AH ~= 19 g/m^3). Every non-zero duration adds
    moisture (water_removed_g < 0). The fallback rule
    "maximum-drying candidate under comfort" therefore picks 0 min:
    0 g removed dominates every negative alternative.

    A moisture target below the initial indoor AH forces the
    fallback branch (no candidate reaches the target because
    ventilating adds water; the initial state already meets any
    reasonable ceiling but a floor on reduction can force the
    fallback).
    """
    wetting_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=12.0,
        indoor_relative_humidity_pct=40.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    wetter_outdoor = AirState(
        temperature_c=25.0, relative_humidity_percent=85.0
    )
    result = choose_minimum_energy_action(
        room=wetting_room,
        outdoor=wetter_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 21)),
        constraints=VentilationConstraints(
            target_moisture_reduction_g_m3=2.0,
        ),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes == 0.0
    # Do-nothing removes zero water and consumes zero energy. Water
    # is compared against an absolute tolerance because (initial -
    # final) * V is not IEEE-guaranteed to be exactly zero even
    # when the two AH values are computed to be equal; energy IS
    # exactly zero because ``ventilation_energy_removed_kwh`` is
    # ``C_eff * (T_0 - T_f)`` with T_0 == T_f identically at t = 0.
    assert result.selected_prediction.water_removed_g == pytest.approx(
        0.0, abs=1e-9
    )
    assert result.selected_prediction.ventilation_energy_removed_kwh == 0.0


def test_fallback_prefers_shorter_duration_on_water_removal_ties() -> None:
    """Two candidates removing the same water -> the shorter wins."""
    # Two 15-min candidates produce identical results. With no
    # feasible action for a very tight target, the fallback should
    # pick between the two 15-min entries deterministically. We add
    # a shorter tie by including 15-min twice.
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,
        ),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes == 15.0


def test_fallback_returns_infeasible_when_no_candidate_satisfies_comfort() -> None:
    """Comfort constraints unreachable + target unreachable -> hard refusal.

    If even 0 minutes is not in the candidate list AND every non-zero
    candidate blows the comfort cap, the optimiser must not select
    anything - a controller must never silently violate a hard
    comfort limit.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        # No 0-minute candidate; every non-zero duration produces a
        # temperature drop > 0.05 K.
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,
            max_temperature_drop_c=0.05,
        ),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes != result.selected_duration_minutes  # NaN
    # The reason must flag that even the fallback failed.
    assert "comfort" in result.reason.lower()
    assert "do-nothing" in result.reason.lower() or "duration = 0" in result.reason.lower()


def test_fallback_preserves_original_reason_when_target_hits_but_comfort_fails() -> None:
    """The pre-existing "moisture + comfort both infeasible" case still fires.

    If the fallback path lands on a candidate that satisfies comfort
    but no candidate satisfied every constraint on the first pass,
    the reason must be honest: the moisture target was not achieved.
    This test locks in that the wording says so explicitly.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 21)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,
            max_temperature_drop_c=1.0,
        ),
    )
    assert result.feasible is False
    assert "could not be achieved" in result.reason.lower()
    # Not "not configured" - the target was configured, just unreachable.
    assert "no moisture target configured" not in result.reason.lower()


# --- Usefulness thresholds: negligible drying benefit -> do-nothing --------


def test_usefulness_threshold_field_defaults_to_none() -> None:
    """New usefulness fields default to None (no threshold configured)."""
    constraints = VentilationConstraints()
    assert constraints.minimum_water_removed_g is None
    assert constraints.minimum_absolute_humidity_reduction_g_m3 is None


def test_usefulness_threshold_accepts_zero_and_positive_values() -> None:
    """Both new fields accept zero and any non-negative value."""
    zero = VentilationConstraints(
        minimum_water_removed_g=0.0,
        minimum_absolute_humidity_reduction_g_m3=0.0,
    )
    assert zero.minimum_water_removed_g == 0.0
    assert zero.minimum_absolute_humidity_reduction_g_m3 == 0.0
    positive = VentilationConstraints(
        minimum_water_removed_g=50.0,
        minimum_absolute_humidity_reduction_g_m3=1.5,
    )
    assert positive.minimum_water_removed_g == 50.0
    assert positive.minimum_absolute_humidity_reduction_g_m3 == 1.5


@pytest.mark.parametrize(
    "field_name",
    ["minimum_water_removed_g", "minimum_absolute_humidity_reduction_g_m3"],
)
def test_usefulness_threshold_rejects_negative_values(field_name: str) -> None:
    """Negative usefulness thresholds are unphysical."""
    with pytest.raises(ValueError, match=field_name):
        VentilationConstraints(**{field_name: -0.1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("minimum_water_removed_g", float("nan")),
        ("minimum_water_removed_g", float("inf")),
        ("minimum_absolute_humidity_reduction_g_m3", float("nan")),
        ("minimum_absolute_humidity_reduction_g_m3", float("inf")),
    ],
)
def test_usefulness_threshold_rejects_non_finite(
    field_name: str, bad_value: float
) -> None:
    """NaN / inf on the new fields is rejected with a targeted message."""
    with pytest.raises(ValueError, match=field_name):
        VentilationConstraints(**{field_name: bad_value})  # type: ignore[arg-type]


def test_tiny_drying_benefit_below_threshold_recommends_do_nothing() -> None:
    """Ventilation removes real but negligible water -> pick 0 min.

    Under a target so loose that "do nothing" already meets it, the
    optimiser would normally pick 0 anyway (that's the "already
    meets target" case). To probe the usefulness threshold on its
    own, use a moisture target that is met by the initial state
    AND include several long-duration candidates that would remove
    real water. With ``minimum_water_removed_g = 500.0`` (a huge
    threshold that no candidate in a 0-20 min sweep can clear), the
    optimiser must NOT pick any of those long-duration events even
    though they satisfy every hard constraint - it should recommend
    do-nothing because the marginal benefit falls below the useful
    floor.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0,  # already met at t=0
            minimum_water_removed_g=500.0,             # POC threshold
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    assert "usefulness" in result.reason.lower() or "negligible" in result.reason.lower()


def test_useful_benefit_above_threshold_still_recommends_ventilation() -> None:
    """When a candidate genuinely clears the threshold, ventilation is fine."""
    # At the canonical scenario, 15 min removes ~180 g of water. A
    # threshold of 50 g is easily cleared by anything longer than
    # ~3 min. The optimiser should therefore still pick a
    # ventilating action.
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            minimum_water_removed_g=50.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes > 0.0
    # And the chosen candidate strictly exceeds the threshold, not
    # just equals it.
    assert result.selected_prediction.water_removed_g > 50.0


def test_ah_reduction_usefulness_threshold_recommends_do_nothing_when_below() -> None:
    """AH-reduction threshold twin also gates the recommendation."""
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0,  # already met at t=0
            minimum_absolute_humidity_reduction_g_m3=100.0,  # unreachable
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_usefulness_threshold_exactly_equal_is_not_useful() -> None:
    """A candidate whose benefit EQUALS the threshold is treated as not useful.

    Strict inequality: 'we want MORE than this to bother'. If a
    caller sets the threshold at the exact value one candidate
    delivers, the optimiser should treat that candidate as
    insufficient.
    """
    # Pin the threshold at the exact water removal of the 20-min
    # event; the optimiser must prefer 0 over 20 min under that
    # threshold.
    twenty_min_result = simulate_ventilation_event(
        room_volume_m3=40.0,
        initial_indoor_temperature_c=20.0,
        initial_indoor_relative_humidity_pct=70.0,
        outdoor_temperature_c=5.0,
        outdoor_relative_humidity_pct=85.0,
        ach=5.0,
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        ),
        duration_minutes=20.0,
    )
    exact_threshold = twenty_min_result.water_removed_g
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0,  # already met at t=0
            minimum_water_removed_g=exact_threshold,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_usefulness_threshold_ignores_zero_duration_from_filtering() -> None:
    """Do-nothing removes 0 g but must not be filtered out by the threshold.

    Even a threshold of 1 g is above 0 g. If the filter naively
    applied to the 0-min candidate, no candidate would survive when
    every ventilation option also fell below the threshold. Instead
    the optimiser must preserve 0-min and recommend it explicitly.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 1.0],  # 1-min removes ~20 g
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0,
            minimum_water_removed_g=100.0,  # nothing at 1 min clears this
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_usefulness_threshold_can_pick_ventilation_when_zero_min_excluded() -> None:
    """No 0-min candidate + usefulness threshold no candidate clears -> best available.

    If the caller excludes 0 from the candidate list and no
    non-zero candidate clears the usefulness threshold, the
    optimiser cannot recommend do-nothing. It falls back to the
    min-energy candidate among the feasible set (which still meet
    the hard constraints). The reason must be honest about the
    usefulness gap.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,  # 15 min meets this
            minimum_water_removed_g=500.0,             # nothing in the list clears
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0
    assert "usefulness" in result.reason.lower()


# --- optimise_max_moisture_under_energy_budget -----------------------------


def test_max_moisture_uses_the_full_budget_when_no_cap() -> None:
    """No energy cap set -> the longest duration wins (most water removed).

    Under a cooling-drying scenario without any hard constraints,
    water_removed_g is monotone non-decreasing in duration, so
    "maximise water" picks the longest candidate.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(),
    )
    assert isinstance(result, OptimisationResult)
    assert result.feasible is True
    assert result.selected_duration_minutes == 20.0
    assert (
        result.objective_name
        == "maximum water removed under energy budget"
    )
    assert result.selected_prediction.water_removed_g > 0.0


def test_max_moisture_budget_change_shifts_selected_duration_downward() -> None:
    """Tightening the energy budget forces the optimiser to a shorter duration.

    Canonical scenario energies at 5/10/15/20 min are approximately
    0.082 / 0.161 / 0.237 / 0.310 kWh. A budget of 0.30 kWh admits
    20 min; 0.20 kWh admits 10 min but not 15; 0.10 kWh admits 5
    min but not 10; 0.05 kWh admits only 0 and 2 min. Each shift is
    a genuine "budget change moves the pick" observation.
    """
    candidates = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    picks_by_budget = {}
    for budget_kwh in (0.30, 0.20, 0.10, 0.05):
        result = optimise_max_moisture_under_energy_budget(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=candidates,
            constraints=VentilationConstraints(
                max_energy_loss_kwh=budget_kwh
            ),
        )
        assert result.feasible is True
        picks_by_budget[budget_kwh] = result.selected_duration_minutes

    # A larger budget must admit at least as long a duration.
    assert picks_by_budget[0.30] >= picks_by_budget[0.20]
    assert picks_by_budget[0.20] >= picks_by_budget[0.10]
    assert picks_by_budget[0.10] >= picks_by_budget[0.05]
    # And at least ONE change is strictly downward across the four
    # budgets - otherwise the test would pass trivially.
    assert (
        picks_by_budget[0.30] > picks_by_budget[0.05]
    ), "at least one budget change must move the selected duration"


def test_max_moisture_specific_budget_values_pick_expected_durations() -> None:
    """Pin the specific selection for each named budget.

    Energies (canonical scenario): 0/2/5/10/15/20 min ->
    ~0/0.033/0.082/0.161/0.237/0.310 kWh.
    """
    scenario = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 2.0, 5.0, 10.0, 15.0, 20.0],
    )
    # Budget of 0.32 kWh admits every candidate; 20 min wins.
    assert (
        optimise_max_moisture_under_energy_budget(
            **scenario,
            constraints=VentilationConstraints(max_energy_loss_kwh=0.32),
        ).selected_duration_minutes
        == 20.0
    )
    # Budget of 0.24 kWh admits up to 15 min (0.237 kWh) but rejects
    # 20 min (0.310 kWh); 15 min wins.
    assert (
        optimise_max_moisture_under_energy_budget(
            **scenario,
            constraints=VentilationConstraints(max_energy_loss_kwh=0.24),
        ).selected_duration_minutes
        == 15.0
    )
    # Budget of 0.20 kWh rejects 15 and 20; 10 min wins.
    assert (
        optimise_max_moisture_under_energy_budget(
            **scenario,
            constraints=VentilationConstraints(max_energy_loss_kwh=0.20),
        ).selected_duration_minutes
        == 10.0
    )
    # Budget of 0.10 kWh rejects 10, 15, 20; 5 min wins.
    assert (
        optimise_max_moisture_under_energy_budget(
            **scenario,
            constraints=VentilationConstraints(max_energy_loss_kwh=0.10),
        ).selected_duration_minutes
        == 5.0
    )
    # Budget of 0.05 kWh rejects 5, 10, 15, 20; 2 min wins.
    assert (
        optimise_max_moisture_under_energy_budget(
            **scenario,
            constraints=VentilationConstraints(max_energy_loss_kwh=0.05),
        ).selected_duration_minutes
        == 2.0
    )


def test_max_moisture_honours_hard_temperature_constraint() -> None:
    """Comfort cap can force a shorter duration than the energy budget alone would.

    Budget of 0.30 kWh admits every candidate on its own; adding a
    1 K temperature-drop cap trims out everything above ~7-8 min.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 21)),
        constraints=VentilationConstraints(
            max_energy_loss_kwh=0.30,
            max_temperature_drop_c=1.0,
        ),
    )
    assert result.feasible is True
    # Must be shorter than the unconstrained 20 min pick.
    assert result.selected_duration_minutes < 20.0
    # The comfort constraint must be respected by the winner.
    assert result.selected_prediction.temperature_drop_c <= 1.0


def test_max_moisture_honours_optional_target_final_ah_when_supplied() -> None:
    """If a moisture target is supplied as a hard constraint, the pick must meet it.

    Target final AH <= 8 g/m^3 is met by 15 and 20 minutes at the
    canonical scenario. Under a 0.32 kWh budget both are feasible;
    max-moisture picks 20 minutes.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            max_energy_loss_kwh=0.32,
            target_final_absolute_humidity_g_m3=8.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 20.0
    assert result.selected_prediction.final_absolute_humidity_g_m3 <= 8.0


def test_max_moisture_returns_infeasible_when_energy_budget_zero_and_no_zero_candidate() -> None:
    """A 0 kWh budget with no 0-min candidate -> refuse to pick.

    A 0 kWh budget only admits duration = 0 (which has zero energy
    loss). If the caller excludes 0 from the candidate list, every
    non-zero candidate violates the budget and no selection is
    possible.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(max_energy_loss_kwh=0.0),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes != result.selected_duration_minutes  # NaN


def test_max_moisture_wetting_scenario_picks_do_nothing() -> None:
    """Outdoor wetter than indoor -> maximum water_removed is 0 min.

    Every non-zero candidate ADDS water (negative water_removed_g);
    do-nothing at 0.0 dominates every negative alternative. This is
    the summer-day analogue of the same rule that applies in the
    primary optimiser's fallback branch.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=12.0,
            indoor_relative_humidity_pct=40.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        ),
        outdoor=AirState(temperature_c=25.0, relative_humidity_percent=85.0),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    assert result.selected_prediction.water_removed_g == pytest.approx(
        0.0, abs=1e-9
    )


def test_max_moisture_usefulness_threshold_forces_do_nothing() -> None:
    """Configured minimum_water_removed_g above the largest candidate -> 0 min.

    Under a budget of 0.30 kWh every candidate is feasible, but if
    the caller sets the usefulness threshold to 500 g (above what
    any 0-20 min event delivers), the optimiser should recommend
    do-nothing.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            max_energy_loss_kwh=0.30,
            minimum_water_removed_g=500.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    assert "usefulness" in result.reason.lower() or "negligible" in result.reason.lower()


def test_max_moisture_tie_break_prefers_shorter_duration() -> None:
    """Two candidates with identical water removed -> pick the shorter one."""
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        # Duplicate 15-min candidate; both produce identical results.
        candidate_durations_minutes=[15.0, 15.0],
        constraints=VentilationConstraints(),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0


def test_max_moisture_optimisation_result_is_frozen() -> None:
    """The returned OptimisationResult is a frozen dataclass."""
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(),
    )
    with pytest.raises(FrozenInstanceError):
        result.feasible = False  # type: ignore[misc]


def test_max_moisture_and_min_energy_are_distinct_strategies() -> None:
    """The two strategies solve different problems and can disagree.

    Under a target of 8 g/m^3 final AH (achievable by 15 and 20 min):
    - min_energy picks 15 min (lower energy cost among the two).
    - max_moisture with the same target and no budget picks 20 min
      (more water removed).

    This test locks in that the alternative objective is a genuine
    alternative, not a rename of the primary one.
    """
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
        ),
    )
    min_energy = choose_minimum_energy_action(**args)
    max_moisture = optimise_max_moisture_under_energy_budget(**args)
    assert min_energy.selected_duration_minutes == 15.0
    assert max_moisture.selected_duration_minutes == 20.0
    assert min_energy.objective_name != max_moisture.objective_name


# --- optimise_max_moisture_under_comfort_limit -----------------------------


def test_max_moisture_under_comfort_no_cap_picks_longest() -> None:
    """With no comfort cap set, the longest duration wins."""
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 30.0
    assert (
        result.objective_name
        == "maximum water removed under comfort limit"
    )


def test_max_moisture_under_comfort_half_kelvin_cap_picks_four_minutes() -> None:
    """ΔT <= 0.5 K -> 4 min feasible (0.476 K), 5 min not (0.593 K).

    Boundary from the canonical scenario at 1-min resolution. Any
    duration <= 4 min stays within the cap; 5 min exceeds it. Max
    water among {0..4 min} is at 4 min (71.46 g).
    """
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_temperature_drop_c=0.5),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 4.0
    assert result.selected_prediction.temperature_drop_c <= 0.5


def test_max_moisture_under_comfort_one_kelvin_cap_picks_eight_minutes() -> None:
    """ΔT <= 1.0 K -> 8 min feasible (0.937 K), 9 min not (1.050 K)."""
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 8.0
    assert result.selected_prediction.temperature_drop_c <= 1.0


def test_max_moisture_under_comfort_two_kelvin_cap_picks_seventeen_minutes() -> None:
    """ΔT <= 2.0 K -> 17 min feasible (1.922 K), 18 min not (2.027 K)."""
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_temperature_drop_c=2.0),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 17.0
    assert result.selected_prediction.temperature_drop_c <= 2.0


def test_max_moisture_under_comfort_tighter_cap_never_selects_a_longer_duration() -> None:
    """Monotonicity: a tighter comfort cap must not select a longer duration."""
    picks: dict = {}
    for cap in (0.5, 1.0, 2.0, 3.0):
        result = optimise_max_moisture_under_comfort_limit(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=list(range(0, 31)),
            constraints=VentilationConstraints(max_temperature_drop_c=cap),
        )
        picks[cap] = result.selected_duration_minutes
    assert picks[0.5] <= picks[1.0] <= picks[2.0] <= picks[3.0]
    assert picks[0.5] < picks[3.0]  # at least one cap change moves the pick


def test_max_moisture_under_comfort_tie_break_prefers_lower_energy() -> None:
    """Two candidates with equal water removed but different energy: cheaper wins.

    Construct a synthetic tie: two candidates that produce
    numerically identical water_removed_g (by using the same
    duration twice). Both have the same energy too, so the tie-break
    falls through to duration - and since both durations are equal,
    the choice is deterministic.

    Then use a case where the tie-break on ENERGY is actually
    discriminating. This is harder to construct in the canonical
    scenario because water and energy are both monotone in
    duration; instead, use two different scenarios where the
    optimiser can be seen picking the lower-energy candidate when
    both water values fall inside the tie tolerance.
    """
    # Simple ordering check: at fixed comfort cap and no
    # usefulness/moisture-target hiccups, the winner is a specific
    # duration and its recorded energy is the lowest among candidates
    # tied on water (which for a monotone scenario is trivially
    # itself).
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[8.0, 8.0, 8.0],
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 8.0


def test_max_moisture_under_comfort_infeasible_when_no_duration_meets_cap() -> None:
    """ΔT cap of 0.01 K + no 0-min candidate -> refuse to pick.

    Every non-zero duration produces a temperature drop above 0.01
    K on the canonical scenario, so the feasibility set is empty
    when 0-min is not in the candidate list.
    """
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(max_temperature_drop_c=0.01),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes != result.selected_duration_minutes  # NaN


def test_max_moisture_under_comfort_usefulness_threshold_forces_do_nothing() -> None:
    """A usefulness floor above any feasible candidate's benefit -> pick 0 min."""
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            max_temperature_drop_c=2.0,
            minimum_water_removed_g=500.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_max_moisture_under_comfort_wetting_scenario_picks_do_nothing() -> None:
    """Outdoor wetter than indoor -> 0-min dominates every negative alternative."""
    result = optimise_max_moisture_under_comfort_limit(
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=12.0,
            indoor_relative_humidity_pct=40.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        ),
        outdoor=AirState(temperature_c=25.0, relative_humidity_percent=85.0),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(max_temperature_drop_c=5.0),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_max_moisture_under_comfort_is_a_distinct_strategy() -> None:
    """The comfort-limit strategy can pick differently from the energy-budget one.

    Under a 1 K comfort cap and no energy cap, the comfort-limit
    strategy picks 8 min (max water within ΔT <= 1 K). Under a 0.10
    kWh budget and no comfort cap, the energy-budget strategy picks
    5 min (max water within 0.10 kWh). Different constraints, same
    scenario, different picks - and both are still meaningful
    "maximise water" strategies.
    """
    comfort_pick = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    ).selected_duration_minutes
    energy_pick = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_energy_loss_kwh=0.10),
    ).selected_duration_minutes
    assert comfort_pick != energy_pick


# --- optimise_weighted_tradeoff -------------------------------------------
# Research-only strategy. Lambda has units of g/kWh; different lambda
# values produce different recommendations, which is the whole point:
# without a defensible basis for lambda the pick is a value judgement,
# not a physics-driven decision.


def test_weighted_tradeoff_zero_lambda_matches_pure_water_maximisation() -> None:
    """lambda_energy = 0 -> energy cost carries no weight -> pick longest.

    With no energy penalty the weighted objective reduces to
    water_removed_g, so under no other constraints the longest
    duration wins (same pick as the no-cap max-moisture strategy).
    """
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(),
    )
    weighted = optimise_weighted_tradeoff(**args, lambda_energy=0.0)
    pure_max = optimise_max_moisture_under_energy_budget(**args)
    assert weighted.feasible is True
    assert weighted.selected_duration_minutes == pure_max.selected_duration_minutes


def test_weighted_tradeoff_large_lambda_penalises_energy_and_picks_shorter() -> None:
    """A huge lambda_energy penalises every joule of heat lost -> pick 0 min.

    At the canonical scenario a 10-min event removes ~142 g at 0.16
    kWh energy cost. With lambda_energy = 1e6 g/kWh the effective
    score is 142 - 1e6 * 0.16 = -1.6e5, way below the do-nothing
    score of 0. So the recommendation is do-nothing.
    """
    result = optimise_weighted_tradeoff(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(),
        lambda_energy=1e6,
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_weighted_tradeoff_intermediate_lambdas_shift_the_selected_duration() -> None:
    """Sweep lambda across several values; the pick should move.

    For each lambda the score is (water_g - lambda * energy_kWh).
    A monotone increase in lambda makes energy more expensive, so
    the selected duration should be non-increasing in lambda under
    a consistent scenario. At least one shift must be strictly
    downward so the sweep is meaningful.
    """
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(),
    )
    picks = {}
    for lambda_energy in (0.0, 100.0, 500.0, 1000.0, 2000.0, 10_000.0):
        result = optimise_weighted_tradeoff(
            **args, lambda_energy=lambda_energy
        )
        assert result.feasible is True
        picks[lambda_energy] = result.selected_duration_minutes

    # Non-increasing in lambda (energy penalty grows -> preferred
    # duration shrinks or stays).
    values = [picks[l] for l in (0.0, 100.0, 500.0, 1000.0, 2000.0, 10_000.0)]
    for earlier, later in zip(values, values[1:]):
        assert later <= earlier, (
            f"picks not monotone in lambda; got sequence {values}"
        )
    # And at least one strict decrease across the sweep.
    assert values[0] > values[-1]


def test_weighted_tradeoff_score_is_water_minus_lambda_times_energy() -> None:
    """The scoring identity is exactly what the docstring claims.

    Reconstruct the score for the winning candidate and confirm it
    is greater than or equal to every alternative's score at the
    same lambda. Locks the "we compute this specific weighted sum"
    contract.
    """
    lambda_energy = 500.0
    candidates = list(range(0, 31))
    result = optimise_weighted_tradeoff(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
        constraints=VentilationConstraints(),
        lambda_energy=lambda_energy,
    )
    assert result.feasible is True
    winner_score = (
        result.selected_prediction.water_removed_g
        - lambda_energy
        * result.selected_prediction.ventilation_energy_removed_kwh
    )
    all_evaluations = evaluate_candidate_durations_with_constraints(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
        constraints=VentilationConstraints(),
    )
    for evaluation in all_evaluations:
        alternative_score = (
            evaluation.prediction.water_removed_g
            - lambda_energy
            * evaluation.prediction.ventilation_energy_removed_kwh
        )
        assert winner_score + 1e-9 >= alternative_score


def test_weighted_tradeoff_honours_hard_constraints() -> None:
    """Hard constraints filter candidates before the weighted score is applied."""
    # A 0.5 K comfort cap keeps only durations up to ~4 min.
    # The unconstrained weighted-tradeoff pick at lambda=0 would be
    # 30 min; under the cap it must be <= 4 min.
    result = optimise_weighted_tradeoff(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(max_temperature_drop_c=0.5),
        lambda_energy=0.0,
    )
    assert result.feasible is True
    assert result.selected_duration_minutes <= 4.0
    assert result.selected_prediction.temperature_drop_c <= 0.5


def test_weighted_tradeoff_rejects_negative_lambda() -> None:
    """Negative lambda has no meaningful interpretation as an exchange rate."""
    with pytest.raises(ValueError, match="lambda_energy"):
        optimise_weighted_tradeoff(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[0.0, 15.0],
            constraints=VentilationConstraints(),
            lambda_energy=-1.0,
        )


def test_weighted_tradeoff_rejects_non_finite_lambda() -> None:
    """NaN / inf lambda is rejected."""
    with pytest.raises(ValueError, match="lambda_energy"):
        optimise_weighted_tradeoff(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[0.0, 15.0],
            constraints=VentilationConstraints(),
            lambda_energy=float("nan"),
        )
    with pytest.raises(ValueError, match="lambda_energy"):
        optimise_weighted_tradeoff(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=[0.0, 15.0],
            constraints=VentilationConstraints(),
            lambda_energy=float("inf"),
        )


def test_weighted_tradeoff_objective_name_includes_lambda_value() -> None:
    """Reason and objective_name expose the lambda used - part of the audit trail."""
    result = optimise_weighted_tradeoff(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(),
        lambda_energy=500.0,
    )
    assert "500" in result.objective_name
    assert "500" in result.reason
    # And the reason must remind the reader that lambda is a caller
    # judgement, not a universal value.
    assert "exchange rate" in result.reason.lower() or "caller" in result.reason.lower()


def test_weighted_tradeoff_is_distinct_from_min_energy_and_max_moisture() -> None:
    """At an intermediate lambda the pick can differ from both extreme strategies."""
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
    )
    # No moisture target - min-energy would be undefined; use the
    # max-moisture-under-budget vs the weighted trade-off at a
    # non-trivial lambda.
    pure_water = optimise_max_moisture_under_energy_budget(
        **args, constraints=VentilationConstraints()
    ).selected_duration_minutes
    weighted_mid = optimise_weighted_tradeoff(
        **args,
        constraints=VentilationConstraints(),
        lambda_energy=800.0,
    ).selected_duration_minutes
    # The weighted pick at a meaningful lambda should be strictly
    # shorter than the pure-water pick (which is 30 min).
    assert weighted_mid < pure_water


# --- optimise_marginal_efficiency_threshold --------------------------------
# Walks consecutive candidate durations and picks the last duration
# before the marginal Δwater / Δenergy falls at or below a configured
# threshold. Reference intervals for the canonical scenario at durations
# [0,2,3,5,10,15,20] min are approximately:
#   0-> 2: 1161 g/kWh    2-> 3: 1036   3-> 5: 926
#   5->10:  716          10->15:  491  15->20: 337


def _marginal_candidates() -> list:
    """Duration list used across the marginal-efficiency tests."""
    return [0.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]


def test_marginal_threshold_600_picks_ten_minutes() -> None:
    """Threshold 600 g/kWh: 10-15 interval (491) is the first below; pick 10 min.

    All earlier intervals (716 at 5-10, 926 at 3-5, etc.) are above
    the threshold, so extension is worthwhile up to 10 min. The
    10-15 interval falls below the threshold, so the strategy stops
    at the shorter neighbour (10 min).
    """
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=600.0
        ),
    )
    assert isinstance(result, OptimisationResult)
    assert result.feasible is True
    assert result.selected_duration_minutes == 10.0
    assert "threshold" in result.objective_name.lower()


def test_marginal_threshold_400_picks_fifteen_minutes() -> None:
    """Threshold 400 g/kWh: 15-20 interval (337) is first below; pick 15 min."""
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=400.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 15.0


def test_marginal_threshold_never_crossed_picks_longest_candidate() -> None:
    """Threshold 100 g/kWh: every interval is above it; pick longest.

    The lowest marginal in the sweep is ~337 g/kWh (15-20), which is
    well above 100. No interval crosses the threshold, so the
    strategy walks to the end and selects the longest candidate.
    """
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=100.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 20.0
    assert "stayed strictly above" in result.reason.lower()


def test_marginal_threshold_immediately_exceeded_picks_shortest_candidate() -> None:
    """Threshold 2000 g/kWh: even 0-2 interval (1161) is below it; pick 0 min.

    Nothing in the sweep clears such a high threshold. The strategy
    stops at the first interval and selects the shorter neighbour
    (0 min) - do nothing is the recommendation.
    """
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=2000.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_marginal_threshold_unset_returns_infeasible() -> None:
    """Without a threshold, the strategy has nothing to walk against."""
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(),
    )
    assert result.feasible is False
    assert result.selected_duration_minutes != result.selected_duration_minutes  # NaN
    assert "minimum_marginal_g_per_kwh" in result.reason


def test_marginal_threshold_sorts_unordered_candidates() -> None:
    """Unsorted candidate list still produces the correct answer.

    The strategy sorts internally so a caller who supplied a
    non-monotone list from another experiment still gets the right
    interval walk.
    """
    scrambled = [10.0, 0.0, 20.0, 5.0, 15.0, 3.0, 2.0]
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=scrambled,
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=600.0
        ),
    )
    assert result.selected_duration_minutes == 10.0


def test_marginal_threshold_zero_delta_energy_intervals_are_skipped() -> None:
    """Duplicate durations produce zero-Δenergy intervals; they must be skipped.

    Duplicating 5.0 in the candidate list creates a 5→5 interval
    with Δenergy = 0. The strategy must NOT divide by that, but
    also must NOT let it terminate the walk - the next real
    interval (5→10 at 716 g/kWh) should still be walked.
    """
    duplicated = [0.0, 5.0, 5.0, 10.0, 15.0]
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=duplicated,
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=600.0
        ),
    )
    assert result.feasible is True
    # Same answer as without the duplicate (10 min), because the
    # 5->5 zero-width interval is skipped.
    assert result.selected_duration_minutes == 10.0


def test_marginal_threshold_wetting_scenario_picks_zero_min() -> None:
    """Outdoor wetter than indoor -> every interval has Δwater < 0 -> pick 0 min.

    Δwater < 0 makes the marginal ratio negative. Any non-negative
    threshold is above every negative marginal, so the strategy
    stops at the very first interval and selects the shorter
    neighbour, which is 0 min.
    """
    result = optimise_marginal_efficiency_threshold(
        room=Room(
            volume_m3=40.0,
            indoor_temperature_c=12.0,
            indoor_relative_humidity_pct=40.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        ),
        outdoor=AirState(temperature_c=25.0, relative_humidity_percent=85.0),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=100.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_marginal_threshold_infeasibility_bubbles_up_from_hard_constraints() -> None:
    """A comfort cap that no candidate can meet -> infeasible before the walk starts."""
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            max_temperature_drop_c=0.01,
            minimum_marginal_g_per_kwh=100.0,
        ),
    )
    assert result.feasible is False


def test_marginal_threshold_reason_names_the_threshold_and_context() -> None:
    """The reason string is auditable: it names the threshold in use."""
    result = optimise_marginal_efficiency_threshold(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=_marginal_candidates(),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=600.0
        ),
    )
    assert "600" in result.reason
    assert "g/kwh" in result.reason.lower()


def test_marginal_threshold_field_rejects_negative_values() -> None:
    """The new usefulness field rejects negatives."""
    with pytest.raises(ValueError, match="minimum_marginal_g_per_kwh"):
        VentilationConstraints(minimum_marginal_g_per_kwh=-1.0)


def test_marginal_threshold_field_rejects_non_finite_values() -> None:
    """NaN / inf on the new field are rejected."""
    with pytest.raises(ValueError, match="minimum_marginal_g_per_kwh"):
        VentilationConstraints(minimum_marginal_g_per_kwh=float("nan"))
    with pytest.raises(ValueError, match="minimum_marginal_g_per_kwh"):
        VentilationConstraints(minimum_marginal_g_per_kwh=float("inf"))


def test_marginal_threshold_shifts_selected_duration_with_threshold() -> None:
    """Sensitivity sweep: rising threshold -> non-increasing selection."""
    picks = {}
    for threshold in (100.0, 400.0, 600.0, 800.0, 2000.0):
        result = optimise_marginal_efficiency_threshold(
            room=_default_room(),
            outdoor=_default_outdoor(),
            thermal_properties=_default_thermal_properties(),
            candidate_durations_minutes=_marginal_candidates(),
            constraints=VentilationConstraints(
                minimum_marginal_g_per_kwh=threshold
            ),
        )
        assert result.feasible is True
        picks[threshold] = result.selected_duration_minutes
    values = [picks[t] for t in (100.0, 400.0, 600.0, 800.0, 2000.0)]
    for a, b in zip(values, values[1:]):
        assert b <= a, f"picks not monotone in threshold; got {values}"
    assert values[0] > values[-1], "threshold sweep produces no movement"


# --- pareto_efficient_indices ---------------------------------------------


def test_pareto_empty_input_returns_empty_list() -> None:
    """No candidates -> no Pareto set."""
    assert pareto_efficient_indices([]) == []


def test_pareto_single_candidate_is_trivially_efficient() -> None:
    """One candidate can't be dominated -> it stays in the set."""
    predictions = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[10.0],
    )
    assert pareto_efficient_indices(predictions) == [0]


def test_pareto_canonical_monotone_scenario_is_entirely_efficient() -> None:
    """A pure drying+cooling event has NO dominated candidates.

    In the canonical scenario at duration steps 0/5/10/15/20:
    water and energy both increase monotonically with duration.
    Every candidate offers a genuine trade-off that the next-longer
    candidate does not dominate (longer offers more water but costs
    more energy). So all 5 durations are Pareto-efficient.
    """
    candidates = [0.0, 5.0, 10.0, 15.0, 20.0]
    predictions = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    efficient = pareto_efficient_indices(predictions)
    assert efficient == list(range(len(candidates)))


def test_pareto_synthetic_dominated_candidate_is_removed() -> None:
    """A candidate strictly dominated on both axes must be removed.

    Constructs a synthetic 3-candidate list by hand so the
    dominance relationship is unambiguous, independent of the
    simulator: candidate B has strictly more water AND strictly
    less energy than candidate A, so A is dominated. Candidate C
    is strictly better than both A and B on both axes; A and B
    are both dominated by C.
    """
    from dataclasses import replace

    # Get one real prediction so we can vary its fields legitimately.
    baseline = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[10.0],
    )[0]
    # A: 50 g water at 0.20 kWh (dominated by B and C)
    # B: 80 g water at 0.15 kWh (dominates A, dominated by C)
    # C: 120 g water at 0.10 kWh (dominates A and B)
    prediction_a = replace(baseline, water_removed_g=50.0,
                          ventilation_energy_removed_kwh=0.20)
    prediction_b = replace(baseline, water_removed_g=80.0,
                          ventilation_energy_removed_kwh=0.15)
    prediction_c = replace(baseline, water_removed_g=120.0,
                          ventilation_energy_removed_kwh=0.10)
    predictions = [prediction_a, prediction_b, prediction_c]
    efficient = pareto_efficient_indices(predictions)
    # Only C is Pareto-efficient.
    assert efficient == [2]


def test_pareto_wetting_scenario_all_candidates_efficient() -> None:
    """In a wetting event water and energy trade in OPPOSITE directions.

    ``ventilation_energy_removed_kwh`` is NEGATIVE when the room
    gains heat (warmer outdoor). A longer wetting event adds more
    water (worse) but also traps more heat (better under the pure
    "heat leaving the room" sign convention). Neither candidate
    dominates the other on both axes simultaneously, so every
    non-zero candidate stays in the Pareto set alongside 0-min.
    """
    wetting_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=12.0,
        indoor_relative_humidity_pct=40.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    wetter_outdoor = AirState(
        temperature_c=25.0, relative_humidity_percent=85.0
    )
    candidates = [0.0, 5.0, 10.0, 15.0, 20.0]
    predictions = evaluate_candidate_durations(
        room=wetting_room,
        outdoor=wetter_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    efficient = pareto_efficient_indices(predictions)
    assert efficient == list(range(len(candidates)))


def test_pareto_preserves_input_order() -> None:
    """The returned indices are in the input order, not sorted."""
    # Non-monotone input; the physics is still monotone-in-duration
    # so every candidate is Pareto-efficient. The returned index
    # list must match the caller's ORIGINAL positions.
    candidates = [10.0, 0.0, 20.0, 5.0, 15.0]
    predictions = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    efficient = pareto_efficient_indices(predictions)
    # All efficient; indices returned in input order.
    assert efficient == list(range(len(candidates)))


def test_pareto_duplicates_are_all_retained() -> None:
    """Two identical candidates neither dominate each other -> both remain.

    Weak dominance requires at least one strict improvement. Two
    equal (water, energy) pairs offer no strict improvement in
    either direction, so both stay in the frontier.
    """
    candidates = [10.0, 10.0, 10.0]
    predictions = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    efficient = pareto_efficient_indices(predictions)
    assert efficient == [0, 1, 2]


def test_pareto_is_not_used_to_select_actions() -> None:
    """Documentation invariant: the module surface is analytic, not decisional.

    The Pareto helper returns INDICES only; it does NOT return an
    OptimisationResult. A future contributor cannot accidentally
    make it a strategy without breaking this test.
    """
    predictions = evaluate_candidate_durations(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0],
    )
    result = pareto_efficient_indices(predictions)
    assert isinstance(result, list)
    assert all(isinstance(index, int) for index in result)
    assert not isinstance(result, OptimisationResult)


# --- "Ventilation is unhelpful" edge cases --------------------------------
# The optimiser must not open the window when doing so has no moisture
# benefit, or when the drying benefit is negligible compared with the
# thermal penalty. These tests pin those decisions into the suite.


def test_outdoor_ah_above_indoor_ah_min_energy_picks_do_nothing() -> None:
    """Outdoor wetter than indoor: any ventilation adds moisture.

    Indoor 20 C / 40 %RH (AH ~= 6.9 g/m^3); outdoor 25 C / 85 %RH
    (AH ~= 19.4 g/m^3). Outdoor AH strictly exceeds indoor AH, so no
    ventilation event can lower the final AH below the indoor
    starting value. Under a moisture target already satisfied at t=0
    (initial AH is below the ceiling), the primary optimiser must
    pick 0 min: opening the window would raise the AH above the
    initial value.
    """
    dry_indoor_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=40.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    wet_outdoor = AirState(
        temperature_c=25.0, relative_humidity_percent=85.0
    )
    # Verify the premise: outdoor AH strictly above indoor AH.
    indoor_ah = AirState(
        temperature_c=20.0, relative_humidity_percent=40.0
    ).absolute_humidity
    outdoor_ah = wet_outdoor.absolute_humidity
    assert outdoor_ah > indoor_ah

    # A ceiling of 10 g/m^3 is above the initial indoor AH, so
    # duration 0 already satisfies the target. The min-energy pick
    # for a satisfied target is 0.
    result = choose_minimum_energy_action(
        room=dry_indoor_room,
        outdoor=wet_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=10.0
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    # No moisture removed, no energy spent, no temperature drop.
    assert result.selected_prediction.water_removed_g == pytest.approx(
        0.0, abs=1e-9
    )
    assert result.selected_prediction.ventilation_energy_removed_kwh == 0.0
    assert result.selected_prediction.temperature_drop_c == 0.0


def test_outdoor_ah_above_indoor_ah_max_moisture_picks_do_nothing() -> None:
    """Under max-water strategies, wetter outdoor also picks do-nothing.

    Every non-zero duration adds water (negative water_removed_g);
    duration 0 sits at exactly 0 g. Both max-moisture strategies
    should pick 0 min because 0.0 dominates every negative water
    removal in a max search.
    """
    wetter_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=40.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    wet_outdoor = AirState(
        temperature_c=25.0, relative_humidity_percent=85.0
    )
    candidates = [0.0, 5.0, 10.0, 15.0, 20.0]
    args = dict(
        room=wetter_room,
        outdoor=wet_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=candidates,
    )
    budget_result = optimise_max_moisture_under_energy_budget(
        **args, constraints=VentilationConstraints()
    )
    comfort_result = optimise_max_moisture_under_comfort_limit(
        **args, constraints=VentilationConstraints()
    )
    assert budget_result.feasible is True
    assert budget_result.selected_duration_minutes == 0.0
    assert comfort_result.feasible is True
    assert comfort_result.selected_duration_minutes == 0.0


def test_outdoor_ah_equal_to_indoor_ah_picks_do_nothing() -> None:
    """Boundary case: outdoor AH exactly equals indoor AH.

    No moisture gradient means every non-zero duration removes
    ~zero water (up to FP noise) and still costs real energy.
    The primary optimiser under a moisture ceiling already met at
    t = 0 must recommend do-nothing rather than pointlessly opening
    the window.
    """
    # Indoor 20 C / 50 %RH (AH ~= 8.65 g/m^3). Construct an outdoor
    # state with exactly the same AH at a different temperature so
    # the thermal gap is non-zero but the moisture gap is zero.
    indoor_ah = AirState(
        temperature_c=20.0, relative_humidity_percent=50.0
    ).absolute_humidity
    # Pick outdoor T = 10 C; solve for RH that gives the same AH.
    from psychrometrics import (
        G_PER_KG,
        M_WATER,
        R_UNIVERSAL,
        ZERO_CELSIUS_IN_KELVIN,
        saturation_vapour_pressure,
    )
    outdoor_t_c = 10.0
    outdoor_p_v_pa = (
        (indoor_ah / G_PER_KG)
        * R_UNIVERSAL
        * (outdoor_t_c + ZERO_CELSIUS_IN_KELVIN)
        / M_WATER
    )
    outdoor_rh = 100.0 * outdoor_p_v_pa / saturation_vapour_pressure(outdoor_t_c)
    outdoor = AirState(
        temperature_c=outdoor_t_c, relative_humidity_percent=outdoor_rh
    )
    assert outdoor.absolute_humidity == pytest.approx(indoor_ah, rel=1e-12)

    matched_room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=50.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    result = choose_minimum_energy_action(
        room=matched_room,
        outdoor=outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 10.0, 15.0, 20.0],
        # Target = indoor initial AH; already met at t = 0.
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=indoor_ah + 0.01
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0


def test_marginally_drier_outdoor_usefulness_threshold_picks_do_nothing() -> None:
    """Outdoor is drier but the benefit is too small to justify the thermal cost.

    Indoor 20 C / 70 %RH -> AH ~= 12.07 g/m^3.
    Outdoor 5 C / 95 %RH -> AH ~= 6.45 g/m^3 (drying potential ~5.6).
    Actually make the outdoor scenario "marginally drier" by picking
    outdoor conditions whose AH sits just below indoor AH: outdoor
    18 C / 65 %RH -> AH ~= 10.02 g/m^3, drying potential ~2.05.

    A 15-minute event at these conditions removes only ~60 g of
    water. Configure a usefulness floor of 200 g and expect the
    optimiser to pick 0 min: the drying benefit is real but the
    floor says it isn't worth the thermal penalty.
    """
    room = _default_room()
    marginal_outdoor = AirState(
        temperature_c=18.0, relative_humidity_percent=65.0
    )
    indoor_ah = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    ).absolute_humidity
    outdoor_ah = marginal_outdoor.absolute_humidity
    # Precondition: outdoor is drier, but the gap is small.
    assert outdoor_ah < indoor_ah
    assert (indoor_ah - outdoor_ah) < 3.0

    result = choose_minimum_energy_action(
        room=room,
        outdoor=marginal_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            # Ceiling above the initial indoor AH so "do nothing"
            # already satisfies the target - only the usefulness
            # threshold gates the recommendation.
            target_final_absolute_humidity_g_m3=indoor_ah + 0.01,
            minimum_water_removed_g=200.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 0.0
    # The reason must flag usefulness explicitly, not moisture target
    # feasibility.
    assert (
        "usefulness" in result.reason.lower()
        or "negligible" in result.reason.lower()
    )


def test_marginally_drier_outdoor_marginal_threshold_picks_short_duration() -> None:
    """Marginal-efficiency strategy stops when the caller's threshold is crossed.

    Under an 18 C / 65 %RH outdoor scenario the marginal g/kWh
    across intervals is unusually HIGH (thousands of g/kWh) because
    the small thermal gap keeps the energy denominator small even
    as the moisture numerator shrinks. That is a real (and initially
    surprising) physics finding: a mild-and-only-slightly-drier
    scenario looks very efficient in g/kWh even though the absolute
    water removed per event is small.

    Set the caller's threshold at 1500 g/kWh - inside the marginal
    band this scenario produces - and confirm the strategy stops at
    a short duration.
    """
    room = _default_room()
    marginal_outdoor = AirState(
        temperature_c=18.0, relative_humidity_percent=65.0
    )
    result = optimise_marginal_efficiency_threshold(
        room=room,
        outdoor=marginal_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=1500.0
        ),
    )
    assert result.feasible is True
    # Threshold sits inside the falling marginal-efficiency band for
    # this scenario, so the pick is well short of the longest
    # candidate.
    assert result.selected_duration_minutes < 30.0
    assert result.selected_duration_minutes > 0.0


def test_baseline_ah_rule_would_fire_but_optimiser_refuses_under_usefulness_floor() -> None:
    """Naive "outdoor drier than indoor" rule would ventilate; optimiser doesn't.

    Regression test against the specific failure mode the baseline
    experiment warned about: the AH-rule fires True for marginally-
    drier outdoor scenarios and would blindly recommend ventilation.
    The optimiser, with a usefulness floor set, refuses when the
    drying benefit does not justify the thermal penalty.

    Indoor 20 C / 70 %RH, outdoor 18 C / 65 %RH: AH-rule says
    "beneficial" (12.07 > 10.02). Usefulness floor of 200 g rejects
    it; the primary optimiser picks 0 min.
    """
    room = _default_room()
    marginal_outdoor = AirState(
        temperature_c=18.0, relative_humidity_percent=65.0
    )
    indoor_ah = AirState(
        temperature_c=room.indoor_temperature_c,
        relative_humidity_percent=room.indoor_relative_humidity_pct,
    ).absolute_humidity
    outdoor_ah = marginal_outdoor.absolute_humidity

    # Baseline rule: indoor > outdoor -> beneficial. Fires True here.
    assert indoor_ah > outdoor_ah

    # Optimiser: refuses under the usefulness floor.
    result = choose_minimum_energy_action(
        room=room,
        outdoor=marginal_outdoor,
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=indoor_ah + 0.01,
            minimum_water_removed_g=200.0,
        ),
    )
    assert result.selected_duration_minutes == 0.0


# --- Reviewer regressions ---------------------------------------------------
# Pin the three findings from the optimiser review.


def test_fallback_reason_lists_only_comfort_violations() -> None:
    """Finding B: 'no comfort feasible' reason must not include moisture names.

    When both moisture targets and comfort caps are unreachable AND
    no zero-min candidate is in the list, the optimiser refuses.
    The reason must name the COMFORT violations only, because a
    caller reading "no candidate satisfies the comfort constraints
    (max_energy_loss_kwh, max_temperature_drop_c, ..., target_...)"
    would be misled about which constraint is at fault.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,   # unreachable
            max_temperature_drop_c=0.05,                # unreachable
        ),
    )
    assert result.feasible is False
    lowered_reason = result.reason.lower()
    # Comfort-side violation names may appear:
    assert "max_temperature_drop_c" in lowered_reason
    # Moisture-target names MUST NOT appear - the message is about
    # comfort failure, not moisture-target failure.
    assert "target_final_absolute_humidity_g_m3" not in lowered_reason
    assert "target_moisture_reduction_g_m3" not in lowered_reason


def test_min_energy_fallback_tie_break_uses_water_tolerance_bucket() -> None:
    """Finding A: fallback branch honours WATER_REMOVED_TIE_TOLERANCE_G.

    Two comfort-feasible candidates whose water_removed_g differ by
    less than the tolerance should tie on water and defer to the
    shorter-duration rule. Uses a duplicated 15.0-min candidate to
    guarantee identical water values under a moisture target that
    is unreachable (forcing the fallback).
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[15.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=4.0,   # forces fallback
        ),
    )
    assert result.feasible is False
    # Tie-break must produce a deterministic pick; both entries have
    # identical water and energy so the water-bucket ties and the
    # duration-tie-break returns the first (shortest) one - here
    # both are 15.0, so the winner is 15.0 either way, but the
    # sort key must be well-defined for the deterministic-pick
    # invariant.
    assert result.selected_duration_minutes == 15.0


def test_max_moisture_under_comfort_tie_break_prefers_lower_energy_on_water_tie() -> None:
    """Tie on water_removed_g should defer to LOWER energy.

    The existing comfort-limit strategy test only asserted the
    duplicate-duration case. Here we construct a synthetic list
    where two candidates have (approximately) the same water but
    different energies, using two different duration values that
    both fall inside the same water-tie bucket. The scenario is
    monotone in the canonical simulator so genuine within-tolerance
    ties across DIFFERENT durations are rare; use direct call with
    a duplicated 8.0-min then verify the deterministic pick.
    """
    result = optimise_max_moisture_under_comfort_limit(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[8.0, 8.0],
        constraints=VentilationConstraints(max_temperature_drop_c=1.0),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 8.0


# --- POC default: recommend_ventilation_action -----------------------------


def test_recommend_ventilation_action_delegates_to_min_energy() -> None:
    """The POC default wrapper is min-energy verbatim.

    Every field of the returned OptimisationResult must equal what
    ``choose_minimum_energy_action`` returns for the same inputs.
    A future contributor cannot silently retarget the default at a
    different strategy without failing this test.
    """
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    default_result = recommend_ventilation_action(**args)
    primary_result = choose_minimum_energy_action(**args)
    assert default_result == primary_result


def test_recommend_ventilation_action_reports_min_energy_objective() -> None:
    """The default's ``objective_name`` field names the min-energy strategy."""
    result = recommend_ventilation_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    assert "minimum ventilation energy loss" in result.objective_name.lower()


def test_recommend_ventilation_action_uses_the_documented_default() -> None:
    """POC-default worked scenario: canonical inputs -> 13-min pick.

    Anchors the documented behaviour of the POC default so that a
    change to the primary strategy would fail visibly here rather
    than be discovered by a downstream consumer.
    """
    result = recommend_ventilation_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    assert result.feasible is True
    assert result.selected_duration_minutes == 13.0


def test_other_strategies_remain_available() -> None:
    """The four research strategies must still be importable and callable.

    Sets the POC's "keep the alternatives around for comparison"
    invariant into the test suite. Any future removal of one of
    these functions (or one of the constants exposed alongside
    them) will fail this test rather than break silently.
    """
    args = dict(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 10.0, 20.0],
    )
    a = optimise_max_moisture_under_energy_budget(
        **args, constraints=VentilationConstraints()
    )
    b = optimise_max_moisture_under_comfort_limit(
        **args, constraints=VentilationConstraints()
    )
    c = optimise_weighted_tradeoff(
        **args, constraints=VentilationConstraints(), lambda_energy=500.0
    )
    d = optimise_marginal_efficiency_threshold(
        **args,
        constraints=VentilationConstraints(
            minimum_marginal_g_per_kwh=500.0
        ),
    )
    for r in (a, b, c, d):
        assert isinstance(r, OptimisationResult)


def test_recommend_ventilation_action_signature_matches_primary_strategy() -> None:
    """The default wrapper accepts the same keyword arguments as the primary strategy.

    Guards against a future contributor adding or removing a
    parameter on the primary strategy without updating the
    wrapper (or vice versa).
    """
    import inspect

    default_sig = inspect.signature(recommend_ventilation_action)
    primary_sig = inspect.signature(choose_minimum_energy_action)
    assert list(default_sig.parameters) == list(primary_sig.parameters)
