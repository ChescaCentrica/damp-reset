"""Tests for the presentation layer.

The presentation layer is a plain-text formatter. It reads named
fields off an ``OptimisationResult`` and returns a string. Tests
verify:

    - Three output shapes: ventilate, do-nothing, infeasible.
    - Named fields are pulled from the correct places.
    - The optimiser's reason string is passed through unchanged.
    - No physics arithmetic happens in ``presentation.py`` (AST
      guard, twin of the one used for the optimiser).
"""

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import Room
from optimiser import (
    VentilationConstraints,
    choose_minimum_energy_action,
    optimise_max_moisture_under_energy_budget,
)
from presentation import format_recommendation
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties


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


# --- Output shape: ventilate ------------------------------------------------


def test_ventilate_recommendation_names_the_selected_duration() -> None:
    """Non-zero duration -> 'open the window for X min' phrasing."""
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    text = format_recommendation(result)
    assert "open the window" in text.lower()
    assert f"{result.selected_duration_minutes:g} min" in text
    assert "do not ventilate" not in text.lower()


def test_ventilate_recommendation_includes_all_predicted_outcome_fields() -> None:
    """Predicted-outcome section names every field the caller asked for.

    The user's example says: water removed, final RH, temp drop,
    ventilation energy loss. The docstring also promises final AH
    and final temperature. Every one of those must appear in the
    text.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    text = format_recommendation(result)
    lowered = text.lower()
    for required in (
        "water removed",
        "final absolute humidity",
        "final relative humidity",
        "final temperature",
        "temperature drop",
        "ventilation energy loss",
    ):
        assert required in lowered, f"missing '{required}' in output"


def test_ventilate_predicted_values_come_directly_from_the_prediction() -> None:
    """Numeric values in the string match the ``selected_prediction`` fields.

    Test uses the same rounding the formatter applies so a byte
    comparison would fail; instead assert the rounded formatted
    substrings appear.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    text = format_recommendation(result)
    prediction = result.selected_prediction
    assert f"{prediction.water_removed_g:+.2f} g" in text
    assert f"{prediction.final_absolute_humidity_g_m3:.2f} g/m^3" in text
    assert f"{prediction.final_relative_humidity_pct:.1f} %" in text
    assert f"{prediction.final_temperature_c:.2f} C" in text
    assert f"{prediction.temperature_drop_c:+.2f} K" in text
    assert (
        f"{prediction.ventilation_energy_removed_kwh:+.4f} kWh" in text
    )


# --- Output shape: do-nothing ----------------------------------------------


def test_do_nothing_recommendation_says_do_not_ventilate() -> None:
    """Duration = 0 -> 'do not ventilate' phrasing.

    Force do-nothing by picking a moisture target already met at
    t = 0. The primary optimiser then correctly recommends 0 min.
    """
    room = _default_room()
    outdoor = _default_outdoor()
    thermal_props = _default_thermal_properties()
    # Indoor AH is ~12 g/m^3; target above that is trivially met.
    result = choose_minimum_energy_action(
        room=room,
        outdoor=outdoor,
        thermal_properties=thermal_props,
        candidate_durations_minutes=[0.0, 5.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    assert result.selected_duration_minutes == 0.0
    text = format_recommendation(result)
    lowered = text.lower()
    assert "do not ventilate" in lowered
    assert "open the window" not in lowered
    assert "no change" in lowered


def test_do_nothing_recommendation_uses_the_optimiser_reason_string() -> None:
    """The reason line copies the optimiser's reason verbatim."""
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 5.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    text = format_recommendation(result)
    assert result.reason in text


# --- Output shape: infeasible ----------------------------------------------


def test_infeasible_recommendation_says_no_action() -> None:
    """Infeasible result -> 'no action can be recommended' phrasing.

    A 0 kWh budget with no 0-min candidate leaves every non-zero
    candidate infeasible.
    """
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(max_energy_loss_kwh=0.0),
    )
    assert result.feasible is False
    text = format_recommendation(result)
    lowered = text.lower()
    assert "no action" in lowered
    assert "open the window" not in lowered
    assert "do not ventilate" not in lowered
    assert result.reason in text


def test_infeasible_recommendation_reads_the_reason_string() -> None:
    """The optimiser's reason string appears in the output verbatim."""
    result = optimise_max_moisture_under_energy_budget(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[5.0, 10.0, 15.0],
        constraints=VentilationConstraints(max_energy_loss_kwh=0.0),
    )
    text = format_recommendation(result)
    assert result.reason in text


# --- Reason string is passed through unchanged ------------------------------


def test_reason_string_is_verbatim() -> None:
    """The reason line of every output shape carries the optimiser's exact text.

    Guards against a future contributor re-flowing / summarising the
    reason inside the presentation layer. The optimiser is the
    source of truth for why the pick was made; the presentation
    layer must not paraphrase.
    """
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=list(range(0, 31)),
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=8.0,
            max_temperature_drop_c=2.0,
        ),
    )
    text = format_recommendation(result)
    assert result.reason in text


# --- Design contract: no physics in the presentation layer -----------------


def test_presentation_module_contains_no_physics_arithmetic() -> None:
    """AST guard: presentation.py must not do arithmetic on prediction fields.

    Twin of the guard on the optimiser module. Every derived value
    the presentation shows must come DIRECTLY from a named field on
    the prediction; the module must not compute anything itself.

    Rules:
        1. imports OptimisationResult from ``optimiser``;
        2. imports no physics constants (rho, cp, /3600, Magnus,
           /1000, +273.15, etc.);
        3. no BinOp on any known prediction-field name at all.
    """
    presentation_source = (
        Path(__file__).resolve().parent.parent / "presentation.py"
    )
    tree = ast.parse(presentation_source.read_text())

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    assert "OptimisationResult" in imported_names

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
        f"presentation must not import physics constants; "
        f"got {imported_names & forbidden}"
    )

    prediction_fields = {
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

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Attribute)
                    and child.attr in prediction_fields
                ):
                    raise AssertionError(  # pragma: no cover - protective
                        "presentation.py performs arithmetic on a "
                        "prediction field. Every derived quantity in "
                        "the recommendation must be read directly from "
                        "a named field on the prediction, not computed "
                        "in the presentation layer."
                    )


# --- Return type ------------------------------------------------------------


def test_format_recommendation_returns_plain_string() -> None:
    """The function returns a single string, not a dict / dataclass / list."""
    result = choose_minimum_energy_action(
        room=_default_room(),
        outdoor=_default_outdoor(),
        thermal_properties=_default_thermal_properties(),
        candidate_durations_minutes=[0.0, 15.0],
        constraints=VentilationConstraints(
            target_final_absolute_humidity_g_m3=15.0
        ),
    )
    text = format_recommendation(result)
    assert isinstance(text, str)
    assert "\n" in text  # multi-line by design (recommendation / reason / outcome)
    # Structure check: three labelled lines in every shape.
    assert text.startswith("Recommendation:")
    assert "Reason:" in text
    assert "Predicted outcome:" in text