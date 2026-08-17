"""Presentation layer: format an ``OptimisationResult`` as human text.

Reads a completed ``OptimisationResult`` from the optimiser and
returns a single formatted string with:

    Recommendation: <action>.
    Reason: <the reason string the optimiser produced>.
    Predicted outcome: <named fields off the result's prediction>.

Design contract:
    * No physics: this module never touches Magnus, ideal-gas,
      ρ·cp, exp(-n·t), or any other equation. Every physically
      meaningful number comes DIRECTLY from
      ``OptimisationResult.selected_prediction``.
    * No derived quantities: no ratios, no differences beyond what
      the prediction already carries. If a future caller wants a
      new derived number in the output, add the number to the
      simulator's output and read it here - do not compute it here.
    * No side effects: the module returns a string. It never prints,
      logs, or writes to disk. Callers decide where the text goes.
    * One-way dependency: this module imports ``OptimisationResult``
      from the optimiser; the optimiser never imports this module.

Explicitly NOT in this module: HTML formatting, JSON serialisation,
localisation, logging, dashboards, notification integrations. Any of
those can be built on top of the string this returns.
"""

from math import isnan

from optimiser import OptimisationResult


def format_recommendation(result: OptimisationResult) -> str:
    """Return a plain-text summary of an ``OptimisationResult``.

    Two shapes:

        Do-nothing case (``selected_duration_minutes`` = 0.0 and
        feasible = True):
            Recommendation: do not ventilate.
            Reason: <the optimiser's reason string>.
            Predicted outcome: no change - the room remains at its
                initial state.

        Ventilate case (``selected_duration_minutes`` > 0.0 and
        feasible = True):
            Recommendation: open the window for X min.
            Reason: <the optimiser's reason string>.
            Predicted outcome: water removed X g, final AH X g/m^3,
                final RH X %, temperature drop X K, ventilation
                energy loss X kWh.

        Infeasible case (feasible = False):
            Recommendation: no action can be recommended.
            Reason: <the optimiser's reason string, which explains
                which constraint could not be met>.
            Predicted outcome: none - no feasible action was found.

    Uses only named fields on the result. Sign conventions and units
    are inherited from ``OptimisationResult.selected_prediction`` (a
    ``VentilationSimulationResult`` - see its docstring for details).

    Args:
        result: the outcome of an optimiser strategy call.

    Returns:
        A single formatted string, suitable for printing, logging,
        or embedding in a report.
    """
    if not result.feasible:
        return _format_infeasible(result)
    duration = result.selected_duration_minutes
    if isnan(duration):
        # Should not happen for a feasible result, but defensive.
        return _format_infeasible(result)
    if duration == 0.0:
        return _format_do_nothing(result)
    return _format_ventilate(result)


def _format_do_nothing(result: OptimisationResult) -> str:
    """Format the 'do not ventilate' recommendation."""
    return (
        "Recommendation: do not ventilate.\n"
        f"Reason: {result.reason}\n"
        "Predicted outcome: no change - the room remains at its "
        "initial state."
    )


def _format_ventilate(result: OptimisationResult) -> str:
    """Format a 'open the window for X min' recommendation."""
    prediction = result.selected_prediction
    duration = result.selected_duration_minutes
    return (
        f"Recommendation: open the window for {duration:g} min.\n"
        f"Reason: {result.reason}\n"
        "Predicted outcome: "
        f"water removed {prediction.water_removed_g:+.2f} g, "
        f"final absolute humidity {prediction.final_absolute_humidity_g_m3:.2f} g/m^3, "
        f"final relative humidity {prediction.final_relative_humidity_pct:.1f} %, "
        f"final temperature {prediction.final_temperature_c:.2f} C, "
        f"temperature drop {prediction.temperature_drop_c:+.2f} K, "
        f"ventilation energy loss {prediction.ventilation_energy_removed_kwh:+.4f} kWh."
    )


def _format_infeasible(result: OptimisationResult) -> str:
    """Format the 'no feasible action' response."""
    return (
        "Recommendation: no action can be recommended.\n"
        f"Reason: {result.reason}\n"
        "Predicted outcome: none - no feasible action was found."
    )
