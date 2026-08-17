"""Walk-through of the psychrometrics module on one representative case.

Indoor: 20 degC, 70 %RH. Outdoor: 5 degC, 85 %RH. Even though outdoor RH is
higher, cool air simply cannot hold as much water in absolute terms, so
ventilating with outdoor air removes moisture. The narrative sentence at the
end is generated from the *computed* values so the example stays honest if
the underlying equations or thresholds are ever changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psychrometrics import AirState, DryingPotential, calculate_drying_potential


def print_state(label: str, state: AirState) -> None:
    """Print the moisture properties of one air state on a single line."""
    print(
        f"{label:<9} T={state.temperature_c:5.1f} C  "
        f"RH={state.relative_humidity_percent:5.1f} %  "
        f"AH={state.absolute_humidity:5.2f} g/m^3  "
        f"W={state.humidity_ratio * 1000:5.2f} g/kg  "
        f"Td={state.dew_point:5.1f} C"
    )


def summary_sentence(
    indoor: AirState, outdoor: AirState, result: DryingPotential
) -> str:
    """Compose a plain-English summary from the computed numbers."""
    diff = result.difference_g_m3
    if diff > 0:
        direction = "less"
        outcome = (
            "so ventilating with outdoor air would remove moisture from the interior"
        )
    elif diff < 0:
        direction = "more"
        outcome = (
            "so ventilating with outdoor air would add moisture to the interior"
        )
    else:
        direction = "the same amount of"
        outcome = "so ventilating with outdoor air would neither add nor remove moisture"
    rh_note = (
        " even though outdoor relative humidity is higher"
        if outdoor.relative_humidity_percent > indoor.relative_humidity_percent
        and diff > 0
        else ""
    )
    return (
        f"At {indoor.temperature_c:.0f} C / {indoor.relative_humidity_percent:.0f} %RH "
        f"indoor air carries {indoor.absolute_humidity:.2f} g of water per cubic metre, "
        f"while at {outdoor.temperature_c:.0f} C / {outdoor.relative_humidity_percent:.0f} %RH "
        f"outdoor air carries {outdoor.absolute_humidity:.2f} g/m^3 - "
        f"that is {abs(diff):.2f} g/m^3 {direction} water outside{rh_note}, "
        f"{outcome} (POC category: {result.category})."
    )


def main() -> None:
    """Run the single-scenario walk-through."""
    indoor = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

    print("Air states")
    print("----------")
    print_state("indoor", indoor)
    print_state("outdoor", outdoor)
    print()

    result = calculate_drying_potential(indoor, outdoor)
    print("Drying potential")
    print("----------------")
    print(
        f"indoor AH  = {result.indoor_absolute_humidity_g_m3:5.2f} g/m^3\n"
        f"outdoor AH = {result.outdoor_absolute_humidity_g_m3:5.2f} g/m^3\n"
        f"difference = {result.difference_g_m3:+.2f} g/m^3\n"
        f"category   = {result.category}"
    )
    print()

    print("Summary")
    print("-------")
    print(summary_sentence(indoor, outdoor, result))


if __name__ == "__main__":
    main()
