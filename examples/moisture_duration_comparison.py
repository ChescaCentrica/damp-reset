"""Compare ventilation moisture predictions across several window-open durations.

Fixes an indoor and outdoor air state plus a room definition, then asks the
moisture model what happens after 0, 2, 5, 10, and 15 minutes with the
window open. Prints the results in a single table so the trajectory of a
first-order-decay ventilation event is easy to eyeball.

Final indoor RH is intentionally NOT reported. The thermal model has not
been implemented yet, so indoor temperature will change during a
ventilation event; deriving an RH from the final absolute humidity would
require a temperature that this project does not yet predict. Once the
thermal model exists, an RH column can be added by feeding the final
temperature + final AH back through the psychrometric layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moisture import Room, predict_room_moisture
from psychrometrics import AirState

DURATIONS_MINUTES = (0.0, 2.0, 5.0, 10.0, 15.0)


def main() -> None:
    """Run the sweep and print the comparison table."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

    print("Scenario")
    print("--------")
    print(
        f"indoor  : T={room.indoor_temperature_c:g} C, "
        f"RH={room.indoor_relative_humidity_pct:g} %, V={room.volume_m3:g} m^3"
    )
    print(
        f"outdoor : T={outdoor.temperature_c:g} C, "
        f"RH={outdoor.relative_humidity_percent:g} %"
    )
    print(
        f"vent    : window open, ACH={room.ach_window_open:g} h^-1 "
        f"(closed baseline {room.ach_closed:g} h^-1)"
    )
    print()

    header = (
        f"{'duration':>10}  "
        f"{'initial AH':>11}  "
        f"{'final AH':>10}  "
        f"{'reduction':>11}  "
        f"{'% reduced':>10}  "
        f"{'water removed':>14}"
    )
    units = (
        f"{'(min)':>10}  "
        f"{'(g/m^3)':>11}  "
        f"{'(g/m^3)':>10}  "
        f"{'(g/m^3)':>11}  "
        f"{'(%)':>10}  "
        f"{'(g)':>14}"
    )
    ruler = "-" * len(header)

    print(header)
    print(units)
    print(ruler)
    for minutes in DURATIONS_MINUTES:
        r = predict_room_moisture(room, outdoor, duration_minutes=minutes)
        print(
            f"{minutes:>10.1f}  "
            f"{r.initial_absolute_humidity_g_m3:>11.3f}  "
            f"{r.final_absolute_humidity_g_m3:>10.3f}  "
            f"{r.absolute_humidity_reduction_g_m3:>+11.3f}  "
            f"{r.percentage_reduction:>+10.2f}  "
            f"{r.water_removed_g:>+14.2f}"
        )
    print()
    print(
        "Note: final indoor RH is not shown - the thermal model is not\n"
        "implemented, so indoor temperature during ventilation is unknown\n"
        "and any RH derived from the final AH would be misleading."
    )


if __name__ == "__main__":
    main()
