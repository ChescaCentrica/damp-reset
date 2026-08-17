"""24-hour synthetic time-domain simulation of a room.

Runs the time_simulation module over one day with:
    - a moisture-generating event (a morning shower);
    - a ventilation event (opening a window for 15 minutes after
      the shower);
    - a constant baseline background moisture generation
      representing two adults present in the room.

Produces three plots:
    outputs/plots/room_indoor_ah_vs_time.png
    outputs/plots/room_indoor_rh_vs_time.png
    outputs/plots/room_indoor_temperature_vs_time.png

Every ventilation and moisture rate used in this experiment is an
ILLUSTRATIVE POC value chosen so the trajectories are interesting
to look at. They are not calibrated to any real building or
occupant, and this experiment must not be quoted as evidence for
any specific residential moisture load.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from moisture import Room
from moisture_sources import (
    MoistureSourceEvent,
    MoistureSourceSchedule,
)
from psychrometrics import AirState
from thermal import ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K, ThermalProperties
from time_simulation import VentilationEvent, simulate_room_period

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "plots"
AH_PLOT_PATH = OUTPUT_DIR / "room_indoor_ah_vs_time.png"
RH_PLOT_PATH = OUTPUT_DIR / "room_indoor_rh_vs_time.png"
T_PLOT_PATH = OUTPUT_DIR / "room_indoor_temperature_vs_time.png"


def _shade_ventilation_events(ax, events, color="tab:red", alpha=0.15) -> None:
    """Overlay light shading on the axis wherever a window is open."""
    for event in events:
        ax.axvspan(
            event.start_time_hours,
            event.end_time_hours,
            color=color,
            alpha=alpha,
        )


def _shade_moisture_events(ax, events, color="tab:orange", alpha=0.15) -> None:
    """Overlay light shading on the axis for moisture-generating events."""
    for event in events:
        ax.axvspan(
            event.start_time_hours,
            event.end_time_hours,
            color=color,
            alpha=alpha,
        )


def main() -> None:
    """Run the 24-hour synthetic simulation and save the three plots."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=8.0, relative_humidity_percent=80.0)
    thermal_props = ThermalProperties(
        effective_thermal_capacity_j_per_k=(
            ILLUSTRATIVE_EFFECTIVE_THERMAL_CAPACITY_J_PER_K
        )
    )

    # Illustrative POC moisture schedule.
    moisture_events = (
        MoistureSourceEvent(
            label="shower",
            start_time_hours=7.0,
            end_time_hours=7.25,
            generation_rate_g_per_hour=1500.0,
        ),
    )
    moisture_schedule = MoistureSourceSchedule(
        # Two-adult baseline; illustrative only.
        constant_background_rate_g_per_hour=60.0,
        events=moisture_events,
    )

    # Illustrative POC ventilation schedule: crack the window open
    # for 15 min after the shower.
    ventilation_events = (
        VentilationEvent(start_time_hours=7.25, end_time_hours=7.5),
    )

    trajectory = simulate_room_period(
        room=room,
        thermal_properties=thermal_props,
        outdoor=outdoor,
        moisture_schedule=moisture_schedule,
        ventilation_events=ventilation_events,
        duration_hours=24.0,
        timestep_minutes=1.0,
    )

    print("Scenario")
    print("--------")
    print(
        f"indoor : T = {room.indoor_temperature_c:g} C, "
        f"RH = {room.indoor_relative_humidity_pct:g} %, "
        f"V = {room.volume_m3:g} m^3"
    )
    print(
        f"outdoor: T = {outdoor.temperature_c:g} C, "
        f"RH = {outdoor.relative_humidity_percent:g} % (constant)"
    )
    print(
        f"C_eff  : {thermal_props.effective_thermal_capacity_j_per_k:,.0f} "
        "J/K (illustrative)"
    )
    print(
        f"background moisture: "
        f"{moisture_schedule.constant_background_rate_g_per_hour:g} g/h "
        "(illustrative POC value)"
    )
    print(
        f"shower event: t = 7.0-7.25 h at "
        f"{moisture_events[0].generation_rate_g_per_hour:g} g/h "
        "(illustrative POC value)"
    )
    print(
        f"window open : t = 7.25-7.5 h at "
        f"{room.ach_window_open:g} ACH (illustrative)"
    )
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: indoor AH vs time ---
    fig, ax = plt.subplots()
    ax.plot(
        trajectory.times_hours,
        trajectory.indoor_absolute_humidity_g_m3,
        color="tab:blue",
        label="indoor AH",
    )
    ax.axhline(
        outdoor.absolute_humidity,
        color="tab:gray",
        linestyle="--",
        label=f"outdoor AH = {outdoor.absolute_humidity:.2f} g/m^3",
    )
    _shade_moisture_events(ax, moisture_events)
    _shade_ventilation_events(ax, ventilation_events)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("indoor absolute humidity (g/m^3)")
    ax.set_title(
        "Indoor absolute humidity over 24 hours\n"
        "orange band = moisture event; red band = window open"
    )
    ax.legend()
    fig.savefig(AH_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {AH_PLOT_PATH}")

    # --- Plot 2: indoor RH vs time ---
    fig, ax = plt.subplots()
    ax.plot(
        trajectory.times_hours,
        trajectory.indoor_relative_humidity_pct,
        color="tab:blue",
        label="indoor RH",
    )
    _shade_moisture_events(ax, moisture_events)
    _shade_ventilation_events(ax, ventilation_events)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("indoor relative humidity (%)")
    ax.set_title(
        "Indoor relative humidity over 24 hours\n"
        "orange band = moisture event; red band = window open"
    )
    ax.legend()
    fig.savefig(RH_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {RH_PLOT_PATH}")

    # --- Plot 3: indoor temperature vs time ---
    fig, ax = plt.subplots()
    ax.plot(
        trajectory.times_hours,
        trajectory.indoor_temperature_c,
        color="tab:blue",
        label="indoor T",
    )
    ax.axhline(
        outdoor.temperature_c,
        color="tab:gray",
        linestyle="--",
        label=f"outdoor T = {outdoor.temperature_c:g} C",
    )
    _shade_ventilation_events(ax, ventilation_events)
    ax.set_xlabel("time (hours)")
    ax.set_ylabel("indoor temperature (C)")
    ax.set_title(
        "Indoor temperature over 24 hours\n"
        "red band = window open"
    )
    ax.legend()
    fig.savefig(T_PLOT_PATH)
    plt.close(fig)
    print(f"saved plot to {T_PLOT_PATH}")

    # --- Text summary at key moments ---
    print()
    print("Indoor state at key moments")
    print("---------------------------")
    print(
        f"  {'t (h)':>6}  {'indoor T':>10}  "
        f"{'indoor AH':>10}  {'indoor RH':>10}  {'window':>7}"
    )
    print(
        f"  {'':>6}  {'(C)':>10}  {'(g/m^3)':>10}  {'(%)':>10}  {'':>7}"
    )
    print("  " + "-" * 52)
    for time_query_hours in (0.0, 6.0, 7.0, 7.25, 7.5, 12.0, 18.0, 24.0):
        idx = min(
            range(len(trajectory.times_hours)),
            key=lambda i: abs(
                trajectory.times_hours[i] - time_query_hours
            ),
        )
        window_str = "open" if trajectory.window_open[idx] else "closed"
        print(
            f"  {trajectory.times_hours[idx]:>6.2f}  "
            f"{trajectory.indoor_temperature_c[idx]:>10.3f}  "
            f"{trajectory.indoor_absolute_humidity_g_m3[idx]:>10.3f}  "
            f"{trajectory.indoor_relative_humidity_pct[idx]:>10.2f}  "
            f"{window_str:>7}"
        )


if __name__ == "__main__":
    main()
