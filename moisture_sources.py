"""Moisture-generation source model for the room air.

This module owns NO physics. It stores caller-supplied moisture
generation rates and answers "what is the total moisture-generation
rate right now?" for a given time in a schedule. The physical
extension of the moisture ODE to include a source term is documented
in ``moisture.predict_final_absolute_humidity_with_source``; this
module supplies the ``G`` (in g/hour) that function consumes.

Two shapes of source are supported:

    1. A constant background rate that applies over the whole
       simulation horizon.
    2. A schedule of discrete moisture events, each with a start
       time, end time, and constant rate over that interval.

Both compose additively: at any query time, the effective rate is
the sum of the constant background rate and every scheduled event
whose interval contains the query time.

Caller responsibility: this module does NOT ship authoritative
moisture-generation values. Every rate a caller supplies is a POC
setting that must be defended with an evidence base (measured room
data, tabulated activity rates from ASHRAE Handbook of Fundamentals /
CIBSE Guide A, or a room-specific study) before any deployment. See
the ``MoistureSourceEvent`` docstring for the specific warning.

Explicitly NOT in this module:
    - Tabulated moisture-generation values for specific activities.
    - Occupancy schedules (someone else supplies the intervals).
    - Any physics equation.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Tuple


@dataclass(frozen=True)
class MoistureSourceEvent:
    """A discrete moisture-generation event with a constant rate.

    Represents "a person is showering from t = 7.0 to t = 7.25
    hours, generating water at some rate". The rate applied is
    piecewise-constant across the interval, and the caller supplies
    it.

    Fields:
        label: caller-set name for logging / audit ("shower",
            "cooking dinner", "adult sleeping"). Free-form.
        start_time_hours: interval start, in hours since the
            schedule's origin. Non-negative.
        end_time_hours: interval end, strictly greater than
            start_time_hours.
        generation_rate_g_per_hour: constant moisture-generation
            rate over the interval, in grams of water per hour.
            Non-negative.

    IMPORTANT: this dataclass does NOT define what "cooking",
    "showering", or "adult sleeping" mean in numeric terms. Every
    rate the caller sets is a POC assumption that must be defended
    on evidence (ASHRAE / CIBSE tabulations, measurements from the
    specific room, or occupancy-study literature) before any real
    deployment. Rates vary strongly with kitchen ventilation, shower
    duration, laundry load, and occupant metabolic rate.
    """

    label: str
    start_time_hours: float
    end_time_hours: float
    generation_rate_g_per_hour: float

    def __post_init__(self) -> None:
        """Validate every numeric field."""
        for field_name, value in (
            ("start_time_hours", self.start_time_hours),
            ("end_time_hours", self.end_time_hours),
            (
                "generation_rate_g_per_hour",
                self.generation_rate_g_per_hour,
            ),
        ):
            if not isfinite(value):
                raise ValueError(
                    f"{field_name} must be finite, got {value!r}"
                )
            if value < 0.0:
                raise ValueError(
                    f"{field_name} must be non-negative, got {value}"
                )
        if self.end_time_hours <= self.start_time_hours:
            raise ValueError(
                f"end_time_hours ({self.end_time_hours}) must be strictly "
                f"greater than start_time_hours ({self.start_time_hours})"
            )


@dataclass(frozen=True)
class MoistureSourceSchedule:
    """A collection of moisture sources composed additively.

    Fields:
        constant_background_rate_g_per_hour: a background rate
            applied at every time in the schedule (e.g. baseline
            occupancy, indoor plants, aquarium). Non-negative.
            Defaults to 0.
        events: tuple of discrete ``MoistureSourceEvent`` values.
            Order is not significant; events may overlap freely,
            and overlapping events sum. Defaults to empty.

    Rate lookup: at a given time t, the effective rate is
    ``constant_background_rate_g_per_hour + sum of every event's
    rate where start <= t < end``. Note the half-open interval
    convention: an event ending at t contributes zero at t.
    """

    constant_background_rate_g_per_hour: float = 0.0
    events: Tuple[MoistureSourceEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the background rate."""
        if not isfinite(self.constant_background_rate_g_per_hour):
            raise ValueError(
                "constant_background_rate_g_per_hour must be finite, got "
                f"{self.constant_background_rate_g_per_hour!r}"
            )
        if self.constant_background_rate_g_per_hour < 0.0:
            raise ValueError(
                "constant_background_rate_g_per_hour must be non-negative, "
                f"got {self.constant_background_rate_g_per_hour}"
            )


def moisture_generation_rate_g_per_hour_at(
    schedule: MoistureSourceSchedule,
    time_hours: float,
) -> float:
    """Total moisture-generation rate at a given time.

    Sums the constant background rate with every scheduled event
    whose interval contains ``time_hours`` using the half-open
    ``start <= t < end`` convention.

    Args:
        schedule: the source schedule to query.
        time_hours: query time, in hours since the schedule's origin.

    Returns:
        The effective moisture-generation rate at that time, in
        grams per hour. Never negative.

    Raises:
        ValueError: if ``time_hours`` is NaN or infinite.
    """
    if not isfinite(time_hours):
        raise ValueError(f"time_hours must be finite, got {time_hours!r}")
    rate = schedule.constant_background_rate_g_per_hour
    for event in schedule.events:
        if event.start_time_hours <= time_hours < event.end_time_hours:
            rate += event.generation_rate_g_per_hour
    return rate
