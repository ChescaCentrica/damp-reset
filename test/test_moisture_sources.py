"""Tests for the moisture-source scheduling module.

The module is pure scheduling: it stores caller-supplied rates and
returns a total rate for a given time. No physics. Tests verify:
    - Data-class validation on both dataclasses.
    - Rate lookup: background alone, single event, overlapping events.
    - Half-open interval convention (start <= t < end).
    - Empty schedule returns the background rate.
"""

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture_sources import (
    MoistureSourceEvent,
    MoistureSourceSchedule,
    moisture_generation_rate_g_per_hour_at,
)


# --- MoistureSourceEvent ---------------------------------------------------


def test_event_accepts_reasonable_positive_values() -> None:
    """A plausible occupancy event constructs without raising."""
    event = MoistureSourceEvent(
        label="shower",
        start_time_hours=7.0,
        end_time_hours=7.25,
        generation_rate_g_per_hour=1500.0,
    )
    assert event.label == "shower"
    assert event.start_time_hours == 7.0
    assert event.end_time_hours == 7.25
    assert event.generation_rate_g_per_hour == 1500.0


def test_event_accepts_zero_rate() -> None:
    """A zero-rate event is unusual but not invalid."""
    MoistureSourceEvent(
        label="empty room",
        start_time_hours=0.0,
        end_time_hours=1.0,
        generation_rate_g_per_hour=0.0,
    )


def test_event_rejects_negative_rate() -> None:
    """Moisture cannot be removed by a source; negative rates rejected."""
    with pytest.raises(ValueError, match="generation_rate_g_per_hour"):
        MoistureSourceEvent(
            label="bad",
            start_time_hours=0.0,
            end_time_hours=1.0,
            generation_rate_g_per_hour=-1.0,
        )


def test_event_rejects_end_before_start() -> None:
    """Interval must have positive width."""
    with pytest.raises(ValueError, match="end_time_hours"):
        MoistureSourceEvent(
            label="reversed",
            start_time_hours=8.0,
            end_time_hours=7.0,
            generation_rate_g_per_hour=50.0,
        )


def test_event_rejects_end_equal_to_start() -> None:
    """Zero-width intervals are not allowed."""
    with pytest.raises(ValueError, match="end_time_hours"):
        MoistureSourceEvent(
            label="instant",
            start_time_hours=5.0,
            end_time_hours=5.0,
            generation_rate_g_per_hour=50.0,
        )


def test_event_rejects_negative_start_time() -> None:
    """Times run forward from the schedule's origin."""
    with pytest.raises(ValueError, match="start_time_hours"):
        MoistureSourceEvent(
            label="pre-history",
            start_time_hours=-1.0,
            end_time_hours=0.5,
            generation_rate_g_per_hour=50.0,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_event_rejects_non_finite_rate(bad_value: float) -> None:
    """NaN / inf on the rate is rejected."""
    with pytest.raises(ValueError, match="generation_rate_g_per_hour"):
        MoistureSourceEvent(
            label="bad",
            start_time_hours=0.0,
            end_time_hours=1.0,
            generation_rate_g_per_hour=bad_value,
        )


def test_event_is_frozen() -> None:
    """Events are immutable."""
    event = MoistureSourceEvent(
        label="cooking",
        start_time_hours=18.0,
        end_time_hours=19.0,
        generation_rate_g_per_hour=500.0,
    )
    with pytest.raises(FrozenInstanceError):
        event.generation_rate_g_per_hour = 100.0  # type: ignore[misc]


# --- MoistureSourceSchedule -----------------------------------------------


def test_schedule_defaults_to_no_background_and_no_events() -> None:
    """MoistureSourceSchedule() -> zero background, empty events tuple."""
    schedule = MoistureSourceSchedule()
    assert schedule.constant_background_rate_g_per_hour == 0.0
    assert schedule.events == ()


def test_schedule_rejects_negative_background_rate() -> None:
    """Background rate cannot be negative."""
    with pytest.raises(ValueError, match="constant_background_rate_g_per_hour"):
        MoistureSourceSchedule(constant_background_rate_g_per_hour=-1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_schedule_rejects_non_finite_background(bad_value: float) -> None:
    """NaN / inf background rejected."""
    with pytest.raises(ValueError, match="constant_background_rate_g_per_hour"):
        MoistureSourceSchedule(
            constant_background_rate_g_per_hour=bad_value
        )


# --- Rate lookup -----------------------------------------------------------


def test_empty_schedule_returns_zero_at_every_time() -> None:
    """A completely empty schedule has zero rate everywhere."""
    schedule = MoistureSourceSchedule()
    for t in (0.0, 5.0, 12.5, 24.0):
        assert moisture_generation_rate_g_per_hour_at(schedule, t) == 0.0


def test_background_rate_applies_at_every_time() -> None:
    """A constant background rate is returned regardless of query time."""
    schedule = MoistureSourceSchedule(
        constant_background_rate_g_per_hour=25.0
    )
    for t in (0.0, 5.0, 12.5, 24.0):
        assert (
            moisture_generation_rate_g_per_hour_at(schedule, t) == 25.0
        )


def test_single_event_rate_active_only_inside_interval() -> None:
    """Half-open interval: [start, end). Test each boundary explicitly."""
    schedule = MoistureSourceSchedule(
        events=(
            MoistureSourceEvent(
                label="cooking",
                start_time_hours=18.0,
                end_time_hours=19.0,
                generation_rate_g_per_hour=500.0,
            ),
        ),
    )
    # Before the event starts.
    assert moisture_generation_rate_g_per_hour_at(schedule, 17.5) == 0.0
    # At the start (inclusive).
    assert moisture_generation_rate_g_per_hour_at(schedule, 18.0) == 500.0
    # Inside the interval.
    assert moisture_generation_rate_g_per_hour_at(schedule, 18.5) == 500.0
    # Just before the end (still active).
    assert (
        moisture_generation_rate_g_per_hour_at(schedule, 18.9999) == 500.0
    )
    # At the end (exclusive - event no longer contributes).
    assert moisture_generation_rate_g_per_hour_at(schedule, 19.0) == 0.0
    # After the event.
    assert moisture_generation_rate_g_per_hour_at(schedule, 20.0) == 0.0


def test_overlapping_events_sum_additively() -> None:
    """Two events active at the same time contribute the sum of their rates."""
    schedule = MoistureSourceSchedule(
        events=(
            MoistureSourceEvent(
                label="cooking",
                start_time_hours=18.0,
                end_time_hours=19.0,
                generation_rate_g_per_hour=500.0,
            ),
            MoistureSourceEvent(
                label="drying laundry",
                start_time_hours=18.5,
                end_time_hours=22.0,
                generation_rate_g_per_hour=150.0,
            ),
        ),
    )
    # 18.0-18.5: only cooking active.
    assert moisture_generation_rate_g_per_hour_at(schedule, 18.25) == 500.0
    # 18.5-19.0: both active.
    assert moisture_generation_rate_g_per_hour_at(schedule, 18.75) == 650.0
    # 19.0-22.0: only laundry.
    assert moisture_generation_rate_g_per_hour_at(schedule, 20.0) == 150.0
    # After 22.0: neither.
    assert moisture_generation_rate_g_per_hour_at(schedule, 22.5) == 0.0


def test_background_plus_event_composes_additively() -> None:
    """The background rate is added to any active event's rate."""
    schedule = MoistureSourceSchedule(
        constant_background_rate_g_per_hour=25.0,
        events=(
            MoistureSourceEvent(
                label="shower",
                start_time_hours=7.0,
                end_time_hours=7.25,
                generation_rate_g_per_hour=1500.0,
            ),
        ),
    )
    # Outside the shower: background alone.
    assert moisture_generation_rate_g_per_hour_at(schedule, 6.0) == 25.0
    # Inside the shower: background + shower.
    assert (
        moisture_generation_rate_g_per_hour_at(schedule, 7.1) == 1525.0
    )
    # After the shower: background alone again.
    assert moisture_generation_rate_g_per_hour_at(schedule, 8.0) == 25.0


def test_rate_lookup_rejects_non_finite_time() -> None:
    """NaN / inf query times are rejected."""
    schedule = MoistureSourceSchedule()
    with pytest.raises(ValueError, match="time_hours"):
        moisture_generation_rate_g_per_hour_at(schedule, float("nan"))
    with pytest.raises(ValueError, match="time_hours"):
        moisture_generation_rate_g_per_hour_at(schedule, float("inf"))


def test_schedule_is_frozen() -> None:
    """Schedules are immutable."""
    schedule = MoistureSourceSchedule(
        constant_background_rate_g_per_hour=25.0
    )
    with pytest.raises(FrozenInstanceError):
        schedule.constant_background_rate_g_per_hour = 50.0  # type: ignore[misc]
