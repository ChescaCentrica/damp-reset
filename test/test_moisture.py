"""Unit tests for the ventilation moisture-model Room dataclass.

Covers valid constructions, every documented validation rule, and the
immutability / equality contract inherited from ``@dataclass(frozen=True)``.
"""

import sys
from dataclasses import FrozenInstanceError
from math import exp, isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from moisture import (
    MoisturePrediction,
    Room,
    predict_final_absolute_humidity,
    predict_final_absolute_humidity_with_source,
    predict_moisture,
    predict_room_moisture,
)
from psychrometrics import AirState


# --- valid constructions ---------------------------------------------------


def test_room_accepts_typical_residential_parameters() -> None:
    """A plausible residential Room constructs without raising."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=60.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    assert room.volume_m3 == 40.0
    assert room.ach_window_open == 5.0


def test_room_accepts_sealed_envelope_with_zero_closed_ach() -> None:
    """ach_closed = 0 is allowed as a limit case (perfectly sealed room)."""
    Room(
        volume_m3=30.0,
        indoor_temperature_c=22.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.0,
        ach_window_open=4.0,
    )


def test_room_accepts_zero_open_ach() -> None:
    """ach_window_open = 0 is allowed (window that provides no exchange)."""
    Room(
        volume_m3=30.0,
        indoor_temperature_c=22.0,
        indoor_relative_humidity_pct=55.0,
        ach_closed=0.3,
        ach_window_open=0.0,
    )


def test_room_accepts_rh_endpoints() -> None:
    """RH exactly 0 and exactly 100 % must both be accepted."""
    Room(
        volume_m3=30.0,
        indoor_temperature_c=22.0,
        indoor_relative_humidity_pct=0.0,
        ach_closed=0.3,
        ach_window_open=4.0,
    )
    Room(
        volume_m3=30.0,
        indoor_temperature_c=22.0,
        indoor_relative_humidity_pct=100.0,
        ach_closed=0.3,
        ach_window_open=4.0,
    )


def test_room_accepts_negative_indoor_temperature() -> None:
    """Indoor temperature can be negative (e.g. unheated space in winter).

    The Room dataclass itself does not range-check temperature; downstream
    psychrometric calls apply the residential range at their own boundary.
    """
    Room(
        volume_m3=30.0,
        indoor_temperature_c=-5.0,
        indoor_relative_humidity_pct=60.0,
        ach_closed=0.3,
        ach_window_open=4.0,
    )


# --- invalid constructions -------------------------------------------------


def test_room_rejects_zero_volume() -> None:
    """volume_m3 must be strictly positive."""
    with pytest.raises(ValueError, match="volume_m3"):
        Room(
            volume_m3=0.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=60.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


def test_room_rejects_negative_volume() -> None:
    """volume_m3 must be strictly positive."""
    with pytest.raises(ValueError, match="volume_m3"):
        Room(
            volume_m3=-10.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=60.0,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


def test_room_rejects_rh_below_zero() -> None:
    """Relative humidity must be at or above 0 %."""
    with pytest.raises(ValueError, match="indoor_relative_humidity_pct"):
        Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=-0.1,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


def test_room_rejects_rh_above_100() -> None:
    """Relative humidity must be at or below 100 %."""
    with pytest.raises(ValueError, match="indoor_relative_humidity_pct"):
        Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=100.01,
            ach_closed=0.5,
            ach_window_open=5.0,
        )


def test_room_rejects_negative_ach_closed() -> None:
    """ach_closed must be non-negative."""
    with pytest.raises(ValueError, match="ach_closed"):
        Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=60.0,
            ach_closed=-0.1,
            ach_window_open=5.0,
        )


def test_room_rejects_negative_ach_window_open() -> None:
    """ach_window_open must be non-negative."""
    with pytest.raises(ValueError, match="ach_window_open"):
        Room(
            volume_m3=40.0,
            indoor_temperature_c=20.0,
            indoor_relative_humidity_pct=60.0,
            ach_closed=0.5,
            ach_window_open=-1.0,
        )


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("volume_m3", float("nan")),
        ("volume_m3", float("inf")),
        ("indoor_temperature_c", float("nan")),
        ("indoor_temperature_c", float("inf")),
        ("indoor_relative_humidity_pct", float("nan")),
        ("indoor_relative_humidity_pct", float("inf")),
        ("ach_closed", float("nan")),
        ("ach_closed", float("inf")),
        ("ach_window_open", float("nan")),
        ("ach_window_open", float("inf")),
    ],
)
def test_room_rejects_non_finite_fields(field_name: str, bad_value: float) -> None:
    """NaN and infinite values are rejected on every field."""
    kwargs = {
        "volume_m3": 40.0,
        "indoor_temperature_c": 20.0,
        "indoor_relative_humidity_pct": 60.0,
        "ach_closed": 0.5,
        "ach_window_open": 5.0,
    }
    kwargs[field_name] = bad_value
    with pytest.raises(ValueError, match=field_name):
        Room(**kwargs)


# --- immutability and equality ---------------------------------------------


def test_room_is_frozen() -> None:
    """Room is a frozen dataclass; field assignment must raise."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=60.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    with pytest.raises(FrozenInstanceError):
        room.volume_m3 = 50.0  # type: ignore[misc]


def test_room_equality_is_by_value() -> None:
    """Two Rooms with identical fields compare equal."""
    a = Room(40.0, 20.0, 60.0, 0.5, 5.0)
    b = Room(40.0, 20.0, 60.0, 0.5, 5.0)
    c = Room(41.0, 20.0, 60.0, 0.5, 5.0)
    assert a == b
    assert a != c


# --- predict_final_absolute_humidity ---------------------------------------
# The analytic solution is C(t) = C_out + (C_0 - C_out) * exp(-n * t).
# Tests cover degenerate cases (t=0, n=0), the three sign scenarios for
# (C_0 - C_out), the long-duration asymptote, and one physically-anchored
# time-constant check that would fail if the sign of the exponent or the
# minutes->hours conversion were wrong.


def test_zero_duration_returns_initial_indoor_humidity() -> None:
    """At t = 0 the room has not yet changed."""
    assert (
        predict_final_absolute_humidity(
            indoor_ah_g_m3=12.0,
            outdoor_ah_g_m3=5.0,
            ach=6.0,
            duration_minutes=0.0,
        )
        == 12.0
    )


def test_zero_ach_returns_initial_indoor_humidity() -> None:
    """With no air exchange the indoor humidity is invariant."""
    assert (
        predict_final_absolute_humidity(
            indoor_ah_g_m3=12.0,
            outdoor_ah_g_m3=5.0,
            ach=0.0,
            duration_minutes=45.0,
        )
        == 12.0
    )


def test_indoor_drier_than_outdoor_gains_moisture() -> None:
    """C_0 < C_out: ventilating adds water; result is between C_0 and C_out."""
    c0, c_out = 5.0, 12.0
    result = predict_final_absolute_humidity(
        indoor_ah_g_m3=c0,
        outdoor_ah_g_m3=c_out,
        ach=4.0,
        duration_minutes=15.0,
    )
    assert c0 < result < c_out


def test_indoor_wetter_than_outdoor_loses_moisture() -> None:
    """C_0 > C_out: ventilating removes water; result is between C_out and C_0."""
    c0, c_out = 12.0, 5.0
    result = predict_final_absolute_humidity(
        indoor_ah_g_m3=c0,
        outdoor_ah_g_m3=c_out,
        ach=4.0,
        duration_minutes=15.0,
    )
    assert c_out < result < c0


def test_indoor_equals_outdoor_stays_constant() -> None:
    """C_0 == C_out: no driving force -> the room stays there for all t and n."""
    for ach in (0.0, 1.0, 6.0):
        for minutes in (0.0, 5.0, 60.0):
            assert (
                predict_final_absolute_humidity(
                    indoor_ah_g_m3=8.0,
                    outdoor_ah_g_m3=8.0,
                    ach=ach,
                    duration_minutes=minutes,
                )
                == 8.0
            )


def test_long_duration_asymptotes_to_outdoor_humidity() -> None:
    """As t grows, the exponential decays and C approaches C_out."""
    result = predict_final_absolute_humidity(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=6.0,
        duration_minutes=600.0,  # 10 hours: n*t = 60 -> exp term ~= 8e-27
    )
    assert result == pytest.approx(5.0, abs=1e-12)


def test_one_time_constant_closes_gap_by_factor_of_e() -> None:
    """After t = 1/n hours the gap C - C_out shrinks by exactly 1/e.

    Independent physical check: the standard first-order-decay time constant
    is tau = 1/n. This anchors the exponent's sign AND the minutes-to-hours
    conversion in one test - if either were wrong (say exp(+n*t), or if we
    forgot to divide by 60), the gap would not equal (C0 - C_out)/e.
    """
    c0, c_out, ach = 12.0, 5.0, 4.0  # tau = 1/4 hour = 15 minutes
    duration_minutes = 15.0
    result = predict_final_absolute_humidity(c0, c_out, ach, duration_minutes)
    expected_gap = (c0 - c_out) / exp(1.0)
    assert (result - c_out) == pytest.approx(expected_gap, rel=1e-12)


def test_five_time_constants_reaches_within_one_percent_of_outdoor() -> None:
    """After 5*tau, exp(-5) ~= 0.0067 -> less than 1 % of the original gap remains."""
    c0, c_out, ach = 12.0, 5.0, 4.0  # tau = 15 minutes; 5*tau = 75 minutes
    result = predict_final_absolute_humidity(c0, c_out, ach, 5.0 * 15.0)
    remaining_fraction = (result - c_out) / (c0 - c_out)
    assert 0.0 < remaining_fraction < 0.01


def test_result_never_overshoots_outdoor() -> None:
    """The analytic solution is monotone toward C_out; it must not cross it.

    Property test across a small grid of (n, t) values in both directions
    of the gradient.
    """
    for c0, c_out in ((12.0, 5.0), (5.0, 12.0), (0.0, 15.0), (15.0, 0.0)):
        for ach in (0.5, 2.0, 10.0):
            for minutes in (1.0, 15.0, 60.0, 300.0):
                result = predict_final_absolute_humidity(c0, c_out, ach, minutes)
                assert min(c0, c_out) <= result <= max(c0, c_out)


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("indoor_ah_g_m3", -0.1),
        ("outdoor_ah_g_m3", -0.1),
        ("ach", -0.1),
        ("duration_minutes", -0.1),
    ],
)
def test_predict_rejects_negative_arguments(arg_name: str, bad_value: float) -> None:
    """Negative absolute humidity, ACH, or duration is unphysical -> rejected."""
    kwargs = dict(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        predict_final_absolute_humidity(**kwargs)


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("indoor_ah_g_m3", float("nan")),
        ("indoor_ah_g_m3", float("inf")),
        ("outdoor_ah_g_m3", float("nan")),
        ("outdoor_ah_g_m3", float("inf")),
        ("ach", float("nan")),
        ("ach", float("inf")),
        ("duration_minutes", float("nan")),
        ("duration_minutes", float("inf")),
    ],
)
def test_predict_rejects_non_finite_arguments(arg_name: str, bad_value: float) -> None:
    """NaN / inf on any argument is rejected."""
    kwargs = dict(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        predict_final_absolute_humidity(**kwargs)


# --- predict_moisture / MoisturePrediction ---------------------------------


# Every predict_moisture test below uses the same room volume, both so the
# expected numbers stay readable and so a water_removed_g regression is easy
# to hand-check: water_removed_g = reduction * ROOM_VOL_M3.
ROOM_VOL_M3 = 40.0


def test_predict_moisture_final_ah_matches_scalar_function() -> None:
    """The wrapper must not reimplement the physics - it delegates."""
    scalar_kwargs = dict(
        indoor_ah_g_m3=12.0, outdoor_ah_g_m3=5.0, ach=4.0, duration_minutes=15.0
    )
    result = predict_moisture(**scalar_kwargs, room_volume_m3=ROOM_VOL_M3)
    assert result.final_absolute_humidity_g_m3 == predict_final_absolute_humidity(
        **scalar_kwargs
    )


def test_predict_moisture_bundles_all_input_state_into_result() -> None:
    """Inputs are echoed back on the result for audit."""
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert isinstance(result, MoisturePrediction)
    assert result.initial_absolute_humidity_g_m3 == 12.0
    assert result.outdoor_absolute_humidity_g_m3 == 5.0
    assert result.ach == 4.0
    assert result.duration_minutes == 15.0
    assert result.room_volume_m3 == ROOM_VOL_M3


def test_predict_moisture_drying_ventilation_reports_positive_reduction() -> None:
    """Indoor wetter than outdoor -> ventilation dries the room."""
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 < result.initial_absolute_humidity_g_m3
    assert result.absolute_humidity_change_g_m3 < 0.0
    assert result.absolute_humidity_reduction_g_m3 > 0.0
    assert result.percentage_reduction > 0.0
    assert result.water_removed_g > 0.0


def test_predict_moisture_wetting_ventilation_reports_negative_reduction() -> None:
    """Indoor drier than outdoor -> ventilation adds moisture."""
    result = predict_moisture(
        indoor_ah_g_m3=5.0,
        outdoor_ah_g_m3=12.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 > result.initial_absolute_humidity_g_m3
    assert result.absolute_humidity_change_g_m3 > 0.0
    assert result.absolute_humidity_reduction_g_m3 < 0.0
    assert result.percentage_reduction < 0.0
    assert result.water_removed_g < 0.0


def test_predict_moisture_no_gradient_reports_zero_everywhere() -> None:
    """Indoor == outdoor -> zero change and zero reduction, all durations / ACH."""
    result = predict_moisture(
        indoor_ah_g_m3=8.0,
        outdoor_ah_g_m3=8.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 == 8.0
    assert result.absolute_humidity_change_g_m3 == 0.0
    assert result.absolute_humidity_reduction_g_m3 == 0.0
    assert result.percentage_reduction == 0.0
    assert result.water_removed_g == 0.0


def test_predict_moisture_zero_duration_reports_no_change() -> None:
    """Duration = 0 -> final equals initial, everything else zero."""
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=0.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 == 12.0
    assert result.absolute_humidity_change_g_m3 == 0.0
    assert result.absolute_humidity_reduction_g_m3 == 0.0
    assert result.percentage_reduction == 0.0
    assert result.water_removed_g == 0.0


def test_predict_moisture_zero_ach_reports_no_change() -> None:
    """ACH = 0 -> no exchange -> final equals initial regardless of duration."""
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=0.0,
        duration_minutes=60.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 == 12.0
    assert result.absolute_humidity_change_g_m3 == 0.0
    assert result.absolute_humidity_reduction_g_m3 == 0.0
    assert result.percentage_reduction == 0.0
    assert result.water_removed_g == 0.0


def test_predict_moisture_change_and_reduction_are_negatives_of_each_other() -> None:
    """By definition, change = final - initial and reduction = initial - final."""
    for indoor, outdoor in ((12.0, 5.0), (5.0, 12.0), (8.0, 8.0), (0.5, 0.0)):
        result = predict_moisture(
            indoor, outdoor, ach=4.0, duration_minutes=15.0, room_volume_m3=ROOM_VOL_M3
        )
        assert result.absolute_humidity_change_g_m3 == pytest.approx(
            -result.absolute_humidity_reduction_g_m3, rel=1e-12, abs=1e-15
        )


def test_predict_moisture_percentage_reduction_uses_initial_ah_as_denominator() -> None:
    """percentage_reduction = 100 * reduction / initial (documented denominator).

    Uses an independently hand-computed expected value: at one time
    constant (n*t = 1) the gap closes by 1/e, so final = 5 + 7/e and
    reduction = 12 - (5 + 7/e). The percentage against a 12 g/m^3
    initial value is therefore 100 * (7 - 7/e) / 12.
    """
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    expected_pct = 100.0 * (7.0 - 7.0 / exp(1.0)) / 12.0
    assert result.percentage_reduction == pytest.approx(expected_pct, rel=1e-12)


def test_predict_moisture_percentage_reduction_hand_computed_at_one_time_constant() -> None:
    """After one time constant, gap shrinks by 1/e; hand-check the percentage.

    initial = 12, outdoor = 5, ach = 4, t = 15 min = 0.25 h -> n*t = 1
    final = 5 + (12-5)/e = 5 + 2.5752... = 7.5752...
    reduction = 12 - 7.5752 = 4.4248...
    percentage = 100 * 4.4248 / 12 = 36.874 %
    """
    result = predict_moisture(
        12.0, 5.0, ach=4.0, duration_minutes=15.0, room_volume_m3=ROOM_VOL_M3
    )
    expected_final = 5.0 + 7.0 / exp(1.0)
    expected_reduction = 12.0 - expected_final
    expected_pct = 100.0 * expected_reduction / 12.0
    assert result.final_absolute_humidity_g_m3 == pytest.approx(expected_final, rel=1e-12)
    assert result.percentage_reduction == pytest.approx(expected_pct, rel=1e-12)


def test_predict_moisture_percentage_reduction_is_nan_when_initial_is_zero() -> None:
    """Percentage is undefined with initial = 0; the raw g/m^3 fields still hold."""
    result = predict_moisture(
        indoor_ah_g_m3=0.0,
        outdoor_ah_g_m3=6.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.absolute_humidity_change_g_m3 > 0.0
    assert result.absolute_humidity_reduction_g_m3 < 0.0
    assert isnan(result.percentage_reduction)


def test_predict_moisture_full_equilibration_to_zero_outdoor_is_100pct_reduction() -> None:
    """Drying the room to outdoor = 0 corresponds to 100 % reduction."""
    result = predict_moisture(
        indoor_ah_g_m3=10.0,
        outdoor_ah_g_m3=0.0,
        ach=6.0,
        duration_minutes=600.0,  # 10 h; exp(-60) ~= 8e-27
        room_volume_m3=ROOM_VOL_M3,
    )
    assert result.final_absolute_humidity_g_m3 == pytest.approx(0.0, abs=1e-12)
    assert result.percentage_reduction == pytest.approx(100.0, abs=1e-10)


def test_moisture_prediction_is_frozen() -> None:
    """MoisturePrediction is a frozen dataclass."""
    result = predict_moisture(
        12.0, 5.0, 4.0, 15.0, room_volume_m3=ROOM_VOL_M3
    )
    with pytest.raises(FrozenInstanceError):
        result.final_absolute_humidity_g_m3 = 0.0  # type: ignore[misc]


# --- water_removed_g -------------------------------------------------------


def test_water_removed_g_equals_reduction_times_volume() -> None:
    """Definition check: water_removed_g = reduction_g_m3 * room_volume_m3."""
    for indoor, outdoor, volume in (
        (12.0, 5.0, 40.0),
        (5.0, 12.0, 40.0),
        (10.0, 3.0, 100.0),
        (8.0, 8.0, 25.0),
    ):
        result = predict_moisture(
            indoor_ah_g_m3=indoor,
            outdoor_ah_g_m3=outdoor,
            ach=4.0,
            duration_minutes=15.0,
            room_volume_m3=volume,
        )
        assert result.water_removed_g == pytest.approx(
            result.absolute_humidity_reduction_g_m3 * volume, rel=1e-12, abs=1e-15
        )


def test_water_removed_g_positive_when_drying() -> None:
    """Drying -> positive mass, hand-computed at one time constant."""
    volume = 40.0
    # initial=12, outdoor=5, ach=4, 15 min: gap shrinks by 1/e as verified elsewhere.
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=volume,
    )
    expected_reduction = 7.0 - 7.0 / exp(1.0)  # (initial - final) with final = 5 + 7/e
    expected_water_g = expected_reduction * volume
    assert result.water_removed_g == pytest.approx(expected_water_g, rel=1e-12)
    assert result.water_removed_g > 0.0


def test_water_removed_g_negative_when_wetting() -> None:
    """Wet outdoor air -> negative mass. Sign preserved deliberately."""
    result = predict_moisture(
        indoor_ah_g_m3=5.0,
        outdoor_ah_g_m3=12.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=40.0,
    )
    assert result.water_removed_g < 0.0


def test_water_removed_g_zero_when_no_gradient() -> None:
    """No AH gradient -> zero mass exchanged in either direction."""
    result = predict_moisture(
        indoor_ah_g_m3=8.0,
        outdoor_ah_g_m3=8.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=40.0,
    )
    assert result.water_removed_g == 0.0


def test_water_removed_g_scales_linearly_with_volume() -> None:
    """Doubling the room volume doubles the water mass at fixed AH change."""
    kwargs = dict(indoor_ah_g_m3=12.0, outdoor_ah_g_m3=5.0, ach=4.0, duration_minutes=15.0)
    small = predict_moisture(**kwargs, room_volume_m3=25.0)
    large = predict_moisture(**kwargs, room_volume_m3=50.0)
    # AH reduction is volume-independent, so mass ratio = volume ratio.
    assert large.water_removed_g == pytest.approx(2.0 * small.water_removed_g, rel=1e-12)


def test_predict_moisture_rejects_zero_or_negative_volume() -> None:
    """Volume must be strictly positive; 0 or negative rejected."""
    kwargs = dict(indoor_ah_g_m3=12.0, outdoor_ah_g_m3=5.0, ach=4.0, duration_minutes=15.0)
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_moisture(**kwargs, room_volume_m3=0.0)
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_moisture(**kwargs, room_volume_m3=-1.0)


def test_predict_moisture_rejects_non_finite_volume() -> None:
    """NaN / inf volume rejected."""
    kwargs = dict(indoor_ah_g_m3=12.0, outdoor_ah_g_m3=5.0, ach=4.0, duration_minutes=15.0)
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_moisture(**kwargs, room_volume_m3=float("nan"))
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_moisture(**kwargs, room_volume_m3=float("inf"))


@pytest.mark.parametrize(
    "arg_name,bad_value",
    [
        ("indoor_ah_g_m3", -0.1),
        ("outdoor_ah_g_m3", -0.1),
        ("ach", -0.1),
        ("duration_minutes", -0.1),
        ("indoor_ah_g_m3", float("nan")),
        ("outdoor_ah_g_m3", float("inf")),
        ("ach", float("nan")),
        ("duration_minutes", float("inf")),
    ],
)
def test_predict_moisture_propagates_scalar_validation(
    arg_name: str, bad_value: float
) -> None:
    """Invalid inputs bubble up from the underlying scalar function."""
    kwargs = dict(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=ROOM_VOL_M3,
    )
    kwargs[arg_name] = bad_value
    with pytest.raises(ValueError, match=arg_name):
        predict_moisture(**kwargs)


# --- predict_room_moisture (integration with AirState / psychrometrics) ----


def test_predict_room_moisture_uses_airstate_for_indoor_absolute_humidity() -> None:
    """The initial AH must come from AirState, not be recomputed by moisture.py."""
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=15.0)

    expected_indoor_ah = AirState(
        temperature_c=20.0, relative_humidity_percent=70.0
    ).absolute_humidity
    assert result.initial_absolute_humidity_g_m3 == expected_indoor_ah


def test_predict_room_moisture_uses_airstate_for_outdoor_absolute_humidity() -> None:
    """Outdoor AH on the result must equal outdoor.absolute_humidity exactly."""
    room = Room(40.0, 20.0, 70.0, 0.5, 5.0)
    outdoor = AirState(5.0, 85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=15.0)
    assert result.outdoor_absolute_humidity_g_m3 == outdoor.absolute_humidity


def test_predict_room_moisture_selects_window_open_ach_when_true() -> None:
    """window_open=True -> the exchange rate is room.ach_window_open."""
    room = Room(40.0, 20.0, 70.0, ach_closed=0.5, ach_window_open=5.0)
    outdoor = AirState(5.0, 85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=15.0, window_open=True)
    assert result.ach == 5.0


def test_predict_room_moisture_selects_closed_ach_when_false() -> None:
    """window_open=False -> the exchange rate is room.ach_closed."""
    room = Room(40.0, 20.0, 70.0, ach_closed=0.5, ach_window_open=5.0)
    outdoor = AirState(5.0, 85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=15.0, window_open=False)
    assert result.ach == 0.5


def test_predict_room_moisture_default_window_open_is_true() -> None:
    """window_open defaults to True (the model exists to answer window-open scenarios)."""
    room = Room(40.0, 20.0, 70.0, ach_closed=0.5, ach_window_open=5.0)
    outdoor = AirState(5.0, 85.0)
    default_result = predict_room_moisture(room, outdoor, duration_minutes=15.0)
    explicit_result = predict_room_moisture(
        room, outdoor, duration_minutes=15.0, window_open=True
    )
    assert default_result == explicit_result


def test_predict_room_moisture_example_scenario_five_minutes() -> None:
    """User-worked scenario: indoor 20/70, outdoor 5/85, ACH=5, 5 minutes.

    Hand computation using the analytic solution and psychrometric AH values:
        indoor_ah  = AH(20 C, 70 %RH) via ideal gas
        outdoor_ah = AH( 5 C, 85 %RH) via ideal gas
        n*t = 5 h^-1 * (5/60) h = 5/12
        final = outdoor_ah + (indoor_ah - outdoor_ah) * exp(-5/12)

    Any drift between the computed final value and this recomposition means
    the moisture layer, the psychrometric layer, or the wiring between them
    has changed.
    """
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)

    result = predict_room_moisture(room, outdoor, duration_minutes=5.0)

    indoor_ah = AirState(20.0, 70.0).absolute_humidity
    outdoor_ah = AirState(5.0, 85.0).absolute_humidity
    expected_final = outdoor_ah + (indoor_ah - outdoor_ah) * exp(-5.0 * (5.0 / 60.0))

    # Physical sanity: initial state matches the psychrometric layer, and
    # after 5 min the room has dropped noticeably but is nowhere near
    # equilibrating with outdoor air (n*t = 5/12, far below one time constant).
    assert result.initial_absolute_humidity_g_m3 == pytest.approx(12.07, abs=0.05)
    assert result.outdoor_absolute_humidity_g_m3 == pytest.approx(5.77, abs=0.05)
    assert result.final_absolute_humidity_g_m3 == pytest.approx(expected_final, rel=1e-12)
    assert result.final_absolute_humidity_g_m3 < result.initial_absolute_humidity_g_m3
    assert result.final_absolute_humidity_g_m3 > result.outdoor_absolute_humidity_g_m3
    assert result.absolute_humidity_reduction_g_m3 > 0.0
    assert 0.0 < result.percentage_reduction < 100.0


def test_predict_room_moisture_matches_predict_moisture_with_manual_ah() -> None:
    """Route equivalence: predict_room_moisture == predict_moisture(indoor.ah, outdoor.ah).

    The wrapper must not introduce any extra transformation on the numbers -
    only look up AH from AirState and pick the right ACH from Room.
    """
    room = Room(50.0, 22.0, 60.0, ach_closed=0.4, ach_window_open=4.5)
    outdoor = AirState(8.0, 90.0)
    duration = 20.0

    wrapped = predict_room_moisture(room, outdoor, duration_minutes=duration)
    manual = predict_moisture(
        indoor_ah_g_m3=AirState(22.0, 60.0).absolute_humidity,
        outdoor_ah_g_m3=outdoor.absolute_humidity,
        ach=4.5,
        duration_minutes=duration,
        room_volume_m3=50.0,
    )
    assert wrapped == manual


def test_predict_room_moisture_zero_duration_leaves_room_unchanged() -> None:
    """A 0-minute event: initial and final match, everything else is zero.

    Tolerances are non-zero because the formula computes
    ``outdoor + (initial - outdoor) * exp(0)`` and the round-trip is only
    exact under IEEE 754 when Sterbenz's theorem applies (initial and
    outdoor within a factor of 2). Real AirState-derived values may or
    may not satisfy that, so use approx with a tight absolute floor.
    """
    room = Room(40.0, 20.0, 70.0, 0.5, 5.0)
    outdoor = AirState(5.0, 85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=0.0)
    assert result.final_absolute_humidity_g_m3 == pytest.approx(
        result.initial_absolute_humidity_g_m3, abs=1e-12
    )
    assert result.absolute_humidity_change_g_m3 == pytest.approx(0.0, abs=1e-12)
    assert result.percentage_reduction == pytest.approx(0.0, abs=1e-12)


def test_predict_room_moisture_wet_outdoor_adds_moisture() -> None:
    """Cool-dry indoor + warm-humid outdoor: opening the window adds water."""
    room = Room(40.0, 12.0, 40.0, 0.5, 5.0)
    outdoor = AirState(25.0, 85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=30.0)
    assert result.absolute_humidity_change_g_m3 > 0.0
    assert result.absolute_humidity_reduction_g_m3 < 0.0
    assert result.percentage_reduction < 0.0


def test_predict_room_moisture_propagates_ach_validation_via_closed_side() -> None:
    """Bad numeric values on the AirState/psy side bubble up as ValueError.

    Constructing an AirState with an out-of-range temperature raises when
    the property is accessed. That happens inside predict_room_moisture,
    so callers see a ValueError from the psychrometrics layer.
    """
    # 200 degC indoor is far outside the residential range and will fail
    # when AirState(...).absolute_humidity is read inside the wrapper.
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=200.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(5.0, 85.0)
    with pytest.raises(ValueError):
        predict_room_moisture(room, outdoor, duration_minutes=15.0)


def test_predict_room_moisture_reports_water_removed_using_room_volume() -> None:
    """The wrapper must feed room.volume_m3 into predict_moisture.

    In the 5-minute worked scenario (indoor 20/70, outdoor 5/85, ACH=5,
    volume=40 m^3) the reduction is roughly 12.07 - 9.93 = 2.15 g/m^3, so
    the water mass leaving the air is about 2.15 * 40 ~= 86 g.
    """
    room = Room(
        volume_m3=40.0,
        indoor_temperature_c=20.0,
        indoor_relative_humidity_pct=70.0,
        ach_closed=0.5,
        ach_window_open=5.0,
    )
    outdoor = AirState(temperature_c=5.0, relative_humidity_percent=85.0)
    result = predict_room_moisture(room, outdoor, duration_minutes=5.0)

    assert result.room_volume_m3 == 40.0
    assert result.water_removed_g == pytest.approx(
        result.absolute_humidity_reduction_g_m3 * 40.0, rel=1e-12
    )
    assert result.water_removed_g > 0.0
    assert 70.0 < result.water_removed_g < 100.0


# --- Design invariant: no post-ventilation RH derived from original T -----


def test_moisture_prediction_does_not_expose_a_final_rh_field() -> None:
    """The result must not report a final RH, dew point, or RH-derived quantity.

    Computing a "final RH" from the model's final absolute humidity and
    the room's ORIGINAL indoor temperature would be physically wrong:
    opening a window changes both moisture content and indoor temperature,
    and RH depends on both. A valid final RH can only come from the
    thermal model (which does not exist yet). The moisture layer
    deliberately stops at absolute humidity to avoid this silent trap.

    This test locks that design in so a well-meaning future addition
    cannot slip a temperature-composed RH onto MoisturePrediction without
    also updating this invariant.
    """
    result = predict_moisture(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        room_volume_m3=40.0,
    )
    forbidden_substrings = ("relative_humidity", "rh", "dew_point")
    for field in result.__dataclass_fields__:
        lowered = field.lower()
        # Guard against "relative_humidity", "final_rh", "dew_point", etc.
        # in any final-state-shaped field. The word 'humidity' on its own
        # is fine because 'absolute_humidity' is a temperature-independent
        # water-content quantity.
        assert not any(
            token in lowered
            for token in forbidden_substrings
        ), (
            f"MoisturePrediction unexpectedly exposes '{field}'. "
            "RH / dew-point outputs require a thermal model to predict "
            "the final indoor temperature and must not be computed from "
            "the moisture layer alone."
        )


# --- predict_final_absolute_humidity_with_source ---------------------------
# Extends dC/dt = n(C_out - C) to dC/dt = n(C_out - C) + G/V.


def test_source_extension_zero_generation_reproduces_existing_model() -> None:
    """G = 0 -> identical output to predict_final_absolute_humidity.

    Load-bearing regression test: the source extension must not
    change the ventilation-only physics. Sweep several (indoor,
    outdoor, ACH, duration) combinations to make the regression
    coverage broad.
    """
    for indoor, outdoor, ach, minutes in (
        (12.0, 5.0, 4.0, 15.0),
        (5.0, 12.0, 4.0, 15.0),
        (8.0, 8.0, 5.0, 30.0),
        (10.0, 3.0, 1.0, 5.0),
    ):
        original = predict_final_absolute_humidity(
            indoor_ah_g_m3=indoor,
            outdoor_ah_g_m3=outdoor,
            ach=ach,
            duration_minutes=minutes,
        )
        extended = predict_final_absolute_humidity_with_source(
            indoor_ah_g_m3=indoor,
            outdoor_ah_g_m3=outdoor,
            ach=ach,
            duration_minutes=minutes,
            moisture_generation_g_per_hour=0.0,
            room_volume_m3=40.0,
        )
        assert extended == pytest.approx(original, rel=1e-12, abs=1e-15)


def test_source_extension_constant_generation_matches_analytic_form() -> None:
    """C(t) = C_eq + (C_0 - C_eq) * exp(-n*t) with C_eq = C_out + G/(n*V).

    Hand-computed anchor. Indoor 12, outdoor 5, ACH 4 h^-1, V = 40
    m^3, G = 60 g/h, duration 15 min. C_eq = 5 + 60/(4*40) = 5.375.
    n*t = 1. C(15 min) = 5.375 + (12 - 5.375)/e = 5.375 + 6.625/e
    ~= 7.812 g/m^3.
    """
    result = predict_final_absolute_humidity_with_source(
        indoor_ah_g_m3=12.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        moisture_generation_g_per_hour=60.0,
        room_volume_m3=40.0,
    )
    equilibrium = 5.0 + 60.0 / (4.0 * 40.0)
    expected = equilibrium + (12.0 - equilibrium) / exp(1.0)
    assert result == pytest.approx(expected, rel=1e-12)


def test_source_extension_balances_ventilation_at_equilibrium() -> None:
    """If C_0 already equals C_eq, C(t) stays there for any t.

    Verifies the C_eq definition: C_out + G/(n*V) is the true fixed
    point of the extended ODE, not a moving equilibrium.
    """
    outdoor_ah = 5.0
    ach = 4.0
    volume = 40.0
    G = 80.0
    equilibrium = outdoor_ah + G / (ach * volume)  # = 5.5
    for minutes in (0.0, 15.0, 60.0, 240.0):
        result = predict_final_absolute_humidity_with_source(
            indoor_ah_g_m3=equilibrium,
            outdoor_ah_g_m3=outdoor_ah,
            ach=ach,
            duration_minutes=minutes,
            moisture_generation_g_per_hour=G,
            room_volume_m3=volume,
        )
        assert result == pytest.approx(equilibrium, rel=1e-12)


def test_source_extension_moisture_grows_when_generation_exceeds_removal() -> None:
    """When G is large enough that C_eq > C_0, indoor AH rises over time.

    Start at C_0 = 8 g/m^3, outdoor 5 g/m^3, ACH 2 h^-1, V = 40
    m^3, G = 500 g/h. C_eq = 5 + 500/(2*40) = 11.25 > C_0. So the
    room should get WETTER, not drier, despite the outdoor gradient
    that would normally dry it.
    """
    result = predict_final_absolute_humidity_with_source(
        indoor_ah_g_m3=8.0,
        outdoor_ah_g_m3=5.0,
        ach=2.0,
        duration_minutes=30.0,
        moisture_generation_g_per_hour=500.0,
        room_volume_m3=40.0,
    )
    # Room is now wetter than it started - source overwhelmed
    # ventilation.
    assert result > 8.0
    # And below the equilibrium (C_eq = 11.25) since the run was
    # finite.
    assert result < 11.25


def test_source_extension_no_ventilation_source_only_linear_growth() -> None:
    """ACH = 0 with a source: dC/dt = G/V, integrates to C_0 + G*t/V.

    Hand-computed anchor. V = 40 m^3, G = 40 g/h, t = 30 min = 0.5 h.
    Expected rise: G*t/V = 40 * 0.5 / 40 = 0.5 g/m^3.
    """
    result = predict_final_absolute_humidity_with_source(
        indoor_ah_g_m3=10.0,
        outdoor_ah_g_m3=5.0,
        ach=0.0,
        duration_minutes=30.0,
        moisture_generation_g_per_hour=40.0,
        room_volume_m3=40.0,
    )
    assert result == pytest.approx(10.5, rel=1e-12)


def test_source_extension_no_ventilation_no_source_room_is_inert() -> None:
    """ACH = 0 AND G = 0 -> the room does not change."""
    result = predict_final_absolute_humidity_with_source(
        indoor_ah_g_m3=10.0,
        outdoor_ah_g_m3=5.0,
        ach=0.0,
        duration_minutes=45.0,
        moisture_generation_g_per_hour=0.0,
        room_volume_m3=40.0,
    )
    assert result == 10.0


def test_source_extension_short_duration_matches_taylor_expansion() -> None:
    """For n*t << 1 the analytic form is well approximated by
    ``C_0 + (n*(C_out - C_0) + G/V) * t`` (linear in t).

    Uses a 30-second event so n*t = 5 * 0.5/60 ~= 0.042, safely in
    the linear-response window. Locks in that the source term is
    added, not multiplied or divided in error.
    """
    n = 5.0
    volume = 40.0
    G = 100.0
    indoor = 10.0
    outdoor = 6.0
    duration_hours = 0.5 / 60.0
    result = predict_final_absolute_humidity_with_source(
        indoor_ah_g_m3=indoor,
        outdoor_ah_g_m3=outdoor,
        ach=n,
        duration_minutes=0.5,
        moisture_generation_g_per_hour=G,
        room_volume_m3=volume,
    )
    linear_estimate = (
        indoor + (n * (outdoor - indoor) + G / volume) * duration_hours
    )
    assert result == pytest.approx(linear_estimate, rel=1e-3)


def test_source_extension_rejects_negative_generation() -> None:
    """Negative moisture generation is unphysical."""
    with pytest.raises(ValueError, match="moisture_generation_g_per_hour"):
        predict_final_absolute_humidity_with_source(
            indoor_ah_g_m3=10.0,
            outdoor_ah_g_m3=5.0,
            ach=4.0,
            duration_minutes=15.0,
            moisture_generation_g_per_hour=-1.0,
            room_volume_m3=40.0,
        )


def test_source_extension_rejects_zero_volume() -> None:
    """Zero volume makes G/V diverge; explicit rejection."""
    with pytest.raises(ValueError, match="room_volume_m3"):
        predict_final_absolute_humidity_with_source(
            indoor_ah_g_m3=10.0,
            outdoor_ah_g_m3=5.0,
            ach=4.0,
            duration_minutes=15.0,
            moisture_generation_g_per_hour=60.0,
            room_volume_m3=0.0,
        )


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("moisture_generation_g_per_hour", float("nan")),
        ("moisture_generation_g_per_hour", float("inf")),
        ("room_volume_m3", float("nan")),
        ("room_volume_m3", float("inf")),
    ],
)
def test_source_extension_rejects_non_finite_new_fields(
    field_name: str, bad_value: float
) -> None:
    """NaN / inf on the two new fields is rejected."""
    kwargs = dict(
        indoor_ah_g_m3=10.0,
        outdoor_ah_g_m3=5.0,
        ach=4.0,
        duration_minutes=15.0,
        moisture_generation_g_per_hour=60.0,
        room_volume_m3=40.0,
    )
    kwargs[field_name] = bad_value
    with pytest.raises(ValueError, match=field_name):
        predict_final_absolute_humidity_with_source(**kwargs)
