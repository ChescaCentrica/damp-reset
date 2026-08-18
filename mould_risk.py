"""Time-integrated surface moisture / condensation risk indicator.

Steps a ``time_simulation.RoomTrajectory`` sample by sample,
computes the surface RH at each step (reusing the fRsi surface
temperature model from ``surface_risk`` and the ideal-gas relation
already validated in ``psychrometrics``), and accumulates four
transparent metrics over the whole trajectory:

    time_above_surface_rh_threshold_hours
    time_in_condensation_hours
    maximum_surface_rh_percent
    cumulative_risk_score

Every threshold and weight the score uses is caller-supplied on a
``RiskConfig`` dataclass. The module does NOT publish default
values that imply biological growth from a single number. The
score is a CONFIGURABLE INDICATOR intended for comparing two
ventilation strategies' sustained surface exposure - it is NOT a
validated mould-growth prediction, and this repo does not claim
otherwise.

Published mould-growth models (VTT, isopleth, ASHRAE 160,
WUFI-Bio) integrate surface T and RH against species-specific
temperature-dependent RH thresholds over days to weeks. That
class of model is out of scope for this slice; adding one is a
separate, literature-review-heavy step.

Design contract:
    - Reads sample points off ``RoomTrajectory`` and derives the
      surface RH at each sample via
      ``surface_risk.surface_temperature_c`` and the psychrometric
      saturation curve. No new physics.
    - Contains no thresholds, weights, or numeric defaults that
      imply biological growth from a single number. Every
      configurable value defaults to zero or None.
    - Reports raw exposure accumulations and a caller-composed
      score; does not label its output as a mould prediction.

Explicitly NOT in this module:
    - Species-specific growth models.
    - Temperature-dependent RH thresholds (Sedlbauer / VTT
      isopleths).
    - Any decision logic ("open the window because mould").
"""

from dataclasses import dataclass
from math import isfinite

from psychrometrics import (
    G_PER_KG,
    M_WATER,
    R_UNIVERSAL,
    ZERO_CELSIUS_IN_KELVIN,
    saturation_vapour_pressure,
)
from surface_risk import SurfaceDescriptor, surface_temperature_c
from time_simulation import RoomTrajectory


@dataclass(frozen=True)
class RiskConfig:
    """Caller-supplied thresholds and weights for the risk indicator.

    Defaults are deliberately conservative in the sense of NOT
    claiming causation: every threshold is a caller decision. The
    module docstring names the reason. See the docstring on each
    field for what it means and what to be careful of.

    Fields:
        elevated_surface_rh_threshold_percent: RH above which a
            surface counts as "exposed". Defaults to 80 %, a
            value often cited in building-services literature
            (Sedlbauer, ISO 13788) as a broad indicator of
            elevated mould risk, but published thresholds are
            temperature-dependent and species-specific, so
            treating 80 % as a hard growth threshold is not
            justified. The caller should override this to match
            whatever evidence base they use.
        condensation_surface_rh_threshold_percent: RH at or above
            which a surface counts as "condensing". Defaults to
            100 % (the physical boundary). Callers can lower it
            (e.g. 95 %) to include near-saturated conditions as
            equivalent risk.
        peak_rh_excess_weight_hours_per_percent: weight applied
            to the amount by which the peak surface RH exceeds
            the elevated threshold. Units make the term add
            hour-equivalents to the score. Defaults to 0.0, i.e.
            the peak does not contribute to the score unless the
            caller opts in.
        elevated_time_weight: dimensionless multiplier on
            ``time_above_surface_rh_threshold_hours``. Defaults
            to 1.0.
        condensation_time_weight: dimensionless multiplier on
            ``time_in_condensation_hours``. Defaults to 1.0.
    """

    elevated_surface_rh_threshold_percent: float = 80.0
    condensation_surface_rh_threshold_percent: float = 100.0
    elevated_time_weight: float = 1.0
    condensation_time_weight: float = 1.0
    peak_rh_excess_weight_hours_per_percent: float = 0.0

    def __post_init__(self) -> None:
        """Validate every field: finite, non-negative, RH thresholds in [0, 200]."""
        for field_name, value in (
            (
                "elevated_surface_rh_threshold_percent",
                self.elevated_surface_rh_threshold_percent,
            ),
            (
                "condensation_surface_rh_threshold_percent",
                self.condensation_surface_rh_threshold_percent,
            ),
            (
                "elevated_time_weight",
                self.elevated_time_weight,
            ),
            (
                "condensation_time_weight",
                self.condensation_time_weight,
            ),
            (
                "peak_rh_excess_weight_hours_per_percent",
                self.peak_rh_excess_weight_hours_per_percent,
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
        # RH thresholds may exceed 100 (a supersaturated indoor
        # scenario can produce a surface RH above 100 - see the
        # surface_risk docstring), so we allow up to 200 % as a
        # loose upper bound on caller mistakes.
        for field_name, value in (
            (
                "elevated_surface_rh_threshold_percent",
                self.elevated_surface_rh_threshold_percent,
            ),
            (
                "condensation_surface_rh_threshold_percent",
                self.condensation_surface_rh_threshold_percent,
            ),
        ):
            if value > 200.0:
                raise ValueError(
                    f"{field_name} must be <= 200, got {value}"
                )


@dataclass(frozen=True)
class MoistureRiskState:
    """Time-integrated surface moisture / condensation risk state.

    NOT a mould-growth prediction. The class name and every field
    are labelled to make that explicit: this is an INDICATOR
    intended to compare two ventilation strategies over the same
    trajectory, not a claim about biological growth.

    Fields:
        surface_label: caller-set descriptor name, echoed for
            audit.
        time_above_surface_rh_threshold_hours: total time across
            the trajectory that the surface RH was strictly above
            ``config.elevated_surface_rh_threshold_percent``.
        time_in_condensation_hours: total time across the
            trajectory that the surface RH was at or above
            ``config.condensation_surface_rh_threshold_percent``.
        maximum_surface_rh_percent: peak surface RH seen in the
            trajectory.
        cumulative_risk_score: a caller-composed weighted sum of
            the three accumulations. Higher = more exposure.
            Meaningful only in relative terms between two runs of
            the same ``RiskConfig``.

    Sign convention: every accumulation is non-negative by
    construction. The score is likewise non-negative.
    """

    surface_label: str
    time_above_surface_rh_threshold_hours: float
    time_in_condensation_hours: float
    maximum_surface_rh_percent: float
    cumulative_risk_score: float


def evaluate_moisture_risk(
    trajectory: RoomTrajectory,
    surface: SurfaceDescriptor,
    config: RiskConfig,
) -> MoistureRiskState:
    """Step the trajectory, accumulate surface exposure, return the risk state.

    Uses the trapezoid-style rule for the time accumulations: each
    step contributes ``Δt`` to the "elevated" or "condensation"
    counter if the surface RH at the START of the step is above
    the respective threshold. This matches the piecewise-constant
    conventions the time-simulation module already uses (window
    state and generation rate are piecewise-constant at the step
    start; risk accumulation follows the same rule).

    The maximum surface RH is a strict maximum across every sample.

    Args:
        trajectory: room trajectory from
            ``time_simulation.simulate_room_period``.
        surface: caller's ``SurfaceDescriptor``.
        config: caller's ``RiskConfig``.

    Returns:
        A ``MoistureRiskState`` bundling the four indicator values.
    """
    n = len(trajectory.times_hours)
    if n == 0:
        return MoistureRiskState(
            surface_label=surface.label,
            time_above_surface_rh_threshold_hours=0.0,
            time_in_condensation_hours=0.0,
            maximum_surface_rh_percent=0.0,
            cumulative_risk_score=0.0,
        )

    # Sample the surface RH at every trajectory point. Uses indoor
    # AH directly rather than routing through AirState(T, RH),
    # because trajectory samples can transiently exceed 100 %RH
    # (a supersaturated indoor moment during a large moisture
    # event), and AirState correctly rejects RH > 100 at
    # construction time. Computing indoor P_v from indoor AH via
    # the ideal-gas relation avoids the boundary while still using
    # the validated saturation curve for the surface saturation
    # pressure.
    surface_rhs = []
    for i in range(n):
        indoor_t = trajectory.indoor_temperature_c[i]
        indoor_ah = trajectory.indoor_absolute_humidity_g_m3[i]
        outdoor_t = trajectory.outdoor_temperature_c[i]
        indoor_p_v_pa = (
            (indoor_ah / G_PER_KG)
            * R_UNIVERSAL
            * (indoor_t + ZERO_CELSIUS_IN_KELVIN)
            / M_WATER
        )
        t_surface = surface_temperature_c(
            indoor_temperature_c=indoor_t,
            outdoor_temperature_c=outdoor_t,
            surface=surface,
        )
        surface_rhs.append(
            100.0 * indoor_p_v_pa / saturation_vapour_pressure(t_surface)
        )

    time_above = 0.0
    time_cond = 0.0
    max_rh = surface_rhs[0]

    # Each step contributes if the START-of-step surface RH is
    # over the threshold. This mirrors the piecewise-constant
    # convention used elsewhere in the pipeline.
    for i in range(n - 1):
        dt_hours = trajectory.times_hours[i + 1] - trajectory.times_hours[i]
        if surface_rhs[i] > config.elevated_surface_rh_threshold_percent:
            time_above += dt_hours
        if surface_rhs[i] >= config.condensation_surface_rh_threshold_percent:
            time_cond += dt_hours
        if surface_rhs[i + 1] > max_rh:
            max_rh = surface_rhs[i + 1]

    peak_excess = max(
        0.0, max_rh - config.elevated_surface_rh_threshold_percent
    )
    score = (
        config.elevated_time_weight * time_above
        + config.condensation_time_weight * time_cond
        + config.peak_rh_excess_weight_hours_per_percent * peak_excess
    )

    return MoistureRiskState(
        surface_label=surface.label,
        time_above_surface_rh_threshold_hours=time_above,
        time_in_condensation_hours=time_cond,
        maximum_surface_rh_percent=max_rh,
        cumulative_risk_score=score,
    )
