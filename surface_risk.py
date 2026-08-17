"""First-order internal-surface temperature and humidity model.

Answers ONE question, per surface, per instantaneous room state:

    given the current indoor air state and outdoor temperature,
    what is the temperature of a defined internal surface, and
    what relative humidity does it see?

The surface is represented by a single dimensionless caller-set
parameter, the temperature factor ``fRsi``, defined by
BS EN ISO 13788 / IStructE / UK Building Regulations Approved
Document L:

    fRsi = (T_surface - T_outdoor) / (T_indoor - T_outdoor)

so

    T_surface = T_outdoor + fRsi * (T_indoor - T_outdoor)

Interpretation of ``fRsi``:
    * 1.0 - the surface is at indoor air temperature. Physically
      idealised: a warm, well-insulated interior wall of a room
      with uniform air.
    * ~0.9 - a good modern interior surface with a well-detailed
      thermal bridge.
    * ~0.75 - the threshold value UK Approved Document L
      references as a limit for avoiding mould growth risk in
      dwellings. NOT re-derived here; the caller supplies their
      own value, and this repo does NOT publish an authoritative
      one.
    * ~0.5 - a severe thermal bridge (e.g. an uninsulated single-
      glazed reveal or an external wall corner).
    * 0.0 - the surface is at outdoor temperature. Physically an
      uninsulated external element.

The module does NOT ship an authoritative fRsi value or an
authoritative mould-risk threshold. Every value the caller sets is
a POC parameter that must be defended on evidence (measured
surface temperature under known conditions, an fRsi calculation
from wall U-values and surface resistances per ISO 13788, or a
building-specific study) before any deployment.

Surface RH is computed from the SAME indoor water-vapour partial
pressure the psychrometric layer already validates, evaluated
against the saturation curve at the SURFACE temperature:

    P_v_indoor    = (RH_indoor / 100) * P_sat(T_indoor)
    P_sat_surface = P_sat(T_surface)
    RH_surface    = 100 * P_v_indoor / P_sat_surface

We reuse ``psychrometrics.vapour_pressure`` for the indoor
partial pressure and ``psychrometrics.saturation_vapour_pressure``
for the surface saturation curve. No new physics equation is
written here.

Note on why we do NOT route through
``relative_humidity_from_absolute_humidity`` with the room's AH
at the surface temperature: absolute humidity is a per-VOLUME
quantity (g/m^3), and the inverse's ideal-gas step uses the
QUERY temperature (T_surface here) to reconstruct P_v. Doing so
would misinterpret the room's water content when applied at the
surface temperature. The vapour PARTIAL PRESSURE is temperature-
independent in a well-mixed room and is the physically correct
quantity to hold constant when moving from air state to surface
state.

Explicitly NOT in this module:
    - Time-integrated mould-risk indicators (VTT, isopleth,
      ASHRAE 160, WUFI-Bio). The architecture proposal lists that
      as a separate slice.
    - Finite-element or hygrothermal wall modelling.
    - Any surface-condensation dynamics (liquid water on a cold
      pane). The module reports the driving condition (surface RH);
      what happens beyond saturation is out of scope.
"""

from dataclasses import dataclass
from math import isfinite

from psychrometrics import (
    AirState,
    dew_point_c,
    saturation_vapour_pressure,
    vapour_pressure,
)


@dataclass(frozen=True)
class SurfaceDescriptor:
    """Configurable description of one critical internal surface.

    Fields:
        label: caller-set free-form name for logging / audit,
            e.g. "kitchen north wall behind fridge",
            "single-glazed bathroom window", "living-room lintel".
        surface_temperature_factor: dimensionless ``fRsi`` per
            BS EN ISO 13788. Must be finite and in the closed
            interval ``[0.0, 1.0]``. Zero means the surface tracks
            outdoor temperature; one means it tracks indoor
            temperature. The caller chooses the value; this
            module does not.

    Illustrative caller-supplied values (NOT authoritative):
        SurfaceDescriptor("warm wall",         0.90)
        SurfaceDescriptor("cold external wall",0.70)
        SurfaceDescriptor("severe thermal bridge", 0.50)

    None of the above are validated. The docstring of
    ``surface_temperature_factor`` in the module docstring above
    describes what caller intents each range represents, but the
    caller is responsible for the number.
    """

    label: str
    surface_temperature_factor: float

    def __post_init__(self) -> None:
        """Validate the temperature factor."""
        if not isfinite(self.surface_temperature_factor):
            raise ValueError(
                "surface_temperature_factor must be finite, got "
                f"{self.surface_temperature_factor!r}"
            )
        if not 0.0 <= self.surface_temperature_factor <= 1.0:
            raise ValueError(
                "surface_temperature_factor must be in [0, 1], got "
                f"{self.surface_temperature_factor}"
            )


def surface_temperature_c(
    indoor_temperature_c: float,
    outdoor_temperature_c: float,
    surface: SurfaceDescriptor,
) -> float:
    """Compute the surface temperature via the fRsi factor.

    Uses::

        T_surface = T_outdoor + fRsi * (T_indoor - T_outdoor)

    Sign convention: when indoor is warmer than outdoor
    (winter-typical), the surface sits between indoor and outdoor,
    with fRsi = 1 giving T_surface = T_indoor and fRsi = 0 giving
    T_surface = T_outdoor. When outdoor is warmer than indoor
    (summer-typical), the formulation still applies and the
    surface sits between indoor and outdoor from the other side.

    Args:
        indoor_temperature_c: indoor air temperature.
        outdoor_temperature_c: outdoor air temperature.
        surface: caller's ``SurfaceDescriptor``.

    Returns:
        Surface temperature in degrees Celsius.

    Raises:
        ValueError: if either temperature is NaN or infinite.
    """
    if not isfinite(indoor_temperature_c):
        raise ValueError(
            f"indoor_temperature_c must be finite, got {indoor_temperature_c!r}"
        )
    if not isfinite(outdoor_temperature_c):
        raise ValueError(
            f"outdoor_temperature_c must be finite, got {outdoor_temperature_c!r}"
        )
    factor = surface.surface_temperature_factor
    return outdoor_temperature_c + factor * (
        indoor_temperature_c - outdoor_temperature_c
    )


def surface_relative_humidity_pct(
    indoor_air_state: AirState,
    outdoor_temperature_c: float,
    surface: SurfaceDescriptor,
) -> float:
    """Compute relative humidity at the surface, given the room's air state.

    The indoor air's water-vapour partial pressure is uniform
    across the well-mixed room; the surface, at its own colder
    temperature, sees the SAME ``P_v`` divided by a SMALLER
    ``P_sat``. So:

        P_v_indoor    = vapour_pressure(T_indoor, RH_indoor)
        P_sat_surface = saturation_vapour_pressure(T_surface)
        RH_surface    = 100 * P_v_indoor / P_sat_surface

    Both helpers already live in ``psychrometrics.py`` and are
    independently validated; this function only assembles their
    outputs.

    Behaviour at and above saturation:
        The underlying psychrometric inverse returns values above
        100 % when the vapour pressure exceeds saturation at the
        query temperature (i.e. when the surface has passed dew
        point and would be condensing). The value is returned
        raw, without clamping, because a caller reading a 130 %
        surface RH would want to see it (dew is forming); a value
        clamped at 100 % would hide the condition. Callers who
        need a strict-[0, 100] answer should clamp themselves.

    Args:
        indoor_air_state: the current indoor air state. Its
            ``absolute_humidity`` provides the room's water content.
        outdoor_temperature_c: outdoor air temperature.
        surface: caller's ``SurfaceDescriptor``.

    Returns:
        Relative humidity at the surface, in percent. May exceed
        100 if the surface is below the room's dew point.

    Raises:
        ValueError: on any invalid input; validation errors from
            the psychrometric layer propagate.
    """
    surface_t = surface_temperature_c(
        indoor_temperature_c=indoor_air_state.temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        surface=surface,
    )
    indoor_vapour_pressure_pa = vapour_pressure(
        temperature_c=indoor_air_state.temperature_c,
        relative_humidity_pct=indoor_air_state.relative_humidity_percent,
    )
    surface_saturation_pa = saturation_vapour_pressure(surface_t)
    return 100.0 * indoor_vapour_pressure_pa / surface_saturation_pa


# Shorter alias matching the caller-facing name convention. Same
# behaviour and same tests as ``surface_relative_humidity_pct`` above.
surface_relative_humidity = surface_relative_humidity_pct


def condensation_margin_c(
    indoor_air_state: AirState,
    outdoor_temperature_c: float,
    surface: SurfaceDescriptor,
) -> float:
    """Return ``T_surface - T_dew_point_indoor``, in kelvin (= degrees Celsius diff).

    Sign convention:
        * positive - the surface is WARMER than the room's dew
          point. Condensation is not predicted on this surface at
          this room state.
        * zero     - the surface is exactly at the dew point.
          The condensation boundary.
        * negative - the surface is COLDER than the room's dew
          point. Surface condensation is predicted to be possible
          at this room state.

    The room's dew point comes from ``psychrometrics.dew_point_c``
    (the analytic inverse of the Magnus curve, already validated).
    The surface temperature comes from ``surface_temperature_c``
    above (the fRsi-based linear formulation).

    NOTE: a negative condensation margin means the model predicts
    that SURFACE CONDENSATION IS POSSIBLE. It does NOT predict
    mould, nor does it predict any specific rate of dew formation
    or drying. Mould-growth models require time-integrated
    exposure over days-to-weeks and are out of scope; this is an
    instantaneous condensation margin only.

    Args:
        indoor_air_state: indoor air state.
        outdoor_temperature_c: outdoor air temperature.
        surface: caller's ``SurfaceDescriptor``.

    Returns:
        Condensation margin in kelvin.

    Raises:
        ValueError: on any invalid input; validation errors from
            the psychrometric layer propagate.
    """
    surface_t = surface_temperature_c(
        indoor_temperature_c=indoor_air_state.temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        surface=surface,
    )
    indoor_dew_point = dew_point_c(
        temperature_c=indoor_air_state.temperature_c,
        relative_humidity_pct=indoor_air_state.relative_humidity_percent,
    )
    return surface_t - indoor_dew_point


@dataclass(frozen=True)
class SurfaceRiskResult:
    """Bundled snapshot of one surface's condensation-risk state.

    Not a mould prediction. This is the instantaneous condensation
    margin at one time on one caller-defined surface. See the
    module docstring and ``condensation_margin_c`` for the sign
    convention and for what this does and does not model.

    Fields:
        surface_label: caller-set descriptor name, echoed for audit.
        indoor_temperature_c: indoor air temperature the assessment
            was made against.
        indoor_relative_humidity_pct: indoor air RH the assessment
            was made against.
        indoor_dew_point_c: room's dew point at this air state, from
            ``psychrometrics.dew_point_c``.
        surface_temperature_c: computed surface temperature from the
            fRsi factor.
        surface_relative_humidity_pct: RH the surface sees, from
            ``surface_relative_humidity(...)``. May exceed 100 %
            when the surface is below the room's dew point; the
            raw value is not clamped.
        condensation_margin_c: ``surface_temperature_c`` minus
            ``indoor_dew_point_c``. Positive = safe from
            condensation, zero = boundary, negative = surface
            condensation predicted possible.
    """

    surface_label: str
    indoor_temperature_c: float
    indoor_relative_humidity_pct: float
    indoor_dew_point_c: float
    surface_temperature_c: float
    surface_relative_humidity_pct: float
    condensation_margin_c: float


def assess_surface(
    indoor_air_state: AirState,
    outdoor_temperature_c: float,
    surface: SurfaceDescriptor,
) -> SurfaceRiskResult:
    """Bundle every named surface metric into one ``SurfaceRiskResult``.

    Convenience wrapper for callers who want the full instantaneous
    picture. Every field is a documented named quantity from the
    functions above; no arithmetic is performed here beyond what
    those functions already do.

    Args:
        indoor_air_state: indoor air state.
        outdoor_temperature_c: outdoor air temperature.
        surface: caller's ``SurfaceDescriptor``.

    Returns:
        A ``SurfaceRiskResult`` bundling the room state, room dew
        point, surface temperature, surface RH, and condensation
        margin.
    """
    surface_t = surface_temperature_c(
        indoor_temperature_c=indoor_air_state.temperature_c,
        outdoor_temperature_c=outdoor_temperature_c,
        surface=surface,
    )
    return SurfaceRiskResult(
        surface_label=surface.label,
        indoor_temperature_c=indoor_air_state.temperature_c,
        indoor_relative_humidity_pct=indoor_air_state.relative_humidity_percent,
        indoor_dew_point_c=dew_point_c(
            temperature_c=indoor_air_state.temperature_c,
            relative_humidity_pct=indoor_air_state.relative_humidity_percent,
        ),
        surface_temperature_c=surface_t,
        surface_relative_humidity_pct=surface_relative_humidity(
            indoor_air_state=indoor_air_state,
            outdoor_temperature_c=outdoor_temperature_c,
            surface=surface,
        ),
        condensation_margin_c=condensation_margin_c(
            indoor_air_state=indoor_air_state,
            outdoor_temperature_c=outdoor_temperature_c,
            surface=surface,
        ),
    )
