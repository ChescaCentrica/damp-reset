"""Psychrometric calculations for moist air at residential conditions.

Provides the moisture math a residential ventilation model needs:

    saturation_vapour_pressure(T)                       -> Pa
    vapour_pressure(T, RH)                              -> Pa
    absolute_humidity_g_per_m3(T, RH)                   -> g/m^3
    humidity_ratio_kg_per_kg(T, RH, P)                  -> kg/kg dry air
    dew_point_c(T, RH)                                  -> degC

    AirState(temperature_c, relative_humidity_percent, atmospheric_pressure_pa)
        aggregates all of the above as properties on one immutable value.

    calculate_drying_potential(indoor, outdoor)         -> DryingPotential
        the first decision-adjacent output: indoor minus outdoor absolute
        humidity in g/m^3, plus a POC heuristic category label.

Every function is a pure conversion between physical quantities and states its
input and output units in both the parameter names and the docstring, so
callers never have to guess a scale. Equations are the Magnus (Alduchov-
Eskridge 1996) saturation curve for P_sat, the ASHRAE Handbook of Fundamentals
form for the humidity ratio, and the ideal-gas relation for absolute humidity;
valid roughly -40 to +50 degC, 0-100 %RH, near 1 atm, over liquid water.
"""

from dataclasses import dataclass
from math import exp, isfinite, log

MAGNUS_A: float = 17.625
MAGNUS_B: float = 243.04
P_SAT_0: float = 610.94

M_WATER: float = 0.018015  # kg/mol, molar mass of water
R_UNIVERSAL: float = 8.31446  # J/(mol.K), universal gas constant
MW_RATIO: float = 0.621945  # M_water / M_dry_air (18.015 / 28.966), dimensionless

DEFAULT_ATM_PRESSURE_PA: float = 101325.0

MIN_VALID_TEMPERATURE_C: float = -50.0
MAX_VALID_TEMPERATURE_C: float = 60.0

ZERO_CELSIUS_IN_KELVIN: float = 273.15
G_PER_KG: float = 1000.0


def saturation_vapour_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure of water over a flat liquid surface.

    Uses the Magnus form given by Alduchov and Eskridge (1996), "Improved
    Magnus Form Approximation of Saturation Vapor Pressure",
    J. Appl. Meteorol., 35(4), 601-609:

        P_sat(T) = 610.94 * exp(17.625 * T / (T + 243.04))

    where T is in degrees Celsius and P_sat is in pascals. The fit is over a
    flat liquid-water surface and agrees with the reference World
    Meteorological Organization / Wexler saturation curve to better than 0.4 %
    across -40 to +50 degC, which comfortably covers all residential
    indoor/outdoor conditions.

    Constants:
        610.94 Pa   saturation vapour pressure at 0 degC
        17.625      dimensionless Magnus coefficient A
        243.04 degC Magnus coefficient B

    Assumptions:
        - Saturation is over water, not ice. Below 0 degC this is the standard
          meteorological convention for supercooled liquid.
        - Ideal-gas / near-1-atm regime; pressure does not enter the formula.

    Args:
        temperature_c: air temperature in degrees Celsius. Must be a finite
            value within the residential range
            [MIN_VALID_TEMPERATURE_C, MAX_VALID_TEMPERATURE_C].

    Returns:
        Saturation vapour pressure in pascals.

    Raises:
        ValueError: if temperature_c is NaN, infinite, or outside the valid
            range. Rejecting unrealistic inputs early avoids silently
            returning wildly extrapolated pressures.
    """
    if not isfinite(temperature_c):
        raise ValueError(f"temperature_c must be finite, got {temperature_c!r}")
    if not MIN_VALID_TEMPERATURE_C <= temperature_c <= MAX_VALID_TEMPERATURE_C:
        raise ValueError(
            f"temperature_c={temperature_c} degC is outside the supported "
            f"residential range [{MIN_VALID_TEMPERATURE_C}, "
            f"{MAX_VALID_TEMPERATURE_C}] degC"
        )
    return P_SAT_0 * exp(MAGNUS_A * temperature_c / (temperature_c + MAGNUS_B))


def vapour_pressure(
    temperature_c: float,
    relative_humidity_pct: float,
) -> float:
    """Actual water-vapour partial pressure from temperature and relative humidity.

    Applies the definition of relative humidity to the saturation curve:

        P_v = (RH / 100) * P_sat(T)

    where P_sat(T) is provided by ``saturation_vapour_pressure`` (Magnus form,
    over water). At RH = 100 % the result equals P_sat exactly to floating
    point precision, since the factor collapses to 1.0.

    Args:
        temperature_c: air temperature in degrees Celsius. Passed through to
            ``saturation_vapour_pressure``, which enforces the residential
            temperature range.
        relative_humidity_pct: relative humidity on a 0-100 scale. Must be a
            finite value in the closed interval [0, 100].

    Returns:
        Water-vapour partial pressure in pascals.

    Raises:
        ValueError: if ``relative_humidity_pct`` is NaN, infinite, or outside
            [0, 100]. Temperature validation errors propagate from
            ``saturation_vapour_pressure``.
    """
    if not isfinite(relative_humidity_pct):
        raise ValueError(
            f"relative_humidity_pct must be finite, got {relative_humidity_pct!r}"
        )
    if not 0.0 <= relative_humidity_pct <= 100.0:
        raise ValueError(
            f"relative_humidity_pct={relative_humidity_pct} is outside the "
            f"valid range [0, 100]"
        )
    return (relative_humidity_pct / 100.0) * saturation_vapour_pressure(temperature_c)


def humidity_ratio_kg_per_kg(
    temperature_c: float,
    relative_humidity_pct: float,
    pressure_pa: float = DEFAULT_ATM_PRESSURE_PA,
) -> float:
    """Humidity ratio W: mass of water vapour per mass of dry air.

    Uses the standard psychrometric relationship (ASHRAE Handbook of
    Fundamentals, Chapter 1):

        W = (M_water / M_dry_air) * P_v / (P - P_v)
          = 0.621945          * P_v / (P - P_v)     [kg water / kg dry air]

    The derivation follows Dalton's law of partial pressures - the total
    pressure P is the sum of the dry-air partial pressure and the water-
    vapour partial pressure P_v, so P_dry = P - P_v. For two ideal gases
    sharing a volume the mass ratio is the mole ratio times the molar-mass
    ratio, giving the coefficient 0.621945 = M_water (18.015 g/mol) divided
    by M_dry_air (28.966 g/mol).

    Reuses ``vapour_pressure`` for P_v, which itself enforces the
    temperature and RH input contracts.

    Why humidity ratio differs from absolute humidity:
        - Absolute humidity (g/m^3) is mass of water vapour per unit
          **volume** of moist air. It depends on temperature via the
          ideal-gas law: cooling the same parcel raises its absolute
          humidity because the volume shrinks, even though no water was
          added.
        - Humidity ratio (kg/kg) is mass of water vapour per unit **mass**
          of dry air. It is invariant when a sealed parcel is heated,
          cooled, or its total pressure changes without adding or removing
          water. That makes it the right conserved quantity for tracking
          how much water an air mass carries as it moves through a building
          envelope or a ventilation duct. It is the standard moisture
          coordinate on psychrometric charts.

    This function is exposed for later components (thermal / ventilation /
    optimisation) that need the mass-based moisture coordinate. The
    drying-potential comparator in this module deliberately does **not**
    consume it yet.

    Assumptions:
        - Ideal-gas behaviour for both dry air and water vapour (excellent
          at residential T, P).
        - Dalton's law: total pressure is the sum of dry-air and vapour
          partial pressures.

    Args:
        temperature_c: air temperature in degrees Celsius.
        relative_humidity_pct: relative humidity on a 0-100 scale.
        pressure_pa: total atmospheric pressure in pascals. Defaults to
            sea-level standard (101325 Pa). Must exceed the vapour
            partial pressure P_v; otherwise the mixture cannot exist as
            described (P_v <= P by definition, and equality would make
            the ratio infinite).

    Returns:
        Humidity ratio in kilograms of water per kilogram of dry air.

    Raises:
        ValueError: if ``pressure_pa`` is non-finite, non-positive, or not
            strictly greater than the water-vapour partial pressure at the
            given temperature and RH.
    """
    if not isfinite(pressure_pa):
        raise ValueError(f"pressure_pa must be finite, got {pressure_pa!r}")
    if pressure_pa <= 0.0:
        raise ValueError(f"pressure_pa must be positive, got {pressure_pa}")
    p_v_pa = vapour_pressure(temperature_c, relative_humidity_pct)
    if pressure_pa <= p_v_pa:
        raise ValueError(
            f"pressure_pa={pressure_pa} Pa must be strictly greater than the "
            f"water-vapour partial pressure P_v={p_v_pa:.3f} Pa at "
            f"T={temperature_c} degC, RH={relative_humidity_pct} %"
        )
    return MW_RATIO * p_v_pa / (pressure_pa - p_v_pa)


def absolute_humidity_g_per_m3(temperature_c: float, relative_humidity_pct: float) -> float:
    """Mass of water vapour per unit volume of moist air.

    Derived from the ideal-gas equation applied to water vapour alone. In a
    mixture, water vapour occupies the same volume as the dry air and exerts
    its own partial pressure P_v, so:

        P_v * V = n_water * R * T_K
        n_water / V = P_v / (R * T_K)                  [mol / m^3]
        m_water / V = P_v * M_water / (R * T_K)        [kg  / m^3]
        AH          = P_v * M_water / (R * T_K) * 1000 [g   / m^3]

    P_v is obtained from ``vapour_pressure`` and stays in pascals throughout
    the calculation; no lookup tables or empirical fits are involved.

    Dimensional check (SI):
        [P_v] = Pa       = J / m^3
        [M_water] = kg / mol
        [R] = J / (mol . K)
        [T_K] = K
        P_v * M_water / (R * T_K)
            = (J / m^3) * (kg / mol) / ((J / (mol.K)) * K)
            = (J * kg) / (m^3 * mol) * (mol.K / J) / K
            = kg / m^3
        Multiplying by G_PER_KG = 1000 converts kg/m^3 to g/m^3.

    Constants (module-level):
        M_WATER            = 0.018015 kg/mol  (molar mass of water)
        R_UNIVERSAL        = 8.31446 J/(mol.K) (universal gas constant)
        ZERO_CELSIUS_IN_KELVIN = 273.15 K
        G_PER_KG           = 1000.0

    Assumptions:
        - Water vapour behaves as an ideal gas (excellent at residential T, P).
        - Temperature and RH validation are delegated to ``vapour_pressure``
          and ``saturation_vapour_pressure``.

    Args:
        temperature_c: air temperature in degrees Celsius.
        relative_humidity_pct: relative humidity on a 0-100 scale.

    Returns:
        Absolute humidity in grams of water per cubic metre of moist air.
    """
    p_v_pa = vapour_pressure(temperature_c, relative_humidity_pct)
    temperature_k = temperature_c + ZERO_CELSIUS_IN_KELVIN
    ah_kg_per_m3 = p_v_pa * M_WATER / (R_UNIVERSAL * temperature_k)
    return ah_kg_per_m3 * G_PER_KG


def relative_humidity_from_absolute_humidity(
    temperature_c: float,
    absolute_humidity_g_m3: float,
) -> float:
    """Recover relative humidity in % from absolute humidity and temperature.

    Physical inverse of ``absolute_humidity_g_per_m3``. Derivation is
    algebraic, using the same ideal-gas relation the forward function
    uses; no new physics is introduced and no saturation equation is
    duplicated.

    Forward direction:
        AH = P_v * M_water / (R * T_K) * G_PER_KG            [g/m^3]

    Inverting for P_v (AH is in g/m^3, so divide by G_PER_KG to reach
    kg/m^3 before applying the ideal-gas relation):
        P_v = (AH / G_PER_KG) * R * T_K / M_water            [Pa]

    Then apply the definition of RH against ``saturation_vapour_pressure``:
        RH = 100 * P_v / P_sat(T)                            [%]

    Dimensional check:
        [AH / G_PER_KG] = kg / m^3
        [R] = J / (mol . K),  [T_K] = K,  [M_water] = kg / mol
        [P_v] = (kg / m^3) * (J / (mol.K)) * K / (kg / mol)
              = J / m^3
              = Pa                                            (as required)
        [RH] = 100 * Pa / Pa = %                              (dimensionless)

    Consistency with the module's other functions:
        - Uses ``saturation_vapour_pressure`` (Magnus form) for P_sat.
          No separate saturation fit is introduced.
        - Uses the module-level constants ``M_WATER``, ``R_UNIVERSAL``,
          ``ZERO_CELSIUS_IN_KELVIN``, ``G_PER_KG``. Any drift in those
          constants would be reflected identically in both the forward
          and inverse calculations.
        - No total-pressure argument: both AH (ideal-gas partial-pressure
          formulation) and P_sat are pressure-independent in this model,
          so RH from AH and T is well-defined without knowing P.

    Behaviour when AH exceeds the saturation value:
        Mathematically the arithmetic still produces an RH value > 100 %.
        This function returns the raw value rather than clamping or
        raising. Preserving the round-trip identity is more useful for
        debugging than silently masking an inconsistent (T, AH) pair;
        callers that need strict [0, 100] behaviour should check the
        result explicitly.

    Args:
        temperature_c: air temperature in degrees Celsius. Passed to
            ``saturation_vapour_pressure``, which enforces the
            residential range.
        absolute_humidity_g_m3: absolute humidity in grams of water per
            cubic metre of moist air. Must be finite and non-negative.

    Returns:
        Relative humidity as a percentage on the 0-100 scale. Values
        above 100 indicate supersaturated inputs (see behaviour note).

    Raises:
        ValueError: if ``absolute_humidity_g_m3`` is not finite or is
            negative; temperature validation propagates from
            ``saturation_vapour_pressure``.
    """
    if not isfinite(absolute_humidity_g_m3):
        raise ValueError(
            f"absolute_humidity_g_m3 must be finite, got {absolute_humidity_g_m3!r}"
        )
    if absolute_humidity_g_m3 < 0.0:
        raise ValueError(
            f"absolute_humidity_g_m3 must be non-negative, "
            f"got {absolute_humidity_g_m3}"
        )
    p_sat_pa = saturation_vapour_pressure(temperature_c)
    temperature_k = temperature_c + ZERO_CELSIUS_IN_KELVIN
    ah_kg_per_m3 = absolute_humidity_g_m3 / G_PER_KG
    p_v_pa = ah_kg_per_m3 * R_UNIVERSAL * temperature_k / M_WATER
    return 100.0 * p_v_pa / p_sat_pa


def dew_point_c(temperature_c: float, relative_humidity_pct: float) -> float:
    """Temperature at which the current water-vapour pressure would saturate.

    Analytic inverse of the Magnus saturation curve used by
    ``saturation_vapour_pressure`` in this module. Reuses the *identical*
    constants ``MAGNUS_A`` (17.625), ``MAGNUS_B`` (243.04 degC) and
    ``P_SAT_0`` (610.94 Pa); no separate fit is introduced. Because the
    inversion is algebraic, the round-trip is exact up to floating-point
    round-off:

        saturation_vapour_pressure(dew_point_c(T, RH)) == vapour_pressure(T, RH)

    Derivation:

        Given P_v = P_SAT_0 * exp(A * T_d / (T_d + B))       (Magnus, at T_d)
              alpha := ln(P_v / P_SAT_0) = A * T_d / (T_d + B)
        Solving for T_d:
              alpha * (T_d + B) = A * T_d
              alpha * T_d + alpha * B = A * T_d
              T_d = B * alpha / (A - alpha)

    Sanity check at 100 %RH: P_v = P_sat(T), so alpha = A*T/(T+B) and
        T_d = B * (A*T/(T+B)) / (A - A*T/(T+B))
            = B * (A*T/(T+B)) / (A*B/(T+B))
            = T.
    Dew point equals ambient temperature exactly (up to floating point).

    Behaviour at 0 %RH: with no water vapour there is no dew-point
    temperature at all - the mathematical limit is -infinity, since
    P_sat(T) is strictly positive for any finite T. Rather than return a
    sentinel that would silently break downstream arithmetic, we raise
    ``ValueError``. Callers that care about "very dry air" should check
    RH > 0 first, or catch this error.

    Args:
        temperature_c: air temperature in degrees Celsius.
        relative_humidity_pct: relative humidity on a 0-100 scale. Must be
            strictly positive.

    Returns:
        Dew-point temperature in degrees Celsius.

    Raises:
        ValueError: if ``relative_humidity_pct`` is exactly 0 (dew point
            not defined); if the computed dew point falls outside the
            module's residential validity range (which happens for
            physically extreme very-low RH values, e.g. RH < ~1 % at
            20 degC yields a dew point below -50 C where the water-phase
            Magnus curve is no longer the right saturation branch);
            temperature and RH range errors propagate from
            ``vapour_pressure`` / ``saturation_vapour_pressure``.
    """
    if relative_humidity_pct == 0.0:
        raise ValueError(
            "dew_point_c is undefined at relative_humidity_pct == 0 "
            "(limit is -infinity); the caller should special-case zero RH"
        )
    p_v_pa = vapour_pressure(temperature_c, relative_humidity_pct)
    alpha = log(p_v_pa / P_SAT_0)
    dew_point = MAGNUS_B * alpha / (MAGNUS_A - alpha)
    if not MIN_VALID_TEMPERATURE_C <= dew_point <= MAX_VALID_TEMPERATURE_C:
        raise ValueError(
            f"computed dew point {dew_point:.2f} degC is outside the module's "
            f"residential validity range "
            f"[{MIN_VALID_TEMPERATURE_C}, {MAX_VALID_TEMPERATURE_C}] degC "
            f"for T={temperature_c} degC, RH={relative_humidity_pct} %; "
            "the water-phase Magnus branch is not appropriate this far from "
            "residential conditions"
        )
    return dew_point


@dataclass(frozen=True)
class AirState:
    """Immutable snapshot of a moist-air state with derived psychrometric properties.

    Bundles the three independent variables - temperature, relative humidity,
    and total pressure - and exposes every derived quantity as a property.
    Each property delegates to the corresponding module-level function, so
    equations live in exactly one place: the standalone functions remain the
    source of truth for calculations and testing, while ``AirState`` is a
    convenience aggregator for callers that want all properties for one set
    of conditions.

    Frozen so that a computed property is a valid function of the state, not
    of the instance's history; two ``AirState`` values with equal fields
    compare equal.

    Example:
        >>> indoor = AirState(temperature_c=20.0, relative_humidity_percent=70.0)
        >>> indoor.absolute_humidity     # g/m^3
        >>> indoor.dew_point             # degC
    """

    temperature_c: float
    relative_humidity_percent: float
    atmospheric_pressure_pa: float = DEFAULT_ATM_PRESSURE_PA

    @property
    def saturation_vapour_pressure(self) -> float:
        """Saturation vapour pressure at this state's temperature, in pascals."""
        return saturation_vapour_pressure(self.temperature_c)

    @property
    def vapour_pressure(self) -> float:
        """Actual water-vapour partial pressure, in pascals."""
        return vapour_pressure(self.temperature_c, self.relative_humidity_percent)

    @property
    def absolute_humidity(self) -> float:
        """Absolute humidity in grams of water per cubic metre of moist air."""
        return absolute_humidity_g_per_m3(
            self.temperature_c, self.relative_humidity_percent
        )

    @property
    def humidity_ratio(self) -> float:
        """Humidity ratio in kilograms of water per kilogram of dry air.

        Uses ``atmospheric_pressure_pa`` from this state, not the module
        default, so altitude effects propagate correctly.
        """
        return humidity_ratio_kg_per_kg(
            self.temperature_c,
            self.relative_humidity_percent,
            self.atmospheric_pressure_pa,
        )

    @property
    def dew_point(self) -> float:
        """Dew-point temperature in degrees Celsius.

        Raises ``ValueError`` at ``relative_humidity_percent == 0`` because
        the dew point is undefined in that limit; see ``dew_point_c``.
        """
        return dew_point_c(self.temperature_c, self.relative_humidity_percent)


# --- Drying-potential decision layer ----------------------------------------
#
# WARNING - POC heuristic thresholds, NOT validated safety limits.
#
# The category bands below (NONE / LOW / MODERATE / HIGH) are qualitative
# convenience labels for a first proof-of-concept build. They are NOT
# medically, scientifically, or industry-validated air-quality, damp-risk,
# mould-risk, or ventilation-effectiveness thresholds. They should not be
# used to make health, building-safety, or regulatory decisions.
#
# The underlying numerical quantity - the absolute-humidity difference in
# g/m^3 - is the meaningful signal. The category is only a coarse
# human-readable summary intended to make demos and dashboards easier to
# read. If any downstream component uses the category to trigger action,
# revisit these thresholds against real occupancy, HVAC, and comfort data
# for the specific building.
#
# Thresholds are exposed as named module constants so calibration by a
# later component is a one-line change and appears in diffs.
DRYING_POTENTIAL_LOW_THRESHOLD_G_M3: float = 0.0
DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3: float = 1.0
DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3: float = 3.0

DRYING_POTENTIAL_CATEGORY_NONE: str = "NONE"
DRYING_POTENTIAL_CATEGORY_LOW: str = "LOW"
DRYING_POTENTIAL_CATEGORY_MODERATE: str = "MODERATE"
DRYING_POTENTIAL_CATEGORY_HIGH: str = "HIGH"


@dataclass(frozen=True)
class DryingPotential:
    """Result of comparing indoor and outdoor absolute humidity.

    The primary signal is ``difference_g_m3``:

        difference_g_m3 = indoor_absolute_humidity_g_m3 - outdoor_absolute_humidity_g_m3

    Interpretation:
        * positive - replacing indoor air with outdoor air has moisture-
          removal potential (outdoor air is drier in absolute terms).
        * zero     - no moisture benefit; the two air masses carry the same
          water per cubic metre.
        * negative - outdoor air currently contains MORE moisture per unit
          volume than indoor air; ventilating would add water.

    ``category`` is a coarse human-readable label. See the module-level
    warning: the category thresholds are POC heuristics, not validated
    safety limits. Downstream logic should prefer ``difference_g_m3``.
    """

    indoor_absolute_humidity_g_m3: float
    outdoor_absolute_humidity_g_m3: float
    difference_g_m3: float
    category: str


def _classify_drying_potential(difference_g_m3: float) -> str:
    """Map an absolute-humidity difference to a POC-heuristic category label.

    Bands (see WARNING above - not validated):
        difference <= DRYING_POTENTIAL_LOW_THRESHOLD_G_M3       -> NONE
        LOW_THRESHOLD   < difference <= MODERATE_THRESHOLD      -> LOW
        MODERATE_THRESHOLD < difference <= HIGH_THRESHOLD       -> MODERATE
        difference > HIGH_THRESHOLD                             -> HIGH
    """
    if difference_g_m3 <= DRYING_POTENTIAL_LOW_THRESHOLD_G_M3:
        return DRYING_POTENTIAL_CATEGORY_NONE
    if difference_g_m3 <= DRYING_POTENTIAL_MODERATE_THRESHOLD_G_M3:
        return DRYING_POTENTIAL_CATEGORY_LOW
    if difference_g_m3 <= DRYING_POTENTIAL_HIGH_THRESHOLD_G_M3:
        return DRYING_POTENTIAL_CATEGORY_MODERATE
    return DRYING_POTENTIAL_CATEGORY_HIGH


def calculate_drying_potential(
    indoor: AirState,
    outdoor: AirState,
) -> DryingPotential:
    """Quantify how much drier the outdoor air is per cubic metre than the indoor air.

    Computes ``indoor.absolute_humidity - outdoor.absolute_humidity`` and
    attaches a POC-heuristic category label. Absolute humidity is used
    because it is the mass of water per unit volume of air, which is what
    physically transfers when air is exchanged between two spaces at
    similar total pressures.

    See the module-level warning about the category thresholds: the
    underlying numerical difference is the meaningful quantity; the
    category is a convenience label only.

    Args:
        indoor: air state inside the building.
        outdoor: air state outside the building.

    Returns:
        A ``DryingPotential`` bundling both absolute humidities, their
        difference in g/m^3, and the heuristic category.
    """
    indoor_ah = indoor.absolute_humidity
    outdoor_ah = outdoor.absolute_humidity
    difference = indoor_ah - outdoor_ah
    return DryingPotential(
        indoor_absolute_humidity_g_m3=indoor_ah,
        outdoor_absolute_humidity_g_m3=outdoor_ah,
        difference_g_m3=difference,
        category=_classify_drying_potential(difference),
    )
