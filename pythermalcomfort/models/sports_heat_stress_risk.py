from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pythermalcomfort.classes_input import SportsHeatStressInputs
from pythermalcomfort.classes_return import SportsHeatStressRisk
from pythermalcomfort.models import phs
from pythermalcomfort.utilities import validate_type


@dataclass
class _ProfileSportValues:
    """Class to hold profile values."""

    height: float
    weight: float
    t_cr_extreme: float
    sweat_loss_g_threshold: float = 0.0  # Will be calculated in __post_init__

    def __post_init__(self):
        validate_type(self.height, "height", (int, float))
        validate_type(self.weight, "weight", (int, float))
        validate_type(self.t_cr_extreme, "t_cr_extreme", (int, float))

        if self.height <= 0:
            msg = f"height must be a positive number > 0, got {self.height}"
            raise ValueError(msg)
        if self.weight <= 0:
            msg = f"weight must be a positive number > 0, got {self.weight}"
            raise ValueError(msg)
        if self.t_cr_extreme <= 0:
            msg = f"t_cr_extreme must be a positive number > 0, got {self.t_cr_extreme}"
            raise ValueError(msg)
        # Calculate sweat_loss_g_threshold using the provided formula
        object.__setattr__(
            self, "sweat_loss_g_threshold", int(self.weight * 0.01 * 1000 * 1.1)
        )


@dataclass(frozen=True)
class ProfileSport:
    """Namespace of predefined profile values.

    Use attributes like `Profile.ADULT` to obtain a _ProfileValues instance.
    This class uses a frozen dataclass decorator to prevent modification of the
    namespace. Attributes are class-level constants, not instance fields.
    """

    AGE_LESS_10 = _ProfileSportValues(height=0.9, weight=25, t_cr_extreme=39.75)
    AGE_10_TO_13 = _ProfileSportValues(height=1.3, weight=43, t_cr_extreme=39.75)
    AGE_14_TO_17 = _ProfileSportValues(height=1.6, weight=60, t_cr_extreme=39.75)
    ADULT = _ProfileSportValues(height=1.8, weight=75, t_cr_extreme=40)


@dataclass
class _SportsValues:
    """Class to hold sport values."""

    clo: float
    met: float
    vr: float
    duration: int

    def __post_init__(self):
        validate_type(self.clo, "clo", (int, float))
        validate_type(self.met, "met", (int, float))
        validate_type(self.vr, "vr", (int, float))
        validate_type(self.duration, "duration", (int,))

        if self.clo <= 0:
            msg = f"clo must be a positive number > 0, got {self.clo}"
            raise ValueError(msg)
        if self.met <= 0:
            msg = f"met must be a positive number > 0, got {self.met}"
            raise ValueError(msg)
        if self.vr <= 0:
            msg = f"vr must be a positive number > 0, got {self.vr}"
            raise ValueError(msg)
        if self.duration < 0:
            msg = f"duration must be a non-negative integer >= 0, got {self.duration}"
            raise ValueError(msg)


@dataclass(frozen=True)
class Sports:
    """Namespace of predefined sport values.

    Use attributes like `Sports.RUNNING` to obtain a `_SportsValues` instance.
    This class uses a frozen dataclass decorator to prevent modification of the
    namespace. Attributes are class-level constants, not instance fields.
    """

    ABSEILING = _SportsValues(clo=0.6, met=6.0, vr=0.5, duration=120)
    ARCHERY = _SportsValues(clo=0.75, met=4.5, vr=0.5, duration=180)
    AUSTRALIAN_FOOTBALL = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    BASEBALL = _SportsValues(clo=0.7, met=6.0, vr=0.75, duration=120)
    BASKETBALL = _SportsValues(clo=0.37, met=7.5, vr=0.75, duration=45)
    BOWLS = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)
    CANOEING = _SportsValues(clo=0.6, met=7.5, vr=2.0, duration=60)
    CLIMBING = _SportsValues(clo=0.6, met=7.5, vr=1.0, duration=45)
    CRICKET = _SportsValues(clo=0.7, met=6.0, vr=0.75, duration=120)
    CYCLING = _SportsValues(clo=0.4, met=7.0, vr=3.0, duration=60)
    EQUESTRIAN = _SportsValues(clo=0.9, met=7.4, vr=3.0, duration=60)
    FIELD_ATHLETICS = _SportsValues(clo=0.3, met=7.0, vr=1.0, duration=60)
    FIELD_HOCKEY = _SportsValues(clo=0.6, met=7.4, vr=0.75, duration=45)
    FISHING = _SportsValues(clo=0.9, met=4.0, vr=0.5, duration=180)
    GOLF = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)
    HORSEBACK = _SportsValues(clo=0.9, met=7.4, vr=3.0, duration=60)
    KAYAKING = _SportsValues(clo=0.6, met=7.5, vr=2.0, duration=60)
    MTB = _SportsValues(clo=0.55, met=7.5, vr=3.0, duration=60)
    NETBALL = _SportsValues(clo=0.37, met=7.5, vr=0.75, duration=45)
    OZTAG = _SportsValues(clo=0.4, met=7.5, vr=0.75, duration=45)
    PICKLEBALL = _SportsValues(clo=0.4, met=6.5, vr=0.5, duration=60)
    ROWING = _SportsValues(clo=0.4, met=7.5, vr=2.0, duration=60)
    RUGBY_LEAGUE = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    RUGBY_UNION = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    RUNNING = _SportsValues(clo=0.37, met=7.5, vr=2.0, duration=60)
    SAILING = _SportsValues(clo=1.0, met=6.5, vr=2.0, duration=180)
    SHOOTING = _SportsValues(clo=0.6, met=5.0, vr=0.5, duration=120)
    SOCCER = _SportsValues(clo=0.47, met=7.5, vr=1.0, duration=45)
    SOFTBALL = _SportsValues(clo=0.9, met=6.1, vr=1.0, duration=120)
    TENNIS = _SportsValues(clo=0.4, met=7.0, vr=0.75, duration=60)
    TOUCH = _SportsValues(clo=0.4, met=7.5, vr=0.75, duration=45)
    VOLLEYBALL = _SportsValues(clo=0.37, met=6.8, vr=0.75, duration=60)
    WALKING = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)


def sports_heat_stress_risk(
    tdb: float | list[float] | np.ndarray,
    tr: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    vr: float | list[float] | np.ndarray,
    sport: _SportsValues,
    profile: _ProfileSportValues = ProfileSport.ADULT,
) -> SportsHeatStressRisk:
    """Calculate sports heat stress risk levels based on environmental conditions and
    sport-specific parameters.

    This function assesses heat stress risk for athletes during outdoor sports by
    combining environmental conditions with sport-specific metabolic rates and clothing
    insulation. It uses the Predicted Heat Strain (PHS) model to determine threshold
    temperatures for different risk categories (Low, Medium, High, Extreme). The method
    is based on the Sports Medicine Australia heat policy framework [SportsHeatStress2025]_,
    with detailed guidelines [SportsHeatPolicy2025]_ and an online implementation available
    at the Sports Heat Tool [SportsHeatTool]_.

    Parameters
    ----------
    tdb : float or list of float
        Dry bulb air temperature [°C].
    tr : float or list of float
        Mean radiant temperature [°C].
    rh : float or list of float
        Relative humidity [%].
    vr : float or list of float
        Relative air speed [m/s].

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. Please note that if this value
            is lower than the self-generated air speed from the activity, the code
            will replace with value with the activity-generated air speed to ensure
            the risk assessment is based on the actual conditions experienced by
            the athlete. To check the activity-generated air speed for each sport,
            refer to the `vr` field in the predefined sport values in the
            :py:class:`Sports` class, e.g., ``Sports.RUNNING.vr`` for running.

    sport : _SportsValues
        Sport-specific activity dataclass with fields ``clo`` (clothing insulation),
        ``met`` (metabolic rate), ``vr`` (relative air speed), and ``duration`` (activity duration).
        Use one of the predefined entries from the :py:class:`Sports` class, e.g., ``Sports.RUNNING``,
        ``Sports.SOCCER``, ``Sports.TENNIS``, etc.
    profile : _ProfileValues, optional
        Profile dataclass with fields ``height``, ``weight``, and ``t_cr_extreme``
        (core temperature threshold for extreme risk). Defaults to ``Profile.ADULT``.
        Use predefined entries from the :py:class:`Profile` class for different age groups, e.g.,
        ``Profile.AGE_LESS_10``, ``Profile.AGE_10_TO_13``, ``Profile.AGE_14_TO_17``, or ``Profile.ADULT``.

    Returns
    -------
    SportsHeatStressRisk
        A dataclass containing the heat stress risk assessment results.
        See :py:class:`~pythermalcomfort.classes_return.SportsHeatStressRisk` for
        more details. To access individual values, use the corresponding attributes
        of the returned instance, e.g., ``result.risk_level_interpolated``.

    Raises
    ------
    ValueError
        If the risk level could not be determined due to NaN thresholds or if the internal
        solver fails to produce thresholds that allow a risk determination.
    TypeError
        If sport is not a valid _SportsValues instance.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models.sports_heat_stress_risk import (
            sports_heat_stress_risk,
            Sports,
        )

        # Example 1: Single condition for running
        result = sports_heat_stress_risk(
            tdb=35, tr=35, rh=40, vr=0.1, sport=Sports.RUNNING
        )
        print(result.risk_level_interpolated)  # 3.0 (Extreme risk)
        print(result.t_medium)  # 23.0 (Temperature threshold for medium risk)
        print(result.t_high)  # 25.0 (Temperature threshold for high risk)
        print(result.t_extreme)  # 28.6 (Temperature threshold for extreme risk)
        print(result.recommendation)  # "Consider suspending play"

        # Example 2: Array inputs for multiple conditions
        result = sports_heat_stress_risk(
            tdb=[30, 35, 40],
            tr=[30, 35, 40],
            rh=[50, 50, 50],
            vr=[0.5, 0.5, 0.5],
            sport=Sports.SOCCER,
        )
        print(result.risk_level_interpolated)  # Array of risk levels

        # Example 3: Different sports
        result_tennis = sports_heat_stress_risk(
            tdb=33, tr=70, rh=60, vr=0.1, sport=Sports.TENNIS
        )
        result_cycling = sports_heat_stress_risk(
            tdb=33, tr=70, rh=60, vr=3.0, sport=Sports.CYCLING
        )
    """

    # Validate inputs using the input dataclass
    inputs = SportsHeatStressInputs(
        tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport, profile=profile
    )

    # Convert to numpy arrays for vectorized calculation
    tdb = np.asarray(inputs.tdb, dtype=float)
    tr = np.asarray(inputs.tr, dtype=float)
    rh = np.asarray(inputs.rh, dtype=float)
    vr = np.asarray(inputs.vr, dtype=float)
    # Ensure air speed is at least the activity-generated air speed
    vr_effective = max(vr, sport.vr) if np.isscalar(vr) else np.maximum(vr, sport.vr)

    # Vectorize the calculation function to handle arrays
    # Returns (risk_level_interpolated, t_medium, t_high, t_extreme, recommendation) for each input
    vectorized_calc = np.vectorize(
        _calc_risk_single_value, otypes=[float, float, float, float, str]
    )
    risk_levels, t_mediums, t_highs, t_extremes, recommendations = vectorized_calc(
        tdb=tdb, tr=tr, rh=rh, vr=vr_effective, sport=sport, profile=profile
    )

    return SportsHeatStressRisk(
        risk_level_interpolated=risk_levels,
        t_medium=t_mediums,
        t_high=t_highs,
        t_extreme=t_extremes,
        recommendation=recommendations,
    )


def _calc_risk_single_value(
    tdb: float,
    tr: float,
    rh: float,
    vr: float,
    sport: _SportsValues,
    profile: _ProfileSportValues,
) -> tuple[float, float, float, float, str]:
    """Calculate the risk level and threshold temperatures for a single set of inputs.

    Parameters
    ----------
    tdb : float
        Dry bulb air temperature, [°C].
    tr : float
        Mean radiant temperature, [°C].
    rh : float
        Relative humidity, [%].
    vr : float
        Relative air speed, [m/s].
    sport : _SportsValues
        Sport-specific parameters (clo, met, vr, duration).

    Returns
    -------
    tuple of (float, float, float, float, str)
        Tuple containing (risk_level_interpolated, t_medium, t_high, t_extreme, recommendation).
    """
    # set the max and min thresholds for the risk levels
    sweat_loss_g = (
        profile.sweat_loss_g_threshold
    )  # sweat loss threshold in grams per hour for the given profile

    max_t_low = 34.5  # maximum tdb for low risk
    max_t_medium = 39  # maximum tdb for medium risk
    max_t_high = 43.5  # maximum tdb for high risk
    min_t_low = 21  # minimum tdb for low risk
    min_t_medium = 23  # minimum tdb for medium risk
    min_t_high = 25  # minimum tdb for high risk
    min_t_extreme = 26  # minimum tdb for extreme risk

    t_cr_extreme = (
        profile.t_cr_extreme
    )  # core temperature threshold for extreme risk for the given profile

    if tdb < min_t_medium:
        # Low risk - use default thresholds and risk level 0
        return (
            0.0,
            min_t_medium,
            min_t_high,
            min_t_extreme,
            _get_recommendation(0.0),
        )
    if tdb > max_t_high:
        # Extreme risk - use maximum thresholds and risk level 3
        return (
            3.0,
            max_t_low,
            max_t_medium,
            max_t_high,
            _get_recommendation(3.0),
        )

    def calculate_threshold_water_loss(x):
        sl = phs(
            tdb=x,
            tr=tr,
            v=vr,
            rh=rh,
            met=sport.met,
            clo=sport.clo,
            posture="standing",
            duration=sport.duration,
            round_output=False,
            limit_inputs=False,
            acclimatized=100,
            i_mst=0.4,
            height=profile.height,
            weight=profile.weight,
        ).sweat_loss_g

        # Ensure a scalar float is returned for the root solver
        sl_scalar = float(np.asarray(sl))
        return float(sl_scalar / float(sport.duration) * 45.0 - float(sweat_loss_g))

    for min_t, max_t in [(0, 36), (20, 50)]:
        try:
            t_medium = brentq(calculate_threshold_water_loss, min_t, max_t)
            break
        except ValueError:
            continue
    else:
        msg = (
            f"Solver did not find a solution for low-medium threshold for {tdb=} and {rh=}: "
            f"all bracket ranges failed. Setting t_medium to max threshold of {max_t_low}°C."
        )
        warnings.warn(msg, stacklevel=2)
        t_medium = max_t_low

    def calculate_threshold_core(x):
        tcr = phs(
            tdb=x,
            tr=tr,
            v=vr,
            rh=rh,
            met=sport.met,
            clo=sport.clo,
            posture="standing",
            duration=sport.duration,
            round_output=False,
            limit_inputs=False,
            acclimatized=100,
            i_mst=0.4,
        ).t_cr

        return float(float(np.asarray(tcr)) - float(t_cr_extreme))

    for min_t, max_t in [(0, 36), (20, 50)]:
        try:
            t_extreme = brentq(calculate_threshold_core, min_t, max_t)
            break
        except ValueError:
            continue
    else:
        msg = (
            f"Solver did not find a solution for high-extreme threshold for {tdb=} and {rh=}: "
            f"all bracket ranges failed. Setting t_extreme to max threshold of {max_t_high}°C."
        )
        warnings.warn(msg, stacklevel=2)
        t_extreme = max_t_high

    # calculate t_high as the average of t_medium and t_extreme
    t_high = (
        (t_medium + t_extreme) / 2
        if not (np.isnan(t_medium) or np.isnan(t_extreme))
        else np.nan
    )

    # check if the thresholds are within the min and max limits defined above
    if t_medium > max_t_low:
        t_medium = max_t_low
    if t_high > max_t_medium:
        t_high = max_t_medium
    if t_extreme > max_t_high:
        t_extreme = max_t_high

    # cap the thresholds to the minimum values defined above
    if t_extreme < min_t_extreme:
        t_extreme = min_t_extreme
    if t_high < min_t_high:
        t_high = min_t_high
    if t_medium < min_t_medium:
        t_medium = min_t_medium

    risk_level_interpolated = np.nan
    # calculate the risk level with one decimal place
    if min_t_low <= tdb < t_medium:
        risk_level_interpolated = (tdb - min_t_medium) / (t_medium - min_t_medium)
    elif t_medium <= tdb < t_high:
        risk_level_interpolated = 1.0 + (tdb - t_medium) / (t_high - t_medium)
    elif t_high <= tdb < t_extreme:
        risk_level_interpolated = 2.0 + (tdb - t_high) / (t_extreme - t_high)
    elif tdb >= t_extreme:
        risk_level_interpolated = 3.0

    if np.isnan(risk_level_interpolated):
        raise ValueError("Risk level could not be determined due to NaN thresholds.")

    # Truncate to one decimal place toward negative infinity.
    risk_level_floor = np.floor(risk_level_interpolated * 10.0) / 10.0

    # Generate recommendation based on the FLOORED risk level for consistency
    recommendation = _get_recommendation(risk_level_floor)

    return (
        risk_level_floor,
        round(t_medium, 1),
        round(t_high, 1),
        round(t_extreme, 1),
        recommendation,
    )


def _get_recommendation(risk_level: float) -> str:
    """Get heat stress management recommendations based on risk level.

    Parameters
    ----------
    risk_level : float
        Interpolated risk level (0.0-3.0).

    Returns
    -------
    str
        Evidence-based recommendation text for managing heat stress at the given
        risk level.
    """
    if risk_level < 1.0:
        return "Increase hydration & modify clothing"
    elif risk_level < 2.0:
        return "Increase frequency and/or duration of rest breaks"
    elif risk_level < 3.0:
        return "Apply active cooling strategies"
    else:
        return "Consider suspending play"


if __name__ == "__main__":
    from pythermalcomfort.models.sports_heat_stress_risk import ProfileSport, Sports

    # todo investigate the following issue with the code
    results = sports_heat_stress_risk(
        tdb=30, tr=30, rh=45, vr=0.1, sport=Sports.SOCCER, profile=ProfileSport.ADULT
    )
    print(results)
    results = sports_heat_stress_risk(
        tdb=32, tr=32, rh=45, vr=0.1, sport=Sports.SOCCER, profile=ProfileSport.ADULT
    )
    print(results)

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    line_styles = ["solid", "dashed", "dotted", "dashdot"]
    profile_names = ["Age < 10", "Age 10-13", "Age 14-17", "Adult"]
    profiles = [
        ProfileSport.AGE_LESS_10,
        ProfileSport.AGE_10_TO_13,
        ProfileSport.AGE_14_TO_17,
        ProfileSport.ADULT,
    ]
    rh_range = np.arange(0, 101, 5)
    t_o = 30

    f, ax = plt.subplots(constrained_layout=True)

    for profile, line_style, profile_name in zip(
        profiles, line_styles, profile_names, strict=True
    ):
        results = sports_heat_stress_risk(
            tdb=t_o, tr=t_o, rh=rh_range, vr=0.1, sport=Sports.SOCCER, profile=profile
        )
        ax.plot(
            results.t_medium,
            rh_range,
            linestyle=line_style,
            label=f"{profile_name} - Medium",
            color="#fd7f00",
        )
        ax.plot(
            results.t_high,
            rh_range,
            linestyle=line_style,
            label=f"{profile_name} - High",
            color="#dc0b00",
        )
        ax.plot(
            results.t_extreme,
            rh_range,
            linestyle=line_style,
            label=f"{profile_name} - Extreme",
            color="#9c001d",
        )

    # Add a legend for risk thresholds (colors)
    from matplotlib.patches import Patch

    color_handles = [
        Patch(color="#fd7f00", label="Medium Risk Threshold"),
        Patch(color="#dc0b00", label="High Risk Threshold"),
        Patch(color="#9c001d", label="Extreme Risk Threshold"),
    ]
    legend1 = ax.legend(
        handles=color_handles, loc="upper right", title="Risk Threshold (color)"
    )
    ax.add_artist(legend1)

    # Add a visual legend for line styles (profiles)
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], color="black", linestyle=ls, lw=2, label=profile_name)
        for profile_name, ls in zip(profile_names, line_styles, strict=True)
    ]
    ax.legend(
        handles=style_handles,
        loc="lower left",
        title="Profile (line style)",
        frameon=False,
    )

    ax.set_xlabel("Operative Temperature (°C)")
    ax.set_ylabel("Relative Humidity (%)")
    sns.despine()
    plt.show()
