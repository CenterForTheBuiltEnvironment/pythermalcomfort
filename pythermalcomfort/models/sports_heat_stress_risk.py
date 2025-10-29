import warnings
from dataclasses import dataclass

# todo not implemented yet in the docs nor in the __init__.py
import numpy as np
import scipy

from pythermalcomfort.models import phs


@dataclass
class _SportsValues:
    """Class to hold sport values."""

    clo: float
    met: float
    vr: float
    duration: int


@dataclass(frozen=True)
class Sports:
    ABSEILING = _SportsValues(clo=0.6, met=6.0, vr=0.5, duration=120)
    ARCHERY = _SportsValues(clo=0.75, met=4.5, vr=0.5, duration=180)
    AUSTRALIAN_FOOTBALL = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    BASEBALL = _SportsValues(clo=0.7, met=6.0, vr=0.75, duration=120)
    BASKETBALL = _SportsValues(clo=0.37, met=7.5, vr=0.75, duration=45)
    BOWLS = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)
    CANOEING = _SportsValues(clo=0.6, met=7.5, vr=2.0, duration=60)
    CRICKET = _SportsValues(clo=0.7, met=6.0, vr=0.75, duration=120)
    CYCLING = _SportsValues(clo=0.4, met=7.0, vr=3.0, duration=60)
    EQUESTRIAN = _SportsValues(clo=0.9, met=7.4, vr=3.0, duration=60)
    FIELD_ATHLETICS = _SportsValues(clo=0.3, met=7.0, vr=1.0, duration=60)
    FIELD_HOCKEY = _SportsValues(clo=0.6, met=7.4, vr=0.75, duration=45)
    FISHING = _SportsValues(clo=0.9, met=4.0, vr=0.5, duration=180)
    GOLF = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)
    HORSEBACK = _SportsValues(clo=0.9, met=7.4, vr=3.0, duration=60)
    KAYAKING = _SportsValues(clo=0.6, met=7.5, vr=2.0, duration=60)
    RUNNING = _SportsValues(clo=0.37, met=7.5, vr=2.0, duration=60)
    MTB = _SportsValues(clo=0.55, met=7.5, vr=3.0, duration=60)
    NETBALL = _SportsValues(clo=0.37, met=7.5, vr=0.75, duration=45)
    OZTAG = _SportsValues(clo=0.4, met=7.5, vr=0.75, duration=45)
    PICKLEBALL = _SportsValues(clo=0.4, met=6.5, vr=0.5, duration=60)
    CLIMBING = _SportsValues(clo=0.6, met=7.5, vr=1.0, duration=45)
    ROWING = _SportsValues(clo=0.4, met=7.5, vr=2.0, duration=60)
    RUGBY_LEAGUE = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    RUGBY_UNION = _SportsValues(clo=0.47, met=7.5, vr=0.75, duration=45)
    SAILING = _SportsValues(clo=1.0, met=6.5, vr=2.0, duration=180)
    SHOOTING = _SportsValues(clo=0.6, met=5.0, vr=0.5, duration=120)
    SOCCER = _SportsValues(clo=0.47, met=7.5, vr=1.0, duration=45)
    SOFTBALL = _SportsValues(clo=0.9, met=6.1, vr=1.0, duration=120)
    TENNIS = _SportsValues(clo=0.4, met=7.0, vr=0.75, duration=60)
    TOUCH = _SportsValues(clo=0.4, met=7.5, vr=0.75, duration=45)
    VOLLEYBALL = _SportsValues(clo=0.37, met=6.8, vr=0.75, duration=60)
    WALKING = _SportsValues(clo=0.5, met=5.0, vr=0.5, duration=180)


def sports_heat_stress_risk(
    tdb: float, tr: float, rh: float, vr: float, sport: _SportsValues
):
    # todo missing docstring
    # todo add references to the docstring
    # todo add examples to the docstring
    # todo the function should accept either float or arrays as all the other functions pythermalcomfort

    # set the max and min thresholds for the risk levels
    sweat_loss_g = 825  # 825 g per hour todo - FT - check this value

    max_t_low = 34.5  # maximum tdb for low risk
    max_t_medium = 39  # maximum tdb for medium risk
    max_t_high = 43.5  # maximum tdb for high risk
    min_t_low = 21  # minimum tdb for low risk
    min_t_medium = 22  # minimum tdb for medium risk
    min_t_high = 23  # minimum tdb for high risk
    min_t_extreme = 25  # minimum tdb for extreme risk

    t_cr_extreme = 40  # core temperature for extreme risk

    if tdb < min_t_low:
        return 0  # we are in the low risk category
    if tdb > max_t_high:
        return 3  # we are in the extreme risk category

    def calculate_threshold_water_loss(x):
        return (
            phs(
                tdb=x,
                tr=tr,
                v=vr,
                rh=rh,
                met=sport.met,
                clo=sport.clo,
                posture="standing",
                # todo check if I can use duration=60 for all sports
                duration=sport.duration,
                round=False,
                limit_inputs=False,
                acclimatized=100,
                i_mst=0.4,
            ).sweat_loss_g
            / sport.duration
            * 45
            # todo I want to remove the above line and calculate a fixed value for all sports over 60 min
            - sweat_loss_g
        )

    for min_t, max_t in [(0, 36), (20, 50)]:
        try:
            t_medium = scipy.optimize.brentq(
                calculate_threshold_water_loss, min_t, max_t
            )
            break
        except ValueError as e:
            msg = f"Solver did not find a solution for low-medium threshold for {tdb=} and {rh=}: {e}"
            msg = f"{msg}. Setting t_medium to max threshold of {max_t_low}°C."
            warnings.warn(msg, stacklevel=2)
            t_medium = max_t_low

    def calculate_threshold_core(x):
        return (
            phs(
                tdb=x,
                tr=tr,
                v=vr,
                rh=rh,
                met=sport.met,
                clo=sport.clo,
                posture="standing",
                duration=sport.duration,
                round=False,
                limit_inputs=False,
                acclimatized=100,
                i_mst=0.4,
            ).t_cr
            - t_cr_extreme
        )

    for min_t, max_t in [(0, 36), (20, 50)]:
        try:
            t_extreme = scipy.optimize.brentq(calculate_threshold_core, min_t, max_t)
            break
        except ValueError as e:
            msg = f"Solver did not find a solution for high-extreme threshold for {tdb=} and {rh=}: {e}"
            msg = f"{msg}. Setting t_extreme to max threshold of {max_t_high}°C."
            warnings.warn(msg, stacklevel=2)
            t_extreme = max_t_high

    # calculate t_high as the average of t_medium and t_extreme
    t_high = (
        (t_medium + t_extreme) / 2
        if not (np.isnan(t_medium) or np.isnan(t_extreme))
        else np.nan
    )

    risk_level = np.nan

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

    # # calcuate the risk level based on the thresholds
    # if tdb < t_medium:
    #     risk_level = 0
    # elif t_medium <= tdb < t_high:
    #     risk_level = 1
    # elif t_high <= tdb < t_extreme:
    #     risk_level = 2
    # elif tdb >= t_extreme:
    #     risk_level = 3

    risk_level_interpolated = np.nan
    # calculate the risk level with one decimal place
    if min_t_low <= tdb < t_medium:
        risk_level_interpolated = (tdb - min_t_low) / (t_medium - min_t_low)
    elif t_medium <= tdb < t_high:
        risk_level_interpolated = 1.0 + (tdb - t_medium) / (t_high - t_medium)
    elif t_high <= tdb < t_extreme:
        risk_level_interpolated = 2.0 + (tdb - t_high) / (t_extreme - t_high)
    elif tdb >= t_extreme:
        risk_level_interpolated = 3.0

    if np.isnan(risk_level_interpolated):
        raise ValueError("Risk level could not be determined due to NaN thresholds.")

    # todo include the recommendations based on the risk level
    # todo return a dataclass with the risk level, risk level interpolated, thresholds, and recommendations

    return risk_level_interpolated


if __name__ == "__main__":
    # first example
    t = 35
    rh = 40
    v = .1
    tr = 70
    sport = "running"
    print(sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB))

    # second example
    results = []
    for t in range(20, 46, 2):
        for rh in range(0, 101, 5):
            risk_interp = sports_heat_stress_risk(
                tdb=t, tr=t, rh=rh, vr=v, sport=Sports.RUNNING
            )
            results.append((t, rh, risk_interp))
    import pandas as pd

    df = pd.DataFrame(
        results, columns=["tdb", "rh", "risk_level_interpolated"]
    )
    df_pivot = df.pivot(index="rh", columns="tdb", values="risk_level_interpolated")
    df_pivot.sort_index(ascending=False, inplace=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_pivot, cmap="YlOrRd")
    plt.title("Sports Heat Stress Risk Level for Running")
    plt.xlabel("Dry Bulb Temperature (°C)")
    plt.ylabel("Relative Humidity (%)")
    plt.show()
