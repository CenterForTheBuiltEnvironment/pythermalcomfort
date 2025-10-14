import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from pythermalcomfort.models import phs
from pythermalcomfort.utilities import mean_radiant_tmp

sports_dict = {
    "abseiling": {
        "clo": 0.6,
        "met": 6.0,
        "vr": 0.5,
        "duration": 120,
        "sport": "Abseiling",
    },
    "archery": {
        "clo": 0.75,
        "met": 4.5,
        "vr": 0.5,
        "duration": 180,
        "sport": "Archery",
    },
    "australian_football": {
        "clo": 0.47,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Australian football",
    },
    "baseball": {
        "clo": 0.7,
        "met": 6.0,
        "vr": 0.75,
        "duration": 120,
        "sport": "Baseball",
    },
    "basketball": {
        "clo": 0.37,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Basketball",
    },
    "bowls": {
        "clo": 0.5,
        "met": 5.0,
        "vr": 0.5,
        "duration": 180,
        "sport": "Bowls",
    },
    "canoeing": {
        "clo": 0.6,
        "met": 7.5,
        "vr": 2.0,
        "duration": 60,
        "sport": "Canoeing",
    },
    "cricket": {
        "clo": 0.7,
        "met": 6.0,
        "vr": 0.75,
        "duration": 120,
        "sport": "Cricket",
    },
    "cycling": {
        "clo": 0.4,
        "met": 7.0,
        "vr": 3.0,
        "duration": 60,
        "sport": "Cycling",
    },
    "equestrian": {
        "clo": 0.9,
        "met": 7.4,
        "vr": 3.0,
        "duration": 60,
        "sport": "Equestrian",
    },
    "field_athletics": {
        "clo": 0.3,
        "met": 7.0,
        "vr": 1.0,
        "duration": 60,
        "sport": "Running (Athletics)",
    },
    "field_hockey": {
        "clo": 0.6,
        "met": 7.4,
        "vr": 0.75,
        "duration": 45,
        "sport": "Field hockey",
    },
    "fishing": {
        "clo": 0.9,
        "met": 4.0,
        "vr": 0.5,
        "duration": 180,
        "sport": "Fishing",
    },
    "golf": {
        "clo": 0.5,
        "met": 5.0,
        "vr": 0.5,
        "duration": 180,
        "sport": "Golf",
    },
    "horseback": {
        "clo": 0.9,
        "met": 7.4,
        "vr": 3.0,
        "duration": 60,
        "sport": "Horseback riding",
    },
    "kayaking": {
        "clo": 0.6,
        "met": 7.5,
        "vr": 2.0,
        "duration": 60,
        "sport": "Kayaking",
    },
    "running": {
        "clo": 0.37,
        "met": 7.5,
        "vr": 2.0,
        "duration": 60,
        "sport": "Long distance running",
    },
    "mtb": {
        "clo": 0.55,
        "met": 7.5,
        "vr": 3.0,
        "duration": 60,
        "sport": "Mountain biking",
    },
    "netball": {
        "clo": 0.37,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Netball",
    },
    "oztag": {
        "clo": 0.4,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Oztag",
    },
    "pickleball": {
        "clo": 0.4,
        "met": 6.5,
        "vr": 0.5,
        "duration": 60,
        "sport": "Pickleball",
    },
    "climbing": {
        "clo": 0.6,
        "met": 7.5,
        "vr": 1.0,
        "duration": 45,
        "sport": "Rock climbing",
    },
    "rowing": {
        "clo": 0.4,
        "met": 7.5,
        "vr": 2.0,
        "duration": 60,
        "sport": "Rowing",
    },
    "rugby_league": {
        "clo": 0.47,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Rugby league",
    },
    "rugby_union": {
        "clo": 0.47,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Rugby union",
    },
    "sailing": {
        "clo": 1.0,
        "met": 6.5,
        "vr": 2.0,
        "duration": 180,
        "sport": "Sailing",
    },
    "shooting": {
        "clo": 0.6,
        "met": 5.0,
        "vr": 0.5,
        "duration": 120,
        "sport": "Shooting",
    },
    "soccer": {
        "clo": 0.47,
        "met": 7.5,
        "vr": 1.0,
        "duration": 45,
        "sport": "Soccer",
    },
    "softball": {
        "clo": 0.9,
        "met": 6.1,
        "vr": 1.0,
        "duration": 120,
        "sport": "Softball",
    },
    "tennis": {
        "clo": 0.4,
        "met": 7.0,
        "vr": 0.75,
        "duration": 60,
        "sport": "Tennis",
    },
    "touch": {
        "clo": 0.4,
        "met": 7.5,
        "vr": 0.75,
        "duration": 45,
        "sport": "Touch football",
    },
    "volleyball": {
        "clo": 0.37,
        "met": 6.8,
        "vr": 0.75,
        "duration": 60,
        "sport": "Volleyball",
    },
    "walking": {
        "clo": 0.5,
        "met": 5.0,
        "vr": 0.5,
        "duration": 180,
        "sport": "Brisk walking",
    },
}


def sports_heat_stress_risk(tdb, rh, vr, sport_id, tg=None, tr=None):
    if sport_id not in sports_dict:
        msg = f"Sport ID '{sport_id}' not recognized. Available sports: {list(sports_dict.keys())}"
        raise ValueError(msg)

    if tg is None and tr is None:
        raise ValueError(
            "Either tg (globe temperature) or tr (mean radiant temperature) must be provided."
        )
    if tg is not None and tr is not None:
        raise ValueError(
            "Only one of tg (globe temperature) or tr (mean radiant temperature) should be provided."
        )
    if tg is not None and tr is None:
        tr = mean_radiant_tmp(tdb=tdb, tg=tg, v=vr)

    # get the sport parameters
    sport_dict = sports_dict[sport_id]

    # set the max and min thresholds for the risk levels
    sweat_loss_g = 825  # 825 g per hour todo check this value
    clo = sport_dict["clo"]
    met = sport_dict["met"]

    max_t_low = 34.5  # maximum tdb for low risk
    max_t_medium = 39  # maximum tdb for medium risk
    max_t_high = 43.5  # maximum tdb for high risk
    min_t_medium = 23  # minimum tdb for medium risk
    min_t_high = 25  # minimum tdb for high risk
    min_t_extreme = 26  # minimum tdb for extreme risk

    t_cr_extreme = 40  # core temperature for extreme risk

    # todo check the code below if want to return risk with one decimal place
    if tdb < min_t_medium:
        return 0  # we are in the low risk category
    if tdb > max_t_high:
        return 3  # we are in the extreme risk category

    if vr < sport_dict["vr"]:  # vr should be higher than the self-generated air speed
        vr = sport_dict["vr"]
        msg = f"Warning: Relative wind speed (vr) increased to {vr=} m/s to match minimum for {sport_id}."
        warnings.warn(msg, stacklevel=2)

    def calculate_threshold_water_loss(x):
        return (
            phs(
                tdb=x,
                tr=tr,
                v=vr,
                rh=rh,
                met=met,
                clo=clo,
                posture="standing",
                # todo check if I can use duration=60 for all sports
                duration=sport_dict["duration"],
                round=False,
                limit_inputs=False,
                acclimatized=100,
                i_mst=0.4,
            ).sweat_loss_g
            / sport_dict["duration"]
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
                met=met,
                clo=clo,
                posture="standing",
                duration=sport_dict["duration"],
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

    if t_extreme < min_t_extreme:
        t_extreme = min_t_extreme
    if t_high < min_t_high:
        t_high = min_t_high
    if t_medium < min_t_medium:
        t_medium = min_t_medium

    # calcuate the risk level based on the thresholds
    if tdb < t_medium:
        risk_level = 0
    elif t_medium <= tdb < t_high:
        risk_level = 1
    elif t_high <= tdb < t_extreme:
        risk_level = 2
    elif tdb >= t_extreme:
        risk_level = 3

    risk_level_interpolated = 0.0
    # calculate the risk level with one decimal place
    if t_medium <= tdb < t_high:
        risk_level_interpolated = 1.0 + (tdb - t_medium) / (t_high - t_medium)
    elif t_high <= tdb < t_extreme:
        risk_level_interpolated = 2.0 + (tdb - t_high) / (t_extreme - t_high)
    elif tdb >= t_extreme:
        risk_level_interpolated = 3.0

    if np.isnan(risk_level):
        raise ValueError("Risk level could not be determined due to NaN thresholds.")

    # todo include the recommendations based on the risk level
    # todo return a dataclass with the risk level, risk level interpolated, thresholds, and recommendations

    return risk_level, risk_level_interpolated


if __name__ == "__main__":
    # get the lowest wind speed across all sports
    min_wind_speed = max([sports_dict[sport]["vr"] for sport in sports_dict.keys()])
    print(f"Minimum wind speed across all sports: {min_wind_speed} m/s")
    max_wind_speed = min(
        [sports_dict[sport]["wind_high"] for sport in sports_dict.keys()]
    )
    print(f"Maximum wind speed across all sports: {max_wind_speed} m/s")

    for sport in sports_dict.keys():
        # sport = "softball"
        tg_delta = 10  # tg - tdb
        sweat_loss_g = 825  # 825 g per hour
        v = sports_dict[sport]["vr"]

        results = []

        for t, rh in product(range(23, 50, 2), range(0, 101, 5)):
            print(f"Calculating for {sport=} {t=} {rh=} {v=}")
            tg = tg_delta + t
            risk, risk_int = sports_heat_stress_risk(
                tdb=t, tg=tg, rh=rh, vr=v, sport_id=sport, sweat_loss_g=sweat_loss_g
            )
            results.append([t, rh, tg_delta, v, risk])

        df_new = pd.DataFrame(results, columns=["tdb", "rh", "tg", "v", "risk"])

        # plot side by side heatmaps
        f, axs = plt.subplots(1, 1, figsize=(7, 7), sharex=True, sharey=True)

        df_pivot = df_new.pivot(index="rh", columns="tdb", values="risk")
        df_pivot.sort_index(ascending=False, inplace=True)
        sns.heatmap(df_pivot, annot=False, cmap="viridis", ax=axs[0], vmin=0, vmax=3)

        axs[0].set_title(
            f"{sports_dict[sport]['sport']} - Heat stress risk (PHS model)", fontsize=14
        )
        axs[0].set_ylabel("Relative Humidity (%)", fontsize=12)
        axs[0].set_xlabel("Air Temperature (°C)", fontsize=12)
        plt.tight_layout()
        plt.savefig(
            f"/Users/ftar3919/downloads/sma/{sport}_v={v}_tg_delta={tg_delta}.png",
            dpi=300,
        )
        plt.show()
