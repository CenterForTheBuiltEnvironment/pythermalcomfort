import numpy as np
import warnings
import math
import re
from pythermalcomfort.psychrometrics import p_sat, t_o
from pythermalcomfort.shared_functions import valid_range

warnings.simplefilter("always")


def transpose_sharp_altitude(sharp, altitude):
    altitude_new = math.degrees(
        math.asin(
            math.sin(math.radians(abs(sharp - 90))) * math.cos(math.radians(altitude))
        )
    )
    sharp = math.degrees(
        math.atan(math.sin(math.radians(sharp)) * math.tan(math.radians(90 - altitude)))
    )
    sol_altitude = altitude_new
    return round(sharp, 3), round(sol_altitude, 3)


def check_standard_compliance(standard, **kwargs):
    params = dict()
    params["standard"] = standard
    for key, value in kwargs.items():
        params[key] = value

    if params["standard"] == "ankle_draft":
        for key, value in params.items():
            if key == "met" and value > 1.3:
                warnings.warn(
                    "The ankle draft model is only valid for met <= 1.3",
                    UserWarning,
                )
            if key == "clo" and value > 0.7:
                warnings.warn(
                    "The ankle draft model is only valid for clo <= 0.7",
                    UserWarning,
                )

    elif params["standard"] == "ashrae":  # based on table 7.3.4 ashrae 55 2020
        for key, value in params.items():
            if key in ["tdb", "tr"]:
                if key == "tdb":
                    parameter = "dry-bulb"
                else:
                    parameter = "mean radiant"
                if value > 40 or value < 10:
                    warnings.warn(
                        f"ASHRAE {parameter} temperature applicability limits between"
                        " 10 and 40 °C",
                        UserWarning,
                    )
            if key in ["v", "vr"] and (value > 2 or value < 0):
                warnings.warn(
                    "ASHRAE air speed applicability limits between 0 and 2 m/s",
                    UserWarning,
                )
            if key == "met" and (value > 4 or value < 1):
                warnings.warn(
                    "ASHRAE met applicability limits between 1.0 and 4.0 met",
                    UserWarning,
                )
            if key == "clo" and (value > 1.5 or value < 0):
                warnings.warn(
                    "ASHRAE clo applicability limits between 0.0 and 1.5 clo",
                    UserWarning,
                )
            if key == "v_limited" and value > 0.2:
                raise ValueError(
                    "This equation is only applicable for air speed lower than 0.2 m/s"
                )

    elif params["standard"] == "iso":  # based on ISO 7730:2005 page 3
        for key, value in params.items():
            if key == "tdb" and (value > 30 or value < 10):
                warnings.warn(
                    "ISO air temperature applicability limits between 10 and 30 °C",
                    UserWarning,
                )
            if key == "tr" and (value > 40 or value < 10):
                warnings.warn(
                    "ISO mean radiant temperature applicability limits between 10 and"
                    " 40 °C",
                    UserWarning,
                )
            if key in ["v", "vr"] and (value > 1 or value < 0):
                warnings.warn(
                    "ISO air speed applicability limits between 0 and 1 m/s",
                    UserWarning,
                )
            if key == "met" and (value > 4 or value < 0.8):
                warnings.warn(
                    "ISO met applicability limits between 0.8 and 4.0 met",
                    UserWarning,
                )
            if key == "clo" and (value > 2 or value < 0):
                warnings.warn(
                    "ISO clo applicability limits between 0.0 and 2 clo",
                    UserWarning,
                )

    elif params["standard"] == "ISO7933":  # based on ISO 7933:2004 Annex A
        if params["tdb"] > 50 or params["tdb"] < 15:
            warnings.warn(
                "ISO 7933:2004 air temperature applicability limits between 15 and"
                " 50 °C",
                UserWarning,
            )
        p_a = p_sat(params["tdb"]) / 1000 * params["rh"] / 100
        rh_max = 4.5 * 100 * 1000 / p_sat(params["tdb"])
        if p_a > 4.5 or p_a < 0:
            warnings.warn(
                f"ISO 7933:2004 rh applicability limits between 0 and {rh_max} %",
                UserWarning,
            )
        if params["tr"] - params["tdb"] > 60 or params["tr"] - params["tdb"] < 0:
            warnings.warn(
                "ISO 7933:2004 t_r - t_db applicability limits between 0 and 60 °C",
                UserWarning,
            )
        if params["v"] > 3 or params["v"] < 0:
            warnings.warn(
                "ISO 7933:2004 air speed applicability limits between 0 and 3 m/s",
                UserWarning,
            )
        if params["met"] > 450 or params["met"] < 100:
            warnings.warn(
                "ISO 7933:2004 met applicability limits between 100 and 450 met",
                UserWarning,
            )
        if params["clo"] > 1 or params["clo"] < 0.1:
            warnings.warn(
                "ISO 7933:2004 clo applicability limits between 0.1 and 1 clo",
                UserWarning,
            )


def check_standard_compliance_array(standard, **kwargs):
    default_kwargs = {"airspeed_control": True}
    params = {**default_kwargs, **kwargs}

    if standard == "ashrae":  # based on table 7.3.4 ashrae 55 2020
        tdb_valid = valid_range(params["tdb"], (10.0, 40.0))
        tr_valid = valid_range(params["tr"], (10.0, 40.0))
        v_valid = valid_range(params["v"], (0.0, 2.0))

        if not params["airspeed_control"]:
            v_valid = np.where(
                (params["v"] > 0.8) & (params["clo"] < 0.7) & (params["met"] < 1.3),
                np.nan,
                v_valid,
            )
            to = t_o(params["tdb"], params["tr"], params["v"])
            v_limit = 50.49 - 4.4047 * to + 0.096425 * to * to
            v_valid = np.where(
                (23 < to)
                & (to < 25.5)
                & (params["v"] > v_limit)
                & (params["clo"] < 0.7)
                & (params["met"] < 1.3),
                np.nan,
                v_valid,
            )
            v_valid = np.where(
                (to <= 23)
                & (params["v"] > 0.2)
                & (params["clo"] < 0.7)
                & (params["met"] < 1.3),
                np.nan,
                v_valid,
            )

        if "met" in params.keys():
            met_valid = valid_range(params["met"], (1.0, 4.0))
            clo_valid = valid_range(params["clo"], (0.0, 1.5))

            return tdb_valid, tr_valid, v_valid, met_valid, clo_valid

        else:
            return tdb_valid, tr_valid, v_valid

    if standard == "fan_heatwaves":
        tdb_valid = valid_range(params["tdb"], (20.0, 50.0))
        tr_valid = valid_range(params["tr"], (20.0, 50.0))
        v_valid = valid_range(params["v"], (0.1, 4.5))
        rh_valid = valid_range(params["rh"], (0, 100))
        met_valid = valid_range(params["met"], (0.7, 2))
        clo_valid = valid_range(params["clo"], (0.0, 1))

        return tdb_valid, tr_valid, v_valid, rh_valid, met_valid, clo_valid

    if standard == "iso":  # based on ISO 7730:2005 page 3
        tdb_valid = valid_range(params["tdb"], (10.0, 30.0))
        tr_valid = valid_range(params["tr"], (10.0, 40.0))
        v_valid = valid_range(params["v"], (0.0, 1.0))
        met_valid = valid_range(params["met"], (0.8, 4.0))
        clo_valid = valid_range(params["clo"], (0.0, 2))

        return tdb_valid, tr_valid, v_valid, met_valid, clo_valid


def body_surface_area(weight, height, formula="dubois"):
    """Returns the body surface area in square meters.

    Parameters
    ----------
    weight : float
        body weight, [kg]
    height : float
        height, [m]
    formula : str, optional,
        formula used to calculate the body surface area. default="dubois"
        Choose a name from "dubois", "takahira", "fujimoto", or "kurazumi".

    Returns
    -------
    body_surface_area : float
        body surface area, [m2]
    """

    if formula == "dubois":
        return 0.202 * (weight**0.425) * (height**0.725)
    elif formula == "takahira":
        return 0.2042 * (weight**0.425) * (height**0.725)
    elif formula == "fujimoto":
        return 0.1882 * (weight**0.444) * (height**0.663)
    elif formula == "kurazumi":
        return 0.2440 * (weight**0.383) * (height**0.693)

def f_svv(w, h, d):
    """Calculates the sky-vault view fraction.

    Parameters
    ----------
    w : float
        width of the window, [m]
    h : float
        height of the window, [m]
    d : float
        distance between the occupant and the window, [m]

    Returns
    -------
    f_svv  : float
        sky-vault view fraction ranges between 0 and 1
    """

    return (
        math.degrees(math.atan(h / (2 * d)))
        * math.degrees(math.atan(w / (2 * d)))
        / 16200
    )


def v_relative(v, met):
    """Estimates the relative air speed which combines the average air speed of
    the space plus the relative air speed caused by the body movement. Vag is assumed to
    be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
    Vag = 0.3 (M – 1) (m/s)

    Parameters
    ----------
    v : float or array-like
        air speed measured by the sensor, [m/s]
    met : float
        metabolic rate, [met]

    Returns
    -------
    vr  : float or array-like
        relative air speed, [m/s]
    """

    return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)


def clo_dynamic(clo, met, standard="ASHRAE"):
    """Estimates the dynamic clothing insulation of a moving occupant. The activity as
    well as the air speed modify the insulation characteristics of the clothing and the
    adjacent air layer. Consequently, the ISO 7730 states that the clothing insulation
    shall be corrected [2]_. The ASHRAE 55 Standard corrects for the effect
    of the body movement for met equal or higher than 1.2 met using the equation
    clo = Icl × (0.6 + 0.4/met)

    Parameters
    ----------
    clo : float or array-like
        clothing insulation, [clo]
    met : float or array-like
        metabolic rate, [met]
    standard: str (default="ASHRAE")
        - If "ASHRAE", uses Equation provided in Section 5.2.2.2 of ASHRAE 55 2020

    Returns
    -------
    clo : float or array-like
        dynamic clothing insulation, [clo]
    """

    standard = standard.lower()

    if standard not in ["ashrae", "iso"]:
        raise ValueError(
            "only the ISO 7730 and ASHRAE 55 2020 models have been implemented"
        )

    if standard == "ashrae":
        return np.where(met > 1.2, np.around(clo * (0.6 + 0.4 / met), 3), clo)
    else:
        return np.where(met > 1, np.around(clo * (0.6 + 0.4 / met), 3), clo)


def running_mean_outdoor_temperature(temp_array, alpha=0.8, units="SI"):
    """Estimates the running mean temperature also known as prevailing mean
    outdoor temperature.

    Parameters
    ----------
    temp_array: list
        array containing the mean daily temperature in descending order (i.e. from
        newest/yesterday to oldest) :math:`[t_{day-1}, t_{day-2}, ... ,
        t_{day-n}]`.
        Where :math:`t_{day-1}` is yesterday's daily mean temperature. The EN
        16798-1 2019 [3]_ states that n should be equal to 7
    alpha : float
        constant between 0 and 1. The EN 16798-1 2019 [3]_ recommends a value of 0.8,
        while the ASHRAE 55 2020 recommends to choose values between 0.9 and 0.6,
        corresponding to a slow- and fast- response running mean, respectively.
        Adaptive comfort theory suggests that a slow-response running mean (alpha =
        0.9) could be more appropriate for climates in which synoptic-scale (day-to-
        day) temperature dynamics are relatively minor, such as the humid tropics.
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    t_rm  : float
        running mean outdoor temperature
    """

    if units.lower() == "ip":
        for ix, x in enumerate(temp_array):
            temp_array[ix] = units_converter(tdb=temp_array[ix])[0]

    coeff = [alpha**ix for ix, x in enumerate(temp_array)]
    t_rm = sum([a * b for a, b in zip(coeff, temp_array)]) / sum(coeff)

    if units.lower() == "ip":
        t_rm = units_converter(tmp=t_rm, from_units="si")[0]

    return round(t_rm, 1)


def units_converter(from_units="ip", **kwargs):
    """Converts IP values to SI units.

    Parameters
    ----------
    from_units: str
        specify system to convert from
    **kwargs : [t, v]

    Returns
    -------
    converted values in SI units
    """
    results = list()
    if from_units == "ip":
        for key, value in kwargs.items():
            if "tmp" in key or key == "tr" or key == "tdb":
                results.append((value - 32) * 5 / 9)
            if key in ["v", "vr", "vel"]:
                results.append(value / 3.281)
            if key == "area":
                results.append(value / 10.764)
            if key == "pressure":
                results.append(value * 101325)

    elif from_units == "si":
        for key, value in kwargs.items():
            if "tmp" in key or key == "tr" or key == "tdb":
                results.append((value * 9 / 5) + 32)
            if key in ["v", "vr", "vel"]:
                results.append(value * 3.281)
            if key == "area":
                results.append(value * 10.764)
            if key == "pressure":
                results.append(value / 101325)

    return results


def mapping(value, map_dictionary, right=True):
    """Maps a temperature array to stress categories.

    Parameters
    ----------
    value : float, array-like
        Temperature to map.
    map_dictionary: dict
        Dictionary used to map the values
    right: bool, optional
        Indicating whether the intervals include the right or the left bin edge.

    Returns
    -------
    Stress category for each input temperature.
    """

    bins = np.array(list(map_dictionary.keys()))
    words = np.append(np.array(list(map_dictionary.values())), "unknown")
    return words[np.digitize(value, bins, right=right)]


#: Met values of typical tasks.
met_typical_tasks = {
    "Sleeping": 0.7,
    "Reclining": 0.8,
    "Seated, quiet": 1.0,
    "Reading, seated": 1.0,
    "Writing": 1.0,
    "Typing": 1.1,
    "Standing, relaxed": 1.2,
    "Filing, seated": 1.2,
    "Flying aircraft, routine": 1.2,
    "Filing, standing": 1.4,
    "Driving a car": 1.5,
    "Walking about": 1.7,
    "Cooking": 1.8,
    "Table sawing": 1.8,
    "Walking 2mph (3.2kmh)": 2.0,
    "Lifting/packing": 2.1,
    "Seated, heavy limb movement": 2.2,
    "Light machine work": 2.2,
    "Flying aircraft, combat": 2.4,
    "Walking 3mph (4.8kmh)": 2.6,
    "House cleaning": 2.7,
    "Driving, heavy vehicle": 3.2,
    "Dancing": 3.4,
    "Calisthenics": 3.5,
    "Walking 4mph (6.4kmh)": 3.8,
    "Tennis": 3.8,
    "Heavy machine work": 4.0,
    "Handling 100lb (45 kg) bags": 4.0,
    "Pick and shovel work": 4.4,
    "Basketball": 6.3,
    "Wrestling": 7.8,
}

#: Total clothing insulation of typical ensembles.
clo_typical_ensembles = {
    "Walking shorts, short-sleeve shirt": 0.36,
    "Typical summer indoor clothing": 0.5,
    "Knee-length skirt, short-sleeve shirt, sandals, underwear": 0.54,
    "Trousers, short-sleeve shirt, socks, shoes, underwear": 0.57,
    "Trousers, long-sleeve shirt": 0.61,
    "Knee-length skirt, long-sleeve shirt, full slip": 0.67,
    "Sweat pants, long-sleeve sweatshirt": 0.74,
    "Jacket, Trousers, long-sleeve shirt": 0.96,
    "Typical winter indoor clothing": 1.0,
}

# Clo values of individual clothing elements. To calculate the total clothing insulation you need to add these values together.
clo_individual_garments = {
    "Metal chair": 0.00,
    "Bra": 0.01,
    "Wooden stool": 0.01,
    "Ankle socks": 0.02,
    "Shoes or sandals": 0.02,
    "Slippers": 0.03,
    "Panty hose": 0.02,
    "Calf length socks": 0.03,
    "Women's underwear": 0.03,
    "Men's underwear": 0.04,
    "Knee socks (thick)": 0.06,
    "Short shorts": 0.06,
    "Walking shorts": 0.08,
    "T-shirt": 0.08,
    "Standard office chair": 0.10,
    "Executive chair": 0.15,
    "Boots": 0.1,
    "Sleeveless scoop-neck blouse": 0.12,
    "Half slip": 0.14,
    "Long underwear bottoms": 0.15,
    "Full slip": 0.16,
    "Short-sleeve knit shirt": 0.17,
    "Sleeveless vest (thin)": 0.1,
    "Sleeveless vest (thick)": 0.17,
    "Sleeveless short gown (thin)": 0.18,
    "Short-sleeve dress shirt": 0.19,
    "Sleeveless long gown (thin)": 0.2,
    "Long underwear top": 0.2,
    "Thick skirt": 0.23,
    "Long-sleeve dress shirt": 0.25,
    "Long-sleeve flannel shirt": 0.34,
    "Long-sleeve sweat shirt": 0.34,
    "Short-sleeve hospital gown": 0.31,
    "Short-sleeve short robe (thin)": 0.34,
    "Short-sleeve pajamas": 0.42,
    "Long-sleeve long gown": 0.46,
    "Long-sleeve short wrap robe (thick)": 0.48,
    "Long-sleeve pajamas (thick)": 0.57,
    "Long-sleeve long wrap robe (thick)": 0.69,
    "Thin trousers": 0.15,
    "Thick trousers": 0.24,
    "Sweatpants": 0.28,
    "Overalls": 0.30,
    "Coveralls": 0.49,
    "Thin skirt": 0.14,
    "Long-sleeve shirt dress (thin)": 0.33,
    "Long-sleeve shirt dress (thick)": 0.47,
    "Short-sleeve shirt dress": 0.29,
    "Sleeveless, scoop-neck shirt (thin)": 0.23,
    "Sleeveless, scoop-neck shirt (thick)": 0.27,
    "Long sleeve shirt (thin)": 0.25,
    "Long sleeve shirt (thick)": 0.36,
    "Single-breasted coat (thin)": 0.36,
    "Single-breasted coat (thick)": 0.44,
    "Double-breasted coat (thin)": 0.42,
    "Double-breasted coat (thick)": 0.48,
}

# This dictionary contains the reflection coefficients, Fr, for different special materials
f_r_garments = {
    "Cotton with aluminium paint": 0.42,
    "Viscose with glossy aluminium foil": 0.19,
    "Aramid (Kevlar) with glossy aluminium foil": 0.14,
    "Wool with glossy aluminium foil": 0.12,
    "Cotton with glossy aluminium foil": 0.04,
    "Viscose vacuum metallized with aluminium": 0.06,
    "Aramid vacuum metallized with aluminium": 0.04,
    "Wool vacuum metallized with aluminium": 0.05,
    "Cotton vacuum metallized with aluminium": 0.05,
    "Glass fiber vacuum metallized with aluminium": 0.07,
}

# This dictionary contains the local and the whole body clothing insulation of typical clothing ensemble.
# It is based on the study by Juyoun et al. (https://escholarship.org/uc/item/18f0r375)
# and by Nomoto et al. (https://doi.org/10.1002/2475-8876.12124)
# Please note that value for the neck is the same as the measured value for the head
# and it does not take into account the insulation effect of the hair.
# Typically, the clothing insulation for the hair are quantified by assuming a head covering of approximately 0.6 to 1.0 clo.
local_clo_typical_ensembles =  {
    "nude (mesh chair)": {
        "whole_body": 0.01,
        "local_body_part": {
            "head": 0.13,
            "neck": 0.13,
            "chest": 0.01,
            "back": 0.01,
            "pelvis": 0.04,
            "left_shoulder": 0.02,
            "left_arm": 0.0,
            "left_hand": 0.01,
            "right_shoulder": 0.02,
            "right_arm": 0.0,
            "right_hand": 0.01,
            "left_thigh": 0.01,
            "left_leg": 0.03,
            "left_foot": 0.05,
            "right_thigh": 0.01,
            "right_leg": 0.03,
            "right_foot": 0.05,
        },
    },
    "nude (nude chair)": {
        "whole_body": -0.02,
        "local_body_part": {
            "head": 0.13,
            "neck": 0.13,
            "chest": 0.05,
            "back": -0.14,
            "pelvis": -0.01,
            "left_shoulder": -0.01,
            "left_arm": -0.01,
            "left_hand": -0.02,
            "right_shoulder": -0.01,
            "right_arm": -0.01,
            "right_hand": -0.02,
            "left_thigh": -0.1,
            "left_leg": 0.0,
            "left_foot": 0.0,
            "right_thigh": -0.1,
            "right_leg": 0.0,
            "right_foot": 0.0,
        },
    },
    "panty": {
        "whole_body": 0.03,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.0,
            "back": 0.0,
            "pelvis": 0.24,
            "left_shoulder": 0.0,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.0,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.05,
            "left_leg": 0.0,
            "left_foot": 0.05,
            "right_thigh": 0.05,
            "right_leg": 0.0,
            "right_foot": 0.05,
        },
    },
    "bra+panty": {
        "whole_body": 0.05,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.22,
            "back": 0.0,
            "pelvis": 0.18,
            "left_shoulder": 0.0,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.0,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.03,
            "left_leg": 0.03,
            "left_foot": 0.08,
            "right_thigh": 0.03,
            "right_leg": 0.03,
            "right_foot": 0.08,
        },
    },
    "bra+panty, tanktop, shorts, sandals": {
        "whole_body": 0.22,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.57,
            "back": 0.27,
            "pelvis": 0.92,
            "left_shoulder": 0.04,
            "left_arm": 0.02,
            "left_hand": 0.02,
            "right_shoulder": 0.04,
            "right_arm": 0.02,
            "right_hand": 0.02,
            "left_thigh": 0.51,
            "left_leg": 0.01,
            "left_foot": 0.38,
            "right_thigh": 0.51,
            "right_leg": 0.01,
            "right_foot": 0.38,
        },
    },
    "bra+panty, long-sleeve shirt, shorts, sandals": {
        "whole_body": 0.43,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.43,
            "back": 1.02,
            "pelvis": 1.45,
            "left_shoulder": 0.29,
            "left_arm": 0.22,
            "left_hand": 0.01,
            "right_shoulder": 0.29,
            "right_arm": 0.22,
            "right_hand": 0.01,
            "left_thigh": 0.57,
            "left_leg": 0.01,
            "left_foot": 0.4,
            "right_thigh": 0.57,
            "right_leg": 0.01,
            "right_foot": 0.4,
        },
    },
    "bra+panty, sleeveless dress, sandals": {
        "whole_body": 0.29,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.85,
            "back": 0.48,
            "pelvis": 0.94,
            "left_shoulder": 0.0,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.0,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.72,
            "left_leg": 0.0,
            "left_foot": 0.41,
            "right_thigh": 0.72,
            "right_leg": 0.0,
            "right_foot": 0.41,
        },
    },
    "bra+panty, T-shirt, long pants, socks, sneakers": {
        "whole_body": 0.52,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.14,
            "back": 0.84,
            "pelvis": 1.04,
            "left_shoulder": 0.42,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.42,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.58,
            "left_leg": 0.62,
            "left_foot": 0.82,
            "right_thigh": 0.58,
            "right_leg": 0.62,
            "right_foot": 0.82,
        },
    },
    "bra+panty, sleeveless dress, cardigan, sandals": {
        "whole_body": 0.53,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.78,
            "back": 1.42,
            "pelvis": 1.19,
            "left_shoulder": 0.65,
            "left_arm": 0.41,
            "left_hand": 0.05,
            "right_shoulder": 0.65,
            "right_arm": 0.41,
            "right_hand": 0.05,
            "left_thigh": 0.77,
            "left_leg": 0.0,
            "left_foot": 0.39,
            "right_thigh": 0.77,
            "right_leg": 0.0,
            "right_foot": 0.39,
        },
    },
    "bra+panty, song-sleeve dress, socks, sneakers": {
        "whole_body": 0.54,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.49,
            "back": 1.1,
            "pelvis": 0.91,
            "left_shoulder": 0.72,
            "left_arm": 0.58,
            "left_hand": 0.03,
            "right_shoulder": 0.72,
            "right_arm": 0.58,
            "right_hand": 0.03,
            "left_thigh": 0.73,
            "left_leg": 0.07,
            "left_foot": 0.77,
            "right_thigh": 0.73,
            "right_leg": 0.07,
            "right_foot": 0.77,
        },
    },
    "bra+panty, long-sleeve dress, cardigan, socks, sneakers": {
        "whole_body": 0.67,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 2.05,
            "back": 1.32,
            "pelvis": 1.39,
            "left_shoulder": 1.14,
            "left_arm": 0.63,
            "left_hand": 0.04,
            "right_shoulder": 1.14,
            "right_arm": 0.63,
            "right_hand": 0.04,
            "left_thigh": 0.84,
            "left_leg": 0.05,
            "left_foot": 0.78,
            "right_thigh": 0.84,
            "right_leg": 0.05,
            "right_foot": 0.78,
        },
    },
    "bra+panty, tank top, skirt, sandals": {
        "whole_body": 0.31,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.83,
            "back": 0.22,
            "pelvis": 0.99,
            "left_shoulder": 0.0,
            "left_arm": 0.0,
            "left_hand": 0.03,
            "right_shoulder": 0.0,
            "right_arm": 0.0,
            "right_hand": 0.03,
            "left_thigh": 0.88,
            "left_leg": 0.05,
            "left_foot": 0.44,
            "right_thigh": 0.88,
            "right_leg": 0.05,
            "right_foot": 0.44,
        },
    },
    "bra+panty, long sleeve shirts, skirt, sandals": {
        "whole_body": 0.52,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.62,
            "back": 0.99,
            "pelvis": 1.41,
            "left_shoulder": 0.31,
            "left_arm": 0.28,
            "left_hand": 0.03,
            "right_shoulder": 0.31,
            "right_arm": 0.28,
            "right_hand": 0.03,
            "left_thigh": 0.82,
            "left_leg": 0.04,
            "left_foot": 0.41,
            "right_thigh": 0.82,
            "right_leg": 0.04,
            "right_foot": 0.41,
        },
    },
    "bra+panty, dress shirts, skirt, stocking, formal shoes": {
        "whole_body": 0.62,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.58,
            "back": 0.99,
            "pelvis": 1.31,
            "left_shoulder": 0.91,
            "left_arm": 0.64,
            "left_hand": 0.04,
            "right_shoulder": 0.91,
            "right_arm": 0.64,
            "right_hand": 0.04,
            "left_thigh": 0.87,
            "left_leg": 0.05,
            "left_foot": 0.81,
            "right_thigh": 0.87,
            "right_leg": 0.05,
            "right_foot": 0.81,
        },
    },
    "bra+panty, dress shirts, skirt, leggings, sandals": {
        "whole_body": 0.65,
        "local_body_part": {
            "head": 0.13,
            "neck": 0.13,
            "chest": 1.59,
            "back": 1.04,
            "pelvis": 1.36,
            "left_shoulder": 0.91,
            "left_arm": 0.67,
            "left_hand": 0.07,
            "right_shoulder": 0.91,
            "right_arm": 0.67,
            "right_hand": 0.07,
            "left_thigh": 1.26,
            "left_leg": 0.12,
            "left_foot": 0.43,
            "right_thigh": 1.26,
            "right_leg": 0.12,
            "right_foot": 0.43,
        },
    },
    "bra+panty, thin dress shirts, long pants, socks, sneakers": {
        "whole_body": 0.82,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 3.35,
            "back": 1.73,
            "pelvis": 1.63,
            "left_shoulder": 1.99,
            "left_arm": 1.49,
            "left_hand": 0.11,
            "right_shoulder": 1.99,
            "right_arm": 1.49,
            "right_hand": 0.11,
            "left_thigh": 0.6,
            "left_leg": 0.43,
            "left_foot": 0.68,
            "right_thigh": 0.6,
            "right_leg": 0.43,
            "right_foot": 0.68,
        },
    },
    "bra+panty, long sleeve shirts, long pants, socks, sneakers": {
        "whole_body": 0.8,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 2.47,
            "back": 1.48,
            "pelvis": 1.58,
            "left_shoulder": 0.98,
            "left_arm": 0.58,
            "left_hand": 0.04,
            "right_shoulder": 0.98,
            "right_arm": 0.58,
            "right_hand": 0.04,
            "left_thigh": 0.69,
            "left_leg": 0.65,
            "left_foot": 0.89,
            "right_thigh": 0.69,
            "right_leg": 0.65,
            "right_foot": 0.89,
        },
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, socks, sneakers": {
        "whole_body": 0.83,
        "local_body_part": {
            "head": 0.25,
            "neck": 0.25,
            "chest": 3.88,
            "back": 2.28,
            "pelvis": 2.07,
            "left_shoulder": 1.89,
            "left_arm": 1.41,
            "left_hand": 0.16,
            "right_shoulder": 1.89,
            "right_arm": 1.41,
            "right_hand": 0.16,
            "left_thigh": 0.83,
            "left_leg": 0.66,
            "left_foot": 0.86,
            "right_thigh": 0.83,
            "right_leg": 0.66,
            "right_foot": 0.86,
        },
    },
    "bra+panty, T-shirt, jeans, socks, sneakers": {
        "whole_body": 0.57,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.29,
            "back": 0.93,
            "pelvis": 1.3,
            "left_shoulder": 0.68,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.68,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.65,
            "left_leg": 0.47,
            "left_foot": 0.73,
            "right_thigh": 0.65,
            "right_leg": 0.47,
            "right_foot": 0.73,
        },
    },
    "bra+panty, long sleeve shirts, jeans, socks, sneakers": {
        "whole_body": 0.74,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.58,
            "back": 0.98,
            "pelvis": 1.35,
            "left_shoulder": 0.86,
            "left_arm": 0.71,
            "left_hand": 0.07,
            "right_shoulder": 0.86,
            "right_arm": 0.71,
            "right_hand": 0.07,
            "left_thigh": 0.74,
            "left_leg": 0.48,
            "left_foot": 0.74,
            "right_thigh": 0.74,
            "right_leg": 0.48,
            "right_foot": 0.74,
        },
    },
    "bra+panty, oxford shirts, long thin pants, socks, sneakers": {
        "whole_body": 0.83,
        "local_body_part": {
            "head": 0.16,
            "neck": 0.16,
            "chest": 1.39,
            "back": 1.02,
            "pelvis": 1.34,
            "left_shoulder": 0.83,
            "left_arm": 0.69,
            "left_hand": 0.22,
            "right_shoulder": 0.83,
            "right_arm": 0.69,
            "right_hand": 0.22,
            "left_thigh": 1.02,
            "left_leg": 0.68,
            "left_foot": 0.8,
            "right_thigh": 1.02,
            "right_leg": 0.68,
            "right_foot": 0.8,
        },
    },
    "bra+panty, thin dress shirts (roll-up), long pants, socks, sneakers": {
        "whole_body": 0.81,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 3.6,
            "back": 1.83,
            "pelvis": 1.71,
            "left_shoulder": 2.16,
            "left_arm": 1.49,
            "left_hand": 0.13,
            "right_shoulder": 2.16,
            "right_arm": 1.49,
            "right_hand": 0.13,
            "left_thigh": 0.64,
            "left_leg": 0.43,
            "left_foot": 0.69,
            "right_thigh": 0.64,
            "right_leg": 0.43,
            "right_foot": 0.69,
        },
    },
    "bra+panty, T-shirt, short sleeve shirt, long pants, socks, sneakers": {
        "whole_body": 0.71,
        "local_body_part": {
            "head": 0.12,
            "neck": 0.12,
            "chest": 2.15,
            "back": 1.4,
            "pelvis": 1.71,
            "left_shoulder": 1.22,
            "left_arm": 0.02,
            "left_hand": 0.05,
            "right_shoulder": 1.22,
            "right_arm": 0.02,
            "right_hand": 0.05,
            "left_thigh": 0.79,
            "left_leg": 0.48,
            "left_foot": 0.67,
            "right_thigh": 0.79,
            "right_leg": 0.48,
            "right_foot": 0.67,
        },
    },
    "bra+panty, sports shirts, long pants, socks, sneakers": {
        "whole_body": 0.8,
        "local_body_part": {
            "head": 0.05,
            "neck": 0.05,
            "chest": 1.92,
            "back": 1.31,
            "pelvis": 1.41,
            "left_shoulder": 1.14,
            "left_arm": 0.86,
            "left_hand": 0.18,
            "right_shoulder": 1.14,
            "right_arm": 0.86,
            "right_hand": 0.18,
            "left_thigh": 0.59,
            "left_leg": 0.49,
            "left_foot": 0.75,
            "right_thigh": 0.59,
            "right_leg": 0.49,
            "right_foot": 0.75,
        },
    },
    "bra+panty, sports shirts, sports pants, sports socks, sports shoes": {
        "whole_body": 0.87,
        "local_body_part": {
            "head": 0.07,
            "neck": 0.07,
            "chest": 1.87,
            "back": 1.17,
            "pelvis": 1.26,
            "left_shoulder": 1.2,
            "left_arm": 1.07,
            "left_hand": 0.09,
            "right_shoulder": 1.2,
            "right_arm": 1.07,
            "right_hand": 0.09,
            "left_thigh": 0.62,
            "left_leg": 0.77,
            "left_foot": 1.58,
            "right_thigh": 0.62,
            "right_leg": 0.77,
            "right_foot": 1.58,
        },
    },
    "bra+panty, thin dress shirts, long pants, wool sweater, socks, sneakers": {
        "whole_body": 0.92,
        "local_body_part": {
            "head": 0.09,
            "neck": 0.09,
            "chest": 2.39,
            "back": 1.64,
            "pelvis": 1.71,
            "left_shoulder": 1.36,
            "left_arm": 1.29,
            "left_hand": 0.21,
            "right_shoulder": 1.36,
            "right_arm": 1.29,
            "right_hand": 0.21,
            "left_thigh": 0.7,
            "left_leg": 0.52,
            "left_foot": 0.77,
            "right_thigh": 0.7,
            "right_leg": 0.52,
            "right_foot": 0.77,
        },
    },
    "bra+panty, thin dress shirts, long pants, cashmere sweater, socks, sneakers": {
        "whole_body": 0.87,
        "local_body_part": {
            "head": 0.1,
            "neck": 0.1,
            "chest": 2.4,
            "back": 1.72,
            "pelvis": 1.67,
            "left_shoulder": 1.33,
            "left_arm": 1.23,
            "left_hand": 0.08,
            "right_shoulder": 1.33,
            "right_arm": 1.23,
            "right_hand": 0.08,
            "left_thigh": 0.61,
            "left_leg": 0.47,
            "left_foot": 0.77,
            "right_thigh": 0.61,
            "right_leg": 0.47,
            "right_foot": 0.77,
        },
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket, socks, sneakers": {
        "whole_body": 1.18,
        "local_body_part": {
            "head": 0.65,
            "neck": 0.65,
            "chest": 5.26,
            "back": 3.07,
            "pelvis": 2.2,
            "left_shoulder": 3.14,
            "left_arm": 2.07,
            "left_hand": 0.08,
            "right_shoulder": 3.14,
            "right_arm": 2.07,
            "right_hand": 0.08,
            "left_thigh": 0.67,
            "left_leg": 0.54,
            "left_foot": 0.77,
            "right_thigh": 0.67,
            "right_leg": 0.54,
            "right_foot": 0.77,
        },
    },
    "bra+panty, T-shirt, long sleeve shirts, jeans, sports jumper, socks, sneakers": {
        "whole_body": 1.07,
        "local_body_part": {
            "head": 0.28,
            "neck": 0.28,
            "chest": 3.99,
            "back": 2.12,
            "pelvis": 2.0,
            "left_shoulder": 1.7,
            "left_arm": 1.36,
            "left_hand": 0.1,
            "right_shoulder": 1.7,
            "right_arm": 1.36,
            "right_hand": 0.1,
            "left_thigh": 0.92,
            "left_leg": 0.48,
            "left_foot": 1.07,
            "right_thigh": 0.92,
            "right_leg": 0.48,
            "right_foot": 1.07,
        },
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, ventura jacket, socks, sneakers": {
        "whole_body": 0.9,
        "local_body_part": {
            "head": 0.09,
            "neck": 0.09,
            "chest": 2.66,
            "back": 1.42,
            "pelvis": 1.57,
            "left_shoulder": 1.32,
            "left_arm": 0.99,
            "left_hand": 0.14,
            "right_shoulder": 1.32,
            "right_arm": 0.99,
            "right_hand": 0.14,
            "left_thigh": 0.73,
            "left_leg": 0.66,
            "left_foot": 0.85,
            "right_thigh": 0.73,
            "right_leg": 0.66,
            "right_foot": 0.85,
        },
    },
    "bra+panty, turtle neck, long pants, short trench coat, socks, sneakers": {
        "whole_body": 1.24,
        "local_body_part": {
            "head": 0.06,
            "neck": 0.06,
            "chest": 3.22,
            "back": 1.99,
            "pelvis": 2.03,
            "left_shoulder": 1.62,
            "left_arm": 1.5,
            "left_hand": 0.37,
            "right_shoulder": 1.62,
            "right_arm": 1.5,
            "right_hand": 0.37,
            "left_thigh": 1.51,
            "left_leg": 0.65,
            "left_foot": 0.8,
            "right_thigh": 1.51,
            "right_leg": 0.65,
            "right_foot": 0.8,
        },
    },
    "bra+panty, tank top, long sleeve shirts, blazer, skirt, sandals": {
        "whole_body": 0.86,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 3.24,
            "back": 1.81,
            "pelvis": 2.06,
            "left_shoulder": 1.98,
            "left_arm": 1.13,
            "left_hand": 0.07,
            "right_shoulder": 1.98,
            "right_arm": 1.13,
            "right_hand": 0.07,
            "left_thigh": 1.19,
            "left_leg": 0.04,
            "left_foot": 0.44,
            "right_thigh": 1.19,
            "right_leg": 0.04,
            "right_foot": 0.44,
        },
    },
    "bra+panty, long sleeve shirts, wool skirt, socks, formal shoes": {
        "whole_body": 0.59,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.21,
            "back": 0.74,
            "pelvis": 1.56,
            "left_shoulder": 0.44,
            "left_arm": 0.24,
            "left_hand": 0.17,
            "right_shoulder": 0.44,
            "right_arm": 0.24,
            "right_hand": 0.17,
            "left_thigh": 1.52,
            "left_leg": 0.09,
            "left_foot": 0.74,
            "right_thigh": 1.52,
            "right_leg": 0.09,
            "right_foot": 0.74,
        },
    },
    "bra+panty, turtleneck, wool skirt, socks, formal shoes": {
        "whole_body": 0.7,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.11,
            "back": 0.94,
            "pelvis": 1.52,
            "left_shoulder": 0.73,
            "left_arm": 0.62,
            "left_hand": 0.14,
            "right_shoulder": 0.73,
            "right_arm": 0.62,
            "right_hand": 0.14,
            "left_thigh": 1.53,
            "left_leg": 0.09,
            "left_foot": 0.85,
            "right_thigh": 1.53,
            "right_leg": 0.09,
            "right_foot": 0.85,
        },
    },
    "bra+panty, long sleeve shirt, wool skirt, sweater, socks, formal shoes": {
        "whole_body": 0.91,
        "local_body_part": {
            "head": 0.14,
            "neck": 0.14,
            "chest": 2.82,
            "back": 1.53,
            "pelvis": 1.79,
            "left_shoulder": 1.22,
            "left_arm": 0.97,
            "left_hand": 0.08,
            "right_shoulder": 1.22,
            "right_arm": 0.97,
            "right_hand": 0.08,
            "left_thigh": 1.53,
            "left_leg": 0.11,
            "left_foot": 0.83,
            "right_thigh": 1.53,
            "right_leg": 0.11,
            "right_foot": 0.83,
        },
    },
    "bra+panty, thin dress shirts, slacks, tie, socks, sneakers": {
        "whole_body": 0.57,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.69,
            "back": 0.8,
            "pelvis": 1.08,
            "left_shoulder": 0.67,
            "left_arm": 0.58,
            "left_hand": 0.07,
            "right_shoulder": 0.67,
            "right_arm": 0.58,
            "right_hand": 0.07,
            "left_thigh": 0.36,
            "left_leg": 0.39,
            "left_foot": 0.74,
            "right_thigh": 0.36,
            "right_leg": 0.39,
            "right_foot": 0.74,
        },
    },
    "bra+panty, thin dress shirts, slacks, blazer, tie, belt, socks, formal shoes": {
        "whole_body": 0.93,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 3.6,
            "back": 1.83,
            "pelvis": 1.71,
            "left_shoulder": 2.16,
            "left_arm": 1.49,
            "left_hand": 0.13,
            "right_shoulder": 2.16,
            "right_arm": 1.49,
            "right_hand": 0.13,
            "left_thigh": 0.64,
            "left_leg": 0.43,
            "left_foot": 0.69,
            "right_thigh": 0.64,
            "right_leg": 0.43,
            "right_foot": 0.69,
        },
    },
    "bra+panty, long sleeve shirts, long pants, blazer, socks, sneakers": {
        "whole_body": 0.96,
        "local_body_part": {
            "head": 0.04,
            "neck": 0.04,
            "chest": 3.3,
            "back": 1.67,
            "pelvis": 2.2,
            "left_shoulder": 2.1,
            "left_arm": 1.43,
            "left_hand": 0.09,
            "right_shoulder": 2.1,
            "right_arm": 1.43,
            "right_hand": 0.09,
            "left_thigh": 0.72,
            "left_leg": 0.42,
            "left_foot": 0.67,
            "right_thigh": 0.72,
            "right_leg": 0.42,
            "right_foot": 0.67,
        },
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket (Notica)": {
        "whole_body": 1.05,
        "local_body_part": {
            "head": 0.04,
            "neck": 0.04,
            "chest": 3.88,
            "back": 2.26,
            "pelvis": 1.97,
            "left_shoulder": 1.82,
            "left_arm": 1.46,
            "left_hand": 0.17,
            "right_shoulder": 1.82,
            "right_arm": 1.46,
            "right_hand": 0.17,
            "left_thigh": 0.81,
            "left_leg": 0.57,
            "left_foot": 0.78,
            "right_thigh": 0.81,
            "right_leg": 0.57,
            "right_foot": 0.78,
        },
    },
    "bra+panty, turtle neck, ski-jumper, skin pants, sports socks, sports shoes": {
        "whole_body": 1.84,
        "local_body_part": {
            "head": 0.89,
            "neck": 0.89,
            "chest": 5.24,
            "back": 2.87,
            "pelvis": 2.64,
            "left_shoulder": 2.55,
            "left_arm": 2.16,
            "left_hand": 0.46,
            "right_shoulder": 2.55,
            "right_arm": 2.16,
            "right_hand": 0.46,
            "left_thigh": 1.49,
            "left_leg": 1.82,
            "left_foot": 1.56,
            "right_thigh": 1.49,
            "right_leg": 1.82,
            "right_foot": 1.56,
        },
    },
    "bra+panty, turtle neck, ski-jumper and hood, skin pants, sports socks, sports shoes": {
        "whole_body": 1.87,
        "local_body_part": {
            "head": 1.63,
            "neck": 1.63,
            "chest": 5.12,
            "back": 2.7,
            "pelvis": 2.57,
            "left_shoulder": 2.58,
            "left_arm": 2.16,
            "left_hand": 0.49,
            "right_shoulder": 2.58,
            "right_arm": 2.16,
            "right_hand": 0.49,
            "left_thigh": 1.44,
            "left_leg": 1.76,
            "left_foot": 1.54,
            "right_thigh": 1.44,
            "right_leg": 1.76,
            "right_foot": 1.54,
        },
    },
    "bra+panty, turtle neck, goose down, ski pants, sports socks, sports shoes": {
        "whole_body": 2.53,
        "local_body_part": {
            "head": 1.17,
            "neck": 1.17,
            "chest": 15.44,
            "back": 5.5,
            "pelvis": 5.2,
            "left_shoulder": 6.55,
            "left_arm": 5.58,
            "left_hand": 0.35,
            "right_shoulder": 6.55,
            "right_arm": 5.58,
            "right_hand": 0.35,
            "left_thigh": 2.12,
            "left_leg": 1.7,
            "left_foot": 1.54,
            "right_thigh": 2.12,
            "right_leg": 1.7,
            "right_foot": 1.54,
        },
    },
    "bra+panty, turtle neck, goose down-with hood, ski pants, sports socks, sports shoes": {
        "whole_body": 2.75,
        "local_body_part": {
            "head": 3.52,
            "neck": 3.52,
            "chest": 12.62,
            "back": 3.99,
            "pelvis": 5.05,
            "left_shoulder": 6.2,
            "left_arm": 5.73,
            "left_hand": 0.53,
            "right_shoulder": 6.2,
            "right_arm": 5.73,
            "right_hand": 0.53,
            "left_thigh": 2.11,
            "left_leg": 1.81,
            "left_foot": 1.58,
            "right_thigh": 2.11,
            "right_leg": 1.81,
            "right_foot": 1.58,
        },
    },
    "bra+panty, turtle neck, goose down-with hood and gloves, ski pants, sports socks, sports shoes": {
        "whole_body": 3.27,
        "local_body_part": {
            "head": 3.92,
            "neck": 3.92,
            "chest": 16.13,
            "back": 4.47,
            "pelvis": 5.71,
            "left_shoulder": 7.12,
            "left_arm": 5.37,
            "left_hand": 2.54,
            "right_shoulder": 7.12,
            "right_arm": 5.37,
            "right_hand": 2.54,
            "left_thigh": 2.14,
            "left_leg": 1.82,
            "left_foot": 1.61,
            "right_thigh": 2.14,
            "right_leg": 1.82,
            "right_foot": 1.61,
        },
    },
    "briefs, socks, T-shirt, half pants, sneakers": {
        "whole_body": 0.53,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.5,
            "back": 1.13,
            "pelvis": 1.21,
            "left_shoulder": 0.39,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.39,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.94,
            "left_leg": 0.07,
            "left_foot": 0.62,
            "right_thigh": 0.94,
            "right_leg": 0.07,
            "right_foot": 0.62,
        },
    },
    "briefs, undershirt, sports t-shirts, sports shorts": {
        "whole_body": 0.7,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.9,
            "back": 1.76,
            "pelvis": 2.14,
            "left_shoulder": 0.48,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.48,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 1.18,
            "left_leg": 0.0,
            "left_foot": 0.01,
            "right_thigh": 1.18,
            "right_leg": 0.0,
            "right_foot": 0.01,
        },
    },
    "briefs, socks, polo shirt, long pants, sneakers": {
        "whole_body": 0.54,
        "local_body_part": {
            "head": 0.02,
            "neck": 0.02,
            "chest": 0.52,
            "back": 0.97,
            "pelvis": 1.12,
            "left_shoulder": 0.34,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.34,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.79,
            "left_leg": 0.56,
            "left_foot": 0.65,
            "right_thigh": 0.79,
            "right_leg": 0.56,
            "right_foot": 0.65,
        },
    },
    "briefs, under shirt, long-sleeved shirt, long pants": {
        "whole_body": 1.01,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.2,
            "back": 1.65,
            "pelvis": 2.29,
            "left_shoulder": 0.98,
            "left_arm": 0.78,
            "left_hand": 0.03,
            "right_shoulder": 0.98,
            "right_arm": 0.78,
            "right_hand": 0.03,
            "left_thigh": 1.46,
            "left_leg": 0.62,
            "left_foot": 0.02,
            "right_thigh": 1.46,
            "right_leg": 0.62,
            "right_foot": 0.02,
        },
    },
    "briefs, socks, undershirt, short-sleeved shirt, long pants, belt, shoes": {
        "whole_body": 0.72,
        "local_body_part": {
            "head": 0.01,
            "neck": 0.01,
            "chest": 0.8,
            "back": 1.4,
            "pelvis": 1.55,
            "left_shoulder": 0.54,
            "left_arm": 0.0,
            "left_hand": 0.0,
            "right_shoulder": 0.54,
            "right_arm": 0.0,
            "right_hand": 0.0,
            "left_thigh": 0.89,
            "left_leg": 0.64,
            "left_foot": 0.99,
            "right_thigh": 0.89,
            "right_leg": 0.64,
            "right_foot": 0.99,
        },
    },
    "briefs, socks, undershirt, long-sleeved shirt, long pants, belt": {
        "whole_body": 0.77,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 0.86,
            "back": 1.45,
            "pelvis": 1.54,
            "left_shoulder": 0.82,
            "left_arm": 0.6,
            "left_hand": 0.01,
            "right_shoulder": 0.82,
            "right_arm": 0.6,
            "right_hand": 0.01,
            "left_thigh": 0.9,
            "left_leg": 0.66,
            "left_foot": 0.64,
            "right_thigh": 0.9,
            "right_leg": 0.66,
            "right_foot": 0.64,
        },
    },
    "briefs, socks, undershirt, long-sleeved shirt, jacket, long pants, belt, shoes": {
        "whole_body": 1.39,
        "local_body_part": {
            "head": 0.02,
            "neck": 0.02,
            "chest": 2.13,
            "back": 2.28,
            "pelvis": 3.04,
            "left_shoulder": 1.8,
            "left_arm": 1.54,
            "left_hand": 0.15,
            "right_shoulder": 1.8,
            "right_arm": 1.54,
            "right_hand": 0.15,
            "left_thigh": 1.33,
            "left_leg": 0.69,
            "left_foot": 0.97,
            "right_thigh": 1.33,
            "right_leg": 0.69,
            "right_foot": 0.97,
        },
    },
    "briefs, socks, undershirt, work jacket, work pants, safety shoes": {
        "whole_body": 0.8,
        "local_body_part": {
            "head": 0.0,
            "neck": 0.0,
            "chest": 1.25,
            "back": 1.39,
            "pelvis": 1.78,
            "left_shoulder": 0.84,
            "left_arm": 0.71,
            "left_hand": 0.08,
            "right_shoulder": 0.84,
            "right_arm": 0.71,
            "right_hand": 0.08,
            "left_thigh": 0.65,
            "left_leg": 0.59,
            "left_foot": 1.12,
            "right_thigh": 0.65,
            "right_leg": 0.59,
            "right_foot": 1.12,
        },
    },
}
