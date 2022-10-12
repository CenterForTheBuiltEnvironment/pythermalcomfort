import numpy as np
import warnings
import math
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
                        f"ASHRAE {parameter} temperature applicability limits between 10 and 40 °C",
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

    elif params["standard"] == "fan_heatwaves":  # based on table 7.3.4 ashrae 55 2020
        for key, value in params.items():
            if key in ["tdb", "tr"]:
                if key == "tdb":
                    parameter = "dry-bulb"
                else:
                    parameter = "mean radiant"
                if value > 50 or value < 30:
                    warnings.warn(
                        f"{parameter} temperature applicability limits between 30 and 50 °C",
                        UserWarning,
                    )
            if key in ["v", "vr"] and (value > 4.5 or value < 0.1):
                warnings.warn(
                    "Air speed applicability limits between 0.4 and 4.5 m/s",
                    UserWarning,
                )
            if key == "met" and (value > 2 or value < 0.7):
                warnings.warn(
                    "Met applicability limits between 0.7 and 2.0 met",
                    UserWarning,
                )
            if key == "clo" and (value > 1.0 or value < 0):
                warnings.warn(
                    "Clo applicability limits between 0.0 and 1.0 clo",
                    UserWarning,
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
                    "ISO mean radiant temperature applicability limits between 10 and 40 °C",
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
                "ISO 7933:2004 air temperature applicability limits between 15 and 50 °C",
                UserWarning,
            )
        p_a = p_sat(params["tdb"]) / 1000 * params["rh"] / 100
        rh_max = 4.5 * 100 * 1000 / p_sat(params["tdb"])
        if p_a > 4.5 or p_a < 0:
            warnings.warn(
                f"ISO 7933:2004 t_r - t_db applicability limits between 0 and {rh_max} %",
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
    formula : {"dubois"}, default="dubois"
        formula used to calculate the body surface area

    Returns
    -------
    body_surface_area : float
        body surface area, [m2]
    """

    if formula == "dubois":
        return 0.202 * (weight ** 0.425) * (height ** 0.725)


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
        newest/yesterday to oldest) :math:`[\Theta_{day-1}, \Theta_{day-2}, \dots ,
        \Theta_{day-n}]`.
        Where :math:`\Theta_{day-1}` is yesterday's daily mean temperature. The EN
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

    coeff = [alpha ** ix for ix, x in enumerate(temp_array)]
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

#: Total clothing insulation of typical typical ensembles.
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
