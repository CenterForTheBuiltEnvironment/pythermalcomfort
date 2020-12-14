import math
from pythermalcomfort.utilities import *

c_to_k = 273.15
cp_vapour = 1805.0
cp_water = 4186
cp_air = 1004
h_fg = 2501000
r_air = 287.055


def f_svv(w, h, d):
    """ Calculates the sky-vault view fraction

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


def p_sat_torr(tdb):
    """ Estimates the saturation vapor pressure in [torr]

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat  : float
        saturation vapor pressure [torr]
    """
    return math.exp(18.6686 - 4030.183 / (tdb + 235.0))


def v_relative(v, met):
    """ Estimates the relative air velocity which combines the average air velocity of
    the space plus the relative air velocity caused by the body movement.

    Parameters
    ----------
    v : float
        air velocity measured by the sensor, [m/s]
    met : float
        metabolic rate, [met]

    Returns
    -------
    vr  : float
        relative air velocity, [m/s]
    """

    if met > 1:
        return round(v + 0.3 * (met - 1), 3)
    else:
        return v


def clo_dynamic(clo, met, standard="ASHRAE"):
    """ Estimates the dynamic clothing insulation of a moving occupant. The activity as
    well as the air speed modify the insulation characteristics of the clothing and the
    adjacent air layer. Consequently the ISO 7730 states that the clothing insulation
    shall be corrected [2]_. The ASHRAE 55 Standard, instead, only corrects for the effect
    of the body movement, and states that the correction is permitted but not required.

    Parameters
    ----------
    clo : float
        clothing insulation, [clo]
    met : float
        metabolic rate, [met]
    standard: str (default="ASHRAE")
        - If "ASHRAE", uses Equation provided in Section 5.2.2.2 of ASHRAE 55 2017

    Returns
    -------
    clo : float
        dynamic clothing insulation, [clo]
    """

    if standard.lower() not in ["ashrae"]:
        raise ValueError(
            "PMV calculations can only be performed in compliance with ISO or ASHRAE "
            "Standards"
        )

    if 1.2 < met < 2:
        return round(clo * (0.6 + 0.4 / met), 3)
    else:
        return clo


def running_mean_outdoor_temperature(temp_array, alpha=0.8, units="SI"):
    """ Estimates the running mean temperature

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
        while the ASHRAE 55 2017 recommends to choose values between 0.9 and 0.6,
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
    """ Converts IP values to SI units

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


def t_o(tdb, tr, v):
    """ Calculates operative temperature in accordance with ISO 7726:1998 [5]_

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    tr: float
        mean radiant temperature temperature, [°C]
    v: float
        air velocity, [m/s]

    Returns
    -------
    to: float
        operative temperature, [°C]
    """

    return (tdb * math.sqrt(10 * v) + tr) / (1 + math.sqrt(10 * v))


def enthalpy(tdb, hr):
    """ Calculates air enthalpy

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    hr: float
        humidity ratio, [kg water/kg dry air]

    Returns
    -------
    enthalpy: float
        enthalpy [J/kg dry air]
    """

    h_dry_air = cp_air * tdb
    h_sat_vap = h_fg + cp_vapour * tdb
    h = h_dry_air + hr * h_sat_vap

    return round(h, 2)


def p_sat(tdb):
    """ Calculates vapour pressure of water at different temperatures

    Parameters
    ----------
    tdb: float
        air temperature, [°C]

    Returns
    -------
    p_sat: float
        operative temperature, [Pa]
    """

    ta_k = tdb + c_to_k
    c1 = -5674.5359
    c2 = 6.3925247
    c3 = -0.9677843 * math.pow(10, -2)
    c4 = 0.62215701 * math.pow(10, -6)
    c5 = 0.20747825 * math.pow(10, -8)
    c6 = -0.9484024 * math.pow(10, -12)
    c7 = 4.1635019
    c8 = -5800.2206
    c9 = 1.3914993
    c10 = -0.048640239
    c11 = 0.41764768 * math.pow(10, -4)
    c12 = -0.14452093 * math.pow(10, -7)
    c13 = 6.5459673

    if ta_k < c_to_k:
        pascals = math.exp(
            c1 / ta_k
            + c2
            + ta_k * (c3 + ta_k * (c4 + ta_k * (c5 + c6 * ta_k)))
            + c7 * math.log(ta_k)
        )
    else:
        pascals = math.exp(
            c8 / ta_k
            + c9
            + ta_k * (c10 + ta_k * (c11 + ta_k * c12))
            + c13 * math.log(ta_k)
        )

    return round(pascals, 1)


def psy_ta_rh(tdb, rh, patm=101325):
    """ Calculates psychrometric values of air based on dry bulb air temperature and
    relative humidity.
    For more accurate results we recommend the use of the the Python package
    `psychrolib`_.

    .. _psychrolib: https://pypi.org/project/PsychroLib/

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    rh: float
        relative humidity, [%]
    patm: float
        atmospheric pressure, [Pa]

    Returns
    -------
    p_vap: float
        partial pressure of water vapor in moist air, [Pa]
    hr: float
        humidity ratio, [kg water/kg dry air]
    t_wb: float
        wet bulb temperature, [°C]
    t_dp: float
        dew point temperature, [°C]
    h: float
        enthalpy [J/kg dry air]
    """
    psat = p_sat(tdb)
    pvap = rh / 100 * psat
    hr = 0.62198 * pvap / (patm - pvap)
    tdp = t_dp(tdb, rh)
    twb = t_wb(tdb, rh)
    h = enthalpy(tdb, hr)

    return {"p_sat": psat, "p_vap": pvap, "hr": hr, "t_wb": twb, "t_dp": tdp, "h": h}


def t_wb(tdb, rh):
    """ Calculates the wet-bulb temperature using the Stull equation [6]_

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    rh: float
        relative humidity, [%]

    Returns
    -------
    tdb: float
        wet-bulb temperature, [°C]
    """
    twb = round(
        tdb * math.atan(0.151977 * (rh + 8.313659) ** (1 / 2))
        + math.atan(tdb + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** (3 / 2) * math.atan(0.023101 * rh)
        - 4.686035,
        1,
    )
    return twb


def t_dp(tdb, rh):
    """ Calculates the dew point temperature.

    Parameters
    ----------
    tdb: float
        dry-bulb air temperature, [°C]
    rh: float
        relative humidity, [%]

    Returns
    -------
    t_dp: float
        dew point temperature, [°C]
    """

    c = 257.14
    b = 18.678
    d = 234.5

    gamma_m = math.log(rh / 100 * math.exp((b - tdb / d) * (tdb / (c + tdb))))

    return round(c * gamma_m / (b - gamma_m), 1)


def t_mrt(tg, tdb, v, d=0.15, emissivity=0.9):
    """ Converts globe temperature reading into mean radiant temperature in accordance
    with ISO 7726:1998 [5]_

    Parameters
    ----------
    tg: float
        globe temperature, [°C]
    tdb: float
        air temperature, [°C]
    v: float
        air velocity, [m/s]
    d: float
        diameter of the globe, [m] default 0.15 m
    emissivity: float
        emissivity of the globe temperature sensor, default 0.9

    Returns
    -------
    tr: float
        mean radiant temperature, [°C]
    """
    tg += c_to_k
    tdb += c_to_k

    # calculate heat transfer coefficient
    h_n = 1.4 * (abs(tg - tdb) / d) ** 0.25  # natural convection
    h_f = 6.3 * v ** 0.6 / d ** 0.4  # forced convection

    # get the biggest between the tow coefficients
    h = max(h_f, h_n)
    print(h_n, h_f, h)

    tr = (tg ** 4 + h * (tg - tdb) / (emissivity * (5.67 * 10 ** -8))) ** 0.25 - c_to_k

    return round(tr, 1)
