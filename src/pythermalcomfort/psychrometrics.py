import math
import numpy as np
from pythermalcomfort.shared_functions import valid_range

c_to_k = 273.15
cp_vapour = 1805.0
cp_water = 4186
cp_air = 1004
h_fg = 2501000
r_air = 287.055
g = 9.81  # m/s2


def p_sat_torr(tdb):
    """Estimates the saturation vapor pressure in [torr]

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat  : float
        saturation vapor pressure [torr]
    """
    return np.exp(18.6686 - 4030.183 / (tdb + 235.0))


def t_o(tdb, tr, v, standard="ISO"):
    """Calculates operative temperature in accordance with ISO 7726:1998 [5]_

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    tr: float
        mean radiant temperature, [°C]
    v: float
        air speed, [m/s]
    standard: str (default="ISO")
        either choose between ISO and ASHRAE


    Returns
    -------
    to: float
        operative temperature, [°C]
    """

    if standard.lower() == "iso":
        return (tdb * np.sqrt(10 * v) + tr) / (1 + np.sqrt(10 * v))
    elif standard.lower() == "ashrae":
        a = np.where(v < 0.6, 0.6, 0.7)
        a = np.where(v < 0.2, 0.5, a)
        return a * tdb + (1 - a) * tr


def enthalpy(tdb, hr):
    """Calculates air enthalpy

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
    """Calculates vapour pressure of water at different temperatures

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


def psy_ta_rh(tdb, rh, p_atm=101325):
    """Calculates psychrometric values of air based on dry bulb air temperature and
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
    p_atm: float
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
    p_saturation = p_sat(tdb)
    p_vap = rh / 100 * p_saturation
    hr = 0.62198 * p_vap / (p_atm - p_vap)
    tdp = t_dp(tdb, rh)
    twb = t_wb(tdb, rh)
    h = enthalpy(tdb, hr)

    return {
        "p_sat": p_saturation,
        "p_vap": p_vap,
        "hr": hr,
        "t_wb": twb,
        "t_dp": tdp,
        "h": h,
    }


def t_wb(tdb, rh):
    """Calculates the wet-bulb temperature using the Stull equation [6]_

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
    """Calculates the dew point temperature.

    Parameters
    ----------
    tdb: float
        dry bulb air temperature, [°C]
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


def t_mrt(tg, tdb, v, d=0.15, emissivity=0.95, standard="Mixed Convection"):
    """Converts globe temperature reading into mean radiant temperature in accordance
    with either the Mixed Convection developed by Teitelbaum E. et al. (2022) or the ISO
    7726:1998 Standard [5]_.

    Parameters
    ----------
    tg : float or array-like
        globe temperature, [°C]
    tdb : float or array-like
        air temperature, [°C]
    v : float or array-like
        air speed, [m/s]
    d : float or array-like
        diameter of the globe, [m] default 0.15 m
    emissivity : float or array-like
        emissivity of the globe temperature sensor, default 0.95
    standard : {"Mixed Convection", "ISO"}
        either choose between the Mixed Convection and ISO formulations.
        The Mixed Convection formulation has been proposed by Teitelbaum E. et al. (2022)
        to better determine the free and forced convection coefficient used in the
        calculation of the mean radiant temperature. They also showed that mean radiant
        temperature measured with ping-pong ball-sized globe thermometers is not reliable
        due to a stochastic convective bias [22]_. The Mixed Convection model has only
        been validated for globe sensors with a diameter between 0.04 and 0.15 m.

    Returns
    -------
    tr: float or array-like
        mean radiant temperature, [°C]
    """
    standard = standard.lower()

    tdb = np.array(tdb)
    tg = np.array(tg)
    v = np.array(v)
    d = np.array(d)

    if standard == "mixed convection":
        mu = 0.0000181  # Pa s
        k_air = 0.02662  # W/m-K
        beta = 0.0034  # 1/K
        nu = 0.0000148  # m2/s
        alpha = 0.00002591  # m2/s
        pr = cp_air * mu / k_air  # Prandtl constants

        o = 0.0000000567
        n = 1.27 * d + 0.57

        ra = g * beta * np.absolute(tg - tdb) * d * d * d / nu / alpha
        re = v * d / nu

        nu_natural = 2 + (0.589 * np.power(ra, (1 / 4))) / (
            np.power(1 + np.power(0.469 / pr, 9 / 16), (4 / 9))
        )
        nu_forced = 2 + (
            0.4 * np.power(re, 0.5) + 0.06 * np.power(re, 2 / 3)
        ) * np.power(pr, 0.4)

        tr = (
            np.power(
                np.power(tg + 273.15, 4)
                - (
                    (
                        (
                            (
                                np.power(
                                    (np.power(nu_forced, n) + np.power(nu_natural, n)),
                                    1 / n,
                                )
                            )
                            * k_air
                            / d
                        )
                        * (-tg + tdb)
                    )
                    / emissivity
                    / o
                ),
                0.25,
            )
            - 273.15
        )

        d_valid = valid_range(d, (0.04, 0.15))
        tr = np.where(~np.isnan(d_valid), tr, np.nan)

        return np.around(tr, 1)

    if standard == "iso":

        tg = np.add(tg, c_to_k)
        tdb = np.add(tdb, c_to_k)

        # calculate heat transfer coefficient
        h_n = np.power(1.4 * (np.abs(tg - tdb) / d), 0.25)  # natural convection
        h_f = 6.3 * np.power(v, 0.6) / np.power(d, 0.4)  # forced convection

        # get the biggest between the two coefficients
        h = np.maximum(h_f, h_n)

        tr = (
            np.power(
                np.power(tg, 4) + h * (tg - tdb) / (emissivity * (5.67 * 10 ** -8)),
                0.25,
            )
            - c_to_k
        )

        return np.around(tr, 1)
