import math
import warnings
from enum import Enum
from typing import NamedTuple, Union

import numpy as np

from pythermalcomfort.classes_return import PsychrometricValues
from pythermalcomfort.shared_functions import valid_range

warnings.simplefilter("always")


c_to_k = 273.15
cp_vapour = 1805.0
cp_water = 4186
cp_air = 1004
h_fg = 2501000
r_air = 287.055
g = 9.81  # m/s2
met_to_w_m2 = 58.15


class Models(Enum):
    ashrae_55_2023 = "55-2023"
    iso_7730_2005 = "7730-2005"
    iso_9920_2007 = "9920-2007"


class Units(Enum):
    SI = "SI"
    IP = "IP"


class Sex(Enum):
    male = "male"
    female = "female"


def p_sat_torr(tdb: Union[float, list[float]]):
    """Estimates the saturation vapor pressure in [torr]

    Parameters
    ----------
    tdb : float or list of floats
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat : float
        saturation vapor pressure [torr]
    """
    return np.exp(18.6686 - 4030.183 / (tdb + 235.0))


def enthalpy_air(
    tdb: Union[float, list[float]],
    hr: Union[float, list[float]],
):
    """Calculates air enthalpy_air.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    hr: float or list of floats
        humidity ratio, [kg water/kg dry air]

    Returns
    -------
    enthalpy_air: float or list of floats
        enthalpy_air [J/kg dry air]
    """
    h_dry_air = cp_air * tdb
    h_sat_vap = h_fg + cp_vapour * tdb
    return h_dry_air + hr * h_sat_vap


# pre-calculated constants for p_sat
c1 = -5674.5359
c2 = 6.3925247
c3 = -0.9677843 * 1e-2
c4 = 0.62215701 * 1e-6
c5 = 0.20747825 * 1e-8
c6 = -0.9484024 * 1e-12
c7 = 4.1635019
c8 = -5800.2206
c9 = 1.3914993
c10 = -0.048640239
c11 = 0.41764768 * 1e-4
c12 = -0.14452093 * 1e-7
c13 = 6.5459673


def p_sat(tdb: Union[float, list[float]]):
    """Calculates vapour pressure of water at different temperatures.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]

    Returns
    -------
    p_sat: float or list of floats
        saturation vapor pressure, [Pa]
    """
    ta_k = tdb + c_to_k
    # pre-calculate the value before passing it to .where
    log_ta_k = np.log(ta_k)
    pascals = np.where(
        ta_k < c_to_k,
        np.exp(
            c1 / ta_k
            + c2
            + ta_k * (c3 + ta_k * (c4 + ta_k * (c5 + c6 * ta_k)))
            + c7 * log_ta_k
        ),
        np.exp(
            c8 / ta_k + c9 + ta_k * (c10 + ta_k * (c11 + ta_k * c12)) + c13 * log_ta_k
        ),
    )

    return pascals


def antoine(tdb: Union[float, np.ndarray]) -> np.ndarray:
    """Calculate saturated vapor pressure using Antoine equation [kPa].

    Parameters
    ----------
    tdb : float or list of floats
        Temperature [°C].

    Returns
    -------
    float or list of floats
        Saturated vapor pressure [kPa].
    """
    tdb = np.array(tdb)
    return math.e ** (16.6536 - 4030.183 / (tdb + 235))


def psy_ta_rh(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    p_atm=101325,
) -> PsychrometricValues:
    """Calculates psychrometric values of air based on dry bulb air temperature and
    relative humidity. For more accurate results we recommend the use of the Python
    package `psychrolib`_.

    .. _psychrolib: https://pypi.org/project/PsychroLib/

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    rh: float or list of floats
        relative humidity, [%]
    p_atm: float or list of floats
        atmospheric pressure, [Pa]

    Returns
    -------
    p_vap: float or list of floats
        partial pressure of water vapor in moist air, [Pa]
    hr: float or list of floats
        humidity ratio, [kg water/kg dry air]
    wet_bulb_tmp: float or list of floats
        wet bulb temperature, [°C]
    dew_point_tmp: float or list of floats
        dew point temperature, [°C]
    h: float or list of floats
        enthalpy_air [J/kg dry air]
    """
    tdb = np.array(tdb)
    rh = np.array(rh)

    p_saturation = p_sat(tdb)
    p_vap = rh / 100 * p_saturation
    hr = 0.62198 * p_vap / (p_atm - p_vap)
    tdp = dew_point_tmp(tdb, rh)
    twb = wet_bulb_tmp(tdb, rh)
    h = enthalpy_air(tdb, hr)

    return PsychrometricValues(
        p_sat=p_saturation,
        p_vap=p_vap,
        hr=hr,
        wet_bulb_tmp=twb,
        dew_point_tmp=tdp,
        h=h,
    )


def wet_bulb_tmp(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
):
    """Calculates the wet-bulb temperature using the Stull equation [Stull2011]_

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    rh: float or list of floats
        relative humidity, [%]

    Returns
    -------
    tdb: float or list of floats
        wet-bulb temperature, [°C]
    """
    twb = (
        tdb * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(tdb + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )

    return np.around(twb, 1)


def dew_point_tmp(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
):
    """Calculates the dew point temperature.

    Parameters
    ----------
    tdb: float or list of floats
        dry bulb air temperature, [°C]
    rh: float or list of floats
        relative humidity, [%]

    Returns
    -------
    dew_point_tmp: float or list of floats
        dew point temperature, [°C]
    """
    tdb = np.array(tdb)
    rh = np.array(rh)

    c = 257.14
    b = 18.678
    d = 234.5

    gamma_m = np.log(rh / 100 * np.exp((b - tdb / d) * (tdb / (c + tdb))))

    return np.round(c * gamma_m / (b - gamma_m), 1)


def mean_radiant_tmp(
    tg: Union[float, list[float]],
    tdb: Union[float, list[float]],
    v: Union[float, list[float]],
    d: Union[float, list[float]] = 0.15,
    emissivity: Union[float, list[float]] = 0.95,
    standard="Mixed Convection",
):
    """Converts globe temperature reading into mean radiant temperature in accordance
    with either the Mixed Convection developed by Teitelbaum E. et al. (2022) or the ISO
    7726:1998 Standard [7726ISO1998]_.

    Parameters
    ----------
    tg : float or list of floats
        globe temperature, [°C]
    tdb : float or list of floats
        air temperature, [°C]
    v : float or list of floats
        air speed, [m/s]
    d : float or list of floats
        diameter of the globe, [m] default 0.15 m
    emissivity : float or list of floats
        emissivity of the globe temperature sensor, default 0.95
    standard : str, optional
        Supported values are 'Mixed Convection' and 'ISO'. Defaults to 'Mixed Convection'.
        either choose between the Mixed Convection and ISO formulations.
        The Mixed Convection formulation has been proposed by Teitelbaum E. et al. (2022)
        to better determine the free and forced convection coefficient used in the
        calculation of the mean radiant temperature. They also showed that mean radiant
        temperature measured with ping-pong ball-sized globe thermometers is not reliable
        due to a stochastic convective bias [Teitelbaum2022]_. The Mixed Convection model has only
        been validated for globe sensors with a diameter between 0.04 and 0.15 m.

    Returns
    -------
    tr: float or list of floats
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

    if standard == "iso":  # pragma: no branch
        tg = np.add(tg, c_to_k)
        tdb = np.add(tdb, c_to_k)

        # calculate heat transfer coefficient
        h_n = 1.4 * np.power(np.abs(tg - tdb) / d, 0.25)  # natural convection
        h_f = 6.3 * np.power(v, 0.6) / np.power(d, 0.4)  # forced convection

        # get the biggest between the two coefficients
        h = np.maximum(h_f, h_n)

        tr = (
            np.power(
                np.power(tg, 4) + h * (tg - tdb) / (emissivity * (5.67 * 10**-8)),
                0.25,
            )
            - c_to_k
        )

        return np.around(tr, 1)


def validate_type(value, name: str, allowed_types: tuple):
    """Validate the type of a value against allowed types."""
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, allowed_types):
        raise TypeError(f"{name} must be one of the following types: {allowed_types}.")


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


def _check_standard_compliance_array(standard, **kwargs):
    default_kwargs = {"airspeed_control": True}
    params = {**default_kwargs, **kwargs}
    values_to_return = {}

    if standard == "ashrae":  # based on table 7.3.4 ashrae 55 2020
        tdb_valid = valid_range(params["tdb"], (10.0, 40.0))
        tr_valid = valid_range(params["tr"], (10.0, 40.0))

        values_to_return["tdb"] = tdb_valid
        values_to_return["tr"] = tr_valid

        if "v" in params.keys():
            v_valid = valid_range(params["v"], (0.0, 2.0))
            values_to_return["v"] = v_valid

        if not params["airspeed_control"]:
            v_valid = np.where(
                (params["v"] > 0.8) & (params["clo"] < 0.7) & (params["met"] < 1.3),
                np.nan,
                v_valid,
            )
            to = operative_tmp(params["tdb"], params["tr"], params["v"])
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

            values_to_return["v"] = v_valid

        if "met" in params.keys():
            met_valid = valid_range(params["met"], (1.0, 4.0))
            clo_valid = valid_range(params["clo"], (0.0, 1.5))

            values_to_return["met"] = met_valid
            values_to_return["clo"] = clo_valid

        if "v_limited" in params.keys():
            valid = valid_range(params["v_limited"], (0.0, 0.2))
            values_to_return["v_limited"] = valid

        return values_to_return.values()

    if standard == "7933":  # based on ISO 7933:2004 Annex A
        tdb_valid = valid_range(params["tdb"], (15.0, 50.0))
        p_a_valid = valid_range(params["p_a"], (0, 4.5))
        tr_valid = valid_range(params["tr"], (0.0, 60.0))
        v_valid = valid_range(params["v"], (0.0, 3))
        met_valid = valid_range(params["met"], (100, 450))
        clo_valid = valid_range(params["clo"], (0.1, 1))

        return tdb_valid, tr_valid, v_valid, p_a_valid, met_valid, clo_valid

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


class Postures(Enum):
    standing = "standing"
    sitting = "sitting"
    sedentary = "sedentary"
    reclining = "reclining"
    lying = "lying"
    supine = "supine"
    crouching = "crouching"


class BodySurfaceAreaEquations(Enum):
    dubois = "dubois"
    takahira = "takahira"
    fujimoto = "fujimoto"
    kurazumi = "kurazumi"


def body_surface_area(
    weight: float, height: float, formula: str = BodySurfaceAreaEquations.dubois.value
) -> float:
    """Calculate the body surface area (BSA) in square meters.

    Parameters
    ----------
    weight : float
        Body weight in kilograms.
    height : float
        Body height in meters.
    formula : str, optional
        Formula used to calculate the body surface area. Default is "dubois".
        Choose one from BodySurfaceAreaEquations.

    Returns
    -------
    float
        Body surface area in square meters.

    Raises
    ------
    ValueError
        If the specified formula is not recognized.

    Examples
    --------
    Calculate BSA using the DuBois formula:

    .. code-block:: python

        bsa = body_surface_area(weight=70, height=1.75, formula="dubois")
        print(bsa)
    """
    if formula == BodySurfaceAreaEquations.dubois.value:
        return 0.202 * (weight**0.425) * (height**0.725)
    elif formula == BodySurfaceAreaEquations.takahira.value:
        return 0.2042 * (weight**0.425) * (height**0.725)
    elif formula == BodySurfaceAreaEquations.fujimoto.value:
        return 0.1882 * (weight**0.444) * (height**0.663)
    elif formula == BodySurfaceAreaEquations.kurazumi.value:
        return 0.2440 * (weight**0.383) * (height**0.693)
    else:
        raise ValueError(
            f"Formula '{formula}' for calculating body surface area is not recognized."
        )


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


def v_relative(v: Union[float, list[float]], met: Union[float, list[float]]):
    """Estimates the relative air speed which combines the average air speed of the
    space plus the relative air speed caused by the body movement. The same equation is
    used in the ASHRAE 55:2023 and ISO 7730:2005 standards.

    Parameters
    ----------
    v : float or list of floats
        air speed measured by the sensor, [m/s]
    met : float or list of floats
        metabolic rate, [met]

    Returns
    -------
    vr  : float or list of floats
        relative air speed, [m/s]
    """
    v = np.array(v)
    met = np.array(met)
    return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)


def clo_dynamic_ashrae(
    clo: Union[float, list[float]],
    met: Union[float, list[float]],
    model: str = Models.ashrae_55_2023.value,
):
    """Estimates the dynamic intrinsic clothing insulation (I :sub:`cl,r`). The ASHRAE
    55:2023 refers to it as (I :sub:`cl,active`). The activity as well as the air speed
    modify the insulation characteristics of the clothing. Consequently, the ASHRAE 55
    standard provides a correction factor for the clothing insulation (I :sub:`cl`)
    based on the metabolic rate.

    Parameters
    ----------
    clo : float or list of floats
        clothing insulation, [clo]

        .. note::
            this is the basic insulation (I :sub:`cl`) also known as the intrinsic
            clothing insulation value under reference conditions

    met : float or list of floats
        metabolic rate, [met]
    model : str, optional
        Select the version of the ASHRAE 55 Standard to use. Currently, the only
        option available is "55-2023".

    Returns
    -------
    clo : float or list of floats
        dynamic clothing insulation (I :sub:`cl,r`), [clo]
    """
    clo = np.array(clo)
    met = np.array(met)

    model = model.lower()
    if model not in [Models.ashrae_55_2023.value]:
        raise ValueError(
            f"PMV calculations can only be performed in compliance with ASHRAE {Models.ashrae_55_2023.value}"
        )

    return np.where(met > 1.2, np.around(clo * (0.6 + 0.4 / met), 3), clo)


def clo_dynamic_iso(
    clo: Union[float, list[float]],
    met: Union[float, list[float]],
    v: Union[float, list[float]],
    i_a: Union[float, list[float]] = 0.7,
    model: str = Models.iso_9920_2007.value,
):
    """Estimates the dynamic intrinsic clothing insulation (I :sub:`cl,r`). The activity
    as well as the air speed modify the insulation characteristics of the clothing.
    Consequently, the ISO standard states that (I :sub:`cl,`) shall be corrected
    [7730ISO2005]_. However, the ISO 7730:2005 contains insufficient information to
    calculate (I :sub:`cl,r`). Therefore, we implemented the equations provided in the
    ISO 9920:2007 standard [ISO9920]_.

    Parameters
    ----------
    clo : float or list of floats
        clothing insulation, [clo]
    met : float or list of floats
        metabolic rate, [met]
    v : float or list of floats
        air speed, [m/s]
    i_a : float or list of floats
        thermal insulation of the boundary (surface) air layer around the outer clothing
        or, when nude, around the skin surface, [clo]
    model : str, optional
        Select the version of the ISO standard to use. Currently, the only
        option available is "9920-2007".

    Returns
    -------
    clo : float or list of floats
        dynamic clothing insulation, [clo]
    """
    model = model.lower()
    if model not in [Models.iso_9920_2007.value]:
        raise ValueError(
            f"PMV calculations can only be performed in compliance with ISO {Models.iso_9920_2007.value}"
        )

    clo = np.array(clo)
    met = np.array(met)
    i_a = np.array(i_a)
    v = np.array(v)

    f_cl = clo_area_factor(i_cl=clo)
    i_t = clo + i_a / f_cl
    v_walk = v_relative(v=v, met=met) - v
    v_r = v_relative(v=v, met=met)
    i_t_r = clo_total_insulation(
        i_t=i_t, vr=v_r, v_walk=v_walk, i_a_static=i_a, i_cl=clo
    )
    i_a_r = clo_insulation_air_layer(vr=v_r, v_walk=v_walk, i_a_static=i_a)
    return i_t_r - i_a_r / f_cl


def running_mean_outdoor_temperature(
    temp_array: list[float], alpha: float = 0.8, units: str = Units.SI.value
):
    """Estimates the running mean temperature also known as prevailing mean outdoor
    temperature.

    Parameters
    ----------
    temp_array: list
        array containing the mean daily temperature in descending order (i.e. from
        newest/yesterday to oldest) :math:`[t_{day-1}, t_{day-2}, ... ,
        t_{day-n}]`.
        Where :math:`t_{day-1}` is yesterday's daily mean temperature. The EN
        16798-1 2019 [16798EN2019]_ states that n should be equal to 7
    alpha : float
        constant between 0 and 1. The EN 16798-1 2019 [16798EN2019]_ recommends a value of 0.8,
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
    units = units.upper()
    if units == Units.IP.value:
        for ix, _x in enumerate(temp_array):
            temp_array[ix] = units_converter(tdb=temp_array[ix])[0]

    coeff = [alpha**ix for ix, x in enumerate(temp_array)]
    t_rm = sum([a * b for a, b in zip(coeff, temp_array)]) / sum(coeff)

    if units == Units.IP.value:
        t_rm = units_converter(tmp=t_rm, from_units=Units.SI.value.lower())[0]

    return round(t_rm, 1)


def units_converter(from_units=Units.IP.value, **kwargs):
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
    from_units = from_units.upper()
    if from_units == Units.IP.value:
        for key, value in kwargs.items():
            if "tmp" in key or key == "tr" or key == "tdb":
                results.append((value - 32) * 5 / 9)
            if key in ["v", "vr", "vel"]:
                results.append(value / 3.281)
            if key == "area":
                results.append(value / 10.764)
            if key == "pressure":
                results.append(value * 101325)

    elif from_units == Units.SI.value:
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


def operative_tmp(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    v: Union[float, list[float]],
    standard: str = "ISO",
):
    """Calculates operative temperature in accordance with ISO 7726:1998
    [7726ISO1998]_

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    tr: float or list of floats
        mean radiant temperature, [°C]
    v: float or list of floats
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


def clo_intrinsic_insulation_ensemble(clo_garments: Union[float, list[float]]):
    """Calculates the intrinsic insulation of a clothing ensemble based on individual
    garments. This equation is in accordance with the ISO 9920:2009 standard [ISO9920]_
    Section 4.3. It should be noted that this equation is only valid for clothing
    ensembles with rather uniform insulation values across the body.

    Parameters
    ----------
    clo_garments:  floats or list of floats
        list of floats containing the clothing insulation for each individual garment

    Returns
    -------
    i_cl: float
        intrinsic insulation of the clothing ensemble, [clo]
    """
    clo_garments = np.array(clo_garments)
    return np.sum(clo_garments) * 0.835 + 0.161


def clo_area_factor(i_cl: Union[float, list[float]]):
    """Calculates the clothing area factor (f_cl) of the clothing ensemble as a function
    of the intrinsic insulation of the clothing ensemble. This equation is in accordance
    with the ISO 9920:2009 standard [ISO9920]_ Section 5. The standard warns that the
    correlation between f_cl and i_cl is low especially for non-western clothing
    ensembles. The application of this equation is limited to clothing ensembles with
    clo values between 0.2 and 1.7 clo.

    Parameters
    ----------
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, [clo]

    Returns
    -------
    f_cl: float or list of floats
        area factor of the clothing ensemble, [m2]
    """
    i_cl = np.array(i_cl)
    return 1 + 0.28 * i_cl


# todo implement the vr and v_walk functions as a function of the met
def clo_insulation_air_layer(
    vr: Union[float, list[float]],
    v_walk: Union[float, list[float]],
    i_a_static: Union[float, list[float]],
):
    """Calculates the insulation of the boundary air layer (`I`:sub:`a,r`). The static
    boundary air value is 0.7 clo (0.109 m2K/W) for air velocities around 0.1 m/s to
    0.15 m/s. Thus, for static conditions, the standard recommends using the value of
    0.7 clo (0.109 m2K/W) for the boundary air layer insulation. For walking conditions,
    the boundary air layer insulation is calculated based on the walking speed (v_walk)
    and the relative air speed (vr). This equation is extracted from the ISO 9920:2009
    standard [ISO9920]_ Section 6.

    Parameters
    ----------
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_a_static: float or list of floats
        static boundary air layer insulation, [clo]

    Returns
    -------
    i_a_r: float or list of floats
        boundary air layer insulation, [clo]
    """
    vr = np.array(vr)
    v_walk = np.array(v_walk)
    i_a_static = np.array(i_a_static)

    return (
        np.exp(
            -0.533 * (vr - 0.15)
            + 0.069 * (vr - 0.15) ** 2
            - 0.462 * v_walk
            + 0.201 * v_walk**2
        )
        * i_a_static
    )


def clo_total_insulation(
    i_t: Union[float, list[float]],
    vr: Union[float, list[float]],
    v_walk: Union[float, list[float]],
    i_a_static: Union[float, list[float]],
    i_cl: Union[float, list[float]],
):
    """Calculates the total insulation of the clothing ensemble (`I`:sub:`T,r`) which is
    the actual thermal insulation from the body surface to the environment, considering
    all clothing, enclosed air layers, and boundary air layers under given environmental
    conditions and activities. It accounts for the effects of movements and wind. The
    ISO 7790 standard [ISO9920]_ provides different equations to calculate it as a
    function of the total thermal insulation of clothing (`I`:sub:`T`), the insulation
    of the boundary air layer (`I`:sub:`a`), the walking speed (`v`:sub:`walk`), and the
    relative air speed (`v`:sub:`r`). These different equations are used if the person
    is clothed in normal clothing (0.6 clo < (`I`:sub:`cl`) < 1.4 clo or 1.2 clo <
    (`I`:sub:`T`) < 2.0 clo), nude (`I`:sub:`cl` = 0 clo), and if the person is clothed
    in very light clothing (`I`:sub:`cl` < 0.6 clo). Here we have not implemented the
    equation for high clothing (`I`:sub:`T` > 2.0 clo). Hence the applicability of this
    function is limited to 0 clo < (`I`:sub:`T`) < 2.0 clo). You can find all the inputs
    required in this function in the ISO 9920:2009 standard [ISO9920]_ Annex A.

    Parameters
    ----------
    i_t: float or list of floats
        total thermal insulation of clothing under static reference conditions [clo]
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_a_static: float or list of floats
        static boundary air layer insulation, [clo]
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, this is the thermal insulation
        from the skin surface to the outer clothing surface [clo]

    Returns
    -------
    i_t_r: float or list of floats
        total insulation of the clothing ensemble, [clo]
    """
    i_t = np.array(i_t)
    vr = np.array(vr)
    v_walk = np.array(v_walk)
    i_a_static = np.array(i_a_static)
    i_cl = np.array(i_cl)

    def normal_clothing(_vr, _vw, _i_t):
        return _i_t * _correction_normal_clothing(_vw=_vw, _vr=_vr)

    def nude(_vr, _vw, _i_a_static):
        return _i_a_static * _correction_nude(_vr=_vr, _vw=_vw)

    def low_clothing(_vr, _vw, _i_a_static, _i_cl, _i_t):
        return (
            (0.6 - _i_cl) * nude(_vr, _vw, _i_a_static)
            + _i_cl * normal_clothing(_vr, _vw, _i_t)
        ) / 0.6

    i_t_r = np.where(
        i_cl <= 0.6,
        low_clothing(_vr=vr, _vw=v_walk, _i_a_static=i_a_static, _i_cl=i_cl, _i_t=i_t),
        normal_clothing(_vr=vr, _vw=v_walk, _i_t=i_t),
    )
    i_t_r = np.where(i_cl == 0, nude(_vr=vr, _vw=v_walk, _i_a_static=i_a_static), i_t_r)
    return i_t_r


def clo_correction_factor_environment(
    vr: Union[float, list[float]],
    v_walk: Union[float, list[float]],
    i_cl: Union[float, list[float]],
):
    """This function returns the correction factor for the total insulation of the
    clothing ensemble (`I`:sub:`T`) or the basic/intrinsic insulation (`I`:sub:`cl`).
    This correction factor takes into account of the fact that the values of
    (`I`:sub:`T`) and (`I`:sub:`cl`) are estimated in static conditions. In real
    environments the person may be walking, activity may pump air through the clothing,
    etc.

    Parameters
    ----------
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, this is the thermal insulation
        from the skin surface to the outer clothing surface [clo]

    Returns
    -------
    correction_factor: float or list of floats
        correction factor for the total insulation of the clothing ensemble
        (`I`:sub:`T,r` / (`I`:sub:`T`)) or the basic/intrinsic insulation
        (`I`:sub:`cl,r` / (`I`:sub:`cl`))
    """
    vr = np.array(vr)
    v_walk = np.array(v_walk)
    i_cl = np.array(i_cl)

    def correction_low_clothing(_vr, _vw, _i_cl):
        return (
            (0.6 - _i_cl) * _correction_nude(_vr, _vw)
            + _i_cl * _correction_normal_clothing(_vr, _vw)
        ) / 0.6

    c_f = np.where(
        i_cl <= 0.6,
        correction_low_clothing(_vr=vr, _vw=v_walk, _i_cl=i_cl),
        _correction_normal_clothing(_vr=vr, _vw=v_walk),
    )
    c_f = np.where(i_cl == 0, _correction_nude(_vr=vr, _vw=v_walk), c_f)
    return c_f


def _correction_nude(_vr, _vw):
    return np.exp(
        -0.533 * (_vr - 0.15) + 0.069 * (_vr - 0.15) ** 2 - 0.462 * _vw + 0.201 * _vw**2
    )


def _correction_normal_clothing(_vr, _vw):
    return np.exp(
        -0.281 * (_vr - 0.15) + 0.044 * (_vr - 0.15) ** 2 - 0.492 * _vw + 0.176 * _vw**2
    )


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

#: Clo values of individual clothing elements. To calculate the total
#: clothing insulation you need to add these values together.
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

#: This dictionary contains the reflection coefficients, Fr, for different
#: special materials
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


class DefaultSkinTemperature(NamedTuple):
    """Default skin temperature in degree Celsius for 17 local body parts
    The data comes from Hui Zhang's experiments
    https://escholarship.org/uc/item/3f4599hx
    """

    head: float = 35.3
    neck: float = 35.6
    chest: float = 35.1
    back: float = 35.3
    pelvis: float = 35.3
    left_shoulder: float = 34.2
    left_arm: float = 34.6
    left_hand: float = 34.4
    right_shoulder: float = 34.2
    right_arm: float = 34.6
    right_hand: float = 34.4
    left_thigh: float = 34.3
    left_leg: float = 32.8
    left_foot: float = 33.3
    right_thigh: float = 34.3
    right_leg: float = 32.8
    right_foot: float = 33.3
