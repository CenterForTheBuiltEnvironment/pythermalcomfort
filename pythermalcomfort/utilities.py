import math
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple, List
from typing import Union

import numpy as np

from pythermalcomfort.shared_functions import valid_range

warnings.simplefilter("always")





c_to_k = 273.15
cp_vapour = 1805.0
cp_water = 4186
cp_air = 1004
h_fg = 2501000
r_air = 287.055
g = 9.81  # m/s2


def p_sat_torr(tdb: Union[float, int, np.ndarray, List[float], List[int]]):
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
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    hr: Union[float, int, np.ndarray, List[float], List[int]],
):
    """Calculates air enthalpy_air

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


def p_sat(tdb: Union[float, int, np.ndarray, List[float], List[int]]):
    """Calculates vapour pressure of water at different temperatures

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


@dataclass
class PsychrometricValues:
    p_sat: Union[float, int, np.ndarray, List[float], List[int]]
    p_vap: Union[float, int, np.ndarray, List[float], List[int]]
    hr: Union[float, int, np.ndarray, List[float], List[int]]
    wet_bulb_tmp: Union[float, int, np.ndarray, List[float], List[int]]
    dew_point_tmp: Union[float, int, np.ndarray, List[float], List[int]]
    h: Union[float, int, np.ndarray, List[float], List[int]]

    def __getitem__(self, item):
        return getattr(self, item)


def psy_ta_rh(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
    p_atm=101325,
) -> PsychrometricValues:
    """Calculates psychrometric values of air based on dry bulb air temperature and
    relative humidity.
    For more accurate results we recommend the use of the Python package
    `psychrolib`_.

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
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
):
    """Calculates the wet-bulb temperature using the Stull equation [6]_

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
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
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
    tg: Union[float, int, np.ndarray, List[float], List[int]],
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    v: Union[float, int, np.ndarray, List[float], List[int]],
    d: Union[float, int, np.ndarray, List[float], List[int]] = 0.15,
    emissivity: Union[float, int, np.ndarray, List[float], List[int]] = 0.95,
    standard="Mixed Convection",
):
    """Converts globe temperature reading into mean radiant temperature in accordance
    with either the Mixed Convection developed by Teitelbaum E. et al. (2022) or the ISO
    7726:1998 Standard [5]_.

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
        due to a stochastic convective bias [22]_. The Mixed Convection model has only
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
                np.power(tg, 4) + h * (tg - tdb) / (emissivity * (5.67 * 10**-8)),
                0.25,
            )
            - c_to_k
        )

        return np.around(tr, 1)


def validate_type(value, name: str, allowed_types: tuple):
    """Validate the type of a value against allowed types."""
    if not isinstance(value, allowed_types):
        raise TypeError(f"{name} must be one of the following types: {allowed_types}.")


@dataclass
class BaseInputs:
    """Base class containing all possible input parameters."""

    body_surface_area: Union[float, int, np.ndarray, list] = field(default=1.8258)
    tdb: Union[float, int, np.ndarray, list] = field(default=None)
    tr: Union[float, int, np.ndarray, list] = field(default=None)
    twb: Union[float, int, np.ndarray, list] = field(default=None)
    tg: Union[float, int, np.ndarray, list] = field(default=None)
    vr: Union[float, int, np.ndarray, list] = field(default=None)
    v: Union[float, int, np.ndarray, list] = field(default=None)
    rh: Union[float, int, np.ndarray, list] = field(default=None)
    met: Union[float, int, np.ndarray, list] = field(default=None)
    clo: Union[float, int, np.ndarray, list] = field(default=None)
    wme: Union[float, int, np.ndarray, list] = field(default=0)
    round_output: bool = field(default=True)
    limit_inputs: bool = field(default=True)
    with_solar_load: bool = field(default=False)
    airspeed_control: bool = field(default=True)
    units: str = field(default="SI")
    standard: str = field(default="ISO")
    a_coefficient: Union[float, int] = field(default=None)
    e_coefficient: Union[float, int] = field(default=None)
    v_ankle: Union[float, int, np.ndarray, list] = field(default=None)
    t_running_mean: Union[float, int, np.ndarray, list] = field(default=None)
    q: Union[float, int, np.ndarray, list] = field(default=None)
    tout: Union[float, int, np.ndarray, list] = field(default=None)
    p_atm: Union[float, int, np.ndarray, list] = field(default=101325)
    age: Union[float, int, np.ndarray, list] = field(default=None)
    weight: Union[float, int, np.ndarray, list] = field(default=None)
    height: Union[float, int, np.ndarray, list] = field(default=None)
    sol_altitude: Union[float, int, np.ndarray, list] = field(default=None)
    sharp: Union[float, int, np.ndarray, list] = field(default=None)
    sol_radiation_dir: Union[float, int, np.ndarray, list] = field(default=None)
    sol_transmittance: Union[float, int, np.ndarray, list] = field(default=None)
    f_svv: Union[float, int, np.ndarray, list] = field(default=None)
    f_bes: Union[float, int, np.ndarray, list] = field(default=None)
    asw: Union[float, int, np.ndarray, list] = field(default=None)
    floor_reflectance: Union[float, int, np.ndarray, list] = field(default=None)
    vertical_tmp_grad: Union[float, int, np.ndarray, list] = field(default=None)
    position: Union[str, np.ndarray, list] = field(default=None)
    sex: Union[str, np.ndarray, list] = field(default=None)
    posture: Union[str, np.ndarray, list] = field(default=None)
    max_skin_blood_flow: Union[float, int, np.ndarray, list] = field(default=80)
    max_sweating: Union[float, int, np.ndarray, list] = field(default=500)
    w_max: Union[float, int, np.ndarray, list] = field(default=None)

    def __post_init__(self):
        # Only validate attributes that are not None
        if self.units.lower() not in ["si", "ip"]:
            raise ValueError("Units must be either 'SI' or 'IP'")
        if self.standard.lower() not in ["ashrae", "iso"]:
            raise ValueError("Standard must be either 'ASHRAE', 'ISO'")
        if self.position is not None:
            if self.position.lower() not in [
                "sitting",
                "standing",
                "standing, forced convection",
            ]:
                raise ValueError(
                    "position must be either 'standing', 'sitting', or 'standing, forced convection'"
                )
        if self.posture is not None:
            if self.posture.lower() not in ["sitting", "standing", "crouching"]:
                raise ValueError(
                    "posture must be either 'sitting', 'standing', or 'crouching'"
                )
        if self.sex is not None:
            if self.sex.lower() not in ["male", "female"]:
                raise ValueError("sex must be either 'male' or 'female'")
        if self.tdb is not None:
            validate_type(self.tdb, "tdb", (float, int, np.ndarray, list))
        if self.tr is not None:
            validate_type(self.tr, "tr", (float, int, np.ndarray, list))
        if self.twb is not None:
            validate_type(self.twb, "twb", (float, int, np.ndarray, list))
        if self.tg is not None:
            validate_type(self.tg, "tg", (float, int, np.ndarray, list))
        if self.vr is not None:
            validate_type(self.vr, "vr", (float, int, np.ndarray, list))
        if self.v is not None:
            validate_type(self.v, "v", (float, int, np.ndarray, list))
        if self.rh is not None:
            validate_type(self.rh, "rh", (float, int, np.ndarray, list))
        if self.met is not None:
            validate_type(self.met, "met", (float, int, np.ndarray, list))
        if self.clo is not None:
            validate_type(self.clo, "clo", (float, int, np.ndarray, list))
        if self.a_coefficient is not None:
            validate_type(self.a_coefficient, "a_coefficient", (float, int))
        if self.e_coefficient is not None:
            validate_type(self.e_coefficient, "e_coefficient", (float, int))
        if self.wme is not None:
            validate_type(self.wme, "wme", (float, int, np.ndarray, list))
        if self.v_ankle is not None:
            validate_type(self.v_ankle, "v_ankle", (float, int, np.ndarray, list))
        if self.t_running_mean is not None:
            validate_type(
                self.t_running_mean, "t_running_mean", (float, int, np.ndarray, list)
            )
        if self.q is not None:
            validate_type(self.q, "q", (float, int, np.ndarray, list))
        if not isinstance(self.round_output, bool):
            raise TypeError("round must be either True or False.")
        if not isinstance(self.limit_inputs, bool):
            raise TypeError("limit_inputs must be either True or False.")
        if not isinstance(self.airspeed_control, bool):
            raise TypeError("airspeed_control must be either True or False.")
        if not isinstance(self.with_solar_load, bool):
            raise TypeError("with_solar_load must be either True or False.")
        if self.tout is not None:
            validate_type(self.tout, "tout", (float, int, np.ndarray, list))
        if self.p_atm is not None:
            validate_type(self.p_atm, "p_atm", (float, int, np.ndarray, list))
        if self.age is not None:
            validate_type(self.age, "age", (float, int, np.ndarray, list))
        if self.weight is not None:
            validate_type(self.weight, "weight", (float, int, np.ndarray, list))
        if self.height is not None:
            validate_type(self.height, "height", (float, int, np.ndarray, list))
        if self.sol_altitude is not None:
            validate_type(
                self.sol_altitude, "sol_altitude", (float, int, np.ndarray, list)
            )
        if self.sharp is not None:
            validate_type(self.sharp, "sharp", (float, int, np.ndarray, list))
        if self.sol_radiation_dir is not None:
            validate_type(
                self.sol_radiation_dir,
                "sol_radiation_dir",
                (float, int, np.ndarray, list),
            )
        if self.sol_transmittance is not None:
            validate_type(
                self.sol_transmittance,
                "sol_transmittance",
                (float, int, np.ndarray, list),
            )
        if self.f_svv is not None:
            validate_type(self.f_svv, "f_svv", (float, int, np.ndarray, list))
        if self.f_bes is not None:
            validate_type(self.f_bes, "f_bes", (float, int, np.ndarray, list))
        if self.asw is not None:
            validate_type(self.asw, "asw", (float, int, np.ndarray, list))
        if self.floor_reflectance is not None:
            validate_type(
                self.floor_reflectance,
                "floor_reflectance",
                (float, int, np.ndarray, list),
            )
        if self.vertical_tmp_grad is not None:
            validate_type(
                self.vertical_tmp_grad,
                "vertical_tmp_grad",
                (float, int, np.ndarray, list),
            )
        if self.body_surface_area is not None:
            validate_type(
                self.body_surface_area,
                "body_surface_area",
                (float, int, np.ndarray, list),
            )
        if self.max_sweating is not None:
            validate_type(self.max_sweating, "max_sweating", (float, int))
        if self.max_skin_blood_flow is not None:
            validate_type(self.max_skin_blood_flow, "max_skin_blood_flow", (float, int))
        if self.w_max is not None:
            validate_type(self.w_max, "w_max", (float, int, np.ndarray, list))


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


def check_standard_compliance_array(standard, **kwargs):
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
    else:
        raise ValueError(
            f"This {formula} to calculate the body_surface_area does not exists."
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


def v_relative(v, met):
    """Estimates the relative air speed which combines the average air speed of
    the space plus the relative air speed caused by the body movement. Vag is assumed to
    be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
    Vag = 0.3 (M – 1) (m/s)

    Parameters
    ----------
    v : float or list of floats
        air speed measured by the sensor, [m/s]
    met : float
        metabolic rate, [met]

    Returns
    -------
    vr  : float or list of floats
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
    clo : float or list of floats
        clothing insulation, [clo]
    met : float or list of floats
        metabolic rate, [met]
    standard: str (default="ASHRAE")
        - If "ASHRAE", uses Equation provided in Section 5.2.2.2 of ASHRAE 55 2020

    Returns
    -------
    clo : float or list of floats
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


def operative_tmp(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    v: Union[float, int, np.ndarray, List[float], List[int]],
    standard: str = "ISO",
):
    """Calculates operative temperature in accordance with ISO 7726:1998 [5]_

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

#: Clo values of individual clothing elements. To calculate the total clothing insulation you need to add these values together.
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

#: This dictionary contains the reflection coefficients, Fr, for different special materials
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
    https://escholarship.org/uc/item/3f4599hx"""

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
