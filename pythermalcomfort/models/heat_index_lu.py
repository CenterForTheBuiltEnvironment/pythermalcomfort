import math
from typing import Union

import numpy as np

from pythermalcomfort.classes_input import HIInputs
from pythermalcomfort.classes_return import HI


def heat_index_lu(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    round_output: bool = True,
) -> HI:
    """The Heat Index (HI) calculated in accordance with the Lu and Romps (2022) model [lu]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HI
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_lu

        result = heat_index_lu(tdb=25, rh=50)
        print(result.hi)  # 25.9
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    hi = _lu_heat_index_vectorized(tdb + 273.15, rh / 100) - 273.15

    if round_output:
        hi = np.around(hi, 1)

    return HI(hi=hi)


# combining the two functions find_eqvar and find_T
@np.vectorize
def _lu_heat_index_vectorized(tdb, rh):  # Thermodynamic parameters
    t_c_k = 273.16  # K
    p_triple_point = 611.65  # Pa
    e0v = 2.3740e6  # J/kg
    e0s = 0.3337e6  # J/kg
    rgasa = 287.04  # J/kg/K
    rgasv = 461.0  # J/kg/K
    cva = 719.0  # J/kg/K
    cvv = 1418.0  # J/kg/K
    cvl = 4119.0  # J/kg/K
    cvs = 1861.0  # J/kg/K
    cpa = cva + rgasa
    cpv = cvv + rgasv

    # The saturation vapor pressure
    def pv_star(t):
        if t == 0.0:
            return 0.0
        elif t < t_c_k:
            return (
                p_triple_point
                * (t / t_c_k) ** ((cpv - cvs) / rgasv)
                * math.exp(
                    (e0v + e0s - (cvv - cvs) * t_c_k) / rgasv * (1.0 / t_c_k - 1.0 / t)
                )
            )
        else:
            return (
                p_triple_point
                * (t / t_c_k) ** ((cpv - cvl) / rgasv)
                * math.exp(
                    (e0v - (cvv - cvl) * t_c_k) / rgasv * (1.0 / t_c_k - 1.0 / t)
                )
            )

    # The latent heat of vaporization of water
    def latent_heat_vap(t):
        return e0v + (cvv - cvl) * (t - t_c_k) + rgasv * t

    # Thermo-regulatory parameters
    sigma = 5.67e-8  # W/m^2/K^4 , Stefan-Boltzmann constant
    epsilon = 0.97  # emissivity of surface, steadman1979
    mass = 83.6  # kg, mass of average US adults, fryar2018
    height = 1.69  # m, height of average US adults, fryar2018
    area = 0.202 * (mass**0.425) * (height**0.725)  # m^2, DuBois formula, parson2014
    cpc = 3492.0  # J/kg/K, specific heat capacity of core, gagge1972
    hc_core = mass * cpc / area  # heat capacity of core
    r = 124.0  # Pa/K, Zf/rf, steadman1979
    q = 180.0  # W/m^2, metabolic rate per skin area, steadman1979
    phi_salt = 0.9  # vapor saturation pressure level of saline solution, steadman1979
    t_cr = 310.0  # K, core temperature, steadman1979
    p_cr = phi_salt * pv_star(t_cr)  # core vapor pressure
    lat_heat = latent_heat_vap(310.0)  # latent heat of vaporization at 310 K
    p = 1.013e5  # Pa, atmospheric pressure
    eta = 1.43e-6  # kg/J, "inhaled mass" / "metabolic rate", steadman1979
    pa0 = 1.6e3  # Pa, reference air vapor pressure in regions III, IV, V, VI, steadman1979

    za = (
        60.6 / 17.4
    )  # Pa m^2/W, mass transfer resistance through air, exposed part of skin
    za_bar = (
        60.6 / 11.6
    )  # Pa m^2/W, mass transfer resistance through air, clothed part of skin
    za_un = (
        60.6 / 12.3
    )  # Pa m^2/W, mass transfer resistance through air, when being naked

    # tolerance and maximum iteration for the root solver
    tol = 1e-8
    tol_t = 1e-8
    max_iter = 100

    # Thermo-regulatory functions
    def qv(ta, pa):  # respiratory heat loss, W/m^2
        return (
            eta * q * (cpa * (t_cr - ta) + lat_heat * rgasa / (p * rgasv) * (p_cr - pa))
        )

    def zs(rs):  # mass transfer resistance through skin, Pa m^2/W
        return 52.1 if rs == 0.0387 else 6.0e8 * rs**5

    def ra(
        ts, ta
    ):  # heat transfer resistance through air, exposed part of skin, K m^2/W
        hc = 17.4
        phi_rad = 0.85
        hr = epsilon * phi_rad * sigma * (ts**2 + ta**2) * (ts + ta)
        return 1.0 / (hc + hr)

    def ra_bar(
        tf, ta
    ):  # heat transfer resistance through air, clothed part of skin, K m^2/W
        hc = 11.6
        phi_rad = 0.79
        hr = epsilon * phi_rad * sigma * (tf**2 + ta**2) * (tf + ta)
        return 1.0 / (hc + hr)

    def ra_un(
        ts, ta
    ):  # heat transfer resistance through air, when being naked, K m^2/W
        hc = 12.3
        phi_rad = 0.80
        hr = epsilon * phi_rad * sigma * (ts**2 + ta**2) * (ts + ta)
        return 1.0 / (hc + hr)

    # Given air temperature and relative humidity, returns the equivalent variables
    def find_eq_var(ta, _rh):
        pa = _rh * pv_star(ta)  # air vapor pressure
        rs = 0.0387  # m^2K/W, heat transfer resistance through skin
        phi = 0.84  # covering fraction
        d_tc_dt = 0.0  # K/s, rate of change in Tc
        m = (p_cr - pa) / (zs(rs) + za)
        m_bar = (p_cr - pa) / (zs(rs) + za_bar)
        ts = solve(
            lambda ts: (ts - ta) / ra(ts, ta)
            + (p_cr - pa) / (zs(rs) + za)
            - (t_cr - ts) / rs,
            max(0.0, min(t_cr, ta) - rs * abs(m)),
            max(t_cr, ta) + rs * abs(m),
            tol,
            max_iter,
        )
        tf = solve(
            lambda tf: (tf - ta) / ra_bar(tf, ta)
            + (p_cr - pa) / (zs(rs) + za_bar)
            - (t_cr - tf) / rs,
            max(0.0, min(t_cr, ta) - rs * abs(m_bar)),
            max(t_cr, ta) + rs * abs(m_bar),
            tol,
            max_iter,
        )
        flux1 = (
            q - qv(ta, pa) - (1.0 - phi) * (t_cr - ts) / rs
        )  # C*dTc/dt when rf=Zf=\inf
        flux2 = (
            q - qv(ta, pa) - (1.0 - phi) * (t_cr - ts) / rs - phi * (t_cr - tf) / rs
        )  # C*dTc/dt when rf=Zf=0
        if flux1 <= 0.0:  # region I
            eq_var_name = "phi"
            phi = 1.0 - (q - qv(ta, pa)) * rs / (t_cr - ts)
            rf = float("inf")
        elif flux2 <= 0.0:  # region II&III
            eq_var_name = "rf"
            ts_bar = (
                t_cr - (q - qv(ta, pa)) * rs / phi + (1.0 / phi - 1.0) * (t_cr - ts)
            )
            tf = solve(
                lambda tf: (tf - ta) / ra_bar(tf, ta)
                + (p_cr - pa)
                * (tf - ta)
                / ((zs(rs) + za_bar) * (tf - ta) + r * ra_bar(tf, ta) * (ts_bar - tf))
                - (t_cr - ts_bar) / rs,
                ta,
                ts_bar,
                tol,
                max_iter,
            )
            rf = ra_bar(tf, ta) * (ts_bar - tf) / (tf - ta)
        else:  # region IV,V,VI
            rf = 0.0
            flux3 = (
                q
                - qv(ta, pa)
                - (t_cr - ta) / ra_un(t_cr, ta)
                - (phi_salt * pv_star(t_cr) - pa) / za_un
            )
            if flux3 < 0.0:  # region IV,V
                ts = solve(
                    lambda ts: (ts - ta) / ra_un(ts, ta)
                    + (p_cr - pa) / (zs((t_cr - ts) / (q - qv(ta, pa))) + za_un)
                    - (q - qv(ta, pa)),
                    0.0,
                    t_cr,
                    tol,
                    max_iter,
                )
                rs = (t_cr - ts) / (q - qv(ta, pa))
                eq_var_name = "rs"
                ps = p_cr - (p_cr - pa) * zs(rs) / (zs(rs) + za_un)
                if ps > phi_salt * pv_star(ts):  # region V
                    ts = solve(
                        lambda ts: (ts - ta) / ra_un(ts, ta)
                        + (phi_salt * pv_star(ts) - pa) / za_un
                        - (q - qv(ta, pa)),
                        0.0,
                        t_cr,
                        tol,
                        max_iter,
                    )
                    rs = (t_cr - ts) / (q - qv(ta, pa))
                    eq_var_name = "rs*"
            else:  # region VI
                rs = 0.0
                eq_var_name = "d_tc_dt"
                d_tc_dt = (1.0 / hc_core) * flux3
        return [eq_var_name, phi, rf, rs, d_tc_dt]

    # given the equivalent variable, find the Heat Index
    def find_t(eq_var_name, eq_var):
        if eq_var_name == "phi":
            t = solve(
                lambda t: find_eq_var(t, 1.0)[1] - eq_var,
                0.0,
                240.0,
                tol_t,
                max_iter,
            )
            _region = "I"
        elif eq_var_name == "rf":
            t = solve(
                lambda t: find_eq_var(t, min(1.0, pa0 / pv_star(t)))[2] - eq_var,
                230.0,
                300.0,
                tol_t,
                max_iter,
            )
            _region = "II" if pa0 > pv_star(t) else "III"
        elif eq_var_name == "rs" or eq_var_name == "rs*":
            t = solve(
                lambda T: find_eq_var(T, pa0 / pv_star(T))[3] - eq_var,
                295.0,
                350.0,
                tol_t,
                max_iter,
            )
            _region = "IV" if eq_var_name == "rs" else "V"
        else:
            t = solve(
                lambda T: find_eq_var(T, pa0 / pv_star(T))[4] - eq_var,
                340.0,
                1000.0,
                tol_t,
                max_iter,
            )
            _region = "VI"
        return t, _region

    def solve(f, x1, x2, _tol, _max_iter):
        a = x1
        b = x2
        fa = f(a)
        fb = f(b)
        if fa * fb > 0.0:
            raise SystemExit("wrong initial interval in the root solver")
        else:
            for i in range(_max_iter):
                c = (a + b) / 2.0
                fc = f(c)
                if fb * fc > 0.0:
                    b = c
                    fb = fc
                else:
                    a = c
                if abs(a - b) < _tol:
                    return c
                if i == _max_iter - 1:
                    raise SystemExit("reaching maximum iteration in the root solver")

    dic = {"phi": 1, "rf": 2, "rs": 3, "rs*": 3, "d_tc_dt": 4}
    eq_vars = find_eq_var(tdb, rh)
    hi, region = find_t(eq_vars[0], eq_vars[dic[eq_vars[0]]])
    if tdb == 0.0:
        hi = 0.0
    return hi
