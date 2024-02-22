from typing import Union, List
import math

import numpy as np
from numba import jit, vectorize, float64

from pythermalcomfort.psychrometrics import p_sat_torr
from pythermalcomfort.utilities import (
    body_surface_area,
)


def two_nodes(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    v: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
    met: Union[float, int, np.ndarray, List[float], List[int]],
    clo: Union[float, int, np.ndarray, List[float], List[int]],
    wme: Union[float, int, np.ndarray, List[float], List[int]] = 0,
    body_surface_area=1.8258,
    p_atmospheric=101325,
    body_position="standing",
    max_skin_blood_flow=90,
    **kwargs,
):
    """Two-node model of human temperature regulation Gagge et al. (1986). [10]_
    This model it can be used to calculate a variety of indices,
    including:

    * Gagge's version of Fanger's Predicted Mean Vote (PMV). This function uses the Fanger's PMV equations but it replaces the heat loss and gain terms with those calculated by the two node model developed by Gagge et al. (1986) [10]_.
    * PMV SET and the predicted thermal sensation based on SET [10]_. This function is similar in all aspects to the :py:meth:`pythermalcomfort.models.pmv_gagge` however, it uses the :py:meth:`pythermalcomfort.models.set` equation to calculate the dry heat loss by convection.
    * Thermal discomfort (DISC) as the relative thermoregulatory strain necessary to restore a state of comfort and thermal equilibrium by sweating [10]_. DISC is described numerically as: comfortable and pleasant (0), slightly uncomfortable but acceptable (1), uncomfortable and unpleasant (2), very uncomfortable (3), limited tolerance (4), and intolerable (S). The range of each category is ± 0.5 numerically. In the cold, the classical negative category descriptions used for Fanger's PMV apply [10]_.
    * Heat gains and losses via convection, radiation and conduction.
    * The Standard Effective Temperature (SET)
    * The New Effective Temperature (ET)
    * The Predicted  Thermal  Sensation  (TSENS)
    * The Predicted  Percent  Dissatisfied  Due  to  Draft  (PD)
    * Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement"   (PS)

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float or array-like
        air speed, default in [m/s] in [fps] if `units` = 'IP'
    rh : float or array-like
        relative humidity, [%]
    met : float or array-like
        metabolic rate, [met]
    clo : float or array-like
        clothing insulation, [clo]
    wme : float or array-like
        external work, [met] default 0
    body_surface_area : float
        body surface area, default value 1.8258 [m2] in [ft2] if `units` = 'IP'

        The body surface area can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.body_surface_area`.
    p_atmospheric : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    body_position: str default="standing" or array-like
        select either "sitting" or "standing"
    max_skin_blood_flow : float
        maximum blood flow from the core to the skin, [kg/h/m2] default 80

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output values, if False it does not round them
    max_sweating : float
        Maximum rate at which regulatory sweat is generated, [kg/h/m2]
    w_max : float
        Maximum skin wettedness (w) adimensional. Ranges from 0 and 1.

    Returns
    -------
    e_skin : float or array-like
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff
    e_rsw : float or array-like
        Rate of evaporative heat loss from sweat evaporation, [W/m2]
    e_diff : float or array-like
        Rate of evaporative heat loss from moisture diffused through the skin, [W/m2]
    e_max : float or array-like
        Maximum rate of evaporative heat loss from skin, [W/m2]
    q_sensible : float or array-like
        Sensible heat loss from skin, [W/m2]
    q_skin : float or array-like
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin
    q_res : float or array-like
        Total rate of heat loss through respiration, [W/m2]
    t_core : float or array-like
        Core temperature, [°C]
    t_skin : float or array-like
        Skin temperature, [°C]
    m_bl : float or array-like
        Skin blood flow, [kg/h/m2]
    m_rsw : float or array-like
        Rate at which regulatory sweat is generated, [kg/h/m2]
    w : float or array-like
        Skin wettedness, adimensional. Ranges from 0 and 1.
    w_max : float or array-like
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 and 1.
    set : float or array-like
        Standard Effective Temperature (SET)
    et : float or array-like
        New Effective Temperature (ET)
    pmv_gagge : float or array-like
        PMV Gagge
    pmv_set : float or array-like
        PMV SET
    pd : float or array-like
        Predicted  Percent  Dissatisfied  Due  to  Draft"
    ps : float or array-like
        Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement
    disc : float or array-like
        Thermal discomfort
    t_sens : float or array-like
        Predicted  Thermal  Sensation

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import two_nodes
        >>> print(two_nodes(tdb=25, tr=25, v=0.3, rh=50, met=1.2, clo=0.5))
        {'e_skin': 15.8, 'e_rsw': 6.5, 'e_diff': 9.3, ... }
        >>> print(two_nodes(tdb=[25, 25], tr=25, v=0.3, rh=50, met=1.2, clo=0.5))
        {'e_skin': array([15.8, 15.8]), 'e_rsw': array([6.5, 6.5]), ... }
    """
    default_kwargs = {
        "round": True,
        "calculate_ce": False,
        "max_sweating": 500,
        "w_max": False,
    }
    kwargs = {**default_kwargs, **kwargs}

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)
    body_position = np.array(body_position)

    vapor_pressure = rh * p_sat_torr(tdb) / 100

    if kwargs["calculate_ce"]:
        result = _two_nodes_optimized_return_set(
            tdb,
            tr,
            v,
            met,
            clo,
            vapor_pressure,
            wme,
            body_surface_area,
            p_atmospheric,
            1,
        )
        return {"_set": result}

    (
        _set,
        e_skin,
        e_rsw,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        et,
        pmv_gagge,
        pmv_set,
        disc,
        t_sens,
    ) = np.vectorize(_two_nodes_optimized, cache=True)(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atmospheric=p_atmospheric,
        body_position=body_position,
        calculate_ce=kwargs["calculate_ce"],
        max_skin_blood_flow=max_skin_blood_flow,
        max_sweating=kwargs["max_sweating"],
        w_max=kwargs["w_max"],
    )

    output = {
        "e_skin": e_skin,
        "e_rsw": e_rsw,
        "e_max": e_max,
        "q_sensible": q_sensible,
        "q_skin": q_skin,
        "q_res": q_res,
        "t_core": t_core,
        "t_skin": t_skin,
        "m_bl": m_bl,
        "m_rsw": m_rsw,
        "w": w,
        "w_max": w_max,
        "_set": _set,
        "et": et,
        "pmv_gagge": pmv_gagge,
        "pmv_set": pmv_set,
        "disc": disc,
        "t_sens": t_sens,
    }

    for key in output.keys():
        # round the results if needed
        if kwargs["round"]:
            output[key] = np.around(output[key], 1)

    return output


@jit(nopython=True)
def _two_nodes_optimized(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atmospheric,
    body_position,
    calculate_ce=False,
    max_skin_blood_flow=90,
    max_sweating=500,
    w_max=False,
):
    # Initial variables as defined in the ASHRAE 55-2020
    air_speed = max(v, 0.1)
    k_clo = 0.25
    body_weight = 70  # body weight in kg
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 120  # driving coefficient for vasodilation ashrae says 50 see page 9.19
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alfa = 0.1
    temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    m_bl = skin_blood_flow_neutral

    # initialize some variables
    e_skin = 0.1 * met  # total evaporative heat loss, W
    q_sensible = 0  # total sensible heat loss, W
    w = 0  # skin wettedness
    _set = 0  # standard effective temperature
    e_rsw = 0  # heat lost by vaporization sweat
    e_diff = 0  # vapor diffusion through skin
    e_max = 0  # maximum evaporative capacity
    m_rsw = 0  # regulatory sweating
    q_res = 0  # heat loss due to respiration
    et = 0  # effective temperature
    e_req = 0  # evaporative heat loss required for tmp regulation
    r_ea = 0
    r_ecl = 0
    c_res = 0  # convective heat loss respiration

    pressure_in_atmospheres = p_atmospheric / 101325
    length_time_simulation = 60  # length time simulation
    n_simulation = 0

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = (met - wme) * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    e_comfort = 0.42 * (rm - met_factor)  # evaporative heat loss during comfort
    if e_comfort < 0:
        e_comfort = 0

    i_cl = 1.0  # permeation efficiency of water vapour naked skin
    if clo > 0:
        i_cl = 0.45  # permeation efficiency of water vapour through the clothing layer

    if not w_max:  # if the user did not pass a value of w_max
        w_max = 0.38 * pow(air_speed, -0.29)  # critical skin wettedness naked
        if clo > 0:
            w_max = 0.59 * pow(air_speed, -0.08)  # critical skin wettedness clothed

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 °C)
    h_fc = 8.600001 * pow((air_speed * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)
    if not calculate_ce and met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_cc = max(h_cc, h_c_met)

    h_r = 4.7  # linearized radiative heat transfer coefficient
    h_t = h_r + h_cc  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_a = 1.0 / (f_a_cl * h_t)  # resistance of air layer to dry heat
    t_op = (h_r * tr + h_cc * tdb) / h_t  # operative temperature

    t_body = alfa * t_skin + (1 - alfa) * t_core  # mean body temperature, °C

    # respiration
    q_res = 0.0023 * m * (44.0 - vapor_pressure)  # latent heat loss due to respiration
    c_res = 0.0014 * m * (34.0 - tdb)  # sensible convective heat loss respiration

    while n_simulation < length_time_simulation:

        n_simulation += 1

        iteration_limit = 150  # for following while loop
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:

            # 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
            if body_position == "sitting":
                # 0.7 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.7
            else:  # if standing
                # 0.73 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.73
            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr + h_cc * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations exceeded")

        q_sensible = (t_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W
        # hf_cs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        s_core = m - hf_cs - q_res - c_res - wme  # rate of energy storage in the core
        s_skin = hf_cs - q_sensible - e_skin  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (
            tc_sk * 60.0
        )  # rate of change skin temperature °C per minute
        d_t_cr = (
            s_core * body_surface_area / (tc_cr * 60.0)
        )  # rate of change core temperature °C per minute
        t_skin = t_skin + d_t_sk
        t_core = t_core + d_t_cr
        t_body = alfa * t_skin + (1 - alfa) * t_core
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = t_skin - temp_skin_neutral
        warm_sk = (sk_sig > 0) * sk_sig  # vasodilation signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the skin, °C
        c_reg_sig = t_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        # bd_sig thermoregulatory control signal from the body
        bd_sig = t_body - temp_body_neutral
        warm_b = (bd_sig > 0) * bd_sig
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1 + c_str * colds)
        if m_bl > max_skin_blood_flow:
            m_bl = max_skin_blood_flow
        if m_bl < 0.5:
            m_bl = 0.5
        m_rsw = c_sw * warm_b * math.exp(warm_sk / 10.7)  # regulatory sweating
        if m_rsw > max_sweating:
            m_rsw = max_sweating
        e_rsw = 0.68 * m_rsw  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        e_req = (
            rm - q_res - c_res - q_sensible
        )  # evaporative heat loss required for tmp regulation
        e_max = (math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (
            r_ea + r_ecl
        )
        if e_max == 0:  # added this otherwise e_rsw / e_max cannot be calculated
            e_max = 0.001
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        w = 0.06 + 0.94 * p_rsw  # skin wetness
        e_diff = w * e_max - e_rsw  # vapor diffusion through skin
        if w > w_max:
            w = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            w = w_max
        e_skin = (
            e_rsw + e_diff
        )  # total evaporative heat loss sweating and vapor diffusion
        m_rsw = (
            e_rsw / 0.68
        )  # back calculating the mass of regulatory sweating as a function of e_rsw
        met_shivering = 19.4 * colds * c_cold  # met shivering W/m2
        m = rm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417)

    q_skin = q_sensible + e_skin  # total heat loss from skin, W
    # p_s_sk saturation vapour pressure of water of the skin
    p_s_sk = math.exp(18.6686 - 4030.183 / (t_skin + 235.0))

    # standard environment - where _s at end of the variable names stands for standard
    h_r_s = h_r  # standard environment radiative heat transfer coefficient

    h_c_s = 3.0 * pow(pressure_in_atmospheres, 0.53)
    if not calculate_ce and met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_c_s = max(h_c_s, h_c_met)
    if h_c_s < 3.0:
        h_c_s = 3.0

    h_t_s = (
        h_c_s + h_r_s
    )  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_clo_s = (
        1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835
    )  # thermal resistance of clothing, °C M^2 /W
    r_cl_s = 0.155 * r_clo_s  # thermal insulation of the clothing in M2K/W
    f_a_cl_s = 1.0 + k_clo * r_clo_s  # increase in body surface area due to clothing
    f_cl_s = 1.0 / (
        1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s
    )  # ratio of surface clothed body over nude body
    i_m_s = 0.45  # permeation efficiency of water vapour through the clothing layer
    i_cl_s = (
        i_m_s * h_c_s / h_t_s * (1 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s)
    )  # clothing vapor permeation efficiency
    r_a_s = 1.0 / (f_a_cl_s * h_t_s)  # resistance of air layer to dry heat
    r_ea_s = 1.0 / (lr * f_a_cl_s * h_c_s)
    r_ecl_s = r_cl_s / (lr * i_cl_s)
    h_d_s = 1.0 / (r_a_s + r_cl_s)
    h_e_s = 1.0 / (r_ea_s + r_ecl_s)

    # calculate Standard Effective Temperature (SET)
    delta = 0.0001
    dx = 100.0
    set_old = round(t_skin - q_skin / h_d_s, 2)
    while abs(dx) > 0.01:
        err_1 = (
            q_skin
            - h_d_s * (t_skin - set_old)
            - w
            * h_e_s
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + 235.0))))
        )
        err_2 = (
            q_skin
            - h_d_s * (t_skin - (set_old + delta))
            - w
            * h_e_s
            * (
                p_s_sk
                - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + delta + 235.0)))
            )
        )
        _set = set_old - delta * err_1 / (err_2 - err_1)
        dx = _set - set_old
        set_old = _set

    # calculate Effective Temperature (ET)
    h_d = 1 / (r_a + r_clo)
    h_e = 1 / (r_ea + r_ecl)
    et_old = t_skin - q_skin / h_d
    delta = 0.0001
    dx = 100.0
    while abs(dx) > 0.01:
        err_1 = (
            q_skin
            - h_d * (t_skin - et_old)
            - w
            * h_e
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + 235.0))))
        )
        err_2 = (
            q_skin
            - h_d * (t_skin - (et_old + delta))
            - w
            * h_e
            * (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + delta + 235.0))))
        )
        et = et_old - delta * err_1 / (err_2 - err_1)
        dx = et - et_old
        et_old = et

    tbm_l = (0.194 / 58.15) * rm + 36.301  # lower limit for evaporative regulation
    tbm_h = (0.347 / 58.15) * rm + 36.669  # upper limit for evaporative regulation

    t_sens = 0.4685 * (t_body - tbm_l)  # predicted thermal sensation
    if (t_body >= tbm_l) & (t_body < tbm_h):
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    elif t_body >= tbm_h:
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h)

    disc = (
        4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff)
    )  # predicted thermal discomfort
    if disc <= 0:
        disc = t_sens

    # PMV Gagge
    pmv_gagge = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff)

    # PMV SET
    dry_set = h_d_s * (t_skin - _set)
    e_req_set = rm - c_res - q_res - dry_set
    pmv_set = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff)

    # Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement
    ps = 100 * (1.13 * (t_op**0.5) - 0.24 * t_op + 2.7 * (v**0.5) - 0.99 * v)

    return (
        _set,
        e_skin,
        e_rsw,
        e_max,
        q_sensible,
        q_skin,
        q_res,
        t_core,
        t_skin,
        m_bl,
        m_rsw,
        w,
        w_max,
        et,
        pmv_gagge,
        pmv_set,
        disc,
        t_sens,
    )


@vectorize(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
)
def _two_nodes_optimized_return_set(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atmospheric,
    body_position,
):
    return _two_nodes_optimized(
        tdb,
        tr,
        v,
        met,
        clo,
        vapor_pressure,
        wme,
        body_surface_area,
        p_atmospheric,
        body_position,
        True,
    )[0]
