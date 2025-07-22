from __future__ import annotations

import math

import numpy as np
from numba import jit

from pythermalcomfort.classes_input import PHSInputs
from pythermalcomfort.classes_return import PHS
from pythermalcomfort.utilities import (
    Models,
    Postures,
    _check_standard_compliance_array,
    met_to_w_m2,
    p_sat,
)


def phs(
    tdb: float | list[float],
    tr: float | list[float],
    v: float | list[float],
    rh: float | list[float],
    met: float | list[float],
    clo: float | list[float],
    posture: str | list[str],
    wme: float | np.ndarray | list[float] | list[int] = 0,
    round_output: bool = True,
    model: str = Models.iso_7933_2023.value,
    **kwargs,
) -> PHS:
    """
    Calculate the Predicted Heat Strain (PHS) index according to ISO 7933:2004 or ISO 7933:2023 standards.
    
    This function predicts physiological heat strain responses—including rectal temperature, skin temperature, core temperature, sweat rate, and maximum allowable exposure times—based on environmental, clothing, and individual parameters. It supports both single values and lists/arrays for batch calculations, and applies model-specific logic and limits as defined by the selected ISO standard.
    
    Parameters:
        tdb (float or list of float): Dry bulb air temperature in °C.
        tr (float or list of float): Mean radiant temperature in °C.
        v (float or list of float): Air speed in m/s.
        rh (float or list of float): Relative humidity in %.
        met (float or list of float): Metabolic rate in met units.
        clo (float or list of float): Clothing insulation in clo units.
        posture (str or list of str): Posture of the person ("sitting", "standing", or "crouching").
        wme (float or list of float, optional): External work in met units. Defaults to 0.
        round_output (bool, optional): If True, rounds output values. Defaults to True.
        model (str, optional): ISO standard version ("7933-2023" or "7933-2004"). Defaults to "7933-2023".
        **kwargs: Additional optional physiological and environmental parameters, such as acclimatization, drink permission, body weight, height, walking speed, clothing properties, initial temperatures, and sweat rate.
    
    Returns:
        PHS: A dataclass instance containing predicted rectal temperature, skin temperature, core temperature, equilibrium core temperature, skin-core weighting factor, sweat rate, total sweat loss, and maximum allowable exposure times for water loss and heat storage.
    """
    if model not in [Models.iso_7933_2004.value, Models.iso_7933_2023.value]:
        error_msg = (
            f"Model '{model}' is not supported. "
            f"Supported models are: {Models.iso_7933_2004.value} and {Models.iso_7933_2023.value}."
        )
        raise ValueError(error_msg)

    PHSInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        round_output=round_output,
        wme=wme,
        posture=posture,
    )

    default_kwargs = {
        "i_mst": 0.38,
        "a_p": 0.54,
        "drink": 1,
        "weight": 75,
        "height": 1.8,
        "walk_sp": 0,
        "theta": 0,
        "acclimatized": 100,
        "duration": 480,
        "f_r": 0.97,
        "t_sk": 34.1,
        "t_cr": 36.8,
        "t_re": False,
        "t_cr_eq": False,
        "t_sk_t_cr_wg": 0.3,
        "sweat_rate": 0,
        "limit_inputs": True,
    }

    if model == Models.iso_7933_2023.value:
        # override changed default kwargs for 2023 standard
        overrides_2023 = {
            "f_r": 0.42,
            "t_re": 36.8,
            "t_cr_eq": 36.8,
        }
        default_kwargs.update(overrides_2023)

    kwargs = {**default_kwargs, **kwargs}

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met) * met_to_w_m2
    clo = np.array(clo)
    wme = np.array(wme) * met_to_w_m2
    posture = np.array(posture)

    i_mst = kwargs["i_mst"]
    a_p = kwargs["a_p"]
    drink = kwargs["drink"]
    weight = kwargs["weight"]
    height = kwargs["height"]
    walk_sp = kwargs["walk_sp"]
    theta = kwargs["theta"]
    acclimatized = kwargs["acclimatized"]
    duration = kwargs["duration"]
    f_r = kwargs["f_r"]
    t_sk = kwargs["t_sk"]
    t_cr = kwargs["t_cr"]
    t_re = kwargs["t_re"]
    t_cr_eq = kwargs["t_cr_eq"]
    t_sk_t_cr_wg = kwargs["t_sk_t_cr_wg"]
    sweat_rate = kwargs["sweat_rate"]
    limit_inputs = kwargs["limit_inputs"]

    if model == Models.iso_7933_2023.value:
        p_a = 0.6105 * np.exp(17.27 * tdb / (tdb + 237.3)) * rh / 100
    else:  # model == Models.iso_7933_2004.value:
        p_a = p_sat(tdb) / 1000 * rh / 100

    acclimatized = int(acclimatized)
    if acclimatized not in [0, 100]:
        raise ValueError("Acclimatized should be 0 or 100")

    if drink not in [0, 1]:
        raise ValueError("Drink should be 0 or 1")

    if weight <= 0 or weight > 1000:
        raise ValueError(
            "The weight of the person should be in kg and it cannot exceed 1000",
        )

    if not t_re:
        t_re = t_cr
    if not t_cr_eq:
        t_cr_eq = t_cr

    (
        t_re,
        t_sk,
        t_cr,
        t_cr_eq,
        t_sk_t_cr_wg,
        sweat_rate,
        sw_tot_g,
        d_lim_loss_50,
        d_lim_loss_95,
        d_lim_t_re,
    ) = _phs_optimized(
        tdb=tdb,
        tr=tr,
        v=v,
        p_a=p_a,
        met=met,
        clo=clo,
        posture=posture,
        drink=drink,
        acclimatized=acclimatized,
        weight=weight,
        wme=wme,
        i_mst=i_mst,
        a_p=a_p,
        height=height,
        walk_sp=walk_sp,
        theta=theta,
        duration=duration,
        f_r=f_r,
        t_sk=t_sk,
        t_cr=t_cr,
        t_re=t_re,
        t_cr_eq=t_cr_eq,
        t_sk_t_cr_wg=t_sk_t_cr_wg,
        sw_tot=sweat_rate,
        model=model,
    )

    output = {
        "t_re": t_re,
        "t_sk": t_sk,
        "t_cr": t_cr,
        "t_cr_eq": t_cr_eq,
        "t_sk_t_cr_wg": t_sk_t_cr_wg,
        "d_lim_loss_50": d_lim_loss_50,
        "d_lim_loss_95": d_lim_loss_95,
        "d_lim_t_re": d_lim_t_re,
        "water_loss_watt": sweat_rate,
        "water_loss": sw_tot_g,
    }

    if limit_inputs:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            p_a_valid,
            met_valid,
            clo_valid,
        ) = _check_standard_compliance_array(
            model,
            tdb=tdb,
            tr=tr,
            v=v,
            met=met,
            clo=clo,
            p_a=p_a,
        )
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(p_a_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        for key in output:
            output[key] = np.where(all_valid, output[key], np.nan)

    if round_output:
        for key in output:
            if key != "t_sk_t_cr_wg":
                output[key] = np.around(output[key], 1)
            else:
                output[key] = np.around(output[key], 2)

    return PHS(**output)


# Constants
const_t_eq = math.exp(-1 / 10)
const_t_sk = math.exp(-1 / 3)
const_sw = math.exp(-1 / 10)


@np.vectorize
@jit(nopython=True, cache=True)
def _phs_optimized(
    tdb,
    tr,
    v,
    p_a,
    met,
    clo,
    posture,
    drink,
    acclimatized,
    weight,
    wme,
    i_mst,
    a_p,
    height,
    walk_sp,
    theta,
    duration,
    f_r,
    t_sk,
    t_cr,
    t_re,
    t_cr_eq,
    t_sk_t_cr_wg,
    sw_tot,
    model,
):
    # DuBois body surface area [m2]
    a_dubois = 0.202 * (weight**0.425) * (height**0.725)
    # specific heat of the body [J/kg/C/min]
    sp_heat = met_to_w_m2 * weight / a_dubois

    d_lim_t_re = 0  # maximum allowable exposure time for heat storage [min]
    # maximum allowable exposure time for water loss, mean subject [min]
    d_lim_loss_50 = 0
    # maximum allowable exposure time for water loss, 95 % of the working population [min]
    d_lim_loss_95 = 0

    if model == Models.iso_7933_2023.value:
        # 2023 standard only has one d_max value
        d_max_50 = (0.03 if drink == 0 else 0.05) * weight * 1000
        d_max_95 = (0.03 if drink == 0 else 0.05) * weight * 1000
    else:  # model == Models.iso_7933_2004.value:
        # maximum water loss to protect a mean subject [g]
        d_max_50 = 0.075 * weight * 1000
        # maximum water loss to protect 95 % of the working population [g]
        d_max_95 = 0.05 * weight * 1000

    sweat_rate = sw_tot

    # def_dir = 1 for unidirectional walking, def_dir = 0 for omni-directional walking
    def_dir = 1 if theta != 0 else 0
    def_speed = 0 if walk_sp == 0 else 1

    # radiating area dubois
    if posture == Postures.standing.value:
        a_r_du = 0.77
    elif posture == Postures.sitting.value:
        a_r_du = 0.7
    else:  # posture == Postures.crouching.value:
        a_r_du = 0.67

    # evaluation of the max sweat rate as a function of the metabolic rate
    if model == Models.iso_7933_2004.value:
        sw_max = (met - 32) * a_dubois
        sw_max = min(sw_max, 400)
        sw_max = max(sw_max, 250)
        if acclimatized == 100:
            sw_max = sw_max * 1.25
    else:  # model == Models.iso_7933_2023.value:
        sw_max = 400 if acclimatized == 0 else 500

    # max skin wettedness
    w_max = 0.85 if acclimatized == 0 else 1

    # static clothing insulation
    i_cl_st = clo * 0.155

    fcl = 1 + 0.28 * clo if model == Models.iso_7933_2023.value else 1 + 0.3 * clo

    # Static boundary layer thermal insulation in quiet air
    i_a_st = 0.111

    # Total static insulation
    i_tot_st = i_cl_st + i_a_st / fcl
    if def_speed > 0:
        if def_dir == 1:  # Unidirectional walking
            v_r = abs(v - walk_sp * math.cos(3.14159 * theta / 180))
        elif v < walk_sp:
            v_r = walk_sp
        else:
            v_r = v
    else:
        walk_sp = 0.0052 * (met - 58)
        walk_sp = min(walk_sp, 0.7)
        v_r = v

    # Dynamic clothing insulation - correction for wind (Var) and walking speed
    v_ux = v_r
    if v_r > 3:
        v_ux = 3
    w_a_ux = walk_sp
    if walk_sp > 1.5:
        w_a_ux = 1.5
    # correction for the dynamic total dry thermal insulation at or above 0.6 clo
    corr_cl = 1.044 * math.exp(
        (0.066 * v_ux - 0.398) * v_ux + (0.094 * w_a_ux - 0.378) * w_a_ux,
    )
    corr_cl = min(corr_cl, 1)
    # correction for the dynamic total dry thermal insulation at 0 clo
    corr_ia = math.exp((0.047 * v_r - 0.472) * v_r + (0.117 * w_a_ux - 0.342) * w_a_ux)
    corr_ia = min(corr_ia, 1)
    corr_tot = corr_cl
    if clo <= 0.6:
        corr_tot = ((0.6 - clo) * corr_ia + clo * corr_cl) / 0.6
    # total dynamic clothing insulation
    i_tot_dyn = i_tot_st * corr_tot
    # dynamic boundary layer thermal insulation
    i_a_dyn = corr_ia * i_a_st
    i_cl_dyn = i_tot_dyn - i_a_dyn / fcl
    # correction for the dynamic permeability index
    corr_e = (2.6 * corr_tot - 6.5) * corr_tot + 4.9
    im_dyn = i_mst * corr_e
    im_dyn = min(im_dyn, 0.9)
    r_t_dyn = i_tot_dyn / im_dyn / 16.7
    t_exp = 28.56 + 0.115 * tdb + 0.641 * p_a  # expired air temperature
    # respiratory convective heat flow [W/m2]
    c_res = 0.001516 * met * (t_exp - tdb)
    # respiratory evaporative heat flow [W/m2]
    e_res = 0.00127 * met * (59.34 + 0.53 * tdb - 11.63 * p_a)
    z = 3.5 + 5.2 * v_r
    if v_r > 1:
        z = 8.7 * v_r**0.6

    # dynamic convective heat transfer coefficient
    if model == Models.iso_7933_2004.value:
        hc_dyn = 2.38 * abs(t_sk - tdb) ** 0.25
    else:  # model == Models.iso_7933_2023.value:
        t_cl = tr + 0.1  # clothing surface temperature
        hc_dyn = 2.38 * abs(t_cl - tdb) ** 0.25

    hc_dyn = max(hc_dyn, z)

    aux_r = 5.67e-08 * a_r_du

    if model == Models.iso_7933_2023.value:
        f_cl_r = (1 - a_p) * 0.97 + a_p * (1 - f_r)
    else:  # model == Models.iso_7933_2004.value:
        f_cl_r = (1 - a_p) * 0.97 + a_p * f_r

    for time in range(1, duration + 1):
        t_sk0 = t_sk
        t_re0 = t_re
        t_cr0 = t_cr
        t_cr_eq0 = t_cr_eq
        t_sk_t_cr_wg0 = t_sk_t_cr_wg

        # equilibrium core temperature associated to the metabolic rate
        t_cr_eq_m = 0.0036 * met + 36.6
        # Core temperature at this minute, by exponential averaging
        t_cr_eq = t_cr_eq0 * const_t_eq + t_cr_eq_m * (1 - const_t_eq)
        # Heat storage associated with this core temperature increase during the last minute
        d_stored_eq = sp_heat * (t_cr_eq - t_cr_eq0) * (1 - t_sk_t_cr_wg0)
        # skin temperature prediction -- clothed model
        t_sk_eq_cl = 12.165 + 0.02017 * tdb + 0.04361 * tr + 0.19354 * p_a - 0.25315 * v
        t_sk_eq_cl = t_sk_eq_cl + 0.005346 * met + 0.51274 * t_re
        # nude model
        t_sk_eq_nu = 7.191 + 0.064 * tdb + 0.061 * tr + 0.198 * p_a - 0.348 * v
        t_sk_eq_nu = t_sk_eq_nu + 0.616 * t_re
        if clo >= 0.6:
            t_sk_eq = t_sk_eq_cl
        elif clo <= 0.2:
            t_sk_eq = t_sk_eq_nu
        else:
            t_sk_eq = t_sk_eq_nu + 2.5 * (t_sk_eq_cl - t_sk_eq_nu) * (clo - 0.2)

        # skin temperature [C]
        t_sk = t_sk0 * const_t_sk + t_sk_eq * (1 - const_t_sk)
        # Saturated water vapour pressure at the surface of the skin
        p_sk = 0.6105 * math.exp(17.27 * t_sk / (t_sk + 237.3))
        t_cl = tr + 0.1  # clothing surface temperature
        while True:
            # radiative heat transfer coefficient
            h_r = f_cl_r * aux_r * ((t_cl + 273) ** 4 - (tr + 273) ** 4) / (t_cl - tr)
            t_cl_new = (fcl * (hc_dyn * tdb + h_r * tr) + t_sk / i_cl_dyn) / (
                fcl * (hc_dyn + h_r) + 1 / i_cl_dyn
            )
            if abs(t_cl - t_cl_new) <= 0.001:
                break
            t_cl = (t_cl + t_cl_new) / 2

        convection = fcl * hc_dyn * (t_cl - tdb)
        radiation = fcl * h_r * (t_cl - tr)
        # maximum evaporative heat flow at the skin surface [W/m2]
        e_max = (p_sk - p_a) / r_t_dyn
        if e_max == 0:  # added this otherwise e_req / e_max cannot be calculated
            e_max = 0.001
        # required evaporative heat flow [W/m2]
        e_req = met - d_stored_eq - wme - c_res - e_res - convection - radiation
        # required skin wettedness
        w_req = e_req / e_max

        if e_req <= 0:
            e_req = 0
            sw_req = 0  # required sweat rate [W/m2]
        elif e_max <= 0:
            e_max = 0
            sw_req = sw_max
        elif w_req >= 1.7:
            sw_req = sw_max
        else:
            e_v_eff = (2 - w_req) ** 2 / 2 if w_req > 1 else 1 - w_req**2 / 2

            e_v_eff = max(0.05, e_v_eff)

            sw_req = e_req / e_v_eff
            sw_req = min(sw_req, sw_max)
        sweat_rate = sweat_rate * const_sw + sw_req * (1 - const_sw)

        if sweat_rate <= 0:
            e_p = 0  # predicted evaporative heat flow [W/m2]
            sweat_rate = 0
        else:
            k = e_max / sweat_rate
            wp = 1
            if k >= 0.5:
                wp = -k + math.sqrt(k * k + 2)
            wp = min(wp, w_max)
            e_p = wp * e_max

        # body heat storage rate [W/m2]
        d_storage = e_req - e_p + d_stored_eq
        t_cr_new = t_cr0
        while True:
            t_sk_t_cr_wg = 0.3 - 0.09 * (t_cr_new - 36.8)
            t_sk_t_cr_wg = min(t_sk_t_cr_wg, 0.3)
            t_sk_t_cr_wg = max(t_sk_t_cr_wg, 0.1)
            t_cr = (
                d_storage / sp_heat
                + t_sk0 * t_sk_t_cr_wg0 / 2
                - t_sk * t_sk_t_cr_wg / 2
            )
            t_cr = (t_cr + t_cr0 * (1 - t_sk_t_cr_wg0 / 2)) / (1 - t_sk_t_cr_wg / 2)
            if abs(t_cr - t_cr_new) <= 0.001:
                break
            t_cr_new = (t_cr_new + t_cr) / 2

        t_re = t_re0 + (2 * t_cr - 1.962 * t_re0 - 1.31) / 9
        if d_lim_t_re == 0 and t_re >= 38:
            d_lim_t_re = time
        sw_tot = sw_tot + sweat_rate + e_res
        sw_tot_g = sw_tot * 2.67 * a_dubois / 1.8 / 60
        if d_lim_loss_50 == 0 and sw_tot_g >= d_max_50:
            d_lim_loss_50 = time
        if d_lim_loss_95 == 0 and sw_tot_g >= d_max_95:
            d_lim_loss_95 = time

    # in the standard the if statement is within the while loop, causing it to decay exponentially
    if drink == 0:
        d_lim_loss_95 = d_lim_loss_95 * 0.6
        d_lim_loss_50 = d_lim_loss_95
    if d_lim_loss_50 == 0:
        d_lim_loss_50 = duration
    if d_lim_loss_95 == 0:
        d_lim_loss_95 = duration
    if d_lim_t_re == 0:
        d_lim_t_re = duration

    return (
        t_re,
        t_sk,
        t_cr,
        t_cr_eq,
        t_sk_t_cr_wg,
        sweat_rate,
        sw_tot_g,
        d_lim_loss_50,
        d_lim_loss_95,
        d_lim_t_re,
    )
