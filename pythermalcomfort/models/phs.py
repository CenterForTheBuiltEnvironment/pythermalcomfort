import math
from typing import Union

import numpy as np
from numba import jit

from pythermalcomfort.classes_input import PHSInputs
from pythermalcomfort.classes_return import PHS
from pythermalcomfort.utilities import (
    Postures,
    _check_standard_compliance_array,
    met_to_w_m2,
    p_sat,
)


def phs(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    v: Union[float, list[float]],
    rh: Union[float, list[float]],
    met: Union[float, list[float]],
    clo: Union[float, list[float]],
    posture: Union[str, list[str]],
    wme: Union[float, int, np.ndarray, list[float], list[int]] = 0,
    round_output: bool = True,
    **kwargs,
) -> PHS:
    """Calculates the Predicted Heat Strain (PHS) index based in compliance
    with the ISO 7933:2004 Standard [7933ISO2004]_. The ISO 7933 provides a method for
    the analytical evaluation and interpretation of the thermal stress
    experienced by a subject in a hot environment. It describes a method for
    predicting the sweat rate and the internal core temperature that the human
    body will develop in response to the working conditions.

    The PHS model can be used to predict the: heat by respiratory convection, heat flow
    by respiratory evaporation, steady state mean skin temperature, instantaneous value
    of skin temperature, heat accumulation associated with the metabolic rate, maximum
    evaporative heat flow at the skin surface, predicted sweat rate, predicted evaporative
    heat flow, and rectal temperature.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    posture: string or list of strings
        a string value presenting posture of person "sitting", "standing", or "crouching"
    wme : float or list of floats
        external work, [met] default 0
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Other Parameters
    ----------------
    limit_inputs : bool, optional
        If True, limits the input parameters to the standard's applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns values even if input values are
            outside the applicability limits of the model.

            The 7933 limits are 15 < tdb [°C] < 50, 0 < tr [°C] < 60,
            0 < vr [m/s] < 3, 1.7 < met [met] < 7.5, and 0.1 < clo [clo] < 1.

    i_mst : float, optional
        Static moisture permeability index, [dimensionless]. Defaults to 0.38.
    a_p : float, optional
        Fraction of the body surface covered by the reflective clothing, [dimensionless]. Defaults to 0.54.
    drink : int, optional
        1 if workers can drink freely, 0 otherwise. Defaults to 1.
    weight : float, optional
        Body weight, [kg]. Defaults to 75.
    height : float, optional
        Height, [m]. Defaults to 1.8.
    walk_sp : float, optional
        Walking speed, [m/s]. Defaults to 0.
    theta : float, optional
        Angle between walking direction and wind direction, [degrees]. Defaults to 0.
    acclimatized : int, optional
        100 if acclimatized subject, 0 otherwise. Defaults to 100.
    duration : int, optional
        Duration of the work sequence, [minutes]. Defaults to 480.
    f_r : float, optional
        Emissivity of the reflective clothing, [dimensionless]. Defaults to 0.97.
    t_sk : float, optional
        Mean skin temperature when worker starts working, [°C]. Defaults to 34.1.
    t_cr : float, optional
        Mean core temperature when worker starts working, [°C]. Defaults to 36.8.
    t_re : float, optional
        Mean rectal temperature when worker starts working, [°C]. Defaults to False.
    t_cr_eq : float, optional
        Mean core temperature as a function of met when worker starts working, [°C]. Defaults to False.
    sweat_rate : float, optional
        Initial sweat rate, [g/h]. Defaults to 0.

    Returns
    -------
    PHS
        A dataclass containing the Predicted Heat Strain. See :py:class:`~pythermalcomfort.classes_return.PHS` for more details.
        To access the individual attributes, use the corresponding attribute of the returned `PHS` instance, e.g., `result.t_re`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import phs

        result = phs(
            tdb=40, tr=40, rh=33.85, v=0.3, met=2.5, clo=0.5, posture="standing", wme=0
        )
        print(result.t_re)  # 37.5

        result = phs(
            tdb=[40, 45],
            tr=[40, 45],
            v=[0.3, 0.4],
            rh=[33.85, 40],
            met=[2.5, 2.6],
            clo=[0.5, 0.6],
            posture=["standing", "standing"],
            wme=[0, 0],
        )
        print(result.t_re)  # [37.5 43.4]
    """
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
        "round": True,
        "limit_inputs": True,
    }
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

    p_a = p_sat(tdb) / 1000 * rh / 100

    acclimatized = int(acclimatized)
    if acclimatized < 0 or acclimatized > 100:
        raise ValueError("Acclimatized should be between 0 and 100")

    if drink not in [0, 1]:
        raise ValueError("Drink should be 0 or 1")

    if weight <= 0 or weight > 1000:
        raise ValueError(
            "The weight of the person should be in kg and it cannot exceed 1000"
        )

    # I needed to create this variable since vectorize accept no more than 34 inputs+outputs
    combined_drink_acc_weight = drink * 1000000 + acclimatized * 1000 + weight

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
        combined_drink_acc_weight=combined_drink_acc_weight,
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
            "7933", tdb=tdb, tr=tr, v=v, met=met, clo=clo, p_a=p_a
        )
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(p_a_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        for key in output.keys():
            output[key] = np.where(all_valid, output[key], np.nan)

    if round_output:
        for key in output.keys():
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
    combined_drink_acc_weight,
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
):
    drink = int(combined_drink_acc_weight / 1000000)
    acclimatized = int((combined_drink_acc_weight - drink * 1000000) / 1000)
    weight = combined_drink_acc_weight - drink * 1000000 - acclimatized * 1000

    # DuBois body surface area [m2]
    a_dubois = 0.202 * (weight**0.425) * (height**0.725)
    sp_heat = 57.83 * weight / a_dubois  # specific heat of the body
    d_lim_t_re = 0  # maximum allowable exposure time for heat storage [min]
    # maximum allowable exposure time for water loss, mean subject [min]
    d_lim_loss_50 = 0
    # maximum allowable exposure time for water loss, 95 % of the working population [min]
    d_lim_loss_95 = 0
    # maximum water loss to protect a mean subject [g]
    d_max_50 = 0.075 * weight * 1000
    # maximum water loss to protect 95 % of the working population [g]
    d_max_95 = 0.05 * weight * 1000
    sweat_rate = sw_tot

    def_dir = 0
    if theta != 0:
        # def_dir = 1 for unidirectional walking, def_dir = 0 for omni-directional walking
        def_dir = 1
    if walk_sp == 0:
        def_speed = 0
    else:
        def_speed = 1

    # radiating area dubois
    a_r_du = 0.7
    if posture == Postures.standing.value:
        a_r_du = 0.77
    if posture == Postures.crouching.value:
        a_r_du = 0.67

    # evaluation of the max sweat rate as a function of the metabolic rate
    sw_max = (met - 32) * a_dubois
    if sw_max > 400:
        sw_max = 400
    if sw_max < 250:
        sw_max = 250
    if acclimatized >= 50:
        sw_max = sw_max * 1.25

    # max skin wettedness
    if acclimatized < 50:
        w_max = 0.85
    else:
        w_max = 1

    # static clothing insulation
    i_cl_st = clo * 0.155
    fcl = 1 + 0.3 * clo

    # Static boundary layer thermal insulation in quiet air
    i_a_st = 0.111

    # Total static insulation
    i_tot_st = i_cl_st + i_a_st / fcl
    if def_speed > 0:
        if def_dir == 1:  # Unidirectional walking
            v_r = abs(v - walk_sp * math.cos(3.14159 * theta / 180))
        else:  # Omni-directional walking IF
            if v < walk_sp:
                v_r = walk_sp
            else:
                v_r = v
    else:
        walk_sp = 0.0052 * (met - 58)
        if walk_sp > 0.7:
            walk_sp = 0.7
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
        (0.066 * v_ux - 0.398) * v_ux + (0.094 * w_a_ux - 0.378) * w_a_ux
    )
    if corr_cl > 1:
        corr_cl = 1
    # correction for the dynamic total dry thermal insulation at 0 clo
    corr_ia = math.exp((0.047 * v_r - 0.472) * v_r + (0.117 * w_a_ux - 0.342) * w_a_ux)
    if corr_ia > 1:
        corr_ia = 1
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
    if im_dyn > 0.9:
        im_dyn = 0.9
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
    hc_dyn = 2.38 * abs(t_sk - tdb) ** 0.25
    if z > hc_dyn:
        hc_dyn = z

    aux_r = 5.67e-08 * a_r_du
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
            e_v_eff = 1 - w_req**2 / 2
            if w_req > 1:
                e_v_eff = (2 - w_req) ** 2 / 2
            sw_req = e_req / e_v_eff
            if sw_req > sw_max:
                sw_req = sw_max
        sweat_rate = sweat_rate * const_sw + sw_req * (1 - const_sw)

        if sweat_rate <= 0:
            e_p = 0  # predicted evaporative heat flow [W/m2]
            sweat_rate = 0
        else:
            k = e_max / sweat_rate
            wp = 1
            if k >= 0.5:
                wp = -k + math.sqrt(k * k + 2)
            if wp > w_max:
                wp = w_max
            e_p = wp * e_max

        # body heat storage rate [W/m2]
        d_storage = e_req - e_p + d_stored_eq
        t_cr_new = t_cr0
        while True:
            t_sk_t_cr_wg = 0.3 - 0.09 * (t_cr_new - 36.8)
            if t_sk_t_cr_wg > 0.3:
                t_sk_t_cr_wg = 0.3
            if t_sk_t_cr_wg < 0.1:
                t_sk_t_cr_wg = 0.1
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
