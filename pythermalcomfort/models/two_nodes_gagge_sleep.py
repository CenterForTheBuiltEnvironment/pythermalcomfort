from __future__ import annotations

import math

import numpy as np

from pythermalcomfort.classes_input import GaggeTwoNodesSleepInputs
from pythermalcomfort.classes_return import GaggeTwoNodesSleep


def two_nodes_gagge_sleep(
    tdb: float | list[float],
    tr: float | list[float],
    v: float | list[float],
    rh: float | list[float],
    clo: float | list[float],
    thickness_quilt: float | list[float],
    wme: float = 0,
    p_atm: float = 101325,
    **kwargs,
) -> GaggeTwoNodesSleep:
    """Adaption of the Gagge two-node model for sleep thermal environment, developed by Yan, S., Xiong, J., Kim, J. and de Dear, R. [Yan2022]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].

        .. note::
            tdb, tr, v, rh, clo and thickness must have the same length.
            This length will be the duration of the simulation.

    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    clo : float or list of floats
        Clothing insulation, [clo].
    thickness_quilt : float or list of floats
        Thickness of the quilt. [cm].
    wme : float, optional
        External work, [met]. Defaults to 0.
    p_atm : float, optional
        Atmospheric pressure, default value 101325 [Pa]. Defaults to 101325.
    **kwargs : dict
        Keyword arguments:

        * ltime : int, optional
            Number of time steps for each iteration. Defaults to 1.
        * height : float, optional
            Height of the person, [cm]. Defaults to 171.
        * weight : float, optional
            Weight of the person, [kg]. Defaults to 70.
        * c_sw : float, optional
            Driving coefficient for regulatory sweating. Defaults to 170.
        * c_dil : float, optional
            Driving coefficient for vasodilation. Defaults to 120.
        * c_str : float, optional
            Driving coefficient for vasoconstriction. Defaults to 0.5.
        * temp_skin_neutral : float, optional
            Skin temperature at neutral conditions, [°C]. Defaults to 33.7.
        * temp_core_neutral : float, optional
            Core temperature at neutral conditions, [°C]. Defaults to 36.8.
        * e_skin : float, optional
            Total evaporative heat loss, [W]. Defaults to 0.094.
        * alfa : float, optional
            Dynamic fraction of total body mass assigned to the skin node. Defaults to 0.1.
        * skin_blood_flow : float, optional
            Skin-blood-flow rate per unit surface area, [kg/h/m2]. Defaults to 6.3.
        * met_shivering : float, optional
            Metabolic rate due to shivering, [met]. Defaults to 0.

    Returns
    -------
    GaggeTwoNodesSleep
        A dataclass containing the results of the Gagge two-node model for sleep thermal environment.
        See :py:class:`~pythermalcomfort.classes_return.GaggeTwoNodesSleep` for more details.
        To access the results, use the corresponding attributes of the returned instance, e.g. `result.e_skin`.
    """

    ltime = kwargs.pop("ltime", 1)
    height = kwargs.pop("height", 171)
    weight = kwargs.pop("weight", 70)
    c_sw = kwargs.pop("c_sw", 170)
    c_dil = kwargs.pop("c_dil", 120)
    c_str = kwargs.pop("c_str", 0.5)
    temp_skin_neutral = kwargs.pop("temp_skin_neutral", 33.7)
    temp_core_neutral = kwargs.pop("temp_core_neutral", 36.8)
    e_skin = kwargs.pop("e_skin", 0.094)
    alfa = kwargs.pop("alfa", 0.1)
    skin_blood_flow = kwargs.pop("skin_blood_flow", 6.3)
    met_shivering = kwargs.pop("met_shivering", 0)

    if kwargs:
        raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")

    GaggeTwoNodesSleepInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        clo=clo,
        thickness_quilt=thickness_quilt,
        wme=wme,
        p_atm=p_atm,
    )

    tdb = np.atleast_1d(tdb)
    tr = np.atleast_1d(tr)
    v = np.atleast_1d(v)
    rh = np.atleast_1d(rh)
    clo = np.atleast_1d(clo)
    thickness_quilt = np.atleast_1d(thickness_quilt)

    # These variables should have the same length, which will be the duration
    lengths = [len(x) for x in (tdb, tr, v, rh, clo, thickness_quilt)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Parameters tdb, tr, v, rh, clo and thickness must have the same length. Got lengths {lengths}"
        )
    duration = lengths[0]

    # this should be the same dataclass as the return of _optimised
    result = {
        "t_core": temp_core_neutral,
        "t_skin": temp_skin_neutral,
        "e_skin": e_skin,
        "met_shivering": met_shivering,
        "alfa": alfa,
        "skin_blood_flow": skin_blood_flow,
    }

    results = []

    for i in range(duration):
        met = (
            -0.000000000000575 * ((i - 1) / 60) ** 5
            + 0.000000000785521 * ((i - 1) / 60) ** 4
            - 0.00000039173563 * ((i - 1) / 60) ** 3
            + 0.000087620232151 * ((i - 1) / 60) ** 2
            - 0.008801558913211 * ((i - 1) / 60)
            + 1.09952538864493
        )

        t_core = 0.022234 * ((i - 1) / 60) ** 2 - 0.27677 * ((i - 1) / 60) + 37.02

        result = _sleep_set(
            tdb[i],
            tr[i],
            v[i],
            rh[i],
            clo[i],
            thickness=thickness_quilt[i],
            met=met,
            wme=wme,
            p_atm=p_atm,
            ltime=ltime,
            height=height,
            weight=weight,
            c_sw=c_sw,
            c_dil=c_dil,
            c_str=c_str,
            temp_skin_neutral=result["t_skin"],
            temp_core_neutral=t_core,
            e_skin=result["e_skin"],
            alfa=result["alfa"],
            skin_blood_flow=result["skin_blood_flow"],
            met_shivering=result["met_shivering"],
        )

        # results should be a list of the local dataclass
        results.append(result)

    if not results:
        output = {}
    else:
        output = {}
        for key in results[0].keys():
            vals = [d[key] for d in results]
            # only wrap in an array if there’s more than one element
            output[key] = np.array(vals) if len(vals) > 1 else vals[0]

    return GaggeTwoNodesSleep(**output)


def _sleep_set(
    tdb,
    tr,
    v,
    rh,
    clo,
    thickness,
    met,
    wme,
    p_atm,
    ltime,
    height,
    weight,
    c_sw,
    c_dil,
    c_str,
    temp_skin_neutral,
    temp_core_neutral,
    e_skin,
    alfa,
    skin_blood_flow,
    met_shivering,
):
    m = met * 58.2
    w = wme * 58.2
    k_clo = 0.25
    temp_body_neutral = 36.49
    skin_blood_flow_neutral = 6.3
    sbc = 5.6697 * 10**-8
    sa = ((height * weight) / 3600) ** 0.5

    v = max(v, 0.1)
    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    rmm = m

    # initialize some variables
    e_rsw = 0  # heat lost by vaporization sweat
    e_diff = 0  # vapor diffusion through skin
    e_max = 0  # maximum evaporative capacity
    dry = 0
    r_ea = 0
    r_ecl = 0
    p_wet = 0
    t_body = 0
    x = 0

    pressure_in_atmospheres = p_atm / 101325
    r_clo = 0.155 * clo
    f_a_cl = 0.0308 * thickness + 0.7695
    lr = 2.2 / pressure_in_atmospheres
    pa = rh * math.exp(18.6686 - (4030.183 / (tdb + 235))) / 100
    if clo <= 0:
        w_max = 0.38 * v**-0.29
        i_cl = 1
    else:
        w_max = 0.59 * v**-0.08
        i_cl = 0.45

    temp_diff = abs(t_skin - tdb)
    chc = 0.881 * temp_diff**0.368
    h_r = 3.235
    ctc = h_r + chc
    r_a = 1 / (f_a_cl * ctc)
    t_op = (h_r * tr + chc * tdb) / ctc
    t_cl = t_op + (t_skin - t_op) / (ctc * (r_a + r_clo))
    t_cl_old = t_cl
    flag = False

    for _ in range(ltime):
        if flag:
            t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl - t_cl_old) > 0.01:
                flag = False
                t_cl_old = t_cl
            else:
                flag = True

        max_iter = 100
        iter_cnt = 0
        while not flag and iter_cnt < max_iter:
            h_r = 4 * sbc * ((t_cl + tr) / 2 + 273.15) ** 3 * 0.72
            ctc = h_r + chc
            r_a = 1 / (f_a_cl * ctc)
            t_op = (h_r * tr + chc * tdb) / ctc
            t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl - t_cl_old) > 0.01:
                flag = False
                t_cl_old = t_cl
            else:
                flag = True
            iter_cnt += 1

        dry = (t_skin - t_op) / (r_a + r_clo)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * skin_blood_flow)
        q_res = 0.0023 * m * (44 - pa)
        c_res = 0.0014 * m * (34 - tdb)
        s_core = m - hf_cs - q_res - c_res - w
        s_skin = hf_cs - dry - e_skin
        tc_sk = 0.97 * alfa * weight
        tc_cr = 0.97 * (1 - alfa) * weight
        d_t_sk = (s_skin * sa) / tc_sk / 60
        d_t_cr = (s_core * sa) / tc_cr / 60
        t_skin += d_t_sk
        t_core += d_t_cr
        t_body = alfa * t_skin + (1 - alfa) * t_core

        warm_sk = max(t_skin - 33.7, 0)
        cold_s = max(33.7 - t_skin, 0)
        warm_c = max(t_core - temp_core_neutral, 0)
        cold_c = max(temp_core_neutral - t_core, 0)
        warm_b = max(t_body - temp_body_neutral, 0)

        skin_blood_flow = (skin_blood_flow_neutral + c_dil * warm_c) / (
            1 + c_str * cold_s
        )
        skin_blood_flow = min(max(skin_blood_flow, 0.5), 90)
        reg_sw = c_sw * warm_b * math.exp(warm_sk / 10.7)
        reg_sw = min(reg_sw, 500)
        e_rsw = 0.68 * reg_sw
        r_ea = 1 / (lr * f_a_cl * chc)
        r_ecl = r_clo / (lr * i_cl)
        e_max = (_fnsvp(t_skin) - pa) / (r_ea + r_ecl)
        p_rsw = e_rsw / e_max
        p_wet = 0.06 + 0.94 * p_rsw
        e_diff = p_wet * e_max - e_rsw
        if p_wet > w_max:
            p_wet = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1 - p_rsw) * e_max
        e_skin = e_rsw + e_diff
        met_shivering = 19.4 * cold_s * cold_c
        m = rmm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (skin_blood_flow + 0.585417)

    q_skin = dry + e_skin
    rn = m - w
    e_comfort = 0.42 * (rn - 58.2)
    e_comfort = max(e_comfort, 0)
    e_max *= w_max
    h_d = 1 / (r_a + r_clo)
    h_e = 1 / (r_ea + r_ecl)
    wet = p_wet
    p_s_sk = _fnsvp(t_skin)
    chrs = h_r
    chcs = max(3, 5.66 * ((met - 0.85) ** 0.39) if met >= 0.85 else 0)
    ctcs = chcs + chrs
    r_clo_s = 1.52 / ((met - wme) + 0.6944) - 0.1835
    r_cl_s = 0.155 * r_clo_s
    f_a_cl_s = 1 + k_clo * r_clo_s
    f_cl_s = 1 / (1 + 0.155 * f_a_cl_s * ctcs * r_clo_s)
    i_m_s = 0.45
    i_cl_s = i_m_s * chcs / ctcs * (1 - f_cl_s) / (chcs / ctcs - f_cl_s * i_m_s)
    r_a_s = 1 / (f_a_cl_s * ctcs)
    r_ea_s = 1 / (lr * f_a_cl_s * chcs)
    r_ecl_s = r_cl_s / (lr * i_cl_s)
    h_d_s = 1 / (r_a_s + r_cl_s)
    h_e_s = 1 / (r_ea_s + r_ecl_s)
    delta = 1e-04
    xold = t_skin - q_skin / h_d

    flag1 = False
    while not flag1:
        err1 = _fnerre(xold, q_skin, h_d, t_skin, wet, h_e, p_s_sk)
        err2 = _fnerre(xold + delta, q_skin, h_d, t_skin, wet, h_e, p_s_sk)
        x = xold - delta * err1 / (err2 - err1)
        if abs(x - xold) > 0.01:
            xold = x
            flag1 = False
        else:
            flag1 = True

    xold = t_skin - q_skin / h_d_s

    flag2 = False
    while not flag2:
        err1 = _fnerrs(xold, q_skin, h_d_s, t_skin, wet, h_e_s, p_s_sk)
        err2 = _fnerrs(xold + delta, q_skin, h_d_s, t_skin, wet, h_e_s, p_s_sk)
        x = xold - delta * err1 / (err2 - err1)
        if abs(x - xold) > 0.01:
            xold = x
            flag2 = False
        else:
            flag2 = True

    set_temp = x
    tbm_l = (0.194 / 58.15) * rn + 36.301
    tbm_h = (0.347 / 58.15) * rn + 36.669
    if t_body < tbm_l:
        t_sens = 0.4685 * (t_body - tbm_l)
    elif tbm_l <= t_body < tbm_h:
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    else:
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h)

    disc = 4.7 * (e_rsw - e_comfort) / (e_max - e_comfort - e_diff)
    if disc < 0:
        disc = t_sens

    return {
        "set": set_temp,
        "t_core": t_core,
        "t_skin": t_skin,
        "wet": wet,
        "t_sens": t_sens,
        "disc": disc,
        "e_skin": e_skin,
        "met_shivering": met_shivering,
        "alfa": alfa,
        "skin_blood_flow": skin_blood_flow,
    }


def _fnsvp(t):
    return math.exp(18.6686 - 4030.183 / (t + 235))


def _fnerre(x, hsk, hd, tsk, w, he, pssk):
    return hsk - hd * (tsk - x) - w * he * (pssk - 0.5 * _fnsvp(x))


def _fnerrs(x, hsk, hd_s, tsk, w, he_s, pssk):
    return hsk - hd_s * (tsk - x) - w * he_s * (pssk - 0.5 * _fnsvp(x))
