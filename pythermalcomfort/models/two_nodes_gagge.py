from __future__ import annotations

import math
import threading

import numpy as np
from numba import float64, jit, prange, vectorize

from pythermalcomfort.classes_input import GaggeTwoNodesInputs, NumericInput
from pythermalcomfort.classes_return import SET, GaggeTwoNodes
from pythermalcomfort.psychrometrics import p_sat_torr
from pythermalcomfort.utilities import Postures, met_to_w_m2

# integer codes for `position`: numba kernels compare integers far faster than strings
_POSITION_SITTING = 0
_POSITION_STANDING = 1

# number of outputs of _gagge_two_nodes_optimized
_N_OUTPUTS = 18

# what _gagge_two_nodes_kernel returns when the clothing temperature does not converge:
# a converged flag of 0.0 followed by NaN for every output
_NOT_CONVERGED = (0.0,) + (math.nan,) * _N_OUTPUTS
# what it returns when an input is NaN or infinite: NaN for every output, but no
# error, so that one missing value does not stop the other points
_NON_FINITE_INPUT = (1.0,) + (math.nan,) * _N_OUTPUTS

# inputs that vary per point, in the order of the rows of the `inputs` array that
# _gagge_two_nodes_point reads
_ARRAY_INPUTS = (
    "tdb",
    "tr",
    "v",
    "met",
    "clo",
    "vapor_pressure",
    "wme",
    "body_surface_area",
    "p_atm",
    "max_skin_blood_flow",
    "max_sweating",
    "w_max",
)

# below this many points the serial loop is used: starting numba's threads costs
# about 80 µs per call, more than computing a few points on one core. On an 8-core
# Apple M1 Pro the parallel loop became faster at about 30 points.
_PARALLEL_MIN_POINTS = 50


def two_nodes_gagge(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    wme: NumericInput = 0,
    body_surface_area: NumericInput = 1.8258,
    p_atm: NumericInput = 101325,
    position: str = Postures.standing.value,
    max_skin_blood_flow: NumericInput = 90,
    round_output: bool = True,
    max_sweating: NumericInput = 500,
    w_max: NumericInput = False,
    calculate_ce: bool = False,
) -> SET | GaggeTwoNodes:
    """Gagge Two-node model of human temperature regulation Gagge et al (1986)
    [Gagge1986]_.

    [Gagge1986]_ This model can be used to calculate a variety of indices, including:

    * Gagge's version of Fanger's Predicted Mean Vote (PMV). This function uses the Fanger's PMV equations but it replaces the heat loss and gain terms with those calculated by the two-node model developed by Gagge et al. (1986) [Gagge1986]_.
    * PMV SET and the predicted thermal sensation based on SET [Gagge1986]_. This function is similar in all aspects to the :py:meth:`pythermalcomfort.models.pmv_gagge` however, it uses the :py:meth:`pythermalcomfort.models.set` equation to calculate the dry heat loss by convection.
    * Thermal discomfort (DISC) as the relative thermoregulatory strain necessary to restore a state of comfort and thermal equilibrium by sweating [Gagge1986]_. DISC is described numerically as: comfortable and pleasant (0), slightly uncomfortable but acceptable (1), uncomfortable and unpleasant (2), very uncomfortable (3), limited tolerance (4), and intolerable (5). The range of each category is ± 0.5 numerically. In the cold, the classical negative category descriptions used for Fanger's PMV apply [Gagge1986]_.
    * Heat gains and losses via convection, radiation, and conduction.
    * The Standard Effective Temperature (SET).
    * The New Effective Temperature (ET).
    * The Predicted Thermal Sensation (TSENS).
    * The Predicted Percent Dissatisfied Due to Draft (PD).
    * Predicted Percent Satisfied With the Level of Air Movement (PS).

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
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    body_surface_area : float or list of floats, optional
        Body surface area, default value 1.8258 [m2]. Defaults to 1.8258.

        .. note::
            The body surface area can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.body_surface_area`.

    p_atm : float or list of floats, optional
        Atmospheric pressure, default value 101325 [Pa]. Defaults to 101325.
    position : str, optional
        Select either "sitting" or "standing". Defaults to "standing".
    max_skin_blood_flow : float or list of floats, optional
        Maximum blood flow from the core to the skin, [kg/h/m2]. Defaults to 90.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.
    max_sweating : float or list of floats, optional
        Maximum rate at which regulatory sweat is generated, [kg/h/m2]. Defaults to 500.
    w_max : float or list of floats, optional
        Maximum skin wettedness (w) adimensional, greater than 0.06 and at most 1. 0.06
        is the wettedness from skin diffusion alone, without sweating, so it is
        excluded. Defaults to False, which computes it from `v` and `clo`. False and 0
        both mean the value is computed; True and NaN are rejected.

    Returns
    -------
    GaggeTwoNodes
        A dataclass containing the results of the two-node model of human temperature regulation.
        See :py:class:`~pythermalcomfort.classes_return.GaggeTwoNodes` for more details.
        To access the results, use the corresponding attributes of the returned `GaggeTwoNodes` instance, e.g., `result.e_skin`.

    Raises
    ------
    StopIteration
        If the clothing surface temperature does not converge for any point, for
        example with an extreme `tr`. A NaN or infinite input does not raise: it gives
        NaN for that point.

    Notes
    -----
    Array inputs are computed in a compiled numba loop. Inputs with 50 or more points,
    called from the main thread, use every CPU core. Calls from other threads, such as
    the worker threads of dask's default scheduler, run on one core so that they do not
    start more threads than there are cores. If you run chunks in separate processes,
    limit the threads of each with ``numba.set_num_threads`` or the
    ``NUMBA_NUM_THREADS`` environment variable.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import two_nodes_gagge

        result = two_nodes_gagge(tdb=25, tr=25, v=0.1, rh=50, clo=0.5, met=1.2)
        print(result.w)  # 0.12

        result = two_nodes_gagge(tdb=[25, 25], tr=25, v=0.3, rh=50, met=1.2, clo=0.5)
        print(result.e_skin)  # [16.17 16.17]
    """
    # Validate inputs using the TwoNodesInputs class
    GaggeTwoNodesInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        max_skin_blood_flow=max_skin_blood_flow,
        round_output=round_output,
        max_sweating=max_sweating,
        w_max=w_max,
    )

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    v = np.asarray(v)
    rh = np.asarray(rh)
    met = np.asarray(met)
    clo = np.asarray(clo)
    wme = np.asarray(wme)

    vapor_pressure = rh * p_sat_torr(tdb) / 100

    if calculate_ce:
        result = _gagge_two_nodes_optimized_return_set(
            tdb,
            tr,
            v,
            met,
            clo,
            vapor_pressure,
            wme,
            body_surface_area,
            p_atm,
            1,
        )
        return SET(set=result)

    position_code = np.where(
        np.asarray(position) == Postures.sitting.value,
        _POSITION_SITTING,
        _POSITION_STANDING,
    )
    # False, None and 0 all mean that the kernel computes w_max from v and clo
    w_max = np.asarray(0.0 if w_max is None else w_max, dtype=np.float64)

    values = {
        "tdb": tdb,
        "tr": tr,
        "v": v,
        "met": met,
        "clo": clo,
        "vapor_pressure": vapor_pressure,
        "wme": wme,
        "body_surface_area": body_surface_area,
        "p_atm": p_atm,
        "max_skin_blood_flow": max_skin_blood_flow,
        "max_sweating": max_sweating,
        "w_max": w_max,
    }
    # broadcast the scalars against the arrays and flatten them into one row per
    # input, so that the numba loop can read point i from column i
    *broadcast, position_code = np.broadcast_arrays(
        *(values[name] for name in _ARRAY_INPUTS), position_code
    )
    shape = position_code.shape
    n_points = position_code.size
    inputs = np.empty((len(_ARRAY_INPUTS), n_points), dtype=np.float64)
    for row, value in enumerate(broadcast):
        inputs[row] = value.ravel()
    position_code = position_code.astype(np.int64).ravel()

    # Use every CPU core only for large inputs called from the main thread. From other
    # threads the loop runs on one core: numba's default thread pool ("workqueue")
    # aborts the whole process if two Python threads use it at once, and callers that
    # run chunks on threads, such as dask, already keep every core busy.
    if (
        n_points >= _PARALLEL_MIN_POINTS
        and threading.current_thread() is threading.main_thread()
    ):
        array_loop = _gagge_two_nodes_array_parallel
    else:
        array_loop = _gagge_two_nodes_array_serial
    results, converged = array_loop(inputs=inputs, position=position_code)
    # numba silently drops exceptions raised inside a prange loop, so the kernel
    # flags non-convergence and the error is raised here instead
    if not converged.all():
        max_iteration_exceeded = "Max iterations exceeded"
        raise StopIteration(max_iteration_exceeded)

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
    ) = (row.reshape(shape) for row in results)

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
        "set": _set,
        "et": et,
        "pmv_gagge": pmv_gagge,
        "pmv_set": pmv_set,
        "disc": disc,
        "t_sens": t_sens,
    }

    if round_output:
        for key in output:
            output[key] = np.around(output[key], 2)

    return GaggeTwoNodes(**output)


@jit(nopython=True, cache=True)
def _gagge_two_nodes_kernel(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atm,
    position,
    calculate_ce=False,
    max_skin_blood_flow=90,
    max_sweating=500,
    w_max=None,
) -> tuple[float, ...]:
    """Two-node model for one point.

    Returns a converged flag then the 18 outputs. It does not raise when the clothing
    temperature fails to converge: numba drops exceptions raised inside a prange loop,
    so it returns `_NOT_CONVERGED` instead and the caller raises. A non-finite input
    returns `_NON_FINITE_INPUT`, NaN outputs without an error. `position` is
    `_POSITION_SITTING` or `_POSITION_STANDING`.
    """
    # A missing or infinite input never converges in the clothing temperature loop
    # below, so return NaN for every output instead of raising StopIteration.
    if not (
        math.isfinite(tdb)
        and math.isfinite(tr)
        and math.isfinite(v)
        and math.isfinite(met)
        and math.isfinite(clo)
        and math.isfinite(vapor_pressure)
    ):
        return _NON_FINITE_INPUT

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

    pressure_in_atmospheres = p_atm / 101325
    length_time_simulation = 60  # length time simulation
    n_simulation = 1

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = (met - wme) * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    e_comfort = 0.42 * (rm - met_factor)  # evaporative heat loss during comfort
    e_comfort = max(e_comfort, 0)

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
            if position == _POSITION_SITTING:
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
                return _NOT_CONVERGED

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
        m_bl = min(m_bl, max_skin_blood_flow)
        m_bl = max(m_bl, 0.5)
        m_rsw = c_sw * warm_b * math.exp(warm_sk / 10.7)  # regulatory sweating
        m_rsw = min(m_rsw, max_sweating)
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
            # inverse of w = 0.06 + 0.94 * p_rsw, as in Gagge et al. (1986). The ASHRAE 55
            # code uses w_max / 0.94, which makes SET discontinuous at the cap (#489)
            p_rsw = (w_max - 0.06) / 0.94
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
    h_c_s = max(h_c_s, 3.0)

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

    tbm_l = (
        0.194 / met_to_w_m2
    ) * rm + 36.301  # lower limit for evaporative regulation
    tbm_h = (
        0.347 / met_to_w_m2
    ) * rm + 36.669  # upper limit for evaporative regulation

    t_sens = 0.4685 * (t_body - tbm_l)  # predicted thermal sensation
    if (t_body >= tbm_l) & (t_body < tbm_h):
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    elif t_body >= tbm_h:
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h)

    if t_sens > 0 > e_max * w_max - e_comfort - e_diff:
        disc = 6.0
    else:
        disc = (
            4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff)
        )  # predicted thermal discomfort
    if disc <= 0:
        disc = t_sens
    if disc > 6:
        disc = 6.0

    # PMV Gagge
    pmv_gagge = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff)

    # PMV SET
    dry_set = h_d_s * (t_skin - _set)
    e_req_set = rm - c_res - q_res - dry_set
    pmv_set = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff)

    # Predicted  Percent  Satisfied  With  the  Level  of  Air  Movement
    # ps = 100 * (1.13 * (t_op**0.5) - 0.24 * t_op + 2.7 * (v**0.5) - 0.99 * v)

    return (
        1.0,  # converged
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


@jit(nopython=True, cache=True)
def _gagge_two_nodes_optimized(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atm,
    position,
    calculate_ce=False,
    max_skin_blood_flow=90,
    max_sweating=500,
    w_max=None,
) -> tuple[float, ...]:
    """Two-node model for one point; raises StopIteration if it does not converge."""
    result = _gagge_two_nodes_kernel(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        calculate_ce=calculate_ce,
        max_skin_blood_flow=max_skin_blood_flow,
        max_sweating=max_sweating,
        w_max=w_max,
    )
    if result[0] == 0.0:
        max_iteration_exceeded = "Max iterations exceeded"
        raise StopIteration(max_iteration_exceeded)
    return result[1:]


@jit(nopython=True, cache=True)
def _gagge_two_nodes_point(i, inputs, position, results, converged):
    """Run the kernel on point i and write its outputs to column i of `results`.

    The inputs of point i are column i of `inputs`, whose rows are in the order of
    `_ARRAY_INPUTS`. Sets `converged[i]` to False if the point did not converge.
    """
    result = _gagge_two_nodes_kernel(
        tdb=inputs[0, i],
        tr=inputs[1, i],
        v=inputs[2, i],
        met=inputs[3, i],
        clo=inputs[4, i],
        vapor_pressure=inputs[5, i],
        wme=inputs[6, i],
        body_surface_area=inputs[7, i],
        p_atm=inputs[8, i],
        position=position[i],
        calculate_ce=False,
        max_skin_blood_flow=inputs[9, i],
        max_sweating=inputs[10, i],
        w_max=inputs[11, i],
    )
    converged[i] = result[0] == 1.0
    for j in range(_N_OUTPUTS):
        results[j, i] = result[j + 1]


# Two compiled loops with the same body: numba caches each function's machine code
# on disk by name, so one function compiled twice with different flags would share
# its cache entry.
@jit(nopython=True, cache=True)
def _gagge_two_nodes_array_serial(inputs, position):
    """Run the two-node model on every column of `inputs`, on one CPU core.

    Returns a (18, n) array with one row per output, in the order of
    `_gagge_two_nodes_optimized`, and a boolean array that is False where the clothing
    temperature did not converge.
    """
    n = inputs.shape[1]
    results = np.empty((_N_OUTPUTS, n), dtype=np.float64)
    converged = np.empty(n, dtype=np.bool_)
    for i in range(n):
        _gagge_two_nodes_point(
            i=i, inputs=inputs, position=position, results=results, converged=converged
        )
    return results, converged


@jit(nopython=True, parallel=True, cache=True)
def _gagge_two_nodes_array_parallel(inputs, position):
    """Run `_gagge_two_nodes_array_serial` on all CPU cores instead of one."""
    n = inputs.shape[1]
    results = np.empty((_N_OUTPUTS, n), dtype=np.float64)
    converged = np.empty(n, dtype=np.bool_)
    for i in prange(n):
        _gagge_two_nodes_point(
            i=i, inputs=inputs, position=position, results=results, converged=converged
        )
    return results, converged


@vectorize(
    [  # signature: output(float64)
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
        ),
    ],
    cache=True,
)
def _gagge_two_nodes_optimized_return_set(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atm,
    position,
):
    return _gagge_two_nodes_optimized(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        calculate_ce=True,
    )[0]
