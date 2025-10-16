import numpy as np
from typing import Union, Dict, Any
import pandas  as pd
from pythermalcomfort.classes_input import IREQInputs
from pythermalcomfort.classes_return import IREQ

def calc_ireq(
    m: Union[float, np.ndarray],
    w_work: Union[float, np.ndarray] = 0,
    tdb: Union[float, np.ndarray] = None,
    tr: Union[float, np.ndarray] = None,
    p_air: Union[float, np.ndarray] = 8.0,
    v_walk: Union[float, np.ndarray] = 0.3,
    v: Union[float, np.ndarray] = 0.4,
    rh: Union[float, np.ndarray] = 50.0,
    clo: Union[float, np.ndarray] = 1.0,
) -> IREQ:  
    """Calculate Required Clothing Insulation (IREQ) and Duration Limited Exposure (DLE) 
    according to ISO 11079:2007 [1]_. The ISO 11079 provides a method for the analytical 
    evaluation and interpretation of cold stress when using required clothing insulation (IREQ) 
    and local cooling effects.

    The IREQ model can be used to predict the:
    - Required clothing insulation for heat balance (IREQ)
    - Required clothing insulation for minimum conditions (IREQ_min)
    - Required clothing insulation for neutral conditions (IREQ_neutral)
    - Duration Limited Exposure (DLE) for both minimum and neutral conditions
    - Intrinsic clothing insulation (ICL) for both minimum and neutral conditions

    Parameters
    ----------
    m : float or np.ndarray
        Metabolic energy production.
    w_work : float or np.ndarray  
        Rate of mechanical work.
    tdb : float or np.ndarray
        Ambient air temperature [°C].
    tr : float or np.ndarray
        Mean radiant temperature [°C].
    p_air : float or np.ndarray
        Air permeability [l/m2s].
    v_walk : float or np.ndarray
        Walking speed (or work-created air movement) [m/s].
    v : float or np.ndarray
        Relative air velocity.
    rh : float or np.ndarray
        Relative humidity [%].
    clo : float or np.ndarray
        Clothing insulation level [clo].

    Returns
    -------
    IREQ
        Dataclass containing IREQ and DLE results with the following attributes:
        
        - ireq_min : Required clothing insulation for minimum conditions [clo]
        - ireq_neutral : Required clothing insulation for neutral conditions [clo]
        - icl_min : Intrinsic clothing insulation for minimum conditions [clo]
        - icl_neutral : Intrinsic clothing insulation for neutral conditions [clo]
        - dle_min : Duration Limited Exposure for minimum conditions [hours]
        - dle_neutral : Duration Limited Exposure for neutral conditions [hours]

    Raises
    ------
    ValueError
        If any input parameter is outside the valid range.
    TypeError
        If inputs have incompatible shapes.

    Examples
    --------
    .. code-block:: python

    from ireq import calc_ireq
    import numpy as np

    # Scalar (single-point) usage
    results = calc_ireq(
        m=116,        # metabolic rate [W/m2]
        w_work=0,     # mechanical work [W/m2]
        tdb=-15,      # dry-bulb temperature [°C]
        tr=-15,       # mean radiant temperature [°C]
        p_air=8,      # air permeability [l/m2s]
        v_walk=0.3,   # walking speed [m/s]
        v=0.4,        # relative air velocity [m/s]
        rh=85,        # relative humidity [%]
        clo=2.5       # clothing insulation [clo]
    )
    # Access results (fields depend on implementation)
    print(results.ireq_neutral)
    print(results.dle_min)
    print(results.icl_neutral)

    # Vectorized usage for multiple scenarios
    m_values = np.array([116, 145, 175])
    tdb_values = np.array([-10, -15, -20])
    results = calc_ireq(
        m=m_values,
        w_work=0,
        tdb=tdb_values,
        tr=tdb_values,
        p_air=8,
        v_walk=0.3,
        v=0.4,
        rh=85,
        clo=2.5
    )
    # When inputs are arrays, corresponding result fields are arrays of the same shape
    print(results.ireq_neutral)  # e.g. array([...])
    print(results.dle_min)       # e.g. array([...]

    Notes
    -----
    The IREQ index represents the clothing insulation required to maintain thermal 
    equilibrium for a person exposed to cold conditions. The model considers:

    - Metabolic heat production
    - Respiratory heat losses
    - Evaporative heat losses
    - Radiative and convective heat exchanges
    - Clothing properties and air permeability

    The DLE indicates the maximum allowable exposure time before physiological strain 
    becomes unacceptable. Values greater than 8 hours are reported as "more than 8".

    References
    ----------
    .. [1] ISO 11079:2007 - Ergonomics of the thermal environment — Determination 
    and interpretation of cold stress when using required clothing insulation (IREQ) 
    and local cooling effects.

    """

    # Convert all inputs to numpy arrays for vectorized operations
    m_arr = np.asarray(m, dtype=float)
    w_work_arr = np.asarray(w_work, dtype=float)
    tdb_arr = np.asarray(tdb, dtype=float)
    tr_arr = np.asarray(tr, dtype=float)
    p_air_arr = np.asarray(p_air, dtype=float)
    v_walk_arr = np.asarray(v_walk, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    clo_arr = np.asarray(clo, dtype=float)

    # Validate input shapes
    _validate_input_shapes(m_arr, w_work_arr, tdb_arr, tr_arr, p_air_arr, 
                          v_walk_arr, v_arr, rh_arr, clo_arr)

    # Validate input ranges
    _validate_input_ranges(m_arr, w_work_arr, tdb_arr, tr_arr, p_air_arr,
                          v_walk_arr, v_arr, rh_arr, clo_arr)

    # Apply input boundaries
    m_arr = np.clip(m_arr, 58.0, 400.0)
    tdb_arr = np.minimum(tdb_arr, 10.0)
    
    # Calculate walking speed boundaries
    w_min_calculated = 0.0052 * (m_arr - 58.0)
    v_walk_arr = np.clip(v_walk_arr, w_min_calculated, 1.2)
    v_arr = np.clip(v_arr, 0.4, 18.0)

    # Convert clothing insulation from clo to m2C/W
    clo_m2cw = clo_arr * 0.155
    
    # Calculate air layer insulation
    r_air = _calculate_air_insulation(v_arr, v_walk_arr)

    # Calculate minimal and neutral conditions
    results_min = _calculate_ireq_conditions(
        m_arr, w_work_arr, tdb_arr, tr_arr, p_air_arr, v_walk_arr, 
        v_arr, rh_arr, clo_m2cw, r_air, "min"
    )
    
    results_neutral = _calculate_ireq_conditions(
        m_arr, w_work_arr, tdb_arr, tr_arr, p_air_arr, v_walk_arr,
        v_arr, rh_arr, clo_m2cw, r_air, "neutral"
    )

    return IREQ(
        ireq_min=results_min["ireq"],
        ireq_neutral=results_neutral["ireq"],
        icl_min=results_min["icl"],
        icl_neutral=results_neutral["icl"],
        dle_min=results_min["dle"],
        dle_neutral=results_neutral["dle"],
    )

def _validate_input_shapes(*arrays) -> None:
    """Validate that all input arrays have compatible shapes."""
    shapes = [arr.shape for arr in arrays if hasattr(arr, 'shape') and arr.shape != ()]
    if shapes and not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays must have the same shape")


def _validate_input_ranges(
    m: np.ndarray,
    w_work: np.ndarray,
    tdb: np.ndarray,
    tr: np.ndarray,
    p_air: np.ndarray,
    v_walk: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    clo: np.ndarray,
) -> None:
    """Validate input parameter ranges."""
    if np.any(m < 58) or np.any(m > 400):
        raise ValueError("Metabolic rate m must be between 58 and 400 W/m2")
    
    if np.any(tdb > 10):
        raise ValueError("Air temperature tdb must be <= 10°C")
    
    if np.any(v < 0.4) or np.any(v > 18):
        raise ValueError("Air velocity v must be between 0.4 and 18 m/s")
    
    if np.any(rh < 0) or np.any(rh > 100):
        raise ValueError("Relative humidity rh must be between 0 and 100%")
    
    if np.any(w_work < 0):
        raise ValueError("Work rate w_work must be non-negative")
    
    if np.any(clo < 0):
        raise ValueError("Clothing insulation clo must be non-negative")


def _calculate_air_insulation(v: np.ndarray, v_walk: np.ndarray) -> np.ndarray:
    """Calculate air layer insulation."""
    return 0.092 * np.exp(-0.15 * v - 0.22 * v_walk) - 0.0045


def _calculate_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Calculate saturation vapor pressure at given temperature."""
    return 0.1333 * np.exp(18.6686 - 4030.183 / (temperature + 235.0))


def _calculate_physiological_state(m: np.ndarray, condition_type: str) -> tuple:
    """Calculate skin temperature and wetness based on condition type."""
    if condition_type == "min":
        t_skin = 33.34 - 0.0354 * m
        wetness = np.full_like(m, 0.06)
    else:  # neutral
        t_skin = 35.7 - 0.0285 * m
        wetness = 0.001 * m
    
    return t_skin, wetness


def _calculate_respiratory_losses(
    m: np.ndarray, tdb: np.ndarray, p_air_vapor: np.ndarray
) -> tuple:
    """Calculate respiratory heat losses and exhaled air temperature."""
    t_exhale = 29.0 + 0.2 * tdb
    p_exhale = _calculate_saturation_vapor_pressure(t_exhale)
    
    q_resp = (1.73e-02 * m * (p_exhale - p_air_vapor) + 
              1.4e-03 * m * (t_exhale - tdb))
    
    return t_exhale, p_exhale, q_resp


def _calculate_heat_transfer_coefficients(
    t_clothing: np.ndarray, tr: np.ndarray, r_air: np.ndarray
) -> tuple:
    """Calculate radiative and convective heat transfer coefficients."""
    ar_adu = 0.77
    t_clothing_k = t_clothing + 273.0
    tr_k = tr + 273.0
    
    # Avoid division by zero
    temp_diff = t_clothing - tr
    mask = np.abs(temp_diff) < 1e-10
    avg_temp_k = 273.0 + (t_clothing + tr) / 2.0
    
    h_rad = np.where(
        mask,
        5.67e-08 * 0.95 * ar_adu * 4 * avg_temp_k**3,
        (5.67e-08 * 0.95 * ar_adu * (t_clothing_k**4 - tr_k**4)) / temp_diff
    )
    
    h_conv = 1.0 / r_air - h_rad
    
    return h_rad, h_conv


def _solve_ireq_iteration(
    m: np.ndarray,
    w_work: np.ndarray,
    tdb: np.ndarray,
    tr: np.ndarray,
    p_air_vapor: np.ndarray,
    t_skin: np.ndarray,
    wetness: np.ndarray,
    r_air: np.ndarray,
    t_exhale: np.ndarray,
    p_exhale: np.ndarray,
) -> np.ndarray:
    """Solve for IREQ value iteratively."""
    ireq = np.full_like(m, 0.5)
    step_factor = np.full_like(m, 0.5)
    energy_balance = np.full_like(m, 1.0)
    
    max_iterations = 100
    iteration = 0
    
    while np.any(np.abs(energy_balance) > 0.01) and iteration < max_iterations:
        iteration += 1
        
        f_clothing = 1.0 + 1.197 * ireq
        r_evap_total = (0.06 / 0.38) * (r_air + ireq)
        
        p_skin = _calculate_saturation_vapor_pressure(t_skin)
        q_evap = wetness * (p_skin - p_air_vapor) / r_evap_total
        q_resp = 1.73e-02 * m * (p_exhale - p_air_vapor) + 1.4e-03 * m * (t_exhale - tdb)
        
        t_clothing = t_skin - ireq * (m - w_work - q_evap - q_resp)
        h_rad, h_conv = _calculate_heat_transfer_coefficients(t_clothing, tr, r_air)
        
        q_rad = f_clothing * h_rad * (t_clothing - tr)
        q_conv = f_clothing * h_conv * (t_clothing - tdb)
        
        energy_balance = m - w_work - q_evap - q_resp - q_rad - q_conv
        
        # Bisection method update
        step_factor = np.where(energy_balance > 0, step_factor / 2, step_factor / 2)
        ireq = np.where(energy_balance > 0, ireq - step_factor, ireq + step_factor)
    
    # Final IREQ calculation
    f_clothing = 1.0 + 1.197 * ireq
    h_rad, h_conv = _calculate_heat_transfer_coefficients(t_clothing, tr, r_air)
    q_rad = f_clothing * h_rad * (t_clothing - tr)
    q_conv = f_clothing * h_conv * (t_clothing - tdb)
    
    return (t_skin - t_clothing) / (q_rad + q_conv)


def _solve_dle_iteration(
    m: np.ndarray,
    w_work: np.ndarray,
    tdb: np.ndarray,
    tr: np.ndarray,
    p_air_vapor: np.ndarray,
    t_skin: np.ndarray,
    wetness: np.ndarray,
    r_air: np.ndarray,
    clo_m2cw: np.ndarray,
    v: np.ndarray,
    v_walk: np.ndarray,
    p_air: np.ndarray,
    t_exhale: np.ndarray,
    p_exhale: np.ndarray,
) -> np.ndarray:
    """Solve for DLE value iteratively."""
    q_store = np.full_like(m, -40.0)
    step_factor = np.full_like(m, 500.0)
    energy_balance = np.full_like(m, 1.0)
    
    max_iterations = 100
    iteration = 0
    
    while np.any(np.abs(energy_balance) > 0.01) and iteration < max_iterations:
        iteration += 1
        
        f_clothing = 1.0 + 1.197 * clo_m2cw
        
        constant_part = (0.54 * np.exp(-0.15 * v - 0.22 * v_walk) * 
                        (p_air**0.075) - 0.06 * np.log(p_air) + 0.5)
        
        iclr = ((clo_m2cw + 0.085 / f_clothing) * constant_part -
                (0.092 * np.exp(-0.15 * v - 0.22 * v_walk) - 0.0045) / f_clothing)
        
        r_evap_total = (0.06 / 0.38) * (r_air + iclr)
        p_skin = _calculate_saturation_vapor_pressure(t_skin)
        q_evap = wetness * (p_skin - p_air_vapor) / r_evap_total
        q_resp = 1.73e-02 * m * (p_exhale - p_air_vapor) + 1.4e-03 * m * (t_exhale - tdb)
        
        t_clothing = t_skin - iclr * (m - w_work - q_evap - q_resp - q_store)
        h_rad, h_conv = _calculate_heat_transfer_coefficients(t_clothing, tr, r_air)
        
        q_rad = f_clothing * h_rad * (t_clothing - tr)
        q_conv = f_clothing * h_conv * (t_clothing - tdb)
        
        energy_balance = m - w_work - q_evap - q_resp - q_rad - q_conv - q_store
        
        # Bisection method update
        step_factor = np.where(energy_balance > 0, step_factor / 2, step_factor / 2)
        q_store = np.where(energy_balance > 0, q_store + step_factor, q_store - step_factor)
    
    return -40.0 / q_store


def _calculate_ireq_conditions(
    m: np.ndarray,
    w_work: np.ndarray,
    tdb: np.ndarray,
    tr: np.ndarray,
    p_air: np.ndarray,
    v_walk: np.ndarray,
    v: np.ndarray,
    rh: np.ndarray,
    clo_m2cw: np.ndarray,
    r_air: np.ndarray,
    condition_type: str,
) -> Dict[str, Any]:
    """Calculate IREQ and DLE for specific conditions (min/neutral)."""
    
    # Calculate physiological state
    t_skin, wetness = _calculate_physiological_state(m, condition_type)
    
    # Calculate ambient water vapor pressure
    p_air_vapor = (rh / 100.0) * _calculate_saturation_vapor_pressure(tdb)
    
    # Calculate respiratory losses
    t_exhale, p_exhale, q_resp = _calculate_respiratory_losses(m, tdb, p_air_vapor)
    
    # Solve for IREQ
    ireq_m2cw = _solve_ireq_iteration(
        m, w_work, tdb, tr, p_air_vapor, t_skin, wetness, r_air, t_exhale, p_exhale
    )
    
    # Solve for DLE
    dle = _solve_dle_iteration(
        m, w_work, tdb, tr, p_air_vapor, t_skin, wetness, r_air, 
        clo_m2cw, v, v_walk, p_air, t_exhale, p_exhale
    )
    
    # Convert IREQ to clo and calculate ICL
    ireq_clo = np.round((ireq_m2cw / 0.155) * 10.0) / 10.0
    
    f_clothing = 1.0 + 1.197 * ireq_m2cw
    constant_part = (0.54 * np.exp(-0.15 * v - 0.22 * v_walk) * 
                    (p_air**0.075) - 0.06 * np.log(p_air) + 0.5)
    
    icl_m2cw = ((ireq_m2cw + r_air / f_clothing) / constant_part - 0.085 / f_clothing)
    icl_clo = np.round((icl_m2cw / 0.155) * 10.0) / 10.0
    
    # Format DLE result
    dle_result = np.where(dle > 8, "more than 8", np.round(dle * 10.0) / 10.0)
    
    return {
        "ireq": ireq_clo,
        "icl": icl_clo,
        "dle": dle_result,
    }

