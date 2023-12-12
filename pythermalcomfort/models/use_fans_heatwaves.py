import numpy as np

from pythermalcomfort.models import two_nodes
from pythermalcomfort.utilities import (
    units_converter,
    check_standard_compliance_array,
    body_surface_area,
)


def use_fans_heatwaves(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    wme=0,
    body_surface_area=1.8258,
    p_atm=101325,
    body_position="standing",
    units="SI",
    max_skin_blood_flow=80,
    **kwargs,
):
    """It helps you to estimate if the conditions you have selected would cause
    heat strain. This occurs when either the following variables reaches its
    maximum value:

    * m_rsw Rate at which regulatory sweat is generated, [mL/h/m2]
    * w : Skin wettedness, adimensional. Ranges from 0 and 1.
    * m_bl : Skin blood flow [kg/h/m2]

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    v : float
        air speed, default in [m/s] in [fps] if `units` = 'IP'
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]
    wme : float
        external work, [met] default 0
    body_surface_area : float
        body surface area, default value 1.8258 [m2] in [ft2] if `units` = 'IP'

        The body surface area can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.body_surface_area`.
    p_atm : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    body_position: str default="standing"
        select either "sitting" or "standing"
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    max_skin_blood_flow : float, [kg/h/m2] default 80
        maximum blood flow from the core to the skin

    Other Parameters
    ----------------
    max_sweating: float, [mL/h/m2] default 500
        max sweating
    round: boolean, default True
        if True rounds output value, if False it does not round it
    limit_inputs : boolean default True
        By default, if the inputs are outside the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The applicability limits are 20 < tdb [°C] < 50, 20 < tr [°C] < 50,
        0.1 < v [m/s] < 4.5, 0.7 < met [met] < 2, and 0 < clo [clo] < 1.

    Returns
    -------
    e_skin : float
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff
    e_rsw : float
        Rate of evaporative heat loss from sweat evaporation, [W/m2]
    e_diff : float
        Rate of evaporative heat loss from moisture diffused through the skin, [W/m2]
    e_max : float
        Maximum rate of evaporative heat loss from skin, [W/m2]
    q_sensible : float
        Sensible heat loss from skin, [W/m2]
    q_skin : float
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin
    q_res : float
        Total rate of heat loss through respiration, [W/m2]
    t_core : float
        Core temperature, [°C]
    t_skin : float
        Skin temperature, [°C]
    m_bl : float
        Skin blood flow, [kg/h/m2]
    m_rsw : float
        Rate at which regulatory sweat is generated, [mL/h/m2]
    w : float
        Skin wettedness, adimensional. Ranges from 0 and 1.
    w_max : float
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 and 1.
    heat_strain : bool
        True if the model predict that the person may be experiencing heat strain
    heat_strain_blood_flow : bool
        True if heat strain is caused by skin blood flow (m_bl) reaching its maximum value
    heat_strain_w : bool
        True if heat strain is caused by skin wettedness (w) reaching its maximum value
    heat_strain_sweating : bool
        True if heat strain is caused by regulatory sweating (m_rsw) reaching its
        maximum value
    """
    # If the SET function is used to calculate the cooling effect then the h_c is
    # calculated in a slightly different way
    default_kwargs = {"round": True, "max_sweating": 500, "limit_inputs": True}
    kwargs = {**default_kwargs, **kwargs}

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    if units.lower() == "ip":
        if body_surface_area == 1.8258:
            body_surface_area = 19.65
        if p_atm == 101325:
            p_atm = 1
        tdb, tr, v, body_surface_area, p_atm = units_converter(
            tdb=tdb, tr=tr, v=v, area=body_surface_area, pressure=p_atm
        )

    output = two_nodes(
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atmospheric=p_atm,
        body_position=body_position,
        max_skin_blood_flow=max_skin_blood_flow,
        round=False,
        output="all",
        max_sweating=kwargs["max_sweating"],
    )

    output_vars = [
        "e_skin",
        "e_rsw",
        "e_max",
        "q_sensible",
        "q_skin",
        "q_res",
        "t_core",
        "t_skin",
        "m_bl",
        "m_rsw",
        "w",
        "w_max",
        "heat_strain_blood_flow",
        "heat_strain_w",
        "heat_strain_sweating",
        "heat_strain",
    ]

    output["heat_strain_blood_flow"] = np.where(
        output["m_bl"] == max_skin_blood_flow, True, False
    )
    output["heat_strain_w"] = np.where(output["w"] == output["w_max"], True, False)
    output["heat_strain_sweating"] = np.where(
        output["m_rsw"] == kwargs["max_sweating"], True, False
    )

    output["heat_strain"] = np.any(
        [
            output["heat_strain_blood_flow"],
            output["heat_strain_w"],
            output["heat_strain_sweating"],
        ],
        axis=0,
    )

    output = {key: output[key] for key in output_vars}

    if kwargs["limit_inputs"]:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            rh_valid,
            met_valid,
            clo_valid,
        ) = check_standard_compliance_array(
            standard="fan_heatwaves", tdb=tdb, tr=tr, v=v, rh=rh, met=met, clo=clo
        )
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        output = {key: np.where(all_valid, output[key], np.nan) for key in output_vars}

    for key in output.keys():
        # round the results if needed
        if (kwargs["round"]) and (type(output[key]) is not bool):
            output[key] = np.around(output[key], 1)

    return output
