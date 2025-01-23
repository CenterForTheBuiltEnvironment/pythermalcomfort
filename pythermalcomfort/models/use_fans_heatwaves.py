from dataclasses import asdict
from typing import Union

import numpy as np

from pythermalcomfort.classes_input import UseFansHeatwavesInputs
from pythermalcomfort.classes_return import UseFansHeatwaves
from pythermalcomfort.models.two_nodes_gagge import two_nodes_gagge
from pythermalcomfort.utilities import (
    Postures,
    _check_standard_compliance_array,
)


def use_fans_heatwaves(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    v: Union[float, list[float]],
    rh: Union[float, list[float]],
    met: Union[float, list[float]],
    clo: Union[float, list[float]],
    wme: Union[float, list[float]] = 0,
    body_surface_area: Union[float, list[float]] = 1.8258,
    p_atm: Union[float, list[float]] = 101325,
    position: str = Postures.standing.value,
    max_skin_blood_flow: float = 80,
    limit_inputs: bool = True,
    round_output: bool = True,
    max_sweating: float = 500,
) -> UseFansHeatwaves:
    """It helps you to estimate if the conditions you have selected would cause
    heat strain. This occurs when either the following variables reaches its
    maximum value:

    * m_rsw Rate at which regulatory sweat is generated, [mL/h/m2]
    * w : Skin wettedness, adimensional. Ranges from 0 and 1.
    * m_bl : Skin blood flow [kg/h/m2]

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
    max_skin_blood_flow : float
        Maximum blood flow from the core to the skin. Defaults to 80.
    limit_inputs : bool, optional
        By default, if the inputs are outside the standard applicability limits the
        function returns nan. If False, returns values even if input values are
        outside the applicability limits of the model. Defaults to True. The
        applicability limits are 20 < tdb [°C] < 50, 20 < tr [°C] < 50,
        0.1 < v [m/s] < 4.5, 0.7 < met [met] < 2, and 0 < clo [clo] < 1.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    UseFansHeatwaves
        A dataclass containing the results of using fans during heatwaves.
        See :py:class:`~pythermalcomfort.classes_return.UseFansHeatwaves` for more details.
        To access the results, use the corresponding attributes of the returned `UseFansHeatwaves` instance, e.g., `result.e_skin`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import use_fans_heatwaves

        result = use_fans_heatwaves(tdb=35, tr=35, v=1.0, rh=50, met=1.2, clo=0.5)
        print(result.e_skin)  # 63.0

    """
    # Validate inputs using the UseFansHeatwavesInputs class
    UseFansHeatwavesInputs(
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
        limit_inputs=limit_inputs,
    )

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    output = two_nodes_gagge(
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        max_skin_blood_flow=max_skin_blood_flow,
        round_output=False,
        max_sweating=max_sweating,
    )

    output = asdict(output)

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
        output["m_rsw"] == max_sweating, True, False
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

    if limit_inputs:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            rh_valid,
            met_valid,
            clo_valid,
        ) = _check_standard_compliance_array(
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

    if round_output:
        output = {key: np.around(output[key], 1) for key in output_vars}

    return UseFansHeatwaves(**output)
