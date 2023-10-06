import numpy as np

from pythermalcomfort.models.two_nodes import two_nodes
from pythermalcomfort.utilities import (
    units_converter,
    check_standard_compliance_array,
    body_surface_area,
)


def set_tmp(
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
    limit_inputs=True,
    **kwargs,
):
    """
    Calculates the Standard Effective Temperature (SET). The SET is the temperature of
    a hypothetical isothermal environment at 50% (rh), <0.1 m/s (20 fpm) average air
    speed (v), and tr = tdb, in which the total heat loss from the skin of an imaginary occupant
    wearing clothing, standardized for the activity concerned is the same as that
    from a person in the actual environment with actual clothing and activity level. [10]_

    Parameters
    ----------
    tdb : float or array-like
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
    p_atm : float
        atmospheric pressure, default value 101325 [Pa] in [atm] if `units` = 'IP'
    body_position: str default="standing" or array-like
        select either "sitting" or "standing"
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the following limits the
        function returns nan. If False returns values regardless of the input values.
        The limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
        0 < v [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.

    Other Parameters
    ----------------
    round : boolean, deafult True
        if True rounds output value, if False it does not round it

    Returns
    -------
    SET : float or array-like
        Standard effective temperature, [°C]

    Notes
    -----
    You can use this function to calculate the `SET`_ temperature in accordance with
    the ASHRAE 55 2020 Standard [1]_.

    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import set_tmp
        >>> set_tmp(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        24.3
        >>> set_tmp(tdb=[25, 25], tr=25, v=0.1, rh=50, met=1.2, clo=.5)
        array([24.3, 24.3])

        >>> # for users who wants to use the IP system
        >>> set_tmp(tdb=77, tr=77, v=0.328, rh=50, met=1.2, clo=.5, units='IP')
        75.8

    """
    # When SET is used to calculate CE then h_c is calculated in a slightly different way
    default_kwargs = {"round": True, "calculate_ce": False}
    kwargs = {**default_kwargs, **kwargs}

    if units.lower() == "ip":
        if body_surface_area == 1.8258:
            body_surface_area = 19.65
        if p_atm == 101325:
            p_atm = 1
        tdb, tr, v, body_surface_area, p_atm = units_converter(
            tdb=tdb, tr=tr, v=v, area=body_surface_area, pressure=p_atm
        )

    tdb = np.array(tdb)
    tr = np.array(tr)
    v = np.array(v)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    set_array = two_nodes(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atmospheric=p_atm,
        body_position=body_position,
        calculate_ce=kwargs["calculate_ce"],
        round=False,
        output="all",
    )["_set"]

    if units.lower() == "ip":
        set_array = units_converter(tmp=set_array, from_units="si")[0]

    if limit_inputs:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            met_valid,
            clo_valid,
        ) = check_standard_compliance_array(
            "ashrae", tdb=tdb, tr=tr, v=v, met=met, clo=clo
        )
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        set_array = np.where(all_valid, set_array, np.nan)

    if kwargs["round"]:
        return np.around(set_array, 1)
    else:
        return set_array
