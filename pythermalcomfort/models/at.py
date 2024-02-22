from pythermalcomfort.psychrometrics import psy_ta_rh


def at(tdb, rh, v, q=None, **kwargs):
    """Calculates the Apparent Temperature (AT). The AT is defined as the
    temperature at the reference humidity level producing the same amount of
    discomfort as that experienced under the current ambient temperature,
    humidity, and solar radiation [17]_. In other words, the AT is an
    adjustment to the dry bulb temperature based on the relative humidity
    value. Absolute humidity with a dew point of 14°C is chosen as a reference.

    [16]_. It includes the chilling effect of the wind at lower temperatures.

    Two formulas for AT are in use by the Australian Bureau of Meteorology: one includes
    solar radiation and the other one does not (http://www.bom.gov.au/info/thermal_stress/
    , 29 Sep 2021). Please specify q if you want to estimate AT with solar load.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature,[°C]
    rh : float
        relative humidity, [%]
    v : float
        wind speed 10m above ground level, [m/s]
    q : float
        Net radiation absorbed per unit area of body surface [W/m2]

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it

    Returns
    -------
    at: float
        apparent temperature, [°C]

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import at
        >>> at(tdb=25, rh=30, v=0.1)
        24.1
    """
    default_kwargs = {
        "round": True,
    }
    kwargs = {**default_kwargs, **kwargs}

    # dividing it by 100 since the at eq. requires p_vap to be in hPa
    p_vap = psy_ta_rh(tdb, rh).p_vap / 100

    # equation sources [16] and http://www.bom.gov.au/info/thermal_stress/#apparent
    if q:
        t_at = tdb + 0.348 * p_vap - 0.7 * v + 0.7 * q / (v + 10) - 4.25
    else:
        t_at = tdb + 0.33 * p_vap - 0.7 * v - 4.00

    if kwargs["round"]:
        t_at = round(t_at, 1)

    return t_at
