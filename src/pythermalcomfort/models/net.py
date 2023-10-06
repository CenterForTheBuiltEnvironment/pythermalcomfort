def net(tdb, rh, v, **kwargs):
    """Calculates the Normal Effective Temperature (NET). Missenard (1933)
    devised a formula for calculating effective temperature. The index
    establishes a link between the same condition of the organism's
    thermoregulatory capability (warm and cold perception) and the surrounding
    environment's temperature and humidity. The index is calculated as a
    function of three meteorological factors: air temperature, relative
    humidity of air, and wind speed. This index allows to calculate the
    effective temperature felt by a person. Missenard original equation was
    then used to calculate the Normal Effective Temperature (NET), by
    considering normal atmospheric pressure and a normal human body temperature
    (37°C). The NET is still in use in Germany, where medical check-ups for
    subjects working in the heat are decided on by prevailing levels of ET,
    depending on metabolic rates. The NET is also constantly monitored by the
    Hong Kong Observatory [16]_. In central Europe the following thresholds are
    in use: <1°C = very cold; 1–9 = cold; 9–17 = cool; 17–21 = fresh; 21–23 = comfortable;
    23–27 = warm; >27°C = hot [16]_.

    Parameters
    ----------
    tdb : float,
        dry bulb air temperature, [°C]
    rh : float
        relative humidity, [%]
    v : float
        wind speed [m/s] at 1.2 m above the ground

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it

    Returns
    -------
    net : float
        Normal Effective Temperature, [°C]

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import net
        >>> net(tdb=37, rh=100, v=0.1)
        37
    """
    default_kwargs = {
        "round": True,
    }
    kwargs = {**default_kwargs, **kwargs}

    frac = 1.0 / (1.76 + 1.4 * v**0.75)

    et = 37 - (37 - tdb) / (0.68 - 0.0014 * rh + frac) - 0.29 * tdb * (1 - 0.01 * rh)

    if kwargs["round"]:
        return round(et, 1)
    else:
        return et
