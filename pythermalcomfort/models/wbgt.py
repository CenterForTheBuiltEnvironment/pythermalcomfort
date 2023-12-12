def wbgt(twb, tg, tdb=None, with_solar_load=False, **kwargs):
    """Calculates the Wet Bulb Globe Temperature (WBGT) index calculated in
    compliance with the ISO 7243 [11]_. The WBGT is a heat stress index that
    measures the thermal environment to which a person is exposed. In most
    situations, this index is simple to calculate. It should be used as a
    screening tool to determine whether heat stress is present. The PHS model
    allows a more accurate estimation of stress. PHS can be calculated using
    the function :py:meth:`pythermalcomfort.models.phs`.

    The WBGT determines the impact of heat on a person throughout the course of a working
    day (up to 8 h). It does not apply to very brief heat exposures. It pertains to
    the evaluation of male and female people who are fit for work in both indoor
    and outdoor occupational environments, as well as other sorts of surroundings [11]_.

    The WBGT is defined as a function of only twb and tg if the person is not exposed to
    direct radiant heat from the sun. When a person is exposed to direct radiant heat,
    tdb must also be specified.

    Parameters
    ----------
    twb : float,
        natural (no forced air flow) wet bulb temperature, [°C]
    tg : float
        globe temperature, [°C]
    tdb : float
        dry bulb air temperature, [°C]. This value is needed as input if the person is
        exposed to direct solar radiation
    with_solar_load: bool
        True if the globe sensor is exposed to direct solar radiation

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it

    Returns
    -------
    wbgt : float
        Wet Bulb Globe Temperature Index, [°C]

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import wbgt
        >>> wbgt(twb=25, tg=32)
        27.1

        >>> # if the persion is exposed to direct solar radiation
        >>> wbgt(twb=25, tg=32, tdb=20, with_solar_load=True)
        25.9
    """
    default_kwargs = {
        "round": True,
    }
    kwargs = {**default_kwargs, **kwargs}

    if with_solar_load and tdb is None:
        raise ValueError("Please enter the dry bulb air temperature")

    if with_solar_load:
        t_wbg = 0.7 * twb + 0.2 * tg + 0.1 * tdb
    else:
        t_wbg = 0.7 * twb + 0.3 * tg

    if kwargs["round"]:
        return round(t_wbg, 1)
    else:
        return t_wbg
