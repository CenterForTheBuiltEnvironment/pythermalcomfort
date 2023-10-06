def wc(tdb, v, **kwargs):
    """Calculates the Wind Chill Index (WCI) in accordance with the ASHRAE 2017 Handbook Fundamentals - Chapter 9 [18]_.

    The wind chill index (WCI) is an empirical index based on cooling measurements
    taken on a cylindrical flask partially filled with water in Antarctica
    (Siple and Passel 1945). For a surface temperature of 33°C, the index describes
    the rate of heat loss from the cylinder via radiation and convection as a function
    of ambient temperature and wind velocity.

    This formulation has been met with some valid criticism. WCI is unlikely to be an
    accurate measure of heat loss from exposed flesh, which differs from plastic in terms
    of curvature, roughness, and radiation exchange qualities, and is always below 33°C
    in a cold environment. Furthermore, the equation's values peak at 90 km/h and then
    decline as velocity increases. Nonetheless, this score reliably represents the
    combined effects of temperature and wind on subjective discomfort for velocities
    below 80 km/h [18]_.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature,[°C]
    v : float
        wind speed 10m above ground level, [m/s]

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it

    Returns
    -------
    wci: float
        wind chill index, [W/m2)]

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import wc
        >>> wc(tdb=-5, v=5.5)
        {"wci": 1255.2}
    """
    default_kwargs = {
        "round": True,
    }
    kwargs = {**default_kwargs, **kwargs}

    wci = (10.45 + 10 * v**0.5 - v) * (33 - tdb)

    # the factor 1.163 is used to convert to W/m2
    wci = wci * 1.163

    if kwargs["round"]:
        wci = round(wci, 1)

    return {"wci": wci}
