def heat_index(tdb, rh, **kwargs):
    """Calculates the Heat Index (HI). It combines air temperature and relative
    humidity to determine an apparent temperature. The HI equation [12]_ is
    derived by multiple regression analysis in temperature and relative
    humidity from the first version of Steadman’s (1979) apparent temperature
    (AT) [13]_.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    rh : float
        relative humidity, [%]

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    hi : float
        Heat Index, default in [°C] in [°F] if `units` = 'IP'

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import heat_index
        >>> heat_index(tdb=25, rh=50)
        25.9
    """
    default_kwargs = {
        "round": True,
        "units": "SI",
    }
    kwargs = {**default_kwargs, **kwargs}

    if kwargs["units"] == "SI":
        # from doi: 10.1007/s00484-011-0453-2
        hi = -8.784695 + 1.61139411 * tdb + 2.338549 * rh - 0.14611605 * tdb * rh
        hi += -1.2308094 * 10**-2 * tdb**2 - 1.6424828 * 10**-2 * rh**2
        hi += 2.211732 * 10**-3 * tdb**2 * rh + 7.2546 * 10**-4 * tdb * rh**2
        hi += -3.582 * 10**-6 * tdb**2 * rh**2

    else:
        # from doi: 10.1007/s00484-021-02105-0
        hi = -42.379 + 2.04901523 * tdb + 10.14333127 * rh
        hi += -0.22475541 * tdb * rh - 6.83783 * 10**-3 * tdb**2
        hi += -5.481717 * 10**-2 * rh**2
        hi += 1.22874 * 10**-3 * tdb**2 * rh + 8.5282 * 10**-4 * tdb * rh**2
        hi += -1.99 * 10**-6 * tdb**2 * rh**2

    if kwargs["round"]:
        return round(hi, 1)
    else:
        return hi
