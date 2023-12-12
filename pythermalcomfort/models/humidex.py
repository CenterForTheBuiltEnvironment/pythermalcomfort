def humidex(tdb, rh, **kwargs):
    """Calculates the humidex (short for "humidity index"). It has been
    developed by the Canadian Meteorological service. It was introduced in 1965
    and then it was revised by Masterson and Richardson (1979) [14]_. It aims
    to describe how hot, humid weather is felt by the average person. The
    Humidex differs from the heat index in being related to the dew point
    rather than relative humidity [15]_.

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, [°C]
    rh : float
        relative humidity, [%]

    Other Parameters
    ----------------
    round: boolean, default True
        if True rounds output value, if False it does not round it

    Returns
    -------
    humidex: float
        Heat Index, [°C]
    discomfort: str
        Degree of Comfort or Discomfort as defined in Havenith and Fiala (2016) [15]_

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import humidex
        >>> humidex(tdb=25, rh=50)
        {"humidex": 28.2, "discomfort": "Little or no discomfort"}
    """
    default_kwargs = {
        "round": True,
    }
    kwargs = {**default_kwargs, **kwargs}

    if rh > 100 or rh < 0:
        raise ValueError

    hi = tdb + 5 / 9 * ((6.112 * 10 ** (7.5 * tdb / (237.7 + tdb)) * rh / 100) - 10)

    if kwargs["round"]:
        hi = round(hi, 1)

    stress_category = "Heat stroke probable"
    if hi <= 30:
        stress_category = "Little or no discomfort"
    elif hi <= 35:
        stress_category = "Noticeable discomfort"
    elif hi <= 40:
        stress_category = "Evident discomfort"
    elif hi <= 45:
        stress_category = "Intense discomfort; avoid exertion"
    elif hi <= 54:
        stress_category = "Dangerous discomfort"

    return {"humidex": hi, "discomfort": stress_category}
