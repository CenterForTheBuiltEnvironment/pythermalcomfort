import math


def p_sat_torr(t):
    """ Estimates the saturation vapor pressure in [torr]

    Parameters
    ----------
    t : float
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat  : float
        saturation vapor pressure [torr]
    """
    return math.exp(18.6686 - 4030.183 / (t + 235.0))


def v_relative(v, met):
    """ Estimates the relative air velocity

    Parameters
    ----------
    v : float
        air velocity measured by the sensor, [m/s]
    met : float
        metabolic rate, [met]

    Returns
    -------
    vr  : float
        relative air velocity, [m/s]
    """

    if met > 1:
        return 0.3 * (met - 1)
    else:
        return v


def units_converter(from_units='ip', **kwargs):
    """ Converts IP values to SI units

    Parameters
    ----------
    from_units: str
        specify system to convert from
    **kwargs : [t, v]

    Returns
    -------
    converted values in SI units
    """
    results = list()
    if from_units == 'ip':
        for key, value in kwargs.items():
            if 'tmp' in key or key == 'tr' or key == 'ta':
                results.append((value - 32) * 5 / 9)
            if key in ['v', 'vr']:
                results.append(value / 3.281)
            if key == 'area':
                results.append(value / 10.764)
            if key == 'pressure':
                results.append(value * 101325)

    elif from_units == 'si':
        for key, value in kwargs.items():
            if 'tmp' in key or key == 'tr' or key == 'ta':
                results.append((value * 9 / 5) + 32)
            if key in ['v', 'vr']:
                results.append(value * 3.281)
            if key == 'area':
                results.append(value * 10.764)
            if key == 'pressure':
                results.append(value / 101325)

    return results
