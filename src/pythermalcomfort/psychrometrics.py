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
