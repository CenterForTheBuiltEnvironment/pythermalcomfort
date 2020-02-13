import math


def p_sat_torr(t):
    """ Estimate the saturation vapor pressure in [torr]

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

