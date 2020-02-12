import math
import warnings
warnings.simplefilter("always")


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


def check_standard_compliance(**kwargs):  # todo enter all the limits
    for key, value in kwargs.items():
        if key == 'met':
            if value > 2:
                warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 clo", UserWarning)
            elif value < 1:
                warnings.warn("ASHRAE met applicability limits between 1.0 and 2.0 clo", UserWarning)
        if key == 'clo':
            if value > 1.5:
                warnings.warn("ASHRAE clo applicability limits between 0.0 and 1.5 clo", UserWarning)
