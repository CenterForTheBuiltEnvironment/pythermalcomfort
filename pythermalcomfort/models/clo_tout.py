import numpy as np

from pythermalcomfort.utilities import (
    units_converter,
)


def clo_tout(tout, units="SI"):
    """Representative clothing insulation Icl as a function of outdoor air
    temperature at 06:00 a.m [4]_.

    Parameters
    ----------
    tout : float or array-like
        outdoor air temperature at 06:00 a.m., default in [°C] in [°F] if `units` = 'IP'
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    clo : float or array-like
         Representative clothing insulation Icl, [clo]

    Notes
    -----
    The ASHRAE 55 2020 states that it is acceptable to determine the clothing
    insulation Icl using this equation in mechanically conditioned buildings [1]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import clo_tout
        >>> clo_tout(tout=27)
        0.46
        >>> clo_tout(tout=[27, 25])
        array([0.46, 0.47])
    """

    tout = np.array(tout)

    if units.lower() == "ip":
        tout = units_converter(tmp=tout)[0]

    clo = np.where(tout < 26, np.power(10, -0.1635 - 0.0066 * tout), 0.46)
    clo = np.where(tout < 5, 0.818 - 0.0364 * tout, clo)
    clo = np.where(tout < -5, 1, clo)

    return np.around(clo, 2)
