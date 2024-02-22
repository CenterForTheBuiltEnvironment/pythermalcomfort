from typing import Union, List

import numpy as np

from pythermalcomfort.models import pmv


def a_pmv(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    vr: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
    met: Union[float, int, np.ndarray, List[float], List[int]],
    clo: Union[float, int, np.ndarray, List[float], List[int]],
    a_coefficient: float,
    wme: Union[float, int, np.ndarray, List[float], List[int]] = 0,
    units="SI",
    limit_inputs=True,
):
    """Returns Adaptive Predicted Mean Vote (aPMV) [25]_. This index was
    developed by Yao, R. et al. (2009). The model takes into account factors
    such as culture, climate, social, psychological and behavioural
    adaptations, which have an impact on the senses used to detect thermal
    comfort. This model uses an adaptive coefficient (λ) representing the
    adaptive factors that affect the sense of thermal comfort.

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float, int, or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float, int, or array-like
        relative air speed, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air speed caused by body movement and not the air
        speed measured by the air speed sensor. The relative air speed is the sum of the
        average air speed measured by the sensor plus the activity-generated air speed
        (Vag). Where Vag is the activity-generated air speed caused by motion of
        individual body parts. vr can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.v_relative`.
    rh : float, int, or array-like
        relative humidity, [%]
    met : float, int, or array-like
        metabolic rate, [met]
    clo : float, int, or array-like
        clothing insulation, [clo]

        Note: The activity as well as the air speed modify the insulation characteristics
        of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
        the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
        for the effect of the body movement for met equal or higher than 1.2 met using
        the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
        can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.clo_dynamic`.
    a_coefficient : float
        adaptive coefficient
    wme : float, int, or array-like
        external work, [met] default 0
    units : str, optional
        select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
        0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.

    Returns
    -------
    pmv : float, int, or array-like
        Predicted Mean Vote

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import a_pmv
        >>> from pythermalcomfort.utilities import v_relative, clo_dynamic
        >>> v = 0.1
        >>> met = 1.4
        >>> clo = 0.5
        >>> # calculate relative air speed
        >>> v_r = v_relative(v=v, met=met)
        >>> # calculate dynamic clothing
        >>> clo_d = clo_dynamic(clo=clo, met=met)
        >>> results = a_pmv(tdb=28, tr=28, vr=v_r, rh=50, met=met, clo=clo_d, a_coefficient=0.293)
        >>> print(results)
        0.74
    """

    # Validate units string
    valid_units: List[str] = ["SI", "IP"]
    if units.upper() not in valid_units:
        raise ValueError(f"Invalid unit: {units}. Supported units are {valid_units}.")

    _pmv = pmv(
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme,
        standard="ISO",
        units=units,
        limit_inputs=limit_inputs,
    )

    return np.around(_pmv / (1 + a_coefficient * _pmv), 2)
