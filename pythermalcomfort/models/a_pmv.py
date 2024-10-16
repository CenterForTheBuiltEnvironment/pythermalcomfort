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
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float, int, or array-like
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    vr : float, int, or array-like
        Relative air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. note::
            vr is the sum of the average air speed measured by the sensor and the activity-generated air speed (Vag). Calculate vr using :py:meth:`pythermalcomfort.utilities.v_relative`.

    rh : float, int, or array-like
        Relative humidity, [%].
    met : float, int, or array-like
        Metabolic rate, [met].
    clo : float, int, or array-like
        Clothing insulation, [clo].

        .. warning::
            Correct for body movement effects using :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    a_coefficient : float
        Adaptive coefficient.
    wme : float, int, or array-like, optional
        External work, [met], default is 0.
    units : str, optional
        Units system, 'SI' or 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, returns nan for inputs outside standard limits. Defaults to True.

        .. warning::
            ISO 7730 2005 limits: 10 < tdb [°C] < 30, 10 < tr [°C] < 40, 0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, -2 < PMV < 2.

    Returns
    -------
    pmv : float, int, or array-like
        Predicted Mean Vote.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import a_pmv
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        v = 0.1
        met = 1.4
        clo = 0.5

        # Calculate relative air speed
        v_r = v_relative(v=v, met=met)

        # Calculate dynamic clothing
        clo_d = clo_dynamic(clo=clo, met=met)

        results = a_pmv(tdb=28, tr=28, vr=v_r, rh=50, met=met, clo=clo_d, a_coefficient=0.293)
        print(results)  # 0.74
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
