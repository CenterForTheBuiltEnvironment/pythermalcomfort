from typing import List, Literal, Union

import numpy as np

from pythermalcomfort.classes_input import APMVInputs
from pythermalcomfort.classes_return import APMV
from pythermalcomfort.models.pmv import pmv


def a_pmv(
    tdb: Union[float, List[float]],
    tr: Union[float, List[float]],
    vr: Union[float, List[float]],
    rh: Union[float, List[float]],
    met: Union[float, List[float]],
    clo: Union[float, List[float]],
    a_coefficient: Union[float, int],
    wme: Union[float, List[float]] = 0,
    units: Literal["SI", "IP"] = "SI",
    limit_inputs: bool = True,
) -> APMV:
    """Returns Adaptive Predicted Mean Vote (aPMV) [25]_. This index was
    developed by Yao, R. et al. (2009). The model takes into account factors
    such as culture, climate, social, psychological, and behavioral
    adaptations, which have an impact on the senses used to detect thermal
    comfort. This model uses an adaptive coefficient (λ) representing the
    adaptive factors that affect the sense of thermal comfort.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    vr : float or list of floats
        Relative air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. warning::
            vr is the sum of the average air speed measured by the sensor and the activity-generated air speed (Vag). Calculate vr using :py:meth:`pythermalcomfort.utilities.v_relative`.

    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].

        .. warning::
            Correct for body movement effects using :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    a_coefficient : float
        Adaptive coefficient.
    wme : float or list of floats, optional
        External work, [met], default is 0.
    units : str, optional
        Units system, 'SI' or 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, returns nan for inputs outside standard limits. Defaults to True.

        .. warning::
            ISO 7730 2005 limits: 10 < tdb [°C] < 30, 10 < tr [°C] < 40, 0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, -2 < PMV < 2.

    Returns
    -------
    APMV
        A dataclass containing the Predicted Mean Vote (a_pmv). See :py:class:`~pythermalcomfort.models.a_pmv.AdaptivePMV` for more details.
        To access the `a_pmv` value, use the `a_pmv` attribute of the returned `AdaptivePMV` instance, e.g., `result.a_pmv`.

    Examples
    --------
    .. code-block:: python
        :emphasize-lines: 9,12,14

        from pythermalcomfort.models import a_pmv
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        v = 0.1
        met = 1.4
        clo = 0.5

        # Calculate relative air speed
        v_r = v_relative(v=v, met=met)

        # Calculate dynamic clothing
        clo_d = clo_dynamic(clo=clo, met=met)

        results = a_pmv(
            tdb=28, tr=28, vr=v_r, rh=50, met=met, clo=clo_d, a_coefficient=0.293
        )
        print(results)  # AdaptivePMV(a_pmv=0.74)
        print(results.a_pmv)  # 0.74
    """
    # Validate inputs using the APMVInputs class
    APMVInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        a_coefficient=a_coefficient,
        wme=wme,
        units=units,
    )

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
    ).pmv

    pmv_value = np.around(_pmv / (1 + a_coefficient * _pmv), 2)

    return APMV(a_pmv=pmv_value)
