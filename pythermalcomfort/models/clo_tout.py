from typing import List, Literal, Union

import numpy as np

from pythermalcomfort.classes_input import CloTOutInputs
from pythermalcomfort.classes_return import CloTOut
from pythermalcomfort.utilities import units_converter


def clo_tout(
    tout: Union[float, List[float]],
    units: Literal["SI", "IP"] = "SI",
) -> CloTOut:
    """Representative clothing insulation Icl as a function of outdoor air
    temperature at 06:00 a.m [4]_.

    Parameters
    ----------
    tout : float or list of floats
        Outdoor air temperature at 06:00 a.m., default in [°C] or [°F] if `units` = 'IP'.
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.

    Returns
    -------
    CloTOut
        A dataclass containing the representative clothing insulation Icl. See :py:class:`~pythermalcomfort.models.clo_tout.CloTOut` for more details.
        To access the `clo_tout` value, use the `clo_tout` attribute of the returned `CloTOut` instance, e.g., `result.clo_tout`.

    Raises
    ------
    TypeError
        If `tout` is not a float, int, NumPy array, or a list of floats or integers.
    ValueError
        If an invalid unit is provided or non-numeric elements are found in `tout`.

    Notes
    -----
    .. note::
        The ASHRAE 55 2020 states that it is acceptable to determine the clothing
        insulation Icl using this equation in mechanically conditioned buildings [1]_.

    .. warning::
        Limitations:
        - This equation may not be accurate for extreme temperature ranges.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import clo_tout

        result = clo_tout(tout=27)
        print(result.clo_tout)  # 0.46

        result = clo_tout(tout=[27, 25])
        print(result.clo_tout)  # array([0.46, 0.47])

    """
    # Validate inputs using the CloTOutInputs class
    CloTOutInputs(
        tout=tout,
        units=units,
    )

    # Convert tout to NumPy array for vectorized operations
    tout = np.array(tout)

    # Convert units if necessary
    if units.lower() == "ip":
        tout = units_converter(tmp=tout)[0]

    clo = np.where(tout < 26, np.power(10, -0.1635 - 0.0066 * tout), 0.46)
    clo = np.where(tout < 5, 0.818 - 0.0364 * tout, clo)
    clo = np.where(tout < -5, 1, clo)

    return CloTOut(clo_tout=np.around(clo, 2))
