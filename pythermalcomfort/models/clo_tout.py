from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import (
    units_converter,
)


def clo_tout(
    tout: Union[float, int, np.ndarray, List[float], List[int]], units: str = "SI"
) -> Union[float, np.ndarray]:
    """Representative clothing insulation Icl as a function of outdoor air
    temperature at 06:00 a.m [4]_.

    Parameters
    ----------
    tout : float, int, or array-like
        outdoor air temperature at 06:00 a.m., default in [°C] in [°F] if `units` = 'IP'
    units : str, optional
        select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.

    Returns
    -------
    clo : float, int, or array-like
         Representative clothing insulation Icl, [clo]

    Raises
    -------
    TypeError
        If `tout` is not a float, int, NumPy array, or a list of floats or integers.
    ValueError
        If an invalid unit is provided or non-numeric elements are found in `tout`.

    Notes
    -----
    The ASHRAE 55 2020 states that it is acceptable to determine the clothing
    insulation Icl using this equation in mechanically conditioned buildings [1]_.

    Limitations:
        - This equation may not be accurate for extreme temperature ranges.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import clo_tout
        >>> clo_tout(tout=27)  # Can be int or float
        0.46
        >>> clo_tout(tout=[27, 25])  # List of ints or floats
        array([0.46, 0.47])
    """

    # Use explicit type hints for sequence
    valid_types = (
        float,
        int,
        np.ndarray,
        list,
    )  # Use 'list' instead of 'List[float]' or 'List[int]'
    if not isinstance(tout, valid_types):
        raise TypeError("tout must be a float, int, NumPy array, or a list.")

    # Check for valid list elements separately
    if isinstance(tout, list):
        if not all(isinstance(item, (float, int)) for item in tout):
            raise TypeError("Elements of tout list must be floats or integers.")

    # Convert tout to NumPy array for vectorized operations
    tout = np.array(tout)

    # Validate units string
    valid_units: List[str] = ["SI", "IP"]
    if units.upper() not in valid_units:
        raise ValueError(f"Invalid unit: {units}. Supported units are {valid_units}.")

    # Convert units if necessary
    if units.lower() == "ip":
        tout = units_converter(tmp=tout)[0]

    clo = np.where(tout < 26, np.power(10, -0.1635 - 0.0066 * tout), 0.46)
    clo = np.where(tout < 5, 0.818 - 0.0364 * tout, clo)
    clo = np.where(tout < -5, 1, clo)

    return np.around(clo, 2)
