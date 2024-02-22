from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import (
    mapping,
)


def discomfort_index(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
):
    """Calculates the Discomfort Index (DI). The index is essentially an
    effective temperature based on air temperature and humidity. The discomfort
    index is usuallly divided in 6 dicomfort categories and it only applies to
    warm environments: [24]_

    * class 1 - DI < 21 °C - No discomfort
    * class 2 - 21 <= DI < 24 °C - Less than 50% feels discomfort
    * class 3 - 24 <= DI < 27 °C - More than 50% feels discomfort
    * class 4 - 27 <= DI < 29 °C - Most of the population feels discomfort
    * class 5 - 29 <= DI < 32 °C - Everyone feels severe stress
    * class 6 - DI >= 32 °C - State of medical emergency

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, [°C]
    rh : float, int, or array-like
        relative humidity, [%]

    Returns
    -------
    di : float, int, or array-like
        Discomfort Index, [°C]
    discomfort_condition : str or array-like
        Classification of the thermal comfort conditions according to the discomfort index

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import discomfort_index
        >>> discomfort_index(tdb=25, rh=50)
        {'di': 22.1, 'discomfort_condition': 'Less than 50% feels discomfort'}
    """

    tdb = np.array(tdb)
    rh = np.array(rh)

    di = tdb - 0.55 * (1 - 0.01 * rh) * (tdb - 14.5)

    di_categories = {
        21: "No discomfort",
        24: "Less than 50% feels discomfort",
        27: "More than 50% feels discomfort",
        29: "Most of the population feels discomfort",
        32: "Everyone feels severe stress",
        99: "State of medical emergency",
    }

    return {
        "di": np.around(di, 1),
        "discomfort_condition": mapping(di, di_categories, right=False),
    }
