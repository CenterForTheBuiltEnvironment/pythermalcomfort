from typing import Union

import numpy as np

from pythermalcomfort.classes_input import HIInputs
from pythermalcomfort.classes_return import HI
from pythermalcomfort.shared_functions import mapping
from pythermalcomfort.utilities import Units, units_converter


def heat_index_rothfusz(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    units: str = Units.SI.value,
    round_output: bool = True,
) -> HI:
    """The Heat Index (HI) calculated in accordance with the Rothfusz (1990) model [Rothfusz]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system. Defaults to 'SI'.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HI
        A dataclass containing the Heat Index and heat stress category.
        See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` and `heat_stress_category` values, use the corresponding attributes
        of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_rothfusz

        result = heat_index_rothfusz(tdb=25, rh=50)
        print(result.hi)    # 29.7
        print(result.stress_category)  # "caution"
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        units=units,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    hi = -8.784695 + 1.61139411 * tdb + 2.338549 * rh - 0.14611605 * tdb * rh
    hi += -1.2308094 * 10**-2 * tdb**2 - 1.6424828 * 10**-2 * rh**2
    hi += 2.211732 * 10**-3 * tdb**2 * rh + 7.2546 * 10**-4 * tdb * rh**2
    hi += -3.582 * 10**-6 * tdb**2 * rh**2

    # heat index should only be calculated for temperatures above 27 °C
    tdb_valid = np.where((tdb >= 27.0), tdb, np.nan)
    all_valid = ~(np.isnan(tdb_valid))
    hi_valid = np.where(all_valid, hi, np.nan)

    heat_index_categories = {
        27.0: "no risk",
        32.0: "caution",
        41.0: "extreme caution",
        54.0: "danger",
        1000.0: "extreme danger",
    }

    if round_output:
        hi_valid = np.around(hi_valid, 1)

    return HI(
        hi=hi_valid,
        stress_category=mapping(hi, heat_index_categories),
    )
