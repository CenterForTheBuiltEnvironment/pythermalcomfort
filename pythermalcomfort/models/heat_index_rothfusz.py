from typing import Union

import numpy as np

from pythermalcomfort.classes_input import HIInputs
from pythermalcomfort.classes_return import HI


def heat_index_rothfusz(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    round_output: bool = True,
) -> HI:
    """The Heat Index (HI) calculated in accordance with the Rothfusz (1990) model [Rothfusz1990]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HI
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_rothfusz

        result = heat_index_rothfusz(tdb=25, rh=50)
        print(result.hi)
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    hi = -8.784695 + 1.61139411 * tdb + 2.338549 * rh - 0.14611605 * tdb * rh
    hi += -1.2308094 * 10**-2 * tdb**2 - 1.6424828 * 10**-2 * rh**2
    hi += 2.211732 * 10**-3 * tdb**2 * rh + 7.2546 * 10**-4 * tdb * rh**2
    hi += -3.582 * 10**-6 * tdb**2 * rh**2

    if round_output:
        hi = np.around(hi, 1)

    return HI(hi=hi)
