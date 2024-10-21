from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class HeatIndex:
    """
    Dataclass to represent the Heat Index (HI).

    Attributes
    ----------
    hi : float or list of floats
        Heat Index, [°C] or [°F] depending on the units.
    """

    hi: Union[float, List[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class HeatIndexInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        units,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            units=units,
            round_output=round_output,
        )


def heat_index(
    tdb: Union[float, List[float]],
    rh: Union[float, List[float]],
    units: str = "SI",
    round_output: bool = True,
) -> HeatIndex:
    """Calculates the Heat Index (HI). It combines air temperature and relative
    humidity to determine an apparent temperature. The HI equation [12]_ is
    derived by multiple regression analysis in temperature and relative
    humidity from the first version of Steadman’s (1979) apparent temperature
    (AT) [13]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'.
    rh : float or list of floats
        Relative humidity, [%].
    units : {'SI', 'IP'}, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HeatIndex
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.models.heat_index.HeatIndex` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HeatIndex` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index

        result = heat_index(tdb=25, rh=50)
        print(result.hi)  # 25.9

        result = heat_index(tdb=[25, 30], rh=[50, 60], units="IP", round_output=False)
        print(result.hi)  # [78.6, 86.7]
    """

    # Validate inputs using the HeatIndexInputs class
    HeatIndexInputs(
        tdb=tdb,
        rh=rh,
        units=units,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    if units == "SI":
        hi = -8.784695 + 1.61139411 * tdb + 2.338549 * rh - 0.14611605 * tdb * rh
        hi += -1.2308094 * 10**-2 * tdb**2 - 1.6424828 * 10**-2 * rh**2
        hi += 2.211732 * 10**-3 * tdb**2 * rh + 7.2546 * 10**-4 * tdb * rh**2
        hi += -3.582 * 10**-6 * tdb**2 * rh**2
    else:
        hi = -42.379 + 2.04901523 * tdb + 10.14333127 * rh
        hi += -0.22475541 * tdb * rh - 6.83783 * 10**-3 * tdb**2
        hi += -5.481717 * 10**-2 * rh**2
        hi += 1.22874 * 10**-3 * tdb**2 * rh + 8.5282 * 10**-4 * tdb * rh**2
        hi += -1.99 * 10**-6 * tdb**2 * rh**2

    if round_output:
        hi = np.around(hi, 1)

    return HeatIndex(hi=hi)
