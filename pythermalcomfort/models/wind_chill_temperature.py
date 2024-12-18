from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class WCT:
    """
    Dataclass to represent the Wind Chill Temperature (WCT).

    Attributes
    ----------
    wct : float or list of floats
        Wind Chill Temperature, [°C].
    """

    wct: Union[float, List[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class WCTInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )


def wct(
    tdb: Union[float, List[float]],
    v: Union[float, List[float]],
    round_output: bool = True,
) -> WCT:
    """Calculates the Wind Chill Temperature (`WCT`_).

    We validated the implementation of this model by comparing the results with the Wind Chill
    Temperature Calculator on `Calculator.net`_

    .. _WCT: https://en.wikipedia.org/wiki/Wind_chill#North_American_and_United_Kingdom_wind_chill_index
    .. _Calculator.net: https://www.calculator.net/wind-chill-calculator.html

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    v : float or list of floats
        Wind speed 10m above ground level, [km/h].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    WCT
        A dataclass containing the Wind Chill Temperature. See :py:class:`~pythermalcomfort.models.wct.WCT` for more details.
        To access the `wct` value, use the `wct` attribute of the returned `WCT` instance, e.g., `result.wct`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import wc

        result = wc(tdb=-5, v=5.5)
        print(result.wct)  # 1255.2

        result = wc(tdb=[-5, -10], v=[5.5, 10], round_output=True)
        print(result.wct)  # [1255.2 1603.9]
    """

    # Validate inputs using the WCYInputs class
    WCTInputs(
        tdb=tdb,
        v=v,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    v = np.array(v)

    wct = 13.12 + 0.6215 * tdb - 11.37 * v**0.16 + 0.3965 * tdb * v**0.16

    if round_output:
        wct = np.around(wct, 1)

    return WCT(wct=wct)


if __name__ == "__main__":
    result = wct(tdb=-20, v=5)
    print(result.wct)

    result = wct(tdb=-20, v=15)
    print(result.wct)

    result = wct(tdb=-20, v=60)
    print(result.wct)
