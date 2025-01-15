from typing import Union

import numpy as np

from pythermalcomfort.classes_input import WCTInputs
from pythermalcomfort.classes_return import WCT


def wct(
    tdb: Union[float, list[float]],
    v: Union[float, list[float]],
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
        A dataclass containing the Wind Chill Temperature. See :py:class:`~pythermalcomfort.classes_return.WCT` for more details.
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

    _wct = 13.12 + 0.6215 * tdb - 11.37 * v**0.16 + 0.3965 * tdb * v**0.16

    if round_output:
        _wct = np.around(_wct, 1)

    return WCT(wct=_wct)
