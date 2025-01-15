from typing import Union

import numpy as np

from pythermalcomfort.classes_input import NETInputs
from pythermalcomfort.classes_return import NET


def net(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    v: Union[float, list[float]],
    round_output: bool = True,
) -> NET:
    """Calculates the Normal Effective Temperature (NET). Missenard (1933)
    devised a formula for calculating effective temperature. The index
    establishes a link between the same condition of the organism's
    thermoregulatory capability (warm and cold perception) and the surrounding
    environment's temperature and humidity. The index is calculated as a
    function of three meteorological factors: air temperature, relative
    humidity of air, and wind speed. This index allows to calculate the
    effective temperature felt by a person. Missenard original equation was
    then used to calculate the Normal Effective Temperature (NET), by
    considering normal atmospheric pressure and a normal human body temperature
    (37°C). The NET is still in use in Germany, where medical check-ups for
    subjects working in the heat are decided on by prevailing levels of ET,
    depending on metabolic rates. The NET is also constantly monitored by the
    Hong Kong Observatory [Blazejczyk2012]_. In central Europe the following thresholds are
    in use: <1°C = very cold; 1–9 = cold; 9–17 = cool; 17–21 = fresh; 21–23 = comfortable;
    23–27 = warm; >27°C = hot [Blazejczyk2012]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    v : float or list of floats
        Wind speed [m/s] at 1.2 m above the ground.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    NET
        A dataclass containing the Normal Effective Temperature. See :py:class:`~pythermalcomfort.classes_return.Net` for more details.
        To access the `net` value, use the `net` attribute of the returned `Net` instance, e.g., `result.net`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import net

        result = net(tdb=37, rh=100, v=0.1)
        print(result.net)  # 37.0

        result = net(tdb=[37, 30], rh=[100, 60], v=[0.1, 0.5], round_output=False)
        print(result.net)  # [37.0, 26.38977535]

    """
    # Validate inputs using the NetInputs class
    NETInputs(
        tdb=tdb,
        rh=rh,
        v=v,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)
    v = np.array(v)

    frac = 1.0 / (1.76 + 1.4 * v**0.75)
    et = 37 - (37 - tdb) / (0.68 - 0.0014 * rh + frac) - 0.29 * tdb * (1 - 0.01 * rh)

    if round_output:
        et = np.around(et, 1)

    return NET(net=et)
