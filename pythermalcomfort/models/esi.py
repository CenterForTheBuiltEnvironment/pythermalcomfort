from typing import Union

import numpy as np

from pythermalcomfort.classes_input import ESIInputs
from pythermalcomfort.classes_return import ESI


def esi(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    sol_radiation_dir: Union[float, list[float]],
    round_output: bool = True,
) -> ESI:
    """Calculates the Environmental Stress Index (ESI) [Moran2001]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh: float or list of floats
        Relative humidity, [%].
    sol_radiation_dir: float or list of floats
        Direct solar radiation, [W/m2].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    ESI
        A dataclass containing the Environmental Stress Index. See :py:class:`~pythermalcomfort.classes_return.ESI` for more details.
        To access the `esi` value, use the `esi` attribute of the returned `ESI` instance, e.g., `result.esi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import esi

        result = esi(tdb=30.2, rh=42.2, sol_radiation_dir=766)
        print(result.esi)  # 26.2

        result = esi(tdb=[30.2, 27.0], rh=[42.2, 68.8], sol_radiation_dir=[766, 289])
        print(result.esi)  # [26.2, 25.6]
    """

    ESIInputs(
        tdb=tdb,
        rh=rh,
        sol_radiation_dir=sol_radiation_dir,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)
    sol_radiation_dir = np.array(sol_radiation_dir)

    esi = (
        0.63 * tdb
        - 0.03 * rh
        + 0.002 * sol_radiation_dir
        + 0.0054 * (tdb * rh)
        - 0.073 * (0.1 + sol_radiation_dir) ** (-1)
    )

    if round_output:
        esi = np.round(esi, 1)
    return ESI(esi=esi)
