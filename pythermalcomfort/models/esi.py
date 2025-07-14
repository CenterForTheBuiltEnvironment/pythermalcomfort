from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import ESIInputs
from pythermalcomfort.classes_return import ESI


def esi(
    tdb: float | list[float],
    rh: float | list[float],
    sol_radiation_global: float | list[float],
    round_output: bool = True,
) -> ESI:
    """Calculate the Environmental Stress Index (ESI) [Moran2001]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh: float or list of floats
        Relative humidity, [%].
    sol_radiation_global: float or list of floats
        Global solar radiation, [W/m2].
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

        result = esi(tdb=30.2, rh=42.2, sol_radiation_global=766)
        print(result.esi)  # 26.2

        result = esi(tdb=[30.2, 27.0], rh=[42.2, 68.8], sol_radiation_global=[766, 289])
        print(result.esi)  # [26.2, 25.6]

    """
    ESIInputs(
        tdb=tdb,
        rh=rh,
        sol_radiation_global=sol_radiation_global,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)
    sol_radiation_global = np.array(sol_radiation_global)

    _esi = (
        0.63 * tdb
        - 0.03 * rh
        + 0.002 * sol_radiation_global
        + 0.0054 * (tdb * rh)
        - 0.073 * (0.1 + sol_radiation_global) ** (-1)
    )

    if round_output:
        _esi = np.round(_esi, 1)

    return ESI(esi=_esi)
