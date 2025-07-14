from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import THIInputs
from pythermalcomfort.classes_return import THI


def thi(
    tdb: float | list[float],
    rh: float | list[float],
    round_output: bool = True,
) -> THI:
    """Calculate the Temperature-Humidity Index (THI) defined in [Yan2025]_, equivalent to
    the definition in [Schlatter1987]_, but uses Celsius instead of Fahrenheit.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh: float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    THI
        A dataclass containing the Temperature-Humidity Index.
        See :py:class:`~pythermalcomfort.classes_return.THI` for more details.
        To access the `thi` value, use the `thi` attribute of the returned `THI`
        instance, e.g., `result.thi`.

    """
    # Validate inputs using the THIInputs class
    THIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    _thi = 1.8 * tdb + 32 - 0.55 * (1 - 0.01 * rh) * (1.8 * tdb - 26)

    if round_output:
        _thi = np.round(_thi, 1)

    return THI(thi=_thi)
