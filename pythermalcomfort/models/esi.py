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
