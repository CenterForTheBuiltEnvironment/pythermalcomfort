from typing import Union

import numpy as np

from pythermalcomfort.classes_input import THIInputs
from pythermalcomfort.classes_return import THI
from pythermalcomfort.utilities import THIModels


def thi(
    tdb: Union[float, list[float]],
    rh: Union[float, list[float]],
    twb: Union[float, list[float]],
    tdp: Union[float, list[float]],
    model: str,
    round_output: bool = True,
) -> THI:

    if model == THIModels.yousef_1985:
        output = tdb + 0.36 * tdp + 41.2

    return THI(thi=output)
