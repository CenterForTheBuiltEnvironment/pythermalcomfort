from dataclasses import dataclass
from enum import Enum
from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import BaseInputs, dew_point_tmp


class HumidexModels(Enum):
    rana = "rana"
    masterson = "masterson"


@dataclass(frozen=True)
class Humidex:
    """
    Dataclass to represent the Humidex and its discomfort category.

    Attributes
    ----------
    humidex : float or list of floats
        Humidex value, [°C].
    discomfort : str or list of str
        Degree of comfort or discomfort as defined in Havenith and Fiala (2016).
    """

    humidex: Union[float, List[float]]
    discomfort: Union[str, List[str]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class HumidexInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
        )


def humidex(
    tdb: Union[float, List[float]],
    rh: Union[float, List[float]],
    round_output: bool = True,
    model: str = "rana",
) -> Humidex:
    """Calculates the humidex (short for "humidity index"). It has been
    developed by the Canadian Meteorological service. It was introduced in 1965
    and then it was revised by Masterson and Richardson (1979) [14]_. It aims
    to describe how hot, humid weather is felt by the average person. The
    Humidex differs from the heat index in being related to the dew point
    rather than relative humidity [15]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.
    model : str, optional
        The model to be used for the calculation. Options are 'rana' and 'masterson'. Defaults to 'rana'.

        .. note::
            The 'rana' model is the Humidex model proposed by `Rana et al. (2013)`_.
            The 'masterson' model is the Humidex model proposed by Masterson and Richardson (1979) [14]_.

            .. _Rana et al. (2013): https://doi.org/10.1016/j.enbuild.2013.04.019

    Returns
    -------
    Humidex
        A dataclass containing the Humidex value and its discomfort category. See :py:class:`~pythermalcomfort.models.humidex.Humidex` for more details.
        To access the `humidex` and `discomfort` values, use the respective attributes of the returned `Humidex` instance, e.g., `result.humidex`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import humidex

        result = humidex(tdb=25, rh=50)
        print(result.humidex)  # 28.2
        print(result.discomfort)  # Little or no discomfort

        result = humidex(tdb=[25, 30], rh=[50, 60], round_output=False)
        print(result.humidex)  # [28.2, 39.1]
        print(result.discomfort)  # ['Little or no discomfort', 'Evident discomfort']
    """

    # Validate inputs using the HumidexInputs class
    HumidexInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    rh = np.array(rh)

    if np.any(rh > 100) or np.any(rh < 0):
        raise ValueError("Relative humidity must be between 0 and 100%")

    if model not in [model.value for model in HumidexModels]:
        raise ValueError(
            "Invalid model. The model must be either 'rana' or 'masterson'"
        )

    hi = tdb + 5 / 9 * ((6.112 * 10 ** (7.5 * tdb / (237.7 + tdb)) * rh / 100) - 10)
    if model == HumidexModels.masterson.value:
        hi = tdb + 5 / 9 * (
            6.11
            * np.exp(
                5417.753 * (1 / 273.15 - 1 / (dew_point_tmp(tdb=tdb, rh=rh) + 273.15))
            )
            - 10
        )

    if round_output:
        hi = np.around(hi, 1)

    stress_category = np.full_like(hi, "Heat stroke probable", dtype=object)
    stress_category[hi <= 30] = "Little or no discomfort"
    stress_category[(hi > 30) & (hi <= 35)] = "Noticeable discomfort"
    stress_category[(hi > 35) & (hi <= 40)] = "Evident discomfort"
    stress_category[(hi > 40) & (hi <= 45)] = "Intense discomfort; avoid exertion"
    stress_category[(hi > 45) & (hi <= 54)] = "Dangerous discomfort"

    return Humidex(humidex=hi, discomfort=stress_category)


if __name__ == "__main__":
    result = humidex(tdb=25, rh=50)
    print(result.humidex)  # 28.2
    print(result.discomfort)  # Little or no discomfort

    result = humidex(tdb=[25, 30], rh=[50, 60], round_output=False)
    print(result.humidex)  # [28.2, 39.1]
    print(result.discomfort)  # ['Little or no discomfort', 'Evident discomfort']

    result = humidex(tdb=21, rh=100, model="masterson")
    print(result.humidex)  # 29.3
