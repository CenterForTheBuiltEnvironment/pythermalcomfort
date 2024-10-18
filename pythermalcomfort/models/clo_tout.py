from typing import Union, List, Literal
from dataclasses import dataclass
import numpy as np
from pythermalcomfort.utilities import units_converter
from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class CloTout:
    """
    Dataclass to represent the clothing insulation Icl as a function of outdoor air temperature.

    Attributes
    ----------
    clo_tout : float or np.ndarray
        Representative clothing insulation Icl.
    """

    clo_tout: Union[float, np.ndarray]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class CloToutInputs(BaseInputs):
    def __init__(
        self,
        tout: Union[float, List[float]],
        units: str = "SI",
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tout=tout,
            units=units,
        )


def clo_tout(
    tout: Union[float, List[float]],
    units: Literal["SI", "IP"] = "SI",
) -> CloTout:
    """Representative clothing insulation Icl as a function of outdoor air
    temperature at 06:00 a.m [4]_.

    Parameters
    ----------
    tout : float or list of floats
        Outdoor air temperature at 06:00 a.m., default in [°C] or [°F] if `units` = 'IP'.
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.

    Returns
    -------
    CloTout
        A dataclass containing the representative clothing insulation Icl. See :py:class:`~pythermalcomfort.models.clo_tout.CloTout` for more details.
        To access the `clo_tout` value, use the `clo_tout` attribute of the returned `CloTout` instance, e.g., `result.clo_tout`.

    Raises
    ------
    TypeError
        If `tout` is not a float, int, NumPy array, or a list of floats or integers.
    ValueError
        If an invalid unit is provided or non-numeric elements are found in `tout`.

    Notes
    -----
    .. note::
        The ASHRAE 55 2020 states that it is acceptable to determine the clothing
        insulation Icl using this equation in mechanically conditioned buildings [1]_.

    .. warning::
        Limitations:
        - This equation may not be accurate for extreme temperature ranges.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import clo_tout

        result = clo_tout(tout=27)
        print(result.clo_tout)  # 0.46

        result = clo_tout(tout=[27, 25])
        print(result.clo_tout)  # array([0.46, 0.47])
    """

    # Validate inputs using the CloToutInputs class
    CloToutInputs(
        tout=tout,
        units=units,
    )

    # Convert tout to NumPy array for vectorized operations
    tout = np.array(tout)

    # Convert units if necessary
    if units.lower() == "ip":
        tout = units_converter(tmp=tout)[0]

    clo = np.where(tout < 26, np.power(10, -0.1635 - 0.0066 * tout), 0.46)
    clo = np.where(tout < 5, 0.818 - 0.0364 * tout, clo)
    clo = np.where(tout < -5, 1, clo)

    return CloTout(clo_tout=np.around(clo, 2))
