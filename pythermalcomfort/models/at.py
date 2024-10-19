from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.psychrometrics import psy_ta_rh
from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class AT:
    """Dataclass to store the results of the Apparent Temperature (AT) calculation.

    Attributes
    ----------
    at : float
        Apparent temperature, [°C]
    """

    at: float

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class ATInputs(BaseInputs):

    def __init__(
        self,
        tdb,
        rh,
        v,
        q=None,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            vr=v,
            rh=rh,
            q=q,
            round_output=round_output,
        )


def at(
    tdb: Union[float, List[float]],
    rh: Union[float, List[float]],
    v: Union[float, List[float]],
    q: Union[float, List[float]] = None,
    round_output: bool = True,
) -> AT:
    """
    Calculates the Apparent Temperature (AT). The AT is defined as the
    temperature at the reference humidity level producing the same amount of
    discomfort as that experienced under the current ambient temperature,
    humidity, and solar radiation [17]_. In other words, the AT is an
    adjustment to the dry bulb temperature based on the relative humidity
    value. Absolute humidity with a dew point of 14°C is chosen as a reference.

    It includes the chilling effect of the wind at lower temperatures. [16]_

    .. note::
        Two formulas for AT are in use by the Australian Bureau of Meteorology: one includes
        solar radiation and the other one does not (http://www.bom.gov.au/info/thermal_stress/
        , 29 Sep 2021). Please specify q if you want to estimate AT with solar load.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C]
    rh : float or list of floats
        Relative humidity, [%]
    v : float or list of floats
        Wind speed 10m above ground level, [m/s]
    q : float or list of floats, optional
        Net radiation absorbed per unit area of body surface [W/m2]
    round_output : bool, default True
        If True, rounds the output value; if False, does not round it.

    Returns
    -------
    AT
        Dataclass containing the apparent temperature, [°C]. See :py:class:`~pythermalcomfort.models.at.AT` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import at
        at(tdb=25, rh=30, v=0.1)
        # AT(at=24.1)
    """
    # Validate inputs
    ATInputs(tdb=tdb, rh=rh, v=v, q=q, round_output=round_output)

    # Convert lists to numpy arrays if necessary
    tdb = np.array(tdb)
    rh = np.array(rh)
    v = np.array(v)

    # Calculate vapor pressure
    p_vap = psy_ta_rh(tdb, rh).p_vap / 100

    # Calculate apparent temperature
    if q is not None:
        q = np.array(q)
        t_at = tdb + 0.348 * p_vap - 0.7 * v + 0.7 * q / (v + 10) - 4.25
    else:
        t_at = tdb + 0.33 * p_vap - 0.7 * v - 4.00

    if round_output:
        t_at = np.around(t_at, 1)

    return AT(at=t_at)


if __name__ == "__main__":
    at(25, 30, 0.1)
    at([25, 25, 25], [30, 30, 30], [0.1, 0.1, 0.1])
    at(25, 30, 0.1, 100)
