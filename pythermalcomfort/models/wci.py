from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class WCI:
    """
    Dataclass to represent the Wind Chill Index (WCI).

    Attributes
    ----------
    wci : float or list of floats
        Wind Chill Index, [W/m^2].
    """

    wci: Union[float, List[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class WCIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )


def wci(
    tdb: Union[float, List[float]],
    v: Union[float, List[float]],
    round_output: bool = True,
) -> WCI:
    """Calculates the Wind Chill Index (WCI) in accordance with the ASHRAE 2017 Handbook Fundamentals - Chapter 9 [18]_.

    The wind chill index (WCI) is an empirical index based on cooling measurements
    taken on a cylindrical flask partially filled with water in Antarctica
    (Siple and Passel 1945). For a surface temperature of 33°C, the index describes
    the rate of heat loss from the cylinder via radiation and convection as a function
    of ambient temperature and wind velocity.

    This formulation has been met with some valid criticism. WCI is unlikely to be an
    accurate measure of heat loss from exposed flesh, which differs from plastic in terms
    of curvature, roughness, and radiation exchange qualities, and is always below 33°C
    in a cold environment. Furthermore, the equation's values peak at 90 km/h and then
    decline as velocity increases. Nonetheless, this score reliably represents the
    combined effects of temperature and wind on subjective discomfort for velocities
    below 80 km/h [18]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    v : float or list of floats
        Wind speed 10m above ground level, [m/s].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    WCI
        A dataclass containing the Wind Chill Index. See :py:class:`~pythermalcomfort.models.wc.WCI` for more details.
        To access the `wci` value, use the `wci` attribute of the returned `WCI` instance, e.g., `result.wci`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import wc

        result = wc(tdb=-5, v=5.5)
        print(result.wci)  # 1255.2

        result = wc(tdb=[-5, -10], v=[5.5, 10], round_output=True)
        print(result.wci)  # [1255.2 1603.9]
    """

    # Validate inputs using the WCYInputs class
    WCIInputs(
        tdb=tdb,
        v=v,
        round_output=round_output,
    )

    tdb = np.array(tdb)
    v = np.array(v)

    wci = (10.45 + 10 * v**0.5 - v) * (33 - tdb)

    # the factor 1.163 is used to convert to W/m^2
    wci = wci * 1.163

    if round_output:
        wci = np.around(wci, 1)

    return WCI(wci=wci)
