from dataclasses import dataclass
from typing import Union

import numpy as np
import numpy.typing as npt

from pythermalcomfort.models.pmv_ppd import _pmv_ppd_optimized


@dataclass
class ATHB:
    """
    Dataclass to store the results of the Adaptive Thermal Heat Balance (ATHB) calculation.

    Attributes
    ----------
    athb_pmv : float or array-like
        Predicted Mean Vote calculated with the Adaptive Thermal Heat Balance framework.
    """

    athb_pmv: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class ATHBInputs:
    tdb: Union[float, int, npt.ArrayLike]
    tr: Union[float, int, npt.ArrayLike]
    vr: Union[float, int, npt.ArrayLike]
    rh: Union[float, int, npt.ArrayLike]
    met: Union[float, int, npt.ArrayLike]
    t_running_mean: Union[float, int, npt.ArrayLike]

    def __post_init__(self):
        if not isinstance(self.tdb, (float, int, np.ndarray, list)):
            raise TypeError("tdb must be a float, int, or array-like.")
        if not isinstance(self.tr, (float, int, np.ndarray, list)):
            raise TypeError("tr must be a float, int, or array-like.")
        if not isinstance(self.vr, (float, int, np.ndarray, list)):
            raise TypeError("vr must be a float, int, or array-like.")
        if not isinstance(self.rh, (float, int, np.ndarray, list)):
            raise TypeError("rh must be a float, int, or array-like.")
        if not isinstance(self.met, (float, int, np.ndarray, list)):
            raise TypeError("met must be a float, int, or array-like.")
        if not isinstance(self.t_running_mean, (float, int, np.ndarray, list)):
            raise TypeError("t_running_mean must be a float, int, or array-like.")


def athb(
    tdb: Union[float, int, npt.ArrayLike],
    tr: Union[float, int, npt.ArrayLike],
    vr: Union[float, int, npt.ArrayLike],
    rh: Union[float, int, npt.ArrayLike],
    met: Union[float, int, npt.ArrayLike],
    t_running_mean: Union[float, int, npt.ArrayLike],
) -> ATHB:
    """
    Return the PMV value calculated with the Adaptive Thermal Heat Balance
    Framework [27]_. The adaptive thermal heat balance (ATHB) framework
    introduced a method to account for the three adaptive principals, namely
    physiological, behavioral, and psychological adaptation, individually
    within existing heat balance models. The objective is a predictive model of
    thermal sensation applicable during the design stage or in international
    standards without knowing characteristics of future occupants.

    Parameters
    ----------
    tdb : float, int, or array-like
        Dry bulb air temperature, in [°C].
    tr : float, int, or array-like
        Mean radiant temperature, in [°C].
    vr : float, int, or array-like
        Relative air speed, in [m/s].

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. The relative air speed is the sum of the
            average air speed measured by the sensor plus the activity-generated air speed
            (Vag). Where Vag is the activity-generated air speed caused by motion of
            individual body parts. vr can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.v_relative`.

    rh : float, int, or array-like
        Relative humidity, [%].
    met : float, int, or array-like
        Metabolic rate, [met].
    t_running_mean: float or array-like
        Running mean temperature, in [°C].

        .. note::
            The running mean temperature can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.running_mean_outdoor_temperature`.

    Returns
    -------
    ATHBResults
        Dataclass containing the results of the ATHB calculation. See :py:class:`~pythermalcomfort.models.athb.ATHBResults` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import athb

        # calculate the predicted mean vote (PMV) using the Adaptive Thermal Heat Balance model
        results = athb(tdb=25, tr=25, vr=0.1, rh=50, met=1.2, t_running_mean=20)
        print(results.athb_pmv)  # returns the PMV value

        # for multiple points
        results = athb(tdb=[25, 25, 25], tr=[25, 25, 25], vr=[0.1, 0.1, 0.1], rh=[50, 50, 50], met=[1.2, 1.2, 1.2], t_running_mean=[20, 20, 20])
        print(results.athb_pmv)
    """
    # Validate inputs using the ATHBInputs class
    ATHBInputs(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, t_running_mean=t_running_mean)

    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    met = np.array(met)
    rh = np.array(rh)
    t_running_mean = np.array(t_running_mean)

    met_adapted = met - (0.234 * t_running_mean) / 58.2

    # Adapted clothing insulation level through behavioral adaptation
    clo_adapted = np.power(
        10,
        (
            -0.17168
            - 0.000485 * t_running_mean
            + 0.08176 * met_adapted
            - 0.00527 * t_running_mean * met_adapted
        ),
    )

    pmv_res = _pmv_ppd_optimized(tdb, tr, vr, rh, met_adapted, clo_adapted, 0)
    ts = 0.303 * np.exp(-0.036 * met_adapted * 58.15) + 0.028
    l_adapted = pmv_res / ts

    # Predicted thermal sensation vote
    athb_pmv = np.around(
        1.484
        + 0.0276 * l_adapted
        - 0.9602 * met_adapted
        - 0.0342 * t_running_mean
        + 0.0002264 * l_adapted * t_running_mean
        + 0.018696 * met_adapted * t_running_mean
        - 0.0002909 * l_adapted * met_adapted * t_running_mean,
        3,
    )

    return ATHB(athb_pmv=athb_pmv)
