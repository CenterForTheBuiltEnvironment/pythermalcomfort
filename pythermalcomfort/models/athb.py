import numpy as np

from pythermalcomfort.models.pmv_ppd import _pmv_ppd_optimized


def athb(tdb, tr, vr, rh, met, t_running_mean):
    """Return the PMV value calculated with the Adaptive Thermal Heat Balance
    Framework [27]_. The adaptive thermal heat balance (ATHB) framework
    introduced a method to account for the three adaptive principals, namely
    physiological, behavioral, and psychological adaptation, individually
    within existing heat balance models. The objective is a predictive model of
    thermal sensation applicable during the design stage or in international
    standards without knowing characteristics of future occupants.

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, in [°C]
    tr : float, int, or array-like
        mean radiant temperature, in [°C]
    vr : float, int, or array-like
        relative air speed, in [m/s]

        Note: vr is the relative air speed caused by body movement and not the air
        speed measured by the air speed sensor. The relative air speed is the sum of the
        average air speed measured by the sensor plus the activity-generated air speed
        (Vag). Where Vag is the activity-generated air speed caused by motion of
        individual body parts. vr can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.v_relative`.
    rh : float, int, or array-like
        relative humidity, [%]
    met : float, int, or array-like
        metabolic rate, [met]
    t_running_mean: float or array-like
        running mean temperature, in [°C]

        The running mean temperature can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.running_mean_outdoor_temperature`.

    Returns
    -------
    athb_pmv : float or array-like
        Predicted Mean Vote calculated with the Adaptive Thermal Heat Balance framework

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import athb
        >>> print(athb( tdb=[25, 27], tr=25, vr=0.1, rh=50, met=1.1, t_running_mean=20))
        [0.2, 0.209]
    """
    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    met = np.array(met)
    rh = np.array(rh)
    t_running_mean = np.array(t_running_mean)

    met_adapted = met - (0.234 * t_running_mean) / 58.2

    # adapted clothing insulation level through behavioural adaptation
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

    # predicted thermal sensation vote
    return np.around(
        1.484
        + 0.0276 * l_adapted
        - 0.9602 * met_adapted
        - 0.0342 * t_running_mean
        + 0.0002264 * l_adapted * t_running_mean
        + 0.018696 * met_adapted * t_running_mean
        - 0.0002909 * l_adapted * met_adapted * t_running_mean,
        3,
    )
