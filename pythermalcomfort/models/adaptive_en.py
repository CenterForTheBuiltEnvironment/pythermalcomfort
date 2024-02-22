from typing import Union, List

import numpy as np

from pythermalcomfort.psychrometrics import t_o
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import (
    units_converter,
)


def adaptive_en(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    t_running_mean: Union[float, int, np.ndarray, List[float], List[int]],
    v: Union[float, int, np.ndarray, List[float], List[int]],
    units: str = "SI",
    limit_inputs: bool = True,
):
    """Determines the adaptive thermal comfort based on EN 16798-1 2019 [3]_

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float, int, or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    t_running_mean: float, int, or array-like
        running mean temperature, default in [°C] in [°C] in [°F] if `units` = 'IP'

        The running mean temperature can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.running_mean_outdoor_temperature`.
    v : float, int, or array-like
        air speed, default in [m/s] in [fps] if `units` = 'IP'

        Note: Indoor operative temperature correction is applicable for buildings equipped
        with fans or personal systems providing building occupants with personal
        control over air speed at occupant level.
        For operative temperatures above 25°C the comfort zone upper limit can be
        increased by 1.2 °C (0.6 < v < 0.9 m/s), 1.8 °C (0.9 < v < 1.2 m/s), 2.2 °C ( v
        > 1.2 m/s)
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

    Returns
    -------
    tmp_cmf : float, int, or array-like
        Comfort temperature at that specific running mean temperature, default in [°C]
        or in [°F]
    acceptability_cat_i : bol or array-like
        If the indoor conditions comply with comfort category I
    acceptability_cat_ii : bol or array-like
        If the indoor conditions comply with comfort category II
    acceptability_cat_iii : bol or array-like
        If the indoor conditions comply with comfort category III
    tmp_cmf_cat_i_up : float, int, or array-like
        Upper acceptable comfort temperature for category I, default in [°C] or in [°F]
    tmp_cmf_cat_ii_up : float, int, or array-like
        Upper acceptable comfort temperature for category II, default in [°C] or in [°F]
    tmp_cmf_cat_iii_up : float, int, or array-like
        Upper acceptable comfort temperature for category III, default in [°C] or in [°F]
    tmp_cmf_cat_i_low : float, int, or array-like
        Lower acceptable comfort temperature for category I, default in [°C] or in [°F]
    tmp_cmf_cat_ii_low : float, int, or array-like
        Lower acceptable comfort temperature for category II, default in [°C] or in [°F]
    tmp_cmf_cat_iii_low : float or array-like
        Lower acceptable comfort temperature for category III, default in [°C] or in [°F]

    Notes
    -----
    You can use this function to calculate if your conditions are within the EN
    adaptive thermal comfort region.
    Calculations with comply with the EN 16798-1 2019 [3]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_en
        >>> results = adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 25.4, 'acceptability_cat_i': True, 'acceptability_cat_ii': True, ... }

        >>> print(results['acceptability_cat_i'])
        True
        # The conditions you entered are considered to comply with Category I

        >>> # for users who wants to use the IP system
        >>> results = adaptive_en(tdb=77, tr=77, t_running_mean=68, v=0.3, units='ip')
        >>> print(results)
        {'tmp_cmf': 77.7, 'acceptability_cat_i': True, 'acceptability_cat_ii': True, ... }

        >>> results = adaptive_en(tdb=25, tr=25, t_running_mean=9, v=0.1)
        {'tmp_cmf': nan, 'acceptability_cat_i': True, 'acceptability_cat_ii': True, ... }
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is between 10 °C and 30 °C
    """

    tdb = np.array(tdb)
    tr = np.array(tr)
    t_running_mean = np.array(t_running_mean)
    v = np.array(v)
    standard = "iso"

    if units.lower() == "ip":
        tdb, tr, t_running_mean, vr = units_converter(
            tdb=tdb, tr=tr, tmp_running_mean=t_running_mean, v=v
        )

    trm_valid = valid_range(t_running_mean, (10.0, 33.5))

    to = t_o(tdb, tr, v, standard=standard)

    # calculate cooling effect (ce) of elevated air speed when top > 25 degC.
    ce = np.where((v >= 0.6) & (to >= 25.0), 999, 0)
    ce = np.where((v < 0.9) & (ce == 999), 1.2, ce)
    ce = np.where((v < 1.2) & (ce == 999), 1.8, ce)
    ce = np.where(ce == 999, 2.2, ce)

    t_cmf = 0.33 * t_running_mean + 18.8

    if limit_inputs:
        all_valid = ~(np.isnan(trm_valid))
        t_cmf = np.where(all_valid, t_cmf, np.nan)

    t_cmf_i_lower = t_cmf - 3.0
    t_cmf_ii_lower = t_cmf - 4.0
    t_cmf_iii_lower = t_cmf - 5.0
    t_cmf_i_upper = t_cmf + 2.0 + ce
    t_cmf_ii_upper = t_cmf + 3.0 + ce
    t_cmf_iii_upper = t_cmf + 4.0 + ce

    acceptability_i = np.where(
        (t_cmf_i_lower <= to) & (to <= t_cmf_i_upper), True, False
    )
    acceptability_ii = np.where(
        (t_cmf_ii_lower <= to) & (to <= t_cmf_ii_upper), True, False
    )
    acceptability_iii = np.where(
        (t_cmf_iii_lower <= to) & (to <= t_cmf_iii_upper), True, False
    )

    if units.lower() == "ip":
        t_cmf, t_cmf_i_upper, t_cmf_ii_upper, t_cmf_iii_upper = units_converter(
            from_units="si",
            tmp_cmf=t_cmf,
            tmp_cmf_cat_i_up=t_cmf_i_upper,
            tmp_cmf_cat_ii_up=t_cmf_ii_upper,
            tmp_cmf_cat_iii_up=t_cmf_iii_upper,
        )
        t_cmf_i_lower, t_cmf_ii_lower, t_cmf_iii_lower = units_converter(
            from_units="si",
            tmp_cmf_cat_i_low=t_cmf_i_lower,
            tmp_cmf_cat_ii_low=t_cmf_ii_lower,
            tmp_cmf_cat_iii_low=t_cmf_iii_lower,
        )

    results = {
        "tmp_cmf": np.around(t_cmf, 1),
        "acceptability_cat_i": acceptability_i,
        "acceptability_cat_ii": acceptability_ii,
        "acceptability_cat_iii": acceptability_iii,
        "tmp_cmf_cat_i_up": np.around(t_cmf_i_upper, 1),
        "tmp_cmf_cat_ii_up": np.around(t_cmf_ii_upper, 1),
        "tmp_cmf_cat_iii_up": np.around(t_cmf_iii_upper, 1),
        "tmp_cmf_cat_i_low": np.around(t_cmf_i_lower, 1),
        "tmp_cmf_cat_ii_low": np.around(t_cmf_ii_lower, 1),
        "tmp_cmf_cat_iii_low": np.around(t_cmf_iii_lower, 1),
    }

    return results
