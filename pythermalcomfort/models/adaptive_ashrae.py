from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.psychrometrics import t_o
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import (
    units_converter,
    check_standard_compliance_array,
)


@dataclass
class AdaptiveASHRAE:
    tmp_cmf: Union[float, int, np.ndarray, List[float], List[int]]
    tmp_cmf_80_low: Union[float, int, np.ndarray, List[float], List[int]]
    tmp_cmf_80_up: Union[float, int, np.ndarray, List[float], List[int]]
    tmp_cmf_90_low: Union[float, int, np.ndarray, List[float], List[int]]
    tmp_cmf_90_up: Union[float, int, np.ndarray, List[float], List[int]]
    acceptability_80: Union[float, int, np.ndarray, List[float], List[int]]
    acceptability_90: Union[float, int, np.ndarray, List[float], List[int]]

    def __getitem__(self, item):
        return getattr(self, item)


def adaptive_ashrae(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    t_running_mean: Union[float, int, np.ndarray, List[float], List[int]],
    v: Union[float, int, np.ndarray, List[float], List[int]],
    units: str = "SI",
    limit_inputs: bool = True,
) -> AdaptiveASHRAE:
    """Determines the adaptive thermal comfort based on ASHRAE 55. The adaptive
    model relates indoor design temperatures or acceptable temperature ranges
    to outdoor meteorological or climatological parameters. The adaptive model
    can only be used in occupant-controlled naturally conditioned spaces that
    meet all the following criteria:

    * There is no mechianical cooling or heating system in operation
    * Occupants have a metabolic rate between 1.0 and 1.5 met
    * Occupants are free to adapt their clothing within a range as wide as 0.5 and 1.0 clo
    * The prevailing mean (runnin mean) outdoor temperature is between 10 and 33.5 °C

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
    units : str, optional
        select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
        0 < vr [m/s] < 2, 10 < t running mean [°C] < 33.5

    Returns
    -------
    tmp_cmf : float, int, or array-like
        Comfort temperature a that specific running mean temperature, default in [°C]
        or in [°F]
    tmp_cmf_80_low : float, int, or array-like
        Lower acceptable comfort temperature for 80% occupants, default in [°C] or in [°F]
    tmp_cmf_80_up : float, int, or array-like
        Upper acceptable comfort temperature for 80% occupants, default in [°C] or in [°F]
    tmp_cmf_90_low : float, int, or array-like
        Lower acceptable comfort temperature for 90% occupants, default in [°C] or in [°F]
    tmp_cmf_90_up : float, int, or array-like
        Upper acceptable comfort temperature for 90% occupants, default in [°C] or in [°F]
    acceptability_80 : bol or array-like
        Acceptability for 80% occupants
    acceptability_90 : bol or array-like
        Acceptability for 90% occupants

    Notes
    -----
    You can use this function to calculate if your conditions are within the `adaptive
    thermal comfort region`.
    Calculations with comply with the ASHRAE 55 2020 Standard [1]_.

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import adaptive_ashrae
        >>> results = adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 24.0, 'tmp_cmf_80_low': 20.5, 'tmp_cmf_80_up': 27.5,
        'tmp_cmf_90_low': 21.5, 'tmp_cmf_90_up': 26.5, 'acceptability_80': array(True),
        'acceptability_90': array(True)}

        >>> print(results.acceptability_80)  # or use print(results["acceptability_80"])
        True
        # The conditions you entered are considered to be comfortable for by 80% of the
        occupants

        >>> # You can also pass arrays as input to the function
        >>> results = adaptive_ashrae(tdb=[25, 26], tr=25, t_running_mean=20, v=0.1)
        >>> print(results)
        {'tmp_cmf': 24.0, 'tmp_cmf_80_low': 20.5, 'tmp_cmf_80_up': 27.5,
        'tmp_cmf_90_low': 21.5, 'tmp_cmf_90_up': 26.5, 'acceptability_80': array(True),
        'acceptability_90': array(True)}

        >>> # for users who want to use the IP system
        >>> results = adaptive_ashrae(tdb=77, tr=77, t_running_mean=68, v=0.3, units='ip')
        >>> print(results)
        {'tmp_cmf': 75.2, 'tmp_cmf_80_low': 68.9, 'tmp_cmf_80_up': 81.5,
        'tmp_cmf_90_low': 70.7, 'tmp_cmf_90_up': 79.7, 'acceptability_80': array(True),
        'acceptability_90': array(True)}

        >>> adaptive_ashrae(tdb=25, tr=25, t_running_mean=9, v=0.1)
        {'tmp_cmf': nan, 'tmp_cmf_80_low': nan, ... }
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is higher than 10°C
    """

    tdb = np.array(tdb)
    tr = np.array(tr)
    t_running_mean = np.array(t_running_mean)
    v = np.array(v)
    standard = "ashrae"

    # Validate units string
    valid_units: List[str] = ["SI", "IP"]
    if units.upper() not in valid_units:
        raise ValueError(f"Invalid unit: {units}. Supported units are {valid_units}.")

    if units.lower() == "ip":
        tdb, tr, t_running_mean, v = units_converter(
            tdb=tdb, tr=tr, tmp_running_mean=t_running_mean, v=v
        )

    (
        tdb_valid,
        tr_valid,
        v_valid,
    ) = check_standard_compliance_array(standard, tdb=tdb, tr=tr, v=v)
    trm_valid = valid_range(t_running_mean, (10.0, 33.5))

    to = t_o(tdb, tr, v, standard=standard)

    # calculate cooling effect (ce) of elevated air speed when top > 25 degC.
    ce = np.where((v >= 0.6) & (to >= 25.0), 999, 0)
    ce = np.where((v < 0.9) & (ce == 999), 1.2, ce)
    ce = np.where((v < 1.2) & (ce == 999), 1.8, ce)
    ce = np.where(ce == 999, 2.2, ce)

    # Relation between comfort and outdoor temperature
    t_cmf = 0.31 * t_running_mean + 17.8

    if limit_inputs:
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(trm_valid)
        )
        t_cmf = np.where(all_valid, t_cmf, np.nan)

    t_cmf = np.around(t_cmf, 1)

    tmp_cmf_80_low = t_cmf - 3.5
    tmp_cmf_90_low = t_cmf - 2.5
    tmp_cmf_80_up = t_cmf + 3.5 + ce
    tmp_cmf_90_up = t_cmf + 2.5 + ce

    acceptability_80 = np.where(
        (tmp_cmf_80_low <= to) & (to <= tmp_cmf_80_up), True, False
    )
    acceptability_90 = np.where(
        (tmp_cmf_90_low <= to) & (to <= tmp_cmf_90_up), True, False
    )

    if units.lower() == "ip":
        (
            t_cmf,
            tmp_cmf_80_low,
            tmp_cmf_80_up,
            tmp_cmf_90_low,
            tmp_cmf_90_up,
        ) = units_converter(
            from_units="si",
            tmp_cmf=t_cmf,
            tmp_cmf_80_low=tmp_cmf_80_low,
            tmp_cmf_80_up=tmp_cmf_80_up,
            tmp_cmf_90_low=tmp_cmf_90_low,
            tmp_cmf_90_up=tmp_cmf_90_up,
        )

    return AdaptiveASHRAE(
        tmp_cmf=t_cmf,
        tmp_cmf_80_low=tmp_cmf_80_low,
        tmp_cmf_80_up=tmp_cmf_80_up,
        tmp_cmf_90_low=tmp_cmf_90_low,
        tmp_cmf_90_up=tmp_cmf_90_up,
        acceptability_80=acceptability_80,
        acceptability_90=acceptability_90,
    )
