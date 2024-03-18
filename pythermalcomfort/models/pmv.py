from typing import Union, List

import numpy as np

from pythermalcomfort.models import pmv_ppd


def pmv(
    tdb: Union[float, int, np.ndarray, List[float], List[int]],
    tr: Union[float, int, np.ndarray, List[float], List[int]],
    vr: Union[float, int, np.ndarray, List[float], List[int]],
    rh: Union[float, int, np.ndarray, List[float], List[int]],
    met: Union[float, int, np.ndarray, List[float], List[int]],
    clo: Union[float, int, np.ndarray, List[float], List[int]],
    wme: Union[float, int, np.ndarray, List[float], List[int]] = 0,
    standard="ISO",
    units="SI",
    limit_inputs=True,
    airspeed_control=True,
):
    """Returns Predicted Mean Vote (`PMV`_) calculated in accordance to main
    thermal comfort Standards. The PMV is an index that predicts the mean value
    of the thermal sensation votes (self-reported perceptions) of a large group
    of people on a sensation scale expressed from –3 to +3 corresponding to the
    categories: cold, cool, slightly cool, neutral, slightly warm, warm, and hot. [1]_

    While the PMV equation is the same for both the ISO and ASHRAE standards, in the
    ASHRAE 55 PMV equation, the SET is used to calculate the cooling effect first,
    this is then subtracted from both the air and mean radiant temperatures, and the
    differences are used as input to the PMV model, while the airspeed is set to 0.1m/s.
    Please read more in the Note below.

    Parameters
    ----------
    tdb : float, int, or array-like
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'
    tr : float, int, or array-like
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float, int, or array-like
        relative air speed, default in [m/s] in [fps] if `units` = 'IP'

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
    clo : float, int, or array-like
        clothing insulation, [clo]

        Note: The activity as well as the air speed modify the insulation characteristics
        of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
        the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
        for the effect of the body movement for met equal or higher than 1.2 met using
        the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
        can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.clo_dynamic`.
    wme : float, int, or array-like
        external work, [met] default 0
    standard : str, optional
        select comfort standard used for calculation.
        Supported values are 'ASHRAE' and 'ISO'. Defaults to 'ISO'.

        - If "ISO", then the ISO Equation is used
        - If "ASHRAE", then the ASHRAE Equation is used

        Note: While the PMV equation is the same for both the ISO and ASHRAE standards,
        the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.10
        m/s (20 fpm).
        When air speeds exceed 0.10 m/s (20 fpm), the comfort zone boundaries are
        adjusted based on the SET model.
        This change was indroduced by the `Addendum C to Standard 55-2020`_
    units : str, optional
        select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
        0 < vr [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.
        The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
        0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.
    airspeed_control : boolean default True
        This only applies if standard = "ASHRAE". By default it is assumed that the
        occupant has control over the airspeed. In this case the ASHRAE 55 Standard does
        not impose any airspeed limits. On the other hand, if the occupant has no control
        over the airspeed the ASHRAE 55 imposes an upper limit for v which varies as a
        function of the operative temperature, for more information please consult the
        Standard.

    Returns
    -------
    pmv : float, int, or array-like
        Predicted Mean Vote

    Notes
    -----
    You can use this function to calculate the `PMV`_ [1]_ [2]_.

    .. _PMV: https://en.wikipedia.org/wiki/Thermal_comfort#PMV/PPD_method
    .. _Addendum C to Standard 55-2020: https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/55_2020_c_20210430.pdf

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import pmv
        >>> from pythermalcomfort.utilities import v_relative, clo_dynamic
        >>> t_db = 25
        >>> t_r = 25
        >>> relative_humidity = 50
        >>> v = 0.1
        >>> met_rate = 1.4
        >>> clo_insulation = 0.5
        >>> # calculate relative air speed
        >>> v_r = v_relative(v=v, met=met_rate)
        >>> # calculate dynamic clothing
        >>> clo_d = clo_dynamic(clo=clo_insulation, met=met_rate)
        >>> results = pmv(tdb=t_db, tr=t_r, vr=v_r, rh=relative_humidity, met=met_rate, clo=clo_d)
        >>> print(results)
        0.06
        >>> # you can also pass an array-like of inputs
        >>> results = pmv(tdb=[22, 25], tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        >>> print(results)
        array([-0.47,  0.06])
    """

    return pmv_ppd(
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme,
        standard=standard,
        units=units,
        limit_inputs=limit_inputs,
        airspeed_control=airspeed_control,
    )["pmv"]
