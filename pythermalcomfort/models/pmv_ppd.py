from typing import Union

import numpy as np
from numba import float64, vectorize

from pythermalcomfort.classes_input import PMVPPDInputs
from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models.cooling_effect import cooling_effect
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import (
    _check_standard_compliance_array,
    units_converter,
)


def pmv_ppd(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    vr: Union[float, list[float]],
    rh: Union[float, list[float]],
    met: Union[float, list[float]],
    clo: Union[float, list[float]],
    wme: Union[float, list[float]] = 0,
    standard: str = "ISO",
    units: str = "SI",
    limit_inputs: bool = True,
    airspeed_control: bool = True,
    round_output: bool = True,
) -> PMVPPD:
    """Returns Predicted Mean Vote (PMV) and Predicted Percentage of
    Dissatisfied (PPD) calculated in accordance with main thermal comfort
    Standards. The PMV is an index that predicts the mean value of the thermal
    sensation votes (self-reported perceptions) of a large group of people on a
    sensation scale expressed from –3 to +3 corresponding to the categories:
    cold, cool, slightly cool, neutral, slightly warm, warm, and hot. [1]_

    While the PMV equation is the same for both the ISO and ASHRAE standards, in the
    ASHRAE 55 PMV equation, the SET is used to calculate the cooling effect first,
    this is then subtracted from both the air and mean radiant temperatures, and the
    differences are used as input to the PMV model, while the airspeed is set to 0.1m/s.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C] in [°F] if `units` = 'IP'
    tr : float or list of floats
        Mean radiant temperature, [°C] in [°F] if `units` = 'IP'
    vr : float or list of floats
        Relative air speed, [m/s] in [fps] if `units` = 'IP'

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. The relative air speed is the sum of the
            average air speed measured by the sensor plus the activity-generated air speed
            (Vag). Where Vag is the activity-generated air speed caused by motion of
            individual body parts. vr can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.v_relative`.

    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].

        .. note::
            The activity as well as the air speed modify the insulation characteristics
            of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
            the clothing insulation shall be corrected. The ASHRAE 55 Standard corrects
            for the effect of the body movement for met equal or higher than 1.2 met using
            the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
            can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    standard : str, optional
        Select comfort standard used for calculation. Supported values are 'ASHRAE' and 'ISO'. Defaults to 'ISO'.

        .. note::
            While the PMV equation is the same for both the ISO and ASHRAE standards,
            the ASHRAE Standard Use of the PMV model is limited to air speeds below 0.10
            m/s (20 fpm). When air speeds exceed 0.10 m/s (20 fpm), the comfort zone boundaries are
            adjusted based on the SET model. This change was introduced by the `Addendum C to Standard 55-2020`.

    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system. Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns pmv and ppd values even if input values are
            outside the applicability limits of the model.

            The ASHRAE 55 2020 limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
            0 < vr [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.
            The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
            0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.

    airspeed_control : bool, optional
        This only applies if standard = "ASHRAE". By default, it is assumed that the
        occupant has control over the airspeed. In this case, the ASHRAE 55 Standard does
        not impose any airspeed limits. On the other hand, if the occupant has no control
        over the airspeed, the ASHRAE 55 imposes an upper limit for v which varies as a
        function of the operative temperature, for more information please consult the
        Standard.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    PMVPPD
        A dataclass containing the Predicted Mean Vote and Predicted Percentage of Dissatisfied. See :py:class:`~pythermalcomfort.models.pmv_ppd.pmv_ppd` for more details.
        To access the `pmv` and `ppd` values, use the corresponding attributes of the returned `pmv_ppd` instance, e.g., `result.pmv`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        tdb = 25
        tr = 25
        rh = 50
        v = 0.1
        met = 1.4
        clo = 0.5
        # calculate relative air speed
        v_r = v_relative(v=v, met=met)
        # calculate dynamic clothing
        clo_d = clo_dynamic(clo=clo, met=met)
        results = pmv_ppd(tdb=tdb, tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        print(results.pmv)  # 0.06
        print(results.ppd)  # 5.1

        result = pmv_ppd(tdb=[22, 25], tr=25, vr=0.1, rh=50, met=1.4, clo=0.5)
        print(result.pmv)  # [-0.  0.41]
        print(result.ppd)  # [5.  8.5]
    """
    # Validate inputs using the PMVPPDInputs class
    PMVPPDInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        standard=standard,
        units=units,
        limit_inputs=limit_inputs,
        airspeed_control=airspeed_control,
    )

    tdb = np.array(tdb)
    tr = np.array(tr)
    rh = np.array(rh)
    vr = np.array(vr)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    if units.lower() == "ip":
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    standard = standard.lower()
    if standard not in ["iso", "ashrae"]:
        raise ValueError(
            "PMV calculations can only be performed in compliance with ISO or ASHRAE "
            "Standards"
        )

    (
        tdb_valid,
        tr_valid,
        v_valid,
        met_valid,
        clo_valid,
    ) = _check_standard_compliance_array(
        standard,
        tdb=tdb,
        tr=tr,
        v=vr,
        met=met,
        clo=clo,
        airspeed_control=airspeed_control,
    )

    # if v_r is higher than 0.1 follow methodology ASHRAE Appendix H, H3
    ce = 0.0
    if standard == "ashrae":
        ce = np.where(
            vr > 0.1,
            cooling_effect(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, wme=wme).ce,
            0.0,
        )

    tdb = tdb - ce
    tr = tr - ce
    vr = np.where(ce > 0, 0.1, vr)

    pmv_array = _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme)

    ppd_array = 100.0 - 95.0 * np.exp(
        -0.03353 * pmv_array**4.0 - 0.2179 * pmv_array**2.0
    )

    # Checks that inputs are within the bounds accepted by the model if not return nan
    if limit_inputs:
        pmv_valid = valid_range(pmv_array, (-2, 2))  # this is the ISO limit
        if standard == "ashrae":
            pmv_valid = valid_range(pmv_array, (-100, 100))

        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
            | np.isnan(pmv_valid)
        )
        pmv_array = np.where(all_valid, pmv_array, np.nan)
        ppd_array = np.where(all_valid, ppd_array, np.nan)

    if round_output:
        pmv_array = np.round(pmv_array, 2)
        ppd_array = np.round(ppd_array, 1)

    return PMVPPD(pmv=pmv_array, ppd=ppd_array)


@vectorize(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
)
def _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme):
    pa = rh * 10 * np.exp(16.6536 - 4030.183 / (tdb + 235))

    icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
    m = met * 58.15  # metabolic rate in W/M2
    w = wme * 58.15  # external work in W/M2
    mw = m - w  # internal heat production in the human body
    # calculation of the clothing area factor
    if icl <= 0.078:
        f_cl = 1 + (1.29 * icl)  # ratio of surface clothed body over nude body
    else:
        f_cl = 1.05 + (0.645 * icl)

    # heat transfer coefficient by forced convection
    hcf = 12.1 * np.sqrt(vr)
    hc = hcf  # initialize variable
    taa = tdb + 273
    tra = tr + 273
    t_cla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)

    p1 = icl * f_cl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = (308.7 - 0.028 * mw) + (p2 * (tra / 100.0) ** 4)
    xn = t_cla / 100
    xf = t_cla / 50
    eps = 0.00015

    n = 0
    while np.abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * np.abs(100.0 * xf - taa) ** 0.25
        if hcf > hcn:
            hc = hcf
        else:
            hc = hcn
        xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise StopIteration("Max iterations exceeded")

    tcl = 100 * xn - 273

    # heat loss diff. through skin
    hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
    # heat loss by sweating
    if mw > 58.15:
        hl2 = 0.42 * (mw - 58.15)
    else:
        hl2 = 0
    # latent respiration heat loss
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    # dry respiration heat loss
    hl4 = 0.0014 * m * (34 - tdb)
    # heat loss by radiation
    hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
    # heat loss by convection
    hl6 = f_cl * hc * (tcl - tdb)

    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    _pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    return _pmv
