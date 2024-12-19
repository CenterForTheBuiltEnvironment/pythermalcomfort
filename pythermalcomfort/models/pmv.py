from typing import Union

import numpy as np

from pythermalcomfort.classes_input import PMVInputs
from pythermalcomfort.classes_return import PMV
from pythermalcomfort.models.pmv_ppd import pmv_ppd


def pmv(
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
) -> PMV:
    """Returns Predicted Mean Vote (PMV) calculated in accordance with main
    thermal comfort Standards. The PMV is an index that predicts the mean value
    of the thermal sensation votes (self-reported perceptions) of a large group
    of people on a sensation scale expressed from –3 to +3 corresponding to the
    categories: cold, cool, slightly cool, neutral, slightly warm, warm, and hot.

    While the PMV equation is the same for both the ISO and ASHRAE standards, in the
    ASHRAE 55 PMV equation, the SET is used to calculate the cooling effect first,
    this is then subtracted from both the air and mean radiant temperatures, and the
    differences are used as input to the PMV model, while the airspeed is set to 0.1m/s.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    vr : float or list of floats
        Relative air speed, [m/s].

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
    PMV
        A dataclass containing the Predicted Mean Vote. See :py:class:`~pythermalcomfort.models.pmv.PMV` for more details.
        To access the `pmv` value, use the `pmv` attribute of the returned `PMV` instance, e.g., `result.pmv`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        t_db = 25
        t_r = 25
        relative_humidity = 50
        v = 0.1
        met_rate = 1.4
        clo_insulation = 0.5
        # calculate relative air speed
        v_r = v_relative(v=v, met=met_rate)
        # calculate dynamic clothing
        clo_d = clo_dynamic(clo=clo_insulation, met=met_rate)
        results = pmv(
            tdb=t_db, tr=t_r, vr=v_r, rh=relative_humidity, met=met_rate, clo=clo_d
        )
        print(result.pmv)  # 0.06
        # you can also pass an array-like of inputs
        results = pmv(tdb=[22, 25], tr=tr, vr=v_r, rh=rh, met=met, clo=clo_d)
        print(result.pmv)  # [-0.47, 0.06]

    """
    # Validate inputs using the PMVInputs class
    PMVInputs(
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

    pmv_value = pmv_ppd(
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
    ).pmv

    if round_output:
        pmv_value = np.round(pmv_value, 2)

    return PMV(pmv=pmv_value)
