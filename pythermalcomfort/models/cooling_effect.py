import warnings
from typing import Literal, Union

import numpy as np
from scipy import optimize

from pythermalcomfort.classes_input import CEInputs
from pythermalcomfort.classes_return import CE
from pythermalcomfort.models.set_tmp import set_tmp
from pythermalcomfort.utilities import Units, units_converter


def cooling_effect(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    vr: Union[float, list[float]],
    rh: Union[float, list[float]],
    met: Union[float, list[float]],
    clo: Union[float, list[float]],
    wme: Union[float, list[float]] = 0,
    units: Literal["SI", "IP"] = Units.SI.value,
) -> CE:
    """Returns the value of the Cooling Effect (`CE`_) calculated in compliance
    with the ASHRAE 55 2020 Standard [55ASHRAE2023]_. The `CE`_ of the elevated air speed
    is the value that, when subtracted equally from both the average air
    temperature and the mean radiant temperature, yields the same `SET`_ under
    still air as in the first `SET`_ calculation under elevated air speed. The
    cooling effect is calculated only for air speed higher than 0.1 m/s.

    .. _CE: https://en.wikipedia.org/wiki/Thermal_comfort#Cooling_Effect
    .. _SET: https://en.wikipedia.org/wiki/Thermal_comfort#Standard_effective_temperature

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    vr : float or list of floats
        Relative air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. note::
            The cooling effect is calculated only for air speed higher than 0.1 m/s. If the air speed is lower than 0.1 m/s the function will return 0.

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
            this is the basic insulation also known as the intrinsic clothing insulation value of the
            clothing ensemble (`I`:sub:`cl,r`), this is the thermal insulation from the skin
            surface to the outer clothing surface, including enclosed air layers, under actual
            environmental conditions. This value is not the total insulation (`I`:sub:`T,r`).
            The dynamic clothing insulation, clo, can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.clo_dynamic_ashrae`.

    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    units : {'SI', 'IP'}, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.

    Returns
    -------
    CE
        A dataclass containing the Cooling Effect value. See :py:class:`~pythermalcomfort.classes_return.CoolingEffect` for more details.
        To access the `ce` value, use the `ce` attribute of the returned `CoolingEffect` instance, e.g., `result.ce`.

    Notes
    -----
    .. note::
        Limitations:
        - This equation may not be accurate for extreme temperature ranges.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import cooling_effect

        result = cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5)
        print(result.ce)  # 1.68

        result = cooling_effect(
            tdb=[25, 77],
            tr=[25, 77],
            vr=[0.3, 1.64],
            rh=[50, 50],
            met=[1.2, 1],
            clo=[0.5, 0.6],
            units="IP",
        )
        print(result.ce)  # [0, 3.95]
    """
    # Validate inputs using the CoolingEffectInputs class
    CEInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        units=units,
    )

    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    wme = np.array(wme)

    if units.upper() == Units.IP.value:
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    still_air_threshold = 0.1

    _ce = _cooling_effect_vectorised(
        tdb=tdb,
        tr=tr,
        still_air_threshold=still_air_threshold,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        vr=vr,
    )

    if units.upper() == Units.IP.value:
        _ce = _ce / 1.8 * 3.28

    return CE(ce=np.around(_ce, 2))


@np.vectorize
def _cooling_effect_vectorised(tdb, tr, still_air_threshold, rh, met, clo, wme, vr):
    if vr <= 0.1:
        return 0.0

    initial_set_tmp = set_tmp(
        tdb=tdb,
        tr=tr,
        v=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        round_output=False,
        calculate_ce=True,
        limit_inputs=False,
    ).set

    def function(x):
        return (
            set_tmp(
                tdb=tdb - x,
                tr=tr - x,
                v=still_air_threshold,
                rh=rh,
                met=met,
                clo=clo,
                wme=wme,
                round_output=False,
                calculate_ce=True,
                limit_inputs=False,
            ).set
            - initial_set_tmp
        )

    try:
        ce = optimize.brentq(function, 0.0, 40)
    except ValueError:
        ce = 0.0

    if ce == 0.0:
        warnings.warn(
            "Cooling effect could not be calculated. Returning 0.",
            UserWarning,
            stacklevel=2,
        )

    return ce
