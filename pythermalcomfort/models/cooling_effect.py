import warnings
from typing import Union, Literal

from scipy import optimize

from pythermalcomfort.models.set_tmp import set_tmp
from pythermalcomfort.utilities import units_converter


def cooling_effect(
    tdb: Union[float, int],
    tr: Union[float, int],
    vr: Union[float, int],
    rh: Union[float, int],
    met: Union[float, int],
    clo: Union[float, int],
    wme: Union[float, int] = 0,
    units: Literal["SI", "IP"] = "SI",
) -> float:
    """Returns the value of the Cooling Effect (`CE`_) calculated in compliance
    with the ASHRAE 55 2020 Standard [1]_. The `CE`_ of the elevated air speed
    is the value that, when subtracted equally from both the average air
    temperature and the mean radiant temperature, yields the same `SET`_ under
    still air as in the first `SET`_ calculation under elevated air speed. The
    cooling effect is calculated only for air speed higher than 0.1 m/s.

    .. _CE: https://en.wikipedia.org/wiki/Thermal_comfort#Cooling_Effect

    Parameters
    ----------
    tdb : float
        Dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'.
    tr : float
        Mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'.
    vr : float
        Relative air speed, default in [m/s] in [fps] if `units` = 'IP'.

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. The relative air speed is the sum of the
            average air speed measured by the sensor plus the activity-generated air speed
            (Vag). Where Vag is the activity-generated air speed caused by motion of
            individual body parts. vr can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.v_relative`.

    rh : float
        Relative humidity, [%].
    met : float
        Metabolic rate, [met].
    clo : float
        Clothing insulation, [clo].

        .. note::
            The activity as well as the air speed modify the insulation characteristics
            of the clothing and the adjacent air layer. Consequently the ISO 7730 states that
            the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
            for the effect of the body movement for met equal or higher than 1.2 met using
            the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
            can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    wme : float, optional
        External work, [met]. Defaults to 0.
    units : {'SI', 'IP'}, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.

    Returns
    -------
    float
        The Cooling Effect value.

    Raises
    ------
    ValueError
        If the cooling effect could not be calculated.

    Notes
    -----
    .. warning::
        Limitations:
        - This equation may not be accurate for extreme temperature ranges.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import cooling_effect

        result = cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5)
        print(result)  # 1.68

        result = cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")
        print(result)  # 3.95
    """

    if units.lower() == "ip":
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    if vr <= 0.1:
        return 0.0

    still_air_threshold = 0.1

    warnings.simplefilter("ignore")

    initial_set_tmp = set_tmp(
        tdb=tdb,
        tr=tr,
        v=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        round=False,
        calculate_ce=True,
        limit_inputs=False,
    )

    def function(x):
        return (
            set_tmp(
                tdb - x,
                tr - x,
                v=still_air_threshold,
                rh=rh,
                met=met,
                clo=clo,
                wme=wme,
                round=False,
                calculate_ce=True,
                limit_inputs=False,
            )
            - initial_set_tmp
        )

    try:
        ce = optimize.brentq(function, 0.0, 40)
    except ValueError:
        ce = 0

    warnings.simplefilter("always")

    if ce == 0:
        warnings.warn(
            "Assuming cooling effect = 0 since it could not be calculated for this set"
            f" of inputs {tdb=}, {tr=}, {rh=}, {vr=}, {clo=}, {met=}",
            UserWarning,
        )

    if units.lower() == "ip":
        ce = ce / 1.8 * 3.28

    return round(ce, 2)


if __name__ == "__main__":
    ce = cooling_effect(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5)
    print(ce)

    ce = cooling_effect(tdb=77, tr=77, vr=1.64, rh=50, met=1, clo=0.6, units="IP")
    print(ce)
