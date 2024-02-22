import math

from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import (
    units_converter,
    check_standard_compliance,
)


def ankle_draft(tdb, tr, vr, rh, met, clo, v_ankle, units: str = "SI"):
    """
    Calculates the percentage of thermally dissatisfied people with the ankle draft (
    0.1 m) above floor level [23]_.
    This equation is only applicable for vr < 0.2 m/s (40 fps).

    Parameters
    ----------
    tdb : float
        dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'

        Note: The air temperature is the average value over two heights: 0.6 m (24 in.)
        and 1.1 m (43 in.) for seated occupants
        and 1.1 m (43 in.) and 1.7 m (67 in.) for standing occupants.
    tr : float
        mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'
    vr : float
        relative air speed, default in [m/s] in [fps] if `units` = 'IP'

        Note: vr is the relative air speed caused by body movement and not the air
        speed measured by the air speed sensor. The relative air speed is the sum of the
        average air speed measured by the sensor plus the activity-generated air speed
        (Vag). Where Vag is the activity-generated air speed caused by motion of
        individual body parts. vr can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.v_relative`.
    rh : float
        relative humidity, [%]
    met : float
        metabolic rate, [met]
    clo : float
        clothing insulation, [clo]

        Note: The activity as well as the air speed modify the insulation characteristics
        of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
        the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
        for the effect of the body movement for met equal or higher than 1.2 met using
        the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
        can be calculated using the function
        :py:meth:`pythermalcomfort.utilities.clo_dynamic`.
    v_ankle : float
        air speed at the 0.1 m (4 in.) above the floor, default in [m/s] in [fps] if
        `units` = 'IP'
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    PPD_ad: float
        Predicted Percentage of Dissatisfied occupants with ankle draft, [%]
    Acceptability: bol
        The ASHRAE 55 2020 standard defines that the value of air speed at the ankle
        level is acceptable if PPD_ad is lower or equal than 20 %

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import ankle_draft
        >>> results = ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")
        >>> print(results)
        {'PPD_ad': 18.5, 'Acceptability': True}

    """
    if units.lower() == "ip":
        tdb, tr, vr, v_ankle = units_converter(tdb=tdb, tr=tr, v=vr, vel=v_ankle)

    check_standard_compliance(
        standard="ashrae", tdb=tdb, tr=tr, v_limited=vr, rh=rh, met=met, clo=clo
    )

    tsv = pmv(tdb, tr, vr, rh, met, clo, standard="ashrae")
    ppd_val = round(
        math.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv)
        / (1 + math.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv))
        * 100,
        1,
    )
    acceptability = ppd_val <= 20
    return {"PPD_ad": ppd_val, "Acceptability": acceptability}
