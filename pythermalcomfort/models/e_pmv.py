import numpy as np

from pythermalcomfort.models import pmv


def e_pmv(tdb, tr, vr, rh, met, clo, e_coefficient, wme=0, **kwargs):
    """Returns Adjusted Predicted Mean Votes with Expectancy Factor (ePMV).
    This index was developed by Fanger, P. O. et al. (2002). In non-air-
    conditioned buildings in warm climates, occupants may sense the warmth as
    being less severe than the PMV predicts. The main reason is low
    expectations, but a metabolic rate that is estimated too high can also
    contribute to explaining the difference. An extension of the PMV model that
    includes an expectancy factor is introduced for use in non-air-conditioned
    buildings in warm climates [26]_.

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
    e_coefficient : float
        expectacy factor
    wme : float, int, or array-like
        external work, [met] default 0

    Other Parameters
    ----------------
    units : {'SI', 'IP'}
        select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : boolean default True
        By default, if the inputs are outsude the standard applicability limits the
        function returns nan. If False returns pmv and ppd values even if input values are
        outside the applicability limits of the model.

        The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
        0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.

    Returns
    -------
    pmv : float, int, or array-like
        Predicted Mean Vote

    Examples
    --------
    .. code-block:: python

        >>> from pythermalcomfort.models import a_pmv
        >>> from pythermalcomfort.utilities import v_relative, clo_dynamic
        >>> tdb = 28
        >>> tr = 28
        >>> rh = 50
        >>> v = 0.1
        >>> met = 1.4
        >>> clo = 0.5
        >>> # calculate relative air speed
        >>> v_r = v_relative(v=v, met=met)
        >>> # calculate dynamic clothing
        >>> clo_d = clo_dynamic(clo=clo, met=met)
        >>> results = e_pmv(tdb, tr, v_r, rh, met, clo_d, e_coefficient=0.6)
        >>> print(results)
        0.51
    """
    default_kwargs = {"units": "SI", "limit_inputs": True}
    kwargs = {**default_kwargs, **kwargs}

    met = np.array(met)
    _pmv = pmv(tdb, tr, vr, rh, met, clo, wme, "ISO", **kwargs)
    met = np.where(_pmv > 0, met * (1 + _pmv * (-0.067)), met)
    _pmv = pmv(tdb, tr, vr, rh, met, clo, wme, "ISO", **kwargs)

    return np.around(_pmv * e_coefficient, 2)
