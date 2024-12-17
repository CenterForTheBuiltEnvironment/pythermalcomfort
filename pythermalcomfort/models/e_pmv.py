from dataclasses import dataclass
from typing import Union, List

import numpy as np

from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import BaseInputs


@dataclass(frozen=True)
class EPMV:
    """
    Dataclass to represent the Adjusted Predicted Mean Votes with Expectancy Factor (ePMV).

    Attributes
    ----------
    e_pmv : float or list of floats
        Adjusted Predicted Mean Votes with Expectancy Factor.
    """

    e_pmv: Union[float, List[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class EPMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        e_coefficient,
        wme,
        units,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            e_coefficient=e_coefficient,
            wme=wme,
            units=units,
        )


def e_pmv(
    tdb: Union[float, List[float]],
    tr: Union[float, List[float]],
    vr: Union[float, List[float]],
    rh: Union[float, List[float]],
    met: Union[float, List[float]],
    clo: Union[float, List[float]],
    e_coefficient: Union[float, List[float]],
    wme: Union[float, List[float]] = 0,
    units: str = "SI",
    limit_inputs: bool = True,
) -> EPMV:
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
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'.
    vr : float or list of floats
        Relative air speed, default in [m/s] in [fps] if `units` = 'IP'.

        .. warning::
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

        .. warning::
            The activity as well as the air speed modify the insulation characteristics
            of the clothing and the adjacent air layer. Consequently, the ISO 7730 states that
            the clothing insulation shall be corrected [2]_. The ASHRAE 55 Standard corrects
            for the effect of the body movement for met equal or higher than 1.2 met using
            the equation clo = Icl × (0.6 + 0.4/met) The dynamic clothing insulation, clo,
            can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    e_coefficient : float or list of floats
        Expectancy factor.
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    units : {'SI', 'IP'}, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        By default, if the inputs are outside the standard applicability limits the
        function returns nan. If False, returns pmv and ppd values even if input values are
        outside the applicability limits of the model. Defaults to True.

        .. note::
            The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
            0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.

    Returns
    -------
    EPMV
        A dataclass containing the Adjusted Predicted Mean Votes with Expectancy Factor. See :py:class:`~pythermalcomfort.models.e_pmv.EPMV` for more details.
        To access the `e_pmv` value, use the `e_pmv` attribute of the returned `e_pmv` instance, e.g., `result.e_pmv`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import e_pmv
        from pythermalcomfort.utilities import v_relative, clo_dynamic

        tdb = 28
        tr = 28
        rh = 50
        v = 0.1
        met = 1.4
        clo = 0.5
        # calculate relative air speed
        v_r = v_relative(v=v, met=met)
        # calculate dynamic clothing
        clo_d = clo_dynamic(clo=clo, met=met)
        results = e_pmv(tdb, tr, v_r, rh, met, clo_d, e_coefficient=0.6)
        print(results.e_pmv)  # 0.51
    """

    # Validate inputs using the EPMVInputs class
    EPMVInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        e_coefficient=e_coefficient,
        wme=wme,
        units=units,
    )

    default_kwargs = {"units": units, "limit_inputs": limit_inputs}
    met = np.array(met)
    _pmv = pmv(tdb, tr, vr, rh, met, clo, wme, "ISO", **default_kwargs).pmv
    met = np.where(_pmv > 0, met * (1 + _pmv * (-0.067)), met)
    _pmv = pmv(tdb, tr, vr, rh, met, clo, wme, "ISO", **default_kwargs).pmv

    epmv_value = np.around(_pmv * e_coefficient, 2)

    return EPMV(e_pmv=epmv_value)


if __name__ == "__main__":
    result = e_pmv(tdb=25, tr=25, vr=0.3, rh=50, met=1.2, clo=0.5, e_coefficient=0.6)
    print(result.e_pmv)  # 0.51
