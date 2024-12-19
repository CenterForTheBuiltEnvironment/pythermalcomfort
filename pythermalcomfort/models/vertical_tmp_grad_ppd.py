from typing import Union, List

import numpy as np

from pythermalcomfort.classes_input import VerticalTGradPPDInputs
from pythermalcomfort.classes_return import VerticalTGradPPD
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import check_standard_compliance_array


def vertical_tmp_grad_ppd(
    tdb: Union[float, List[float]],
    tr: Union[float, List[float]],
    vr: Union[float, List[float]],
    rh: Union[float, List[float]],
    met: Union[float, List[float]],
    clo: Union[float, List[float]],
    vertical_tmp_grad: Union[float, List[float]],
    round_output: bool = True,
) -> VerticalTGradPPD:
    """Calculates the percentage of thermally dissatisfied people with a
    vertical temperature gradient between feet and head [1]_. This equation is
    only applicable for vr < 0.2 m/s (40 fps).

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].

        .. note::
            The air temperature is the average value over two heights: 0.6 m (24 in.)
            and 1.1 m (43 in.) for seated occupants
            and 1.1 m (43 in.) and 1.7 m (67 in.) for standing occupants.

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

    vertical_tmp_grad : float or list of floats
        Vertical temperature gradient between the feet and the head, [°C/m].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    VerticalTGradPPD
        A dataclass containing the Predicted Percentage of Dissatisfied occupants with vertical temperature gradient and acceptability.
        See :py:class:`~pythermalcomfort.models.vertical_tmp_grad_ppd.VerticalTmpGradPPD` for more details.
        To access the `ppd_vg` and `acceptability` values, use the corresponding attributes of the returned `VerticalTmpGradPPD` instance, e.g., `result.ppd_vg`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import vertical_tmp_grad_ppd

        result = vertical_tmp_grad_ppd(tdb=25, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, vertical_tmp_grad=7)
        print(result.ppd_vg)  # 12.6
        print(result.acceptability)  # False
    """

    # Validate inputs using the VerticalTmpGradPPDInputs class
    VerticalTGradPPDInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        vertical_tmp_grad=vertical_tmp_grad,
    )

    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    met = np.array(met)
    clo = np.array(clo)
    vertical_tmp_grad = np.array(vertical_tmp_grad)

    (
        tdb_valid,
        tr_valid,
        v_valid,
        met_valid,
        clo_valid,
    ) = check_standard_compliance_array(
        standard="ashrae", tdb=tdb, tr=tr, v_limited=vr, met=met, clo=clo
    )

    tsv = pmv(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ashrae").pmv
    numerator = np.exp(0.13 * (tsv - 1.91) ** 2 + 0.15 * vertical_tmp_grad - 1.6)
    ppd_val = (numerator / (1 + numerator) - 0.345) * 100
    acceptability = ppd_val <= 5

    if round_output:
        ppd_val = np.round(ppd_val, 1)

    all_valid = ~(
        np.isnan(tdb_valid)
        | np.isnan(tr_valid)
        | np.isnan(v_valid)
        | np.isnan(met_valid)
        | np.isnan(clo_valid)
    )

    ppd_val = np.where(all_valid, ppd_val, np.nan)
    acceptability = np.where(all_valid, acceptability, np.nan)

    return VerticalTGradPPD(ppd_vg=ppd_val, acceptability=acceptability)


if __name__ == "__main__":
    result = vertical_tmp_grad_ppd(
        tdb=25, tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, vertical_tmp_grad=7
    )
    print(result.ppd_vg)

    result = vertical_tmp_grad_ppd(
        tdb=[25, 23], tr=25, vr=0.1, rh=50, met=1.2, clo=0.5, vertical_tmp_grad=7
    )
    print(result.ppd_vg)
