from typing import Union

import numpy as np

from pythermalcomfort.classes_input import AnkleDraftInputs
from pythermalcomfort.classes_return import AnkleDraft
from pythermalcomfort.models.pmv_ppd_ashrae import pmv_ppd_ashrae
from pythermalcomfort.utilities import (
    Models,
    Units,
    _check_standard_compliance_array,
    units_converter,
)


def ankle_draft(
    tdb: Union[float, list[float]],
    tr: Union[float, list[float]],
    vr: Union[float, list[float]],
    rh: Union[float, list[float]],
    met: Union[float, list[float]],
    clo: Union[float, list[float]],
    v_ankle: Union[float, list[float]],
    units: str = Units.SI.value,
) -> AnkleDraft:
    """Calculates the percentage of thermally dissatisfied people with the
    ankle draft (0.1 m) above floor level [Liu2017]_.

    This equation is only applicable for vr < 0.2 m/s (40 fps).

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.

        .. note::
            The air temperature is the average value over two heights: 0.6 m (24 in.)
            and 1.1 m (43 in.) for seated occupants, and 1.1 m (43 in.) and 1.7 m (67 in.) for standing occupants.

    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.

    vr : float or list of floats
        Relative air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. note::
            `vr` is the relative air speed caused by body movement and not the air speed measured by the air speed sensor.
            The relative air speed is the sum of the average air speed measured by the sensor plus the activity-generated air speed (Vag).
            Vag is the activity-generated air speed caused by motion of individual body parts.
            `vr` can be calculated using the function :py:meth:`pythermalcomfort.utilities.v_relative`.

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

    v_ankle : float or list of floats
        Air speed at 0.1 m (4 in.) above the floor, default in [m/s] or [fps] if `units` = 'IP'.

    units : {'SI', 'IP'}
        Select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    AnkleDraft
        Dataclass containing the results of the ankle draft calculation. See :py:class:`~pythermalcomfort.classes_return.AnkleDraft` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import ankle_draft

        results = ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")
        print(results)
        # AnkleDraft(ppd_ad=18.5, acceptability=True)
    """
    # Validate inputs using the AnkleDraftInputs class
    AnkleDraftInputs(
        tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, v_ankle=v_ankle, units=units
    )

    # Convert lists to numpy arrays
    tdb = np.array(tdb)
    tr = np.array(tr)
    vr = np.array(vr)
    rh = np.array(rh)
    met = np.array(met)
    clo = np.array(clo)
    v_ankle = np.array(v_ankle)

    if units.upper() == Units.IP.value:
        tdb, tr, vr, v_ankle = units_converter(tdb=tdb, tr=tr, vr=vr, vel=v_ankle)

    tdb_valid, tr_valid, v_valid, v_limited = _check_standard_compliance_array(
        standard="ashrae",
        tdb=tdb,
        tr=tr,
        v_limited=vr,
        v=vr,
    )

    if np.all(np.isnan(v_limited)):
        raise ValueError(
            "This equation is only applicable for air speed lower than 0.2 m/s"
        )

    tsv = pmv_ppd_ashrae(
        tdb, tr, vr, rh, met, clo, model=Models.ashrae_55_2023.value
    ).pmv
    ppd_val = np.around(
        np.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv)
        / (1 + np.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv))
        * 100,
        1,
    )
    acceptability = ppd_val <= 20
    return AnkleDraft(ppd_ad=ppd_val, acceptability=acceptability)
