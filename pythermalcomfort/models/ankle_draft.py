from dataclasses import dataclass
from typing import Union, List, Literal

import numpy as np
import numpy.typing as npt

from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import (
    units_converter,
    check_standard_compliance_array,
    validate_units,
    validate_type,
)


@dataclass(frozen=True)
class AnkleDraft:
    """
    Dataclass to store the results of the ankle draft calculation.

    Attributes
    ----------
    PPD_ad : float
        Predicted Percentage of Dissatisfied occupants with ankle draft, [%].
    Acceptability : bool
        Indicates if the air speed at the ankle level is acceptable according to ASHRAE 55 2020 standard.
    """

    PPD_ad: Union[float, npt.ArrayLike]
    Acceptability: Union[bool, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class AnkleDraftInputs:
    tdb: Union[float, int, npt.ArrayLike]
    tr: Union[float, int, npt.ArrayLike]
    vr: Union[float, int, npt.ArrayLike]
    rh: Union[float, int, npt.ArrayLike]
    met: Union[float, int, npt.ArrayLike]
    clo: Union[float, int, npt.ArrayLike]
    v_ankle: Union[float, int, npt.ArrayLike]
    units: Literal["SI", "IP"] = "SI"

    def __post_init__(self):
        validate_units(self.units)
        validate_type(self.tdb, "tdb", (float, int, np.ndarray, list))
        validate_type(self.tr, "tr", (float, int, np.ndarray, list))
        validate_type(self.vr, "vr", (float, int, np.ndarray, list))
        validate_type(self.rh, "rh", (float, int, np.ndarray, list))
        validate_type(self.met, "met", (float, int, np.ndarray, list))
        validate_type(self.clo, "clo", (float, int, np.ndarray, list))
        validate_type(self.v_ankle, "v_ankle", (float, int, np.ndarray, list))


def ankle_draft(
    tdb: Union[float, List[float]],
    tr: Union[float, List[float]],
    vr: Union[float, List[float]],
    rh: Union[float, List[float]],
    met: Union[float, List[float]],
    clo: Union[float, List[float]],
    v_ankle: Union[float, List[float]],
    units: str = "SI",
) -> AnkleDraft:
    """
    Calculates the percentage of thermally dissatisfied people with the ankle draft (0.1 m) above floor level.

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

        .. warning::
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

        .. warning::
            The activity as well as the air speed modify the insulation characteristics of the clothing and the adjacent air layer.
            Consequently, the ISO 7730 states that the clothing insulation shall be corrected.
            The ASHRAE 55 Standard corrects for the effect of the body movement for met equal or higher than 1.2 met using the equation
            `clo = Icl × (0.6 + 0.4/met)`. The dynamic clothing insulation, `clo`, can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.clo_dynamic`.

    v_ankle : float or list of floats
        Air speed at 0.1 m (4 in.) above the floor, default in [m/s] or [fps] if `units` = 'IP'.

    units : {'SI', 'IP'}
        Select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    AnkleDraft
        Dataclass containing the results of the ankle draft calculation. See :py:class:`~pythermalcomfort.models.ankle_draft.AnkleDraft` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import ankle_draft
        results = ankle_draft(25, 25, 0.2, 50, 1.2, 0.5, 0.3, units="SI")
        print(results)
        # AnkleDraft(PPD_ad=18.5, Acceptability=True)
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

    if units.lower() == "ip":
        tdb, tr, vr, v_ankle = units_converter(tdb=tdb, tr=tr, vr=vr, vel=v_ankle)

    tdb_valid, tr_valid, v_valid, v_limited = check_standard_compliance_array(
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

    tsv = pmv(tdb, tr, vr, rh, met, clo, standard="ashrae")
    ppd_val = np.around(
        np.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv)
        / (1 + np.exp(-2.58 + 3.05 * v_ankle - 1.06 * tsv))
        * 100,
        1,
    )
    acceptability = ppd_val <= 20
    return AnkleDraft(PPD_ad=ppd_val, Acceptability=acceptability)


if __name__ == "__main__":
    results = ankle_draft(
        tdb=[50, 26, 27],
        tr=[25, 26, 27],
        vr=[0.1, 0.1, 0.1],
        rh=[50, 50, 50],
        met=[1.2, 1.2, 1.2],
        clo=[0.5, 0.5, 0.5],
        v_ankle=[0.1, 0.1, 0.1],
        units="SI",
    )
