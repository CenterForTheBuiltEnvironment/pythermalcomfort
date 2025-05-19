from typing import Union

import numpy as np

from pythermalcomfort.classes_input import WorkCapacityHothapsInputs
from pythermalcomfort.classes_input import WorkIntensity
from pythermalcomfort.classes_return import WorkCapacity


def work_capacity_dunne(
    wbgt: Union[float, list[float]],
    intensity: str = WorkIntensity.HEAVY.value,
) -> WorkCapacity:
    """
    Estimate work capacity due to heat based ISO standards as described by Dunne et al

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    This is based upon NIOSH safety standards. See:
    Dunne JP, Stouffer RJ, John JG. Reductions in labour capacity from heat
    stress under climate warming. Nature Climate Change. 2013 Jun;3(6):563–6.

    Heavy intensity work is sometimes labelled as 400 W, but this is only
    nominal. Medium work is assumed to be half as much as heavy intensity, and
    light half as much as medium.

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    intensity : str
        Which work intensity to use for the calculation, choice of "heavy",
        "medium" or "light".

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.

    """

    # validate inputs
    WorkCapacityHothapsInputs(wbgt=wbgt, intensity=intensity)

    # convert str to enum
    intensity = WorkIntensity(intensity.lower())
    wbgt = np.array(wbgt)

    capacity = np.clip((100 - (25 * (np.maximum(0, wbgt - 25)) ** (2 / 3))), 0, 100)

    factor_map = {
        WorkIntensity.HEAVY: 1,
        WorkIntensity.MEDIUM: 2,
        WorkIntensity.LIGHT: 4,
    }

    capacity = np.clip(capacity * factor_map[intensity], 0, 100)

    return WorkCapacity(capacity=capacity)
