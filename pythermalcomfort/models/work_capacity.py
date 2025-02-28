from typing import Union

import numpy as np

from pythermalcomfort.classes_return import WorkCapacity


def workcapacity_dunne(wbgt: Union[float, list[float]], intensity: str):
    """
    Estimate work capacity due to heat based on Dunne et al

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    This is based upon NIOSH safety standards. See:
    Dunne JP, Stouffer RJ, John JG. Reductions in labour capacity from heat
    stress under climate warming. Nature Climate Change. 2013 Jun;3(6):563–6.

    Heavy intensity work is sometimes labelled as 400 W, medium 300 W, light 200
    W, but this is only nominal.

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    intensity : str
        Which work intensity to use for the calculation, choice of "heavy",
        "med" or "light".

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.

    """
    wbgt = np.array(wbgt)
    capacity = np.clip((100 - (25 * (np.maximum(0, wbgt - 25)) ** (2 / 3))), 0, 100)
    if intensity == "heavy":
        pass
    elif intensity == "med":
        capacity = np.clip(capacity * 2, 0, 100)
    elif intensity == "light":
        capacity = np.clip(capacity * 4, 0, 100)
    else:
        raise ValueError("intensity should = heavy, med, or light")

    return WorkCapacity(capacity=capacity)


def workcapacity_hothaps(wbgt: Union[float, list[float]], intensity: str):
    """
    Estimate work capacity due to heat based on Kjellstrom et al.

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    Note that for this function "the empirical evidence is from studies in
    heavyly distinct locations, including a gold mine (Wyndham, 1969), 124 rice
    harvesters in West Bengal in India (Sahu et al., 2013), and six women
    observed in a climatic chamber (Nag and Nag, 1992)."
    (https://adaptecca.es/sites/default/files/documentos/2018_jrc_pesetaiii_impact_labour_productivity.pdf).
    The shape of the function is just an assumption, and the fit of the
    sigmoid to the data it is analysing is not especially good.

    Heavy intensity work is sometimes labelled as 400 W, medium 300 W, light 200
    W, but this is only nominal.

    Sometimes the maximum capacity of work is assumed to be 95% rather than 100%,
    but we do not apply that here.

    The relevant definitions of the functions can be found most clearly in:
    Orlov A, Sillmann J, Aunan K, Kjellstrom T, Aaheim A. Economic costs of
    heat-induced reductions in worker productivity due to global warming. Global
    Environmental Change [Internet]. 2020 Jul;63. Available from:
    https://doi.org/10.1016/j.gloenvcha.2020.102087

    But often people cite:
    See Kjellstrom, T., Freyberg, C., Lemke, B. et al. Estimating population
    heat exposure and impacts on working people in conjunction with climate
    change. Int J Biometeorol 62, 291–306 (2018).
    https://doi.org/10.1007/s00484-017-1407-0

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    intensity : str
        Which work intensity to use for the calculation, choice of "heavy",
        "med" or "light".

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.


    """
    wbgt = np.array(wbgt)
    if intensity == "heavy":
        capacity = 100 - 100 * (0.1 + (0.9 / (1 + (wbgt / 30.94) ** 16.64)))
    elif intensity == "med":
        capacity = 100 - 100 * (0.1 + (0.9 / (1 + (wbgt / 32.93) ** 17.81)))
    elif intensity == "light":
        capacity = 100 - 100 * (0.1 + (0.9 / (1 + (wbgt / 34.64) ** 22.72)))
    else:
        raise ValueError("intensity should = heavy, med, or light")

    capacity = np.clip(capacity, 0, 100)

    return WorkCapacity(capacity=capacity)
