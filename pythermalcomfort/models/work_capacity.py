from typing import Union

import numpy as np

from pythermalcomfort.classes_input import WorkCapacityInputs
from pythermalcomfort.classes_return import WorkCapacity



def workcapacity_dunne(wbgt: Union[float, list[float]], intensity: str) -> WorkCapacity:
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
        "med" or "light".

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.

    """
    #Validate inputs
    WorkCapacityInputs(wbgt=wbgt) 

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


def workcapacity_hothaps(wbgt: Union[float, list[float]], intensity: str) -> WorkCapacity:
    """
    Estimate work capacity due to heat based on Kjellstrom et al.

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done. Note that in this version the functions do not reach
    0% as it is assumed that it is always possible to work in short bursts for
    10% of the time.

    Note that for this function "the empirical evidence is from studies in
    heavyly distinct locations, including a gold mine (Wyndham, 1969), 124 rice
    harvesters in West Bengal in India (Sahu et al., 2013), and six women
    observed in a climatic chamber (Nag and Nag, 1992)."
    (https://adaptecca.es/sites/default/files/documentos/2018_jrc_pesetaiii_impact_labour_productivity.pdf).
    The shape of the function is just an assumption, and the fit of the
    sigmoid to the data it is analysing is not especially good.

    Heavy intensity work is sometimes labelled as 400 W, medium 300 W, light 200
    W, but this is only nominal.

    The correction citation is: Bröde P, Fiala D, Lemke B, Kjellstrom T.
    Estimated work ability in warm outdoor environments depends on the chosen
    heat stress assessment metric. International Journal of Biometeorology.
    2018 Mar;62(3):331–45. 


    The relevant definitions of the functions can be found most clearly in:
    Orlov A, Sillmann J, Aunan K, Kjellstrom T, Aaheim A. Economic costs of
    heat-induced reductions in worker productivity due to global warming. Global
    Environmental Change [Internet]. 2020 Jul;63. Available from:
    https://doi.org/10.1016/j.gloenvcha.2020.102087

    For a comparison of different functions see Fig 1 of Day E, Fankhauser S, Kingsmill
    N, Costa H, Mavrogianni A. Upholding labour productivity under climate
    change: an assessment of adaptation options. Climate Policy. 2019
    Mar;19(3):367–85. 

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
    #Validate inputs
    WorkCapacityInputs(wbgt=wbgt) 

    wbgt = np.array(wbgt)
    if intensity == "heavy":
        capacity = 100 * (0.1 + (0.9 / (1 + (wbgt / 30.94) ** 16.64)))
    elif intensity == "med":
        capacity = 100 * (0.1 + (0.9 / (1 + (wbgt / 32.93) ** 17.81)))
    elif intensity == "light":
        capacity = 100 * (0.1 + (0.9 / (1 + (wbgt / 34.64) ** 22.72)))
    else:
        raise ValueError("intensity should = heavy, med, or light")

    capacity = np.clip(capacity, 0, 100)

    return WorkCapacity(capacity=capacity)

def workcapacity_niosh(wbgt: Union[float, list[float]], M: Union[float,list[float]]) -> WorkCapacity:
    """
    Estimate work capacity due to heat based on NIOSH standards as described by Brode et al

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    The function definitions / parameters can be found in: 1. Bröde P, Fiala D,
    Lemke B, Kjellstrom T. Estimated work ability in warm outdoor environments
    depends on the chosen heat stress assessment metric. International Journal
    of Biometeorology. 2018 Mar;62(3):331–45. 

    For a comparison of different functions see Fig 1 of Day E, Fankhauser S, Kingsmill
    N, Costa H, Mavrogianni A. Upholding labour productivity under climate
    change: an assessment of adaptation options. Climate Policy. 2019
    Mar;19(3):367–85. 

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    M : float or list of floats
        Metabolic heat production in Watts

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.


    """
    #Validate inputs
    WorkCapacityInputs(wbgt=wbgt) 

    wbgt = np.array(wbgt)
    M = np.array(M)
    if (M<0).any() or (M>2500).any():
        raise ValueError("Metabolic rate out of plausible range")
    Mrest = 117 # assumed resting metabolic rate

    wbgt_lim = 56.7 - 11.5*np.log10(M)
    wbgt_lim_rest = 56.7 - 11.5*np.log10(Mrest)
    capacity = ((wbgt_lim_rest-wbgt)/(wbgt_lim_rest-wbgt_lim))*100
    capacity = np.clip(capacity, 0, 100)

    return WorkCapacity(capacity=capacity)

def workcapacity_iso(wbgt: Union[float, list[float]], M: Union[float,list[float]]) -> WorkCapacity:
    """
    Estimate work capacity due to heat based on ISO standards as described by Brode et al

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    The function definitions / parameters can be found in: 1. Bröde P, Fiala D,
    Lemke B, Kjellstrom T. Estimated work ability in warm outdoor environments
    depends on the chosen heat stress assessment metric. International Journal
    of Biometeorology. 2018 Mar;62(3):331–45. 

    For a comparison of different functions see Fig 1 of Day E, Fankhauser S, Kingsmill
    N, Costa H, Mavrogianni A. Upholding labour productivity under climate
    change: an assessment of adaptation options. Climate Policy. 2019
    Mar;19(3):367–85. 

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    M : float or list of floats
        Metabolic heat production in Watts

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.


    """
    #Validate inputs
    WorkCapacityInputs(wbgt=wbgt) 

    wbgt = np.array(wbgt)
    M = np.array(M)
    if (M<0).any() or (M>2500).any():
        raise ValueError("Metabolic rate out of plausible range")
    Mrest = 117 # assumed resting metabolic rate

    wbgt_lim = 34.9-M/46
    wbgt_lim_rest = 34.9-Mrest/46
    capacity = ((wbgt_lim_rest-wbgt)/(wbgt_lim_rest-wbgt_lim))*100
    capacity = np.clip(capacity, 0, 100)

    return WorkCapacity(capacity=capacity)
