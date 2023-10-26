# -*- coding: utf-8 -*-

"""This code provides models for calculating the surface area of different
body parts, weight ratio, basal blood flow ratio, and thermal conductance and
thermal capacity.

The values of a NumPy array containing 17 elements correspond to the
following order: "head", "neck", "chest", "back", "pelvis",
"left_shoulder", "left_arm", "left_hand", "right_shoulder", "right_arm",
"right_hand", "left_thigh", "left_leg", "left_hand", "right_thigh",
"right_leg" and "right_hand".
"""

import numpy as np
from pythermalcomfort.jos3_functions.matrix import NUM_NODES, IDICT, BODY_NAMES
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.utilities import body_surface_area


def validate_body_parameters(
    height=Default.height,
    weight=Default.weight,
    age=Default.age,
    body_fat=Default.body_fat,
):
    """
    Validate the parameters: height, weight, age, and body fat percentage.

    Parameters
    ----------
    height : float
        The height of the person in meters.
    weight : float
        The weight of the person in kilograms.
    age : int
        The age of the person in years.
    body_fat : float
        The body fat percentage as a fraction of total body mass.

    Raises
    ------
    ValueError
        If any of the parameters are out of the specified range.
    """

    if not (0.5 <= height <= 3.0):
        raise ValueError("Height must be in the range [0.5, 3.0] meters.")

    if not (20.0 <= weight <= 200.0):
        raise ValueError("Weight must be in the range [20.0, 200.0] kilograms.")

    if not (5 <= age <= 100):
        raise ValueError("Age must be in the range [5, 100] years.")

    if not (1 <= body_fat <= 90):
        raise ValueError("Body Fat must be in the range [1, 90] (1% to 90%).")


def _to17array(inp):
    """Create a NumPy array of shape (17,) with the given input.

    Parameters
    ----------
    inp : int, float, dict, list, ndarray
        The value(s) to use when creating the 17-element array.
        Supports int, float, dict (with BODY_NAMES as keys), list, and ndarray types.

    Returns
    -------
    ndarray
        A NumPy array of shape (17,) with the specified values.

    Raises
    ------
    ValueError
        If the input type is not supported or if the input list or ndarray is not of length 17.
    """
    if isinstance(inp, (int, float)):
        array = np.ones(17) * inp
    elif isinstance(inp, dict):
        return np.array([inp[key] for key in BODY_NAMES])
    elif isinstance(inp, (list, np.ndarray)):
        inp = np.asarray(inp)
        if inp.shape == (17,):
            array = inp
        else:
            raise ValueError("The input list or ndarray is not of length 17")
    else:
        raise ValueError(
            "Unsupported input type. Supported types: int, float, list, dict, ndarray"
        )

    return array.copy()


def bsa_rate(
    height=Default.height,
    weight=Default.weight,
    bsa_equation=Default.bsa_equation,
):
    """Calculate the rate of bsa to standard body.

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".

    Returns
    -------
    bsa_rate : float
        The ratio of bsa to the standard body [-].
    """
    validate_body_parameters(height=height, weight=weight)

    bsa_all = body_surface_area(
        height=height,
        weight=weight,
        formula=bsa_equation,
    )
    return (
        bsa_all / Default.local_bsa.sum()
    )  # The bsa ratio to the standard body (1.87m2)


def local_bsa(
    height=Default.height,
    weight=Default.weight,
    bsa_equation=Default.bsa_equation,
):
    """Calculate local body surface area (bsa) [m2].

    The local body surface area has been derived from 65MN.
    The head have been divided to head and neck based on Smith's model.
        head = 0.1396*0.1117/0.1414 (65MN_Head * Smith_Head / Smith_Head+neck)
        neck = 0.1396*0.0297/0.1414 (65MN_Head * Smith_Neck / Smith_Head+neck)

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".

    Returns
    -------
    local_bsa : ndarray(17,)
        Local body surface area (bsa) [m2].
    """
    validate_body_parameters(height=height, weight=weight)
    _bsa_rate = bsa_rate(
        height=height,
        weight=weight,
        bsa_equation=bsa_equation,
    )  # The bsa ratio to the standard body (1.87m2)
    local_bsa = Default.local_bsa * _bsa_rate
    return local_bsa


def weight_rate(
    weight=Default.weight,
):
    """Calculate the ratio of the body weight to the standard body (74.43 kg).

    The standard values of local body weights are as below.
        weight_local = np.array([
            3.18, 0.84, 12.4, 11.03, 17.57,
            2.16, 1.37, 0.34, 2.16, 1.37, 0.34,
            7.01, 3.34, 0.48, 7.01, 3.34, 0.48])
    The data have been derived from 65MN.
    The weight of neck is extracted from the weight of 65MN's head based on
    the local body surface area of Smith's model.

    Parameters
    ----------
    weight : float, optional
        The body weight [kg]. The default is 74.43.

    Returns
    -------
    weight_rate : float
        The ratio of the body weight to the standard body (74.43 kg).
        weight_rate = weight / 74.43
    """
    validate_body_parameters(weight=weight)
    return weight / Default.weight


def bfb_rate(
    height=Default.height,
    weight=Default.weight,
    bsa_equation=Default.bsa_equation,
    age=Default.age,
    ci=Default.cardiac_index,
):
    """Calculate the ratio of basal blood flow (BFB) of the standard body (290
    L/h).

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        age [years]. The default is 20.
    ci : float, optional
        Cardiac index [L/min/㎡]. The default is 2.59.

    Returns
    -------
    bfb_rate : float
        Basal blood flow rate.
    """
    validate_body_parameters(
        height=height,
        weight=weight,
        age=age,
    )

    ci *= 60  # Change unit [L/min/㎡] to [L/h/㎡]

    # Decrease of BFB by aging
    if age < 50:
        ci *= 1
    elif age < 60:
        ci *= 0.85
    elif age < 70:
        ci *= 0.75
    else:  # age >= 70
        ci *= 0.7

    bfb_all = (
        ci * bsa_rate(height, weight, bsa_equation) * Default.local_bsa.sum()
    )  # [L/h]
    return bfb_all / Default.blood_flow_rate


def conductance(
    height=Default.height,
    weight=Default.weight,
    bsa_equation=Default.bsa_equation,
    fat=Default.body_fat,
):
    """Calculate thermal conductance between layers [W/K].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    fat : float, optional
        Body fat rate [%]. The default is 15.

    Returns
    -------
    conductance : numpy.ndarray
        Thermal conductance between layers [W/K].
        The shape is (NUM_NODES, NUM_NODES).
    """

    validate_body_parameters(
        height=height,
        weight=weight,
        body_fat=fat,
    )

    if fat < 12.5:
        cdt_cr_sk = np.array(
            [
                1.341,
                0.930,
                1.879,
                1.729,
                2.370,
                1.557,
                1.018,
                2.210,
                1.557,
                1.018,
                2.210,
                2.565,
                1.378,
                3.404,
                2.565,
                1.378,
                3.404,
            ]
        )
    elif fat < 17.5:
        cdt_cr_sk = np.array(
            [
                1.311,
                0.909,
                1.785,
                1.643,
                2.251,
                1.501,
                0.982,
                2.183,
                1.501,
                0.982,
                2.183,
                2.468,
                1.326,
                3.370,
                2.468,
                1.326,
                3.370,
            ]
        )
    elif fat < 22.5:
        cdt_cr_sk = np.array(
            [
                1.282,
                0.889,
                1.698,
                1.563,
                2.142,
                1.448,
                0.947,
                2.156,
                1.448,
                0.947,
                2.156,
                2.375,
                1.276,
                3.337,
                2.375,
                1.276,
                3.337,
            ]
        )
    elif fat < 27.5:
        cdt_cr_sk = np.array(
            [
                1.255,
                0.870,
                1.618,
                1.488,
                2.040,
                1.396,
                0.913,
                2.130,
                1.396,
                0.913,
                2.130,
                2.285,
                1.227,
                3.304,
                2.285,
                1.227,
                3.304,
            ]
        )
    else:  # fat >= 27.5
        cdt_cr_sk = np.array(
            [
                1.227,
                0.852,
                1.542,
                1.419,
                1.945,
                1.346,
                0.880,
                1.945,
                1.346,
                0.880,
                1.945,
                2.198,
                1.181,
                3.271,
                2.198,
                1.181,
                3.271,
            ]
        )

    cdt_cr_ms = np.zeros(17)  # core to muscle [W/K]
    cdt_ms_fat = np.zeros(17)  # muscle to fat [W/K]
    cdt_fat_sk = np.zeros(17)  # fat to skin [W/K]

    # head and pelvis consists of 65MN's conductances
    cdt_cr_ms[0] = 1.601  # head
    cdt_ms_fat[0] = 13.222
    cdt_fat_sk[0] = 16.008
    cdt_cr_ms[4] = 3.0813  # pelvis
    cdt_ms_fat[4] = 10.3738
    cdt_fat_sk[4] = 41.4954

    # vessel to core
    # The shape is a cylinder.
    # It is assumed that the inner is vascular radius, 2.5mm and the outer is
    # stolwijk's core radius.
    # The heat transfer coefficient of the core is assumed as the Michel's
    # counter-flow model 0.66816 [W/(m･K)].
    cdt_ves_cr = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0.586,
            0.383,
            1.534,
            0.586,
            0.383,
            1.534,
            0.810,
            0.435,
            1.816,
            0.810,
            0.435,
            1.816,
        ]
    )
    # superficial vein to skin
    cdt_sfv_sk = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            57.735,
            37.768,
            16.634,
            57.735,
            37.768,
            16.634,
            102.012,
            54.784,
            24.277,
            102.012,
            54.784,
            24.277,
        ]
    )

    # art to vein (counter-flow) [W/K]
    # The data has been derived Mitchell's model.
    # THe values = 15.869 [W/(m･K)] * the segment length [m]
    cdt_art_vein = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0.537,
            0.351,
            0.762,
            0.537,
            0.351,
            0.762,
            0.826,
            0.444,
            0.992,
            0.826,
            0.444,
            0.992,
        ]
    )

    # Changes values by body size based on the standard body.
    wr = weight_rate(weight)
    bsar = bsa_rate(height, weight, bsa_equation)
    # head, neck (Sphere shape)
    cdt_cr_sk[:2] *= wr / bsar
    cdt_cr_ms[:2] *= wr / bsar
    cdt_ms_fat[:2] *= wr / bsar
    cdt_fat_sk[:2] *= wr / bsar
    cdt_ves_cr[:2] *= wr / bsar
    cdt_sfv_sk[:2] *= wr / bsar
    cdt_art_vein[:2] *= wr / bsar
    # Others (Cylinder shape)
    cdt_cr_sk[2:] *= bsar**2 / wr
    cdt_cr_ms[2:] *= bsar**2 / wr
    cdt_ms_fat[2:] *= bsar**2 / wr
    cdt_fat_sk[2:] *= bsar**2 / wr
    cdt_ves_cr[2:] *= bsar**2 / wr
    cdt_sfv_sk[2:] *= bsar**2 / wr
    cdt_art_vein[2:] *= bsar**2 / wr

    cdt_whole = np.zeros((NUM_NODES, NUM_NODES))
    for i, bn in enumerate(BODY_NAMES):
        # Dictionary of indices in each body segment
        # key = layer name, value = index of matrix
        index_of = IDICT[bn]

        # Common
        cdt_whole[index_of["artery"], index_of["vein"]] = cdt_art_vein[i]  # art to vein
        cdt_whole[index_of["artery"], index_of["core"]] = cdt_ves_cr[i]  # art to cr
        cdt_whole[index_of["vein"], index_of["core"]] = cdt_ves_cr[i]  # vein to cr

        # Only limbs
        if i >= 5:
            cdt_whole[index_of["sfvein"], index_of["skin"]] = cdt_sfv_sk[i]  # sfv to sk

        # If the segment has a muscle or fat layer
        if index_of["muscle"] is not None:  # or not indexof["fat"] is None
            cdt_whole[index_of["core"], index_of["muscle"]] = cdt_cr_ms[i]  # cr to ms
            cdt_whole[index_of["muscle"], index_of["fat"]] = cdt_ms_fat[i]  # ms to fat
            cdt_whole[index_of["fat"], index_of["skin"]] = cdt_fat_sk[i]  # fat to sk

        else:
            cdt_whole[index_of["core"], index_of["skin"]] = cdt_cr_sk[i]  # cr to sk

    # Creates a symmetrical matrix
    cdt_whole = cdt_whole + cdt_whole.T

    return cdt_whole.copy()


def capacity(
    height=Default.height,
    weight=Default.weight,
    bsa_equation=Default.bsa_equation,
    age=Default.age,
    ci=Default.cardiac_index,
):
    """Calculate the thermal capacity [J/K].

    The values of vascular and central blood capacity have been derived from
    Yokoyama's model.
    The specific heat of blood is assumed as 1.0 [kcal/L.K].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose a name from "dubois",
        "takahira", "fujimoto", or "kurazumi". The default is "dubois".
    age : float, optional
        age [years]. The default is 20.
    ci : float, optional
        Cardiac index [L/min/㎡]. The default is 2.59.

    Returns
    -------
    capacity : numpy.ndarray.
        Thermal capacity [W/K].
        The shape is (NUM_NODES).
    """
    validate_body_parameters(
        height=height,
        weight=weight,
        age=age,
    )
    # artery [Wh/K]
    cap_art = np.array(
        [
            0.096,
            0.025,
            0.12,
            0.111,
            0.265,
            0.0186,
            0.0091,
            0.0044,
            0.0186,
            0.0091,
            0.0044,
            0.0813,
            0.04,
            0.0103,
            0.0813,
            0.04,
            0.0103,
        ]
    )

    # vein [Wh/K]
    cap_vein = np.array(
        [
            0.321,
            0.085,
            0.424,
            0.39,
            0.832,
            0.046,
            0.024,
            0.01,
            0.046,
            0.024,
            0.01,
            0.207,
            0.1,
            0.024,
            0.207,
            0.1,
            0.024,
        ]
    )

    # superficial vein [Wh/K]
    cap_sfv = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0.025,
            0.015,
            0.011,
            0.025,
            0.015,
            0.011,
            0.074,
            0.05,
            0.021,
            0.074,
            0.05,
            0.021,
        ]
    )

    # central blood [Wh/K]
    cap_cb = 1.999

    # core [Wh/K]
    cap_cr = np.array(
        [
            1.7229,
            0.564,
            10.2975,
            9.3935,
            4.488,
            1.6994,
            1.1209,
            0.1536,
            1.6994,
            1.1209,
            0.1536,
            5.3117,
            2.867,
            0.2097,
            5.3117,
            2.867,
            0.2097,
        ]
    )

    # muscle [Wh/K]
    cap_ms = np.array(
        [
            0.305,
            0.0,
            0.0,
            0.0,
            7.409,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # fat [Wh/K]
    cap_fat = np.array(
        [
            0.203,
            0.0,
            0.0,
            0.0,
            1.947,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    # skin [Wh/K]
    cap_sk = np.array(
        [
            0.1885,
            0.058,
            0.441,
            0.406,
            0.556,
            0.126,
            0.084,
            0.088,
            0.126,
            0.084,
            0.088,
            0.334,
            0.169,
            0.107,
            0.334,
            0.169,
            0.107,
        ]
    )

    # Changes the values based on the standard body
    bfbr = bfb_rate(height, weight, bsa_equation, age, ci)
    wr = weight_rate(weight)
    cap_art *= bfbr
    cap_vein *= bfbr
    cap_sfv *= bfbr
    cap_cb *= bfbr
    cap_cr *= wr
    cap_ms *= wr
    cap_fat *= wr
    cap_sk *= wr

    cap_whole = np.zeros(NUM_NODES)
    cap_whole[0] = cap_cb

    for i, bn in enumerate(BODY_NAMES):
        # Dictionary of indices in each body segment
        # key = layer name, value = index of matrix
        index_of = IDICT[bn]

        # Common
        cap_whole[index_of["artery"]] = cap_art[i]
        cap_whole[index_of["vein"]] = cap_vein[i]
        cap_whole[index_of["core"]] = cap_cr[i]
        cap_whole[index_of["skin"]] = cap_sk[i]

        # Only limbs
        if i >= 5:
            cap_whole[index_of["sfvein"]] = cap_sfv[i]

        # If the segment has a muscle or fat layer
        if index_of["muscle"] is not None:  # or not indexof["fat"] is None
            cap_whole[index_of["muscle"]] = cap_ms[i]
            cap_whole[index_of["fat"]] = cap_fat[i]

    cap_whole *= 3600  # Changes unit [Wh/K] to [J/K]
    return cap_whole
