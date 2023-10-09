import numpy as np
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.utilities import DefaultSkinTemperature

# Coefficients for sensation prediction
c1_plus = {
    "head": 0.9,
    "neck": 0.9,
    "chest": 1.0,
    "back": 1.0,
    "pelvis": 0.4,
    "left_shoulder": 0.4,
    "left_arm": 0.7,
    "left_hand": 0.45,
    "right_shoulder": 0.4,
    "right_arm": 0.7,
    "right_hand": 0.45,
    "left_thigh": 0.29,
    "left_leg": 0.4,
    "left_foot": 0.26,
    "right_thigh": 0.29,
    "right_leg": 0.4,
    "right_foot": 0.26,
}

c1_minus = {
    "head": 0.38,
    "neck": 0.38,
    "chest": 0.35,
    "back": 0.3,
    "pelvis": 0.2,
    "left_shoulder": 0.29,
    "left_arm": 0.3,
    "left_hand": 0.2,
    "right_shoulder": 0.29,
    "right_arm": 0.3,
    "right_hand": 0.2,
    "left_thigh": 0.2,
    "left_leg": 0.29,
    "left_foot": 0.25,
    "right_thigh": 0.2,
    "right_leg": 0.29,
    "right_foot": 0.25,
}

k1 = {
    "head": 0.18,
    "neck": 0.18,
    "chest": 0.1,
    "back": 0.1,
    "pelvis": 0.15,
    "left_shoulder": 0.1,
    "left_arm": 0.1,
    "left_hand": 0.15,
    "right_shoulder": 0.1,
    "right_arm": 0.1,
    "right_hand": 0.15,
    "left_thigh": 0.11,
    "left_leg": 0.1,
    "left_foot": 0.15,
    "right_thigh": 0.11,
    "right_leg": 0.1,
    "right_foot": 0.15,
}

c2_plus = {
    "head": 90,
    "neck": 90,
    "chest": 136,
    "back": 192,
    "pelvis": 137,
    "left_shoulder": 167,
    "left_arm": 125,
    "left_hand": 46,
    "right_shoulder": 167,
    "right_arm": 125,
    "right_hand": 46,
    "left_thigh": 263,
    "left_leg": 212,
    "left_foot": 162,
    "right_thigh": 263,
    "right_leg": 212,
    "right_foot": 162,
}

c2_minus = {
    "head": 543,
    "neck": 543,
    "chest": 39,
    "back": 88,
    "pelvis": 75,
    "left_shoulder": 156,
    "left_arm": 144,
    "left_hand": 19,
    "right_shoulder": 156,
    "right_arm": 144,
    "right_hand": 19,
    "left_thigh": 151,
    "left_leg": 206,
    "left_foot": 109,
    "right_thigh": 151,
    "right_leg": 206,
    "right_foot": 109,
}

c3 = {
    "head": 0,
    "neck": 0,
    "chest": -2135,
    "back": -4054,
    "pelvis": -5053,
    "left_shoulder": 0,
    "left_arm": 0,
    "left_hand": 0,
    "right_shoulder": 0,
    "right_arm": 0,
    "right_hand": 0,
    "left_thigh": 0,
    "left_leg": 0,
    "left_foot": 0,
    "right_thigh": 0,
    "right_leg": 0,
    "right_foot": 0,
}

a_if_delta_sens_local_is_less_than_minus_2 = {
    "head": 0.54,
    "neck": 0.65,
    "chest": 0.91,
    "back": 0.91,
    "pelvis": 0.94,
    "left_shoulder": 0.43,
    "left_arm": 0.37,
    "left_hand": 0.25,
    "right_shoulder": 0.43,
    "right_arm": 0.37,
    "right_hand": 0.25,
    "left_thigh": 0.81,
    "left_leg": 0.7,
    "left_foot": 0.5,
    "right_thigh": 0.81,
    "right_leg": 0.7,
    "right_foot": 0.5,
}

a_if_delta_sens_local_is_between_minus_2_to_plus_2 = {
    "head": 0.5,
    "neck": 0.46,
    "chest": 0.57,
    "back": 0.46,
    "pelvis": 0.32,
    "left_shoulder": 0.28,
    "left_arm": 0.38,
    "left_hand": 0.0,
    "right_shoulder": 0.28,
    "right_arm": 0.38,
    "right_hand": 0.0,
    "left_thigh": 0.3,
    "left_leg": 0.29,
    "left_foot": 0.0,
    "right_thigh": 0.3,
    "right_leg": 0.29,
    "right_foot": 0.0,
}

a_if_delta_sens_local_is_plus_2_or_more = {
    "head": 0.4,
    "neck": 0.4,
    "chest": 0.4,
    "back": 0.2,
    "pelvis": 0.6,
    "left_shoulder": 0.4,
    "left_arm": 0.3,
    "left_hand": 0.1,
    "right_shoulder": 0.4,
    "right_arm": 0.3,
    "right_hand": 0.1,
    "left_thigh": 0.4,
    "left_leg": 0.4,
    "left_foot": 0.6,
    "right_thigh": 0.4,
    "right_leg": 0.4,
    "right_foot": 0.6,
}

b_if_delta_sens_local_is_less_than_minus_2 = {
    "head": -1.1,
    "neck": -0.92,
    "chest": -1.14,
    "back": -0.92,
    "pelvis": -0.64,
    "left_shoulder": -0.56,
    "left_arm": -0.73,
    "left_hand": 0.0,
    "right_shoulder": -0.56,
    "right_arm": -0.73,
    "right_hand": 0.0,
    "left_thigh": -0.6,
    "left_leg": -0.59,
    "left_foot": 0.0,
    "right_thigh": -0.6,
    "right_leg": -0.59,
    "right_foot": 0.0,
}

b_if_delta_sens_local_is_between_minus_2_to_plus_2 = {
    "head": 0.0,
    "neck": 0.0,
    "chest": 0.0,
    "back": 0.0,
    "pelvis": 0.0,
    "left_shoulder": 0.0,
    "left_arm": 0.0,
    "left_hand": 0.0,
    "right_shoulder": 0.0,
    "right_arm": 0.0,
    "right_hand": 0.0,
    "left_thigh": 0.0,
    "left_leg": 0.0,
    "left_foot": 0.0,
    "right_thigh": 0.0,
    "right_leg": 0.0,
    "right_foot": 0.0,
}

b_if_delta_sens_local_is_plus_2_or_more = {
    "head": 1.1,
    "neck": 0.63,
    "chest": 1.14,
    "back": 0.92,
    "pelvis": 0.64,
    "left_shoulder": 0.56,
    "left_arm": 0.73,
    "left_hand": 0.0,
    "right_shoulder": 0.56,
    "right_arm": 0.73,
    "right_hand": 0.0,
    "left_thigh": 0.6,
    "left_leg": 0.59,
    "left_foot": 0.0,
    "right_thigh": 0.6,
    "right_leg": 0.59,
    "right_foot": 0.0,
}

c_if_delta_sens_local_is_less_than_minus_2 = {
    "head": -2,
    "neck": -2,
    "chest": -2,
    "back": -2,
    "pelvis": -2,
    "left_shoulder": -2,
    "left_arm": -2,
    "left_hand": -2,
    "right_shoulder": -2,
    "right_arm": -2,
    "right_hand": -2,
    "left_thigh": -2,
    "left_leg": -2,
    "left_foot": -2,
    "right_thigh": -2,
    "right_leg": -2,
    "right_foot": -2,
}

c_if_delta_sens_local_is_between_minus_2_to_plus_2 = {
    "head": 0.0,
    "neck": 0.0,
    "chest": 0.0,
    "back": 0.0,
    "pelvis": 0.0,
    "left_shoulder": 0.0,
    "left_arm": 0.0,
    "left_hand": 0.0,
    "right_shoulder": 0.0,
    "right_arm": 0.0,
    "right_hand": 0.0,
    "left_thigh": 0.0,
    "left_leg": 0.0,
    "left_foot": 0.0,
    "right_thigh": 0.0,
    "right_leg": 0.0,
    "right_foot": 0.0,
}

c_if_delta_sens_local_is_plus_2_or_more = {
    "head": 2,
    "neck": 2,
    "chest": 2,
    "back": 2,
    "pelvis": 2,
    "left_shoulder": 2,
    "left_arm": 2,
    "left_hand": 2,
    "right_shoulder": 2,
    "right_arm": 2,
    "right_hand": 2,
    "left_thigh": 2,
    "left_leg": 2,
    "left_foot": 2,
    "right_thigh": 2,
    "right_leg": 2,
    "right_foot": 2,
}

# Coefficients for comfort prediction
C31 = {
    "head": -0.35,
    "neck": 0.0,
    "chest": -0.66,
    "back": -0.45,
    "pelvis": -0.59,
    "left_shoulder": -0.3,
    "left_arm": -0.23,
    "left_hand": -0.8,
    "right_shoulder": -0.3,
    "right_arm": -0.23,
    "right_hand": -0.8,
    "left_thigh": 0.0,
    "left_leg": -0.2,
    "left_foot": -0.91,
    "right_thigh": 0.0,
    "right_leg": -0.2,
    "right_foot": -0.91,
}

C32 = {
    "head": 0.35,
    "neck": 0.0,
    "chest": 0.66,
    "back": 0.45,
    "pelvis": 0.0,
    "left_shoulder": 0.35,
    "left_arm": 0.23,
    "left_hand": 0.8,
    "right_shoulder": 0.35,
    "right_arm": 0.23,
    "right_hand": 0.8,
    "left_thigh": 0.0,
    "left_leg": 0.61,
    "left_foot": 0.4,
    "right_thigh": 0.0,
    "right_leg": 0.61,
    "right_foot": 0.4,
}

C6 = {
    "head": 2.17,
    "neck": 1.96,
    "chest": 2.1,
    "back": 2.1,
    "pelvis": 2.06,
    "left_shoulder": 2.14,
    "left_arm": 2.0,
    "left_hand": 1.98,
    "right_shoulder": 2.14,
    "right_arm": 2.0,
    "right_hand": 1.98,
    "left_thigh": 1.98,
    "left_leg": 2.0,
    "left_foot": 2.13,
    "right_thigh": 1.98,
    "right_leg": 2.0,
    "right_foot": 2.13,
}

C71 = {
    "head": 0.28,
    "neck": 0.0,
    "chest": 1.39,
    "back": 0.96,
    "pelvis": 0.5,
    "left_shoulder": 0.0,
    "left_arm": 0.0,
    "left_hand": 0.48,
    "right_shoulder": 0.0,
    "right_arm": 0.0,
    "right_hand": 0.48,
    "left_thigh": 0.0,
    "left_leg": 1.67,
    "left_foot": 0.5,
    "right_thigh": 0.0,
    "right_leg": 1.67,
    "right_foot": 0.5,
}

C72 = {
    "head": 0.4,
    "neck": 0.0,
    "chest": 0.9,
    "back": 0.0,
    "pelvis": 0.0,
    "left_shoulder": 0.0,
    "left_arm": 1.71,
    "left_hand": 0.48,
    "right_shoulder": 0.0,
    "right_arm": 1.71,
    "right_hand": 0.48,
    "left_thigh": 0.0,
    "left_leg": 0.0,
    "left_foot": 0.3,
    "right_thigh": 0.0,
    "right_leg": 0.0,
    "right_foot": 0.3,
}

C8 = {
    "head": 0.5,
    "neck": -0.19,
    "chest": 0.0,
    "back": 0.0,
    "pelvis": -0.51,
    "left_shoulder": -0.4,
    "left_arm": -0.68,
    "left_hand": 0.0,
    "right_shoulder": -0.4,
    "right_arm": -0.68,
    "right_hand": 0.0,
    "left_thigh": 0.0,
    "left_leg": 0.0,
    "left_foot": 0.0,
    "right_thigh": 0.0,
    "right_leg": 0.0,
    "right_foot": 0.0,
}
c1_plus_array = np.array(list(c1_plus.values()))
c1_minus_array = np.array(list(c1_minus.values()))
k1_array = np.array(list(k1.values()))
c2_plus_array = np.array(list(c2_plus.values()))
c2_minus_array = np.array(list(c2_minus.values()))
c3_array = np.array(list(c3.values()))
a_if_delta_sens_local_is_less_than_minus_2_array = np.array(
    list(a_if_delta_sens_local_is_less_than_minus_2.values())
)
a_if_delta_sens_local_is_between_minus_2_to_plus_2_array = np.array(
    list(a_if_delta_sens_local_is_between_minus_2_to_plus_2.values())
)
a_if_delta_sens_local_is_plus_2_or_more_array = np.array(
    list(a_if_delta_sens_local_is_plus_2_or_more.values())
)
b_if_delta_sens_local_is_less_than_minus_2_array = np.array(
    list(b_if_delta_sens_local_is_less_than_minus_2.values())
)
b_if_delta_sens_local_is_between_minus_2_to_plus_2_array = np.array(
    list(b_if_delta_sens_local_is_between_minus_2_to_plus_2.values())
)
b_if_delta_sens_local_is_plus_2_or_more_array = np.array(
    list(b_if_delta_sens_local_is_plus_2_or_more.values())
)
c_if_delta_sens_local_is_less_than_minus_2_array = np.array(
    list(c_if_delta_sens_local_is_less_than_minus_2.values())
)
c_if_delta_sens_local_is_between_minus_2_to_plus_2_array = np.array(
    list(c_if_delta_sens_local_is_between_minus_2_to_plus_2.values())
)
c_if_delta_sens_local_is_plus_2_or_more_array = np.array(
    list(c_if_delta_sens_local_is_plus_2_or_more.values())
)
C31_array = np.array(list(C31.values()))
C32_array = np.array(list(C32.values()))
C6_array = np.array(list(C6.values()))
C71_array = np.array(list(C71.values()))
C72_array = np.array(list(C72.values()))
C8_array = np.array(list(C8.values()))

# 17 body segment names (The name corresponds to JOS-3 model)
body_name_list = [
    "head",
    "neck",
    "chest",
    "back",
    "pelvis",
    "left_shoulder",
    "left_arm",
    "left_hand",
    "right_shoulder",
    "right_arm",
    "right_hand",
    "left_thigh",
    "left_leg",
    "left_foot",
    "right_thigh",
    "right_leg",
    "right_foot",
]


def convert_to_17_segments_array_from_any_data_type(data):
    """Convert various input data types to a 17-segment numpy array.

    This function is designed to handle a variety of input types including integers, floats, dictionaries,
    lists, and numpy arrays, and returns a standardized 17-segment numpy array. This can be useful when
    ensuring consistent array formats across different input types.

    Parameters
    ----------
    data : int, float, dict, list, ndarray or None
        Input data that needs to be converted. If `data` is:
        - int or float: The returned array will have all 17 values set to this number.
        - dict: The keys of the dictionary should match the `body_name_list` and the returned array will be constructed based on this order.
        - list or ndarray: Must be of length 17.

    Returns
    -------
    data_17_array : numpy.ndarray
        A numpy array of shape (17,) containing the standardized data.

    Raises
    ------
    ValueError
        If the input data is not one of the supported types, or if it is a list or ndarray of length other than 17.

    Notes
    -----
    Ensure that the `body_name_list` is defined elsewhere in the code if using this function with dictionary input data.
    """
    if data is None:
        return None
    # Convert input data to sens_third_coldest numpy array for consistent handling
    if isinstance(data, (int, float)):
        data_17_array = np.ones(17) * data
    elif isinstance(data, dict):
        data_17_array = np.array([data[key] for key in body_name_list])
    elif isinstance(data, (list, np.ndarray)):
        data_17_array = np.asarray(data)
        if data.shape == (17,):
            data_17_array = data
        else:
            ValueError("The input list or ndarray is not of length 17")
    else:
        raise ValueError(
            "Unsupported input type. Supported types: int, float, list, dict, ndarray"
        )
    return data_17_array


def convert_17_segments_dict_from_array(array, body_name_list):
    """
    Convert a 17-segment numpy array to a dictionary using the provided body name list.

    Parameters
    ----------
    array : numpy.ndarray
        A numpy array of shape (17,) containing the data to be converted.

    body_name_list : list
        A list of body part names with length 17. Each name corresponds to a value in the `array`.

    Returns
    -------
    dict
        A dictionary with keys as body part names from `body_name_list` and values from the input `array`.
    """
    return {part: array[i] for i, part in enumerate(body_name_list)}


def calculate_local_thermal_sensation(
    t_skin_local_array,
    dt_skin_local_dt_array,
    t_skin_local_set_array,
    t_skin_mean_value,
    t_skin_mean_set_value,
    dt_core_local_dt_array,
    sensation_scale,
):
    """Calculate local thermal sensation based on physiological parameters and the selected sensation scale.

    Parameters
    -----------
    t_skin_local_array : numpy.ndarray
        Array of local skin temperatures [°C].
    dt_skin_local_dt_array : numpy.ndarray
        Time derivative of local skin temperatures [°C/s].
    t_skin_local_set_array : numpy.ndarray
        Array of local skin temperature setpoints [°C].
    t_skin_mean_value : float
        Mean skin temperature value [°C].
    t_skin_mean_set_value : float
        Setpoint for mean skin temperature [°C].
    dt_core_local_dt_array : numpy.ndarray
        Time derivative of core temperatures [°C/s].
    sensation_scale : str
        Thermal sensation scale to be used. Accepts either '9-point' or '7-point'. Default is "9-point".
        - '9-point': Ranges from -4 (very cold) to 4 (very hot).
        - '7-point': Ranges from -3 (cold) to 3 (hot).

    Return
    ------
    sensation_local_array : numpy.ndarray
        Local thermal sensation on sens_third_coldest 9-point/7-point sensation scale [-].
    """

    def get_coefficient_c1(t_skin_local_array, t_skin_local_set_array):
        """Return c1_minus or c1_plus based on the comparison of skin temperature to its setpoint."""
        return np.where(
            t_skin_local_array - t_skin_local_set_array < 0,
            c1_minus_array,
            c1_plus_array,
        )

    def get_coefficient_c2(dt_skin_local_dt_array):
        """Return c2_minus or c2_plus based on the local skin temperature derivative."""
        return np.where(dt_skin_local_dt_array < 0, c2_minus_array, c2_plus_array)

    # Set coefficients for the equation
    c1 = get_coefficient_c1(
        t_skin_local_array=t_skin_local_array,
        t_skin_local_set_array=t_skin_local_set_array,
    )

    k1 = k1_array
    c2 = get_coefficient_c2(dt_skin_local_dt_array=dt_skin_local_dt_array)

    c3 = c3_array

    # Calculate static component of local thermal sensation
    sensation_local_static_array = 4 * (
        (
            2
            / (
                1
                + np.exp(
                    -c1 * (t_skin_local_array - t_skin_local_set_array)
                    - k1
                    * (
                        (t_skin_local_array - t_skin_local_set_array)
                        - (t_skin_mean_value - t_skin_mean_set_value)
                    )
                )
            )
        )
        - 1
    )

    # Calculate dynamic component of local thermal sensation
    sensation_local_dynamic_array = (
        c2 * dt_skin_local_dt_array + c3 * dt_core_local_dt_array
    )

    # Calculate local thermal sensation
    sensation_local_array = sensation_local_static_array + sensation_local_dynamic_array

    # Apply the selected thermal sensation scale
    if sensation_scale == "7-point":
        # Clip the array between -3 and 3
        sensation_local_array = np.clip(sensation_local_array, -3, 3)
    elif sensation_scale == "9-point":
        # Clip the array between -4 and 4
        sensation_local_array = np.clip(sensation_local_array, -4, 4)
    else:
        raise ValueError("Please select '7-point' or '9-point'")

    return sensation_local_array


def calculate_overall_sensation(sensation_local_array):
    """Calculate overall thermal sensation based on local sensation for each body part.

    Sensations in the cold or warm range are emphasized more in the average.
    If the most extreme sensations are from the hands or feet, they are excluded from the weighted average.

    Parameters
    ----------
    sensation_local_array : ndarray
        Array or list of local thermal sensations from various body parts.

    Returns
    -------
    overall_sensation : float
        Overall thermal sensation.
    """

    # Define local thermal comfort as a dictionary
    local_17_segments_dict = convert_17_segments_dict_from_array(
        array=sensation_local_array, body_name_list=body_name_list
    )

    def are_most_extreme_sensations_hands_or_feet(
        sensation_local_array,
        left_hand_sensation,
        right_hand_sensation,
        left_foot_sensation,
        right_foot_sensation,
    ):
        """Determine if the most extreme sensations in a given array are from the hands or feet.

        This function examines the input array of sensations to determine if the most
        extreme sensations (both maximum and minimum) come from the hands or feet. The
        sensations of the hands and feet are provided separately for comparison.

        Parameters
        ----------
        sensation_local_array : list or ndarray
            An array or list containing sensations from various body parts.
        left_hand_sensation : float
            The sensation value corresponding to the left hand.
        right_hand_sensation : float
            The sensation value corresponding to the right hand.
        left_foot_sensation : float
            The sensation value corresponding to the left foot.
        right_foot_sensation : float
            The sensation value corresponding to the right foot.

        Returns
        -------
        bool
            Returns True if the most extreme sensations (either the two largest or two smallest)
            correspond to both hands or both feet. Otherwise, returns False.

        Notes
        -----
        The function assumes that the `sensation_local_array` contains at least two elements.
        If not, it will directly return False without further checks.

        """

        # Sensation array must have at least two elements for this function to work properly
        if len(sensation_local_array) < 2:
            return False

        # Extract the top two maximum and minimum sensations
        max_two = sorted(sensation_local_array, reverse=True)[:2]
        min_two = sorted(sensation_local_array)[:2]

        # Set up hand and foot sensations
        hand_sensations = {left_hand_sensation, right_hand_sensation}
        foot_sensations = {left_foot_sensation, right_foot_sensation}

        # Check if the most extreme sensations correspond to two hands or two feet
        return (set(max_two) == hand_sensations or set(max_two) == foot_sensations) or (
            set(min_two) == hand_sensations or set(min_two) == foot_sensations
        )

    def calculate_weighted_overall_sensation(
        sensation_local_array,
        cold_threshold=-2,
        warm_threshold=2,
        coefficient=2,
    ):
        """Calculate the weighted average of provided thermal sensations.

        This function calculates a weighted average of given local thermal sensations, adjusting
        weights dynamically based on the provided thresholds. Sensations that fall within the
        cold or warm threshold range are given increased importance in the average using
        a specified coefficient.

        If the most extreme sensations are from the hands or feet, they are excluded from the weighted average.

        Parameters
        ----------
        sensation_local_array : list or ndarray
            An array or list containing thermal sensations for various body parts.
        cold_threshold : float, optional
            The threshold below which sensations are considered cold and given higher weight.
            Default is -2.
        warm_threshold : float, optional
            The threshold above which sensations are considered warm and given higher weight.
            Default is 2.
        coefficient : float, optional
            The coefficient to increase the weight of sensations that fall within the
            cold or warm threshold. Default is 2.

        Returns
        -------
        overall_sensation_weighted : float
            The weighted overall sensation.
        """

        # Calculate the absolute values of sensations as initial weights
        abs_sensations = np.abs(sensation_local_array)

        # Check if most extreme sensations are hands or feet
        if are_most_extreme_sensations_hands_or_feet(
            sensation_local_array=sensation_local_array,
            left_hand_sensation=local_17_segments_dict["left_hand"],
            right_hand_sensation=local_17_segments_dict["right_hand"],
            left_foot_sensation=local_17_segments_dict["left_foot"],
            right_foot_sensation=local_17_segments_dict["right_foot"],
        ):
            # Set the weight of the sensation with the maximum absolute value to 0
            max_abs_index = np.argmax(np.abs(sensation_local_array))
            abs_sensations[max_abs_index] = 0

        # Apply the coefficient to sensations that fall within the specified ranges
        abs_sensations[sensation_local_array <= cold_threshold] *= coefficient
        abs_sensations[sensation_local_array >= warm_threshold] *= coefficient

        # Normalize the weights so that their sum is 1
        total_weight = np.sum(abs_sensations)
        normalized_weights = (
            abs_sensations / total_weight if total_weight != 0 else abs_sensations
        )

        # Calculate the weighted average
        overall_sensation_weighted = np.average(
            sensation_local_array, weights=normalized_weights
        )

        return overall_sensation_weighted

    overall_sensation = calculate_weighted_overall_sensation(
        sensation_local_array, cold_threshold=-2, warm_threshold=2, coefficient=2
    )

    return overall_sensation


def calculate_local_thermal_comfort(overall_sensation, sensation_local_array):
    """Calculate local thermal comfort based on overall sensation and local sensations.

    This function computes the local thermal comfort by considering the overall sensation and
    an array of local sensations. It utilizes predefined coefficients and applies specific formulas
    based on whether the overall sensation is positive or negative.

    Parameters
    ----------
    overall_sensation : float
        The overall sensation which can be a positive or negative float.
    sensation_local_array : np.ndarray
        A numpy array containing local sensations.

    Returns
    -------
    local_comfort_array : np.ndarray
        local thermal comfort values. Each value is constrained to the range of -4 (very uncomfortable) to 4 (very comfortable).

    """

    C31 = C31_array
    C32 = C32_array
    C6 = C6_array
    C71 = C71_array
    C72 = C72_array
    C8 = C8_array

    def get_elements_for_calculating_local_thermal_comfort(overall_sensation):
        """Calculate and return elements used for determining local thermal comfort.

        Parameters
        ----------
        overall_sensation : float
            The overall thermal sensation which could be either positive or negative.

        Returns
        -------
        tuple
            A tuple containing the computed elements.
        """
        if overall_sensation >= 0:
            A1 = -8 * (C6 + C72 * overall_sensation + 4)
            A2 = (
                (4 - (C8 + C32 * overall_sensation))
                * (-4 - (C8 + C32 * overall_sensation))
                * (1 + np.exp(5 * (sensation_local_array + C32 * overall_sensation)))
            )
            A4 = (4 + (C6 + C72 * overall_sensation)) / (
                -4 - (C8 + C32 * overall_sensation)
            )
            A5 = sensation_local_array + C8 + C32 * overall_sensation
            A6 = C6 + C72 * overall_sensation
        else:
            A1 = -8 * (C6 - C71 * overall_sensation + 4)
            A2 = (
                (4 - (C8 - C31 * overall_sensation))
                * (-4 - (C8 - C31 * overall_sensation))
                * (1 + np.exp(5 * (sensation_local_array - C31 * overall_sensation)))
            )
            A4 = (4 + (C6 - C71 * overall_sensation)) / (
                -4 - (C8 - C31 * overall_sensation)
            )
            A5 = sensation_local_array + C8 - C31 * overall_sensation
            A6 = C6 - C71 * overall_sensation

        return A1, A2, A4, A5, A6

    sensation_local_array = np.array(sensation_local_array)
    local_comfort_array = np.zeros(len(sensation_local_array))

    for i in range(len(sensation_local_array)):
        A1, A2, A4, A5, A6 = get_elements_for_calculating_local_thermal_comfort(
            overall_sensation
        )

        local_comfort_array = (A1 / A2 + A4) * A5 + A6

        # Constrain local_comfort_values[i] to the range -4 to 4
        if local_comfort_array[i] < -4:
            local_comfort_array[i] = -4
        elif local_comfort_array[i] > 4:
            local_comfort_array[i] = 4

    return local_comfort_array


def calculate_overall_thermal_comfort(
    local_comfort_array, is_transient=False, is_controlled=False
):
    """Calculate the overall thermal comfort based on local comfort values and additional conditions.

    This function calculates the overall thermal comfort by considering local thermal comforts
    and their relation to the hands and feet.

    It uses two rules:
    1) When in steady state or controlled environments, and
    2) In transient or uncontrolled conditions.
    The function adjusts the weights of the local comfort values based on these rules.

    Parameters
    ----------
    local_comfort_array : list or ndarray
        An array or list containing local thermal comfort values for various body segments.
    is_transient : bool, optional
        A flag indicating if the environment is in a transient state. Default is False.

        Note : The way to determine whether transient or not is not described in the paper.
        Original CBE's C++ code always returns False (It might be determined by the time derivative of skin temperature and so on).

    is_controlled : bool, optional
        A flag indicating if the environment is controlled. Default is False.
        This can be changed to "True" if simulated people can adjust their own thermal environment,
        such as in an automobile or personal comfort system.

    Returns
    -------
    overall_comfort_value : float
        The overall thermal comfort value.

    Notes
    -----
    - Rule 1: In steady state or controlled environments, the overall comfort is the average of
      the two lowest local comfort votes unless the two lowest sensations are from hands or feet.
    - Rule 2: In transient or uncontrolled conditions, the overall comfort is the average of the
      two lowest and the highest comfort votes, unless the two lowest sensations are from hands or feet.
    - The function makes use of the `are_lowest_comfort_values_hands_or_feet` inner function to
      determine if the two most uncomfortable segments are the hands or feet.
    - The function assumes that `body_name_list` and `convert_17_segments_dict_from_array` are available
      in the surrounding code.
    """

    # Define local comfort dictionary
    local_comfort_dict = convert_17_segments_dict_from_array(
        array=local_comfort_array, body_name_list=body_name_list
    )

    # Define the highest local thermal comfort (most comfortable)
    highest_local_comfort_value = np.max(local_comfort_array)

    # Sort in ascending order
    local_comfort_ascending_order_array = np.sort(local_comfort_array)

    # Define the TOP-4 lowest local thermal comfort (TOP-4 most uncomfortable)
    (
        lowest_local_comfort_value,
        second_lowest_local_comfort_value,
        third_lowest_local_comfort_value,
        fourth_lowest_local_comfort_value,
    ) = local_comfort_ascending_order_array[:4]

    # Define the mean value of TOP-2 lowest local thermal comfort
    mean_top2_lowest_comfort = np.mean(
        [lowest_local_comfort_value, second_lowest_local_comfort_value]
    )

    def are_lowest_comfort_values_hands_or_feet(local_comfort_dict):
        """
        Determine if the two most uncomfortable sensations are from the hands or feet.

        Parameters
        ----------
        local_comfort_dict : dict
            A dictionary containing body segments as keys and their associated thermal comfort values.
            It should at least have entries for "left_hand", "right_hand", "left_foot", and "right_foot".

        Returns
        -------
        bool
            True if the two most uncomfortable sensations correspond to the hands or feet. False otherwise.
        """

        # Find the two minimum comfort values
        two_lowest_values = sorted(local_comfort_dict.values())[:2]

        # Extract comfort values for hands and feet
        comfort_left_hand = local_comfort_dict["left_hand"]
        comfort_right_hand = local_comfort_dict["right_hand"]
        comfort_left_foot = local_comfort_dict["left_foot"]
        comfort_right_foot = local_comfort_dict["right_foot"]

        # Compare the two lowest values with the comfort values for hands and feet
        # If the two lowest values match the values for either hands or feet, return True
        return (
            sorted([comfort_left_hand, comfort_right_hand]) == two_lowest_values
            or sorted([comfort_left_foot, comfort_right_foot]) == two_lowest_values
        )

    # Rule 1 (Steady state or controlled, such as in sens_third_coldest climate chamber)
    # Rule 1: Overall comfort is the average of the two minimum local comfort votes unless Rule 2 applies.
    if not is_transient or is_controlled:
        if are_lowest_comfort_values_hands_or_feet == True:
            overall_comfort_value = np.mean(
                [
                    mean_top2_lowest_comfort,
                    third_lowest_local_comfort_value,
                    fourth_lowest_local_comfort_value,
                ]
            )
        else:  # False
            overall_comfort_value = np.mean(
                [
                    lowest_local_comfort_value,
                    second_lowest_local_comfort_value,
                    third_lowest_local_comfort_value,
                ]
            )

    # Rule 2 (Transient state or not controlled space)
    # Rule 2: Overall comfort is the average of the two minimum votes and the maximum comfort vote:
    else:
        if are_lowest_comfort_values_hands_or_feet == True:
            overall_comfort_value = np.mean(
                [
                    mean_top2_lowest_comfort,
                    third_lowest_local_comfort_value,
                    highest_local_comfort_value,
                ]
            )
        else:  # False
            overall_comfort_value = np.mean(
                [
                    lowest_local_comfort_value,
                    second_lowest_local_comfort_value,
                    highest_local_comfort_value,
                ]
            )

    return overall_comfort_value


def zhang_sensation_comfort(
    t_skin_local,
    dt_skin_local_dt,
    dt_core_local_dt,
    t_skin_local_set=DefaultSkinTemperature()._asdict(),
    options=None,
):
    """Zhang's local and overall sensation and comfort model.

    This model predicts thermal sensation and comfort based on physiological responses,
    such as skin temperature and its time variation.

    It consists of four main helper models that predict:
    1. Local thermal sensation
    2. Whole-body thermal sensation
    3. Local thermal comfort
    4. Whole-body thermal comfort

    The results of these predictions are returned as a dictionary.

    Accepted input data types for conditions like skin temperature are int, float, list, dict, and ndarray.
    To avoid input errors, it's recommended to provide the data as a dictionary.
    Each key in the dictionary should represent the name of a body segment, aligning with the JOS-3 model.

    body_name_list = [
        "head", "neck", "chest", "back", "pelvis",
        "left_shoulder", "left_arm", "left_hand",
        "right_shoulder", "right_arm", "right_hand",
        "left_thigh", "left_leg", "left_foot",
        "right_thigh", "right_leg", "right_foot"
    ]

    When data is provided as an int or float format, all body segments will be assigned the same value.
    If data is provided as a list or ndarray format, it should be of length 17, with values corresponding to the keys listed above.


    Parameters
    -----------
    t_skin_local : int, float, list, numpy.ndarray, dict
        Local skin temperature [°C].
    t_skin_local_set : int, float, list, numpy.ndarray, dict
        Local skin temperature setpoint [°C].

        Note: t_skin_local_set is the local skin temperature when people feel thermally neutral.
        Neutral skin temperatures vary by site, generally higher for the trunk and lower for the extremities.
        It is stated that multi-node segment model is used to get these parameters under the condition where
        people thermally feel neutral, calculated by the PMV model at a certain metabolic rate and clothing level.

        The experimental data on neutral skin temperature from Zhang's experiments : https://escholarship.org/uc/item/3f4599hx

    dt_skin_local_dt : int, float, list, numpy.ndarray, dict
        Time derivetive of local skin temperature [°C/s].
    dt_core_local_dt : int, float, list, numpy.ndarray, dict
        Time derivetive of core temperature [°C/s].
    options : dict, optional
        Dictionary containing options for the calculation.

        Keys include:
            "sensation_scale" (str):
                Scale for evaluating thermal sensation. Default is "9-point".
                Either '9-point' or '7-point'.
                9-point sensation scale (-4:very cold, -3:cold, -2:cool, -1:slightly cool, 0:neutral, +1:slightly warm, +2:warm, +3:hot, +4:very hot)
                7-point sensation scale (-3:cold, -2:cool, -1:slightly cool, 0:neutral, +1:slightly warm, +2:warm, +3:hot)

            "is_transient" (bool):
                Indicates if the environment is transient. The original paper does not describe how to define.
                Default is False.

            "is_controlled" (bool):
                Indicates if the environment is controlled. Default is False.

            "show_extra_info" (bool):
                Indicates if extra information should be displayed. Default is False.

    Returns
    -------
    sensation_local : local thermal sensation (each body segment) [-]
    sensation_overall : overall thermal sensation [-]
    comfort_local : local thermal comfort for each body segment [-]
    comfort_overall : overall thermal sensation [-]
    OPTIONAL PARAMETERS : the paramters listed below are returned if show_extra_info = True
    t_skin_local : local skin temperature [°C].
    t_skin_local_set : local skin temperature setpoint [°C].
    dt_skin_local_dt : time derivetive of local skin temperature [°C/s].
    t_skin_mean : mean skin temperature for the whole body [°C].
    t_skin_mean_set : mean skin temperature setpoint for the whole body [°C].
    dt_core_local_dt : time derivetive of core temperature [°C/s].


    Examples
    ---------
    .. code-block:: python
    >>> from pythermalcomfort.models import zhang_sensation_comfort
    >>> dict_results = zhang_sensation_comfort(
    >>>     t_skin_local={
    >>>         "head": 34.3,
    >>>         "neck": 34.6,
    >>>         "chest": 34.1,
    >>>         "back": 34.3,
    >>>         "pelvis": 34.3,
    >>>         "left_shoulder": 33.2,
    >>>         "left_arm": 33.6,
    >>>         "left_hand": 33.4,
    >>>         "right_shoulder": 33.2,
    >>>         "right_arm": 33.6,
    >>>         "right_hand": 33.4,
    >>>         "left_thigh": 33.3,
    >>>         "left_leg": 31.8,
    >>>         "left_foot": 32.3,
    >>>         "right_thigh": 33.3,
    >>>         "right_leg": 31.8,
    >>>         "right_foot": 32.3,
    >>>     },
    >>>     dt_skin_local_dt={
    >>>         "head": 0.01,
    >>>         "neck": 0.01,
    >>>         "chest": 0.01,
    >>>         "back": 0.01,
    >>>         "pelvis": 0.01,
    >>>         "left_shoulder": 0.01,
    >>>         "left_arm": 0.01,
    >>>         "left_hand": 0.01,
    >>>         "right_shoulder": 0.01,
    >>>         "right_arm": 0.01,
    >>>         "right_hand": 0.01,
    >>>         "left_thigh": 0.01,
    >>>         "left_leg": 0.01,
    >>>         "left_foot": 0.01,
    >>>         "right_thigh": 0.01,
    >>>         "right_leg": 0.01,
    >>>         "right_foot": 0.01,
    >>>     },
    >>>     dt_core_local_dt=0,
    >>>     options={
    >>>         "sensation_scale": "7-point",
    >>>     },
    >>> )
    >>>
    >>> pprint(dict_results)
    {
    'comfort_back': -0.48,
    'comfort_chest': 0.98,
    'comfort_head': 1.35,
    'comfort_left_arm': 3.74,
    'comfort_left_foot': 0.13,
    'comfort_left_hand': 1.12,
    'comfort_left_leg': -0.97,
    'comfort_left_shoulder': 0.36,
    'comfort_left_thigh': -1.36,
    'comfort_neck': 1.98,
    'comfort_overall': -1.23,
    'comfort_pelvis': 1.27,
    'comfort_right_arm': 3.74,
    'comfort_right_foot': 0.13,
    'comfort_right_hand': 1.12,
    'comfort_right_leg': -0.97,
    'comfort_right_shoulder': 0.36,
    'comfort_right_thigh': -1.36,
    'sensation_back': 1.32,
    'sensation_chest': 0.67,
    'sensation_head': 0.15,
    'sensation_left_arm': 0.65,
    'sensation_left_foot': 1.12,
    'sensation_left_hand': 0.06,
    'sensation_left_leg': 1.54,
    'sensation_left_shoulder': 1.09,
    'sensation_left_thigh': 2.23,
    'sensation_neck': 0.15,
    'sensation_overall': 1.42,
    'sensation_pelvis': 0.97,
    'sensation_right_arm': 0.65,
    'sensation_right_foot': 1.12,
    'sensation_right_hand': 0.06,
    'sensation_right_leg': 1.54,
    'sensation_right_shoulder': 1.09,
    'sensation_right_thigh': 2.23
     }
    """

    default_options = {
        "sensation_scale": "9-point",
        "is_transient": False,
        "is_controlled": False,
        "show_extra_info": False,
    }

    # If options is provided, update the default options with the provided ones
    if options:
        default_options.update(options)

    options = default_options  # Now use the merged dictionary in your function

    # Define phygiological input parameters
    t_skin_local_array = convert_to_17_segments_array_from_any_data_type(t_skin_local)
    dt_skin_local_dt_array = convert_to_17_segments_array_from_any_data_type(
        dt_skin_local_dt
    )
    t_skin_local_set_array = convert_to_17_segments_array_from_any_data_type(
        t_skin_local_set
    )
    t_skin_mean_value = np.average(t_skin_local_array, weights=Default.local_bsa)
    t_skin_mean_set_value = np.average(
        t_skin_local_set_array, weights=Default.local_bsa
    )
    dt_core_local_dt_array = convert_to_17_segments_array_from_any_data_type(
        dt_core_local_dt
    )

    # Define optional input parameters
    sensation_scale = options["sensation_scale"]
    is_transient = options["is_transient"]
    is_controlled = options["is_controlled"]
    show_extra_info = options["show_extra_info"]

    # ------------------------------------------------------
    # Process Main
    # ------------------------------------------------------

    # Calculate local thermal sensation
    sensation_local_array = calculate_local_thermal_sensation(
        t_skin_local_array=t_skin_local_array,
        dt_skin_local_dt_array=dt_skin_local_dt_array,
        t_skin_local_set_array=t_skin_local_set_array,
        t_skin_mean_value=t_skin_mean_value,
        t_skin_mean_set_value=t_skin_mean_set_value,
        dt_core_local_dt_array=dt_core_local_dt_array,
        sensation_scale=sensation_scale,
    )

    # Calculate overall thermal sensation
    overall_sensation_value = calculate_overall_sensation(
        sensation_local_array=sensation_local_array
    )

    # Calculate local thermal comfort
    comfort_local_array = calculate_local_thermal_comfort(
        overall_sensation=overall_sensation_value,
        sensation_local_array=sensation_local_array,
    )

    # Calculate overall thermal comfort
    overall_comfort_value = calculate_overall_thermal_comfort(
        local_comfort_array=comfort_local_array,
        is_transient=is_transient,
        is_controlled=is_controlled,
    )

    # Define result dict
    dict_data = {
        "sensation_local": np.around(sensation_local_array, 2),
        "sensation_overall": np.around(overall_sensation_value, 2),
        "comfort_local": np.around(comfort_local_array, 2),
        "comfort_overall": np.around(overall_comfort_value, 2),
    }

    if show_extra_info:
        dict_data.update(
            {
                "t_skin_local": t_skin_local_array,
                "dt_skin_local_dt": dt_skin_local_dt_array,
                "t_skin_local_set": t_skin_local_set_array,
                "t_skin_mean_value": t_skin_mean_value,
                "t_skin_mean_set_value": t_skin_mean_set_value,
                "dt_core_local_dt_array": dt_core_local_dt_array,
                "sensation_scale": sensation_scale,
                "is_transient": is_transient,
            }
        )

    # Create new dictionary to store the new keys and values
    new_data_dict = {}

    # Map 17 arrays of values to new keys for keys that contain 'local'
    for key in dict_data.keys():
        if "local" in key:
            # Map 17 arrays of values to new keys
            for i, body_name in enumerate(body_name_list):
                # Replace 'local' with body_name in the key
                new_key = key.replace("local", body_name)
                new_value = dict_data[key][i]
                new_data_dict[new_key] = new_value

    # Merge the new dictionary with the original dictionary
    dict_data.update(new_data_dict)

    # Delete the original 'local_sensation' and 'local_comfort' keys
    for key in list(dict_data.keys()):
        if "local" in key and not any(part in key for part in body_name_list):
            del dict_data[key]

    return dict_data
