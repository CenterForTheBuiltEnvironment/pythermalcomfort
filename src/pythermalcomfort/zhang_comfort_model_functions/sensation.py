import numpy as np
from pythermalcomfort.zhang_comfort_model_functions import coefficients, utilities

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
        Thermal sensation scale to be used. Accepts either '9-point' or '7-point'.
        - '9-point': Ranges from -4 (very cold) to 4 (very hot).
        - '7-point': Ranges from -3 (cold) to 3 (hot).

    Return
    ------
    sensation_local_array : numpy.ndarray
        Local thermal sensation on sens_third_coldest 9-point/7-point sensation scale [-].
    """

    def get_coefficient_c1(t_skin_local_array, t_skin_local_set_array):
        """Return c1_minus or c1_plus based on the comparison of skin temperature to its setpoint."""
        return np.where(t_skin_local_array - t_skin_local_set_array < 0, c1_minus_array, c1_plus_array)

    def get_coefficient_c2(dt_skin_local_dt_array):
        """Return c2_minus or c2_plus based on the local skin temperature derivative."""
        return np.where(dt_skin_local_dt_array < 0, c2_minus_array, c2_plus_array)

    # Set coefficients for the equation
    c1_plus_array = np.array(list(coefficients.c1_plus.values()))
    c1_minus_array = np.array(list(coefficients.c1_minus.values()))
    c1 = get_coefficient_c1(
        t_skin_local_array=t_skin_local_array,
        t_skin_local_set_array=t_skin_local_set_array,
    )
    k1 = np.array(list(coefficients.k1.values()))
    c2_plus_array = np.array(list(coefficients.c2_plus.values()))
    c2_minus_array = np.array(list(coefficients.c2_minus.values()))
    c2 = get_coefficient_c2(dt_skin_local_dt_array=dt_skin_local_dt_array)
    c3 = np.array(list(coefficients.c3.values()))

    # Calculate static component of local thermal sensation
    sensation_local_static_array = 4 * (
            (
                    2
                    / (
                            1
                            + np.exp(
                        - c1 * (t_skin_local_array - t_skin_local_set_array)
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
    sensation_local_array = (
            sensation_local_static_array + sensation_local_dynamic_array
    )

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
    """Calculates overall sensation.

    Parameters:
    sensation_local_array (array): Array of sensations from various body parts.

    :returns
    overall_sensation_dict (dict): Dict ....

    """

    # Define local thermal comfort as a dictionary
    local_17_segments_dict = utilities.convert_17_segments_dict_from_array(
        array=sensation_local_array, body_name_list=utilities.body_name_list
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
        return (
                       set(max_two) == hand_sensations or set(max_two) == foot_sensations
               ) or (set(min_two) == hand_sensations or set(min_two) == foot_sensations)

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

    return calculate_weighted_overall_sensation(
        sensation_local_array, cold_threshold=-2, warm_threshold=2, coefficient=2
    )
