import numpy as np
from pythermalcomfort.zhang_comfort_model_functions import coefficients, utilities


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

    C31 = np.array(list(coefficients.C31.values()))
    C32 = np.array(list(coefficients.C32.values()))
    C6 = np.array(list(coefficients.C6.values()))
    C71 = np.array(list(coefficients.C71.values()))
    C72 = np.array(list(coefficients.C72.values()))
    C8 = np.array(list(coefficients.C8.values()))

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
    local_comfort_dict = utilities.convert_17_segments_dict_from_array(
        array=local_comfort_array, body_name_list=utilities.body_name_list
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
