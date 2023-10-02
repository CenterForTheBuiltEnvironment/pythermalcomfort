import numpy as np

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
