from collections.abc import Mapping
from typing import Any

import numpy as np


def valid_range(x, valid) -> np.ndarray:
    """Filter values based on a valid range."""
    return np.where((x >= valid[0]) & (x <= valid[1]), x, np.nan)


def mapping(
    value: float | np.ndarray, map_dictionary: Mapping[float, Any], right: bool = True
) -> np.ndarray:
    """Map a temperature array to stress categories.

    Parameters
    ----------
    value : float or array-like
        Temperature(s) to map.
    map_dictionary : dict
        Dictionary mapping bin edges to categories.
    right : bool, optional
        If True, intervals include the right bin edge.

    Returns
    -------
    np.ndarray
        Stress category for each input temperature. np.nan for unmapped.

    Raises
    ------
    TypeError
        If input types are invalid.

    Examples
    --------
    >>> mapping([20, 25, 30], {15: "low", 25: "medium", 35: "high"})
    array(['low', 'medium', 'high'], dtype=object)
    """
    if not isinstance(map_dictionary, dict):
        raise TypeError("map_dictionary must be a dict")
    value_arr = np.asarray(value)
    bins = np.asarray(list(map_dictionary.keys()))
    categories = np.array(list(map_dictionary.values()), dtype=object)
    # Append np.nan for out-of-range values
    categories = np.append(categories, np.nan)
    idx = np.digitize(value_arr, bins, right=right)
    return categories[idx]
