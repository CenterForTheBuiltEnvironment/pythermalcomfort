from collections.abc import Mapping
from typing import Any

import numpy as np


def valid_range(x, valid) -> np.ndarray:
    """Filter values based on a valid range."""
    return np.where((x >= valid[0]) & (x <= valid[1]), x, np.nan)


def _finalize_scalar_or_array(arr: Any) -> Any:
    """Convert 0d arrays to Python scalars, preserve np.nan, return arrays as-is.

    Args:
        arr: np.ndarray, scalar, or array-like.

    Returns:
        Python scalar (with np.nan preserved) if input is scalar, else array.

    Examples
    --------
    >>> _finalize_scalar_or_array(np.array(True, dtype=object))
    True
    >>> _finalize_scalar_or_array(np.array(np.nan, dtype=object))
    nan
    >>> _finalize_scalar_or_array(np.array([True, False, np.nan], dtype=object))
    array([True, False, nan], dtype=object)
    """
    arr = np.asarray(arr, dtype=object)
    if arr.shape == ():
        val = arr.item()
        if isinstance(val, float) and np.isnan(val):
            return np.nan
        # Convert np.bool_ to Python bool
        if isinstance(val, (np.bool_ | bool)):
            return bool(val)
        return val
    return arr


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
