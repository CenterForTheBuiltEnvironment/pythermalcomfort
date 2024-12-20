import numpy as np


def valid_range(x, valid):
    """Filter values based on a valid range."""
    return np.where((x >= valid[0]) & (x <= valid[1]), x, np.nan)


def mapping(value, map_dictionary, right=True):
    """Maps a temperature array to stress categories.

    Parameters
    ----------
    value : float, array-like
        Temperature to map.
    map_dictionary: dict
        Dictionary used to map the values
    right: bool, optional
        Indicating whether the intervals include the right or the left bin edge.

    Returns
    -------
    Stress category for each input temperature.
    """
    bins = np.array(list(map_dictionary.keys()))
    words = np.append(np.array(list(map_dictionary.values())), "unknown")
    return words[np.digitize(value, bins, right=right)]
