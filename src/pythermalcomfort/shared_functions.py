import numpy as np


def valid_range(x, valid):
    """Filter values based on a valid range."""
    return np.where((x >= valid[0]) & (x <= valid[1]), x, np.nan)
