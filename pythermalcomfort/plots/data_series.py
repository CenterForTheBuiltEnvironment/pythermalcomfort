"""DataSeries - Overlay data points for scenes.

This module provides the DataSeries class for adding scatter data, lines or annotations
on top of Scene visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from pythermalcomfort.plots.scenes.base import BaseScene


@dataclass(frozen=True)
class DataSeries:
    """Overlay data points for a scene.

    DataSeries holds x/y coordinates and optional values for scatter plots
    that can be overlaid on Scene visualizations. It is frozen (immutable)
    after creation.

    Attributes
    ----------
    x : np.ndarray
        X-coordinates of data points.
    y : np.ndarray
        Y-coordinates of data points.
    values : np.ndarray or None
        Optional metric values for coloring points.

    Examples
    --------
    >>> data = DataSeries.from_arrays([22, 24, 26], [50, 60, 70])
    >>> categories = data.compute_categories(scene)
    """

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray | None = None

    def __post_init__(self):
        """Validate array shapes."""
        if len(self.x) != len(self.y):
            msg = f"x and y must have same length (got {len(self.x)} and {len(self.y)})"
            raise ValueError(msg)
        if self.values is not None and len(self.values) != len(self.x):
            msg = f"values must have same length as x (got {len(self.values)} and {len(self.x)})"
            raise ValueError(msg)

    @classmethod
    def from_arrays(
        cls,
        x: list | np.ndarray,
        y: list | np.ndarray,
        values: list | np.ndarray | None = None,
    ) -> DataSeries:
        """Create DataSeries from array-like data.

        Parameters
        ----------
        x : array-like
            X-coordinates.
        y : array-like
            Y-coordinates.
        values : array-like or None
            Optional metric values for coloring.

        Returns
        -------
        DataSeries
            New DataSeries instance.

        Examples
        --------
        >>> data = DataSeries.from_arrays(
        ...     x=[22, 24, 26, 28],
        ...     y=[50, 60, 70, 80],
        ...     values=[-0.5, 0.0, 0.3, 0.8],
        ... )
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        values_arr = np.asarray(values, dtype=float) if values is not None else None

        return cls(x=x_arr, y=y_arr, values=values_arr)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str | None = None,
    ) -> DataSeries:
        """Create DataSeries from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Source DataFrame.
        x_col : str
            Column name for x-coordinates.
        y_col : str
            Column name for y-coordinates.
        value_col : str or None
            Optional column name for values.

        Returns
        -------
        DataSeries
            New DataSeries instance.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"temp": [22, 24], "rh": [50, 60]})
        >>> data = DataSeries.from_dataframe(df, x_col="temp", y_col="rh")
        """
        x_arr = df[x_col].to_numpy(dtype=float)
        y_arr = df[y_col].to_numpy(dtype=float)
        values_arr = df[value_col].to_numpy(dtype=float) if value_col else None

        return cls(x=x_arr, y=y_arr, values=values_arr)

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.x)

    def compute_categories(self, scene: BaseScene) -> np.ndarray:
        """Compute category for each data point using the scene.

        Parameters
        ----------
        scene : BaseScene
            Scene to use for classification.

        Returns
        -------
        np.ndarray
            Array of category labels (strings).

        Examples
        --------
        >>> categories = data.compute_categories(scene)
        >>> print(categories)  # ['Neutral', 'Warm', 'Neutral', ...]
        """
        categories = []
        for xi, yi in zip(self.x, self.y):
            cat = scene.get_category(float(xi), float(yi))
            categories.append(cat)
        return np.array(categories, dtype=object)

    def compute_category_counts(self, scene: BaseScene) -> dict[str, int]:
        """Compute counts for each category.

        Parameters
        ----------
        scene : BaseScene
            Scene to use for classification.

        Returns
        -------
        dict[str, int]
            Dictionary mapping category labels to counts.
        """
        categories = self.compute_categories(scene)
        unique, counts = np.unique(categories, return_counts=True)
        return dict(zip(unique, counts))

    def compute_category_percentages(self, scene: BaseScene) -> dict[str, float]:
        """Compute percentages for each category.

        Parameters
        ----------
        scene : BaseScene
            Scene to use for classification.

        Returns
        -------
        dict[str, float]
            Dictionary mapping category labels to percentages.
        """
        counts = self.compute_category_counts(scene)
        total = len(self)
        if total == 0:
            return {}
        return {cat: count / total * 100 for cat, count in counts.items()}
