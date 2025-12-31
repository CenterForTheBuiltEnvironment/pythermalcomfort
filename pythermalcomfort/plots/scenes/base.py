"""Base class for all Scene types.

Scenes are frozen (immutable) objects that represent the context/background
of a thermal comfort plot. They compute and render threshold-based regions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import matplotlib.pyplot as plt

    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class BaseScene(ABC):
    """Abstract base class for all Scene types.

    A Scene represents the context/background of a plot, typically showing
    threshold-based regions computed from thermal comfort models. All Scene
    subclasses are frozen (immutable) after creation.

    Subclasses must implement:
    - render(): Draw the scene on matplotlib axes
    - get_category(): Classify a data point into a category
    - get_labels(): Return category labels for the scene

    Attributes
    ----------
    thresholds : Sequence[float]
        Threshold values that define region boundaries.
    labels : Sequence[str] or None
        Labels for each region. If None, auto-generated from thresholds.

    Examples
    --------
    >>> scene = RangeScene.create(utci, fixed_params={...})
    >>> artists = scene.render(ax, style)
    >>> category = scene.get_category(25.0, 50.0)
    """

    # Note: All fields have defaults to allow inheritance in frozen dataclasses
    thresholds: Sequence[float] = ()
    labels: Sequence[str] | None = None

    @abstractmethod
    def render(self, ax: plt.Axes, style: Style) -> dict[str, Any]:
        """Render the scene on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on.
        style : Style
            Style configuration for rendering.

        Returns
        -------
        dict[str, Any]
            Dictionary of matplotlib artists created during rendering.
            Typically includes 'bands', 'curves', 'legend' keys.
        """
        ...

    @abstractmethod
    def get_category(self, x: float, y: float) -> str:
        """Get the category label for a data point.

        Used by DataSeries to classify points for the summary distribution.

        Parameters
        ----------
        x : float
            X-coordinate of the point.
        y : float
            Y-coordinate of the point.

        Returns
        -------
        str
            Category label for this point.
        """
        ...

    @abstractmethod
    def get_labels(self) -> list[str]:
        """Get the category labels for this scene.

        Returns
        -------
        list[str]
            List of category labels (one per region).
        """
        ...

    @abstractmethod
    def get_colors(self, style: Style) -> list:
        """Get the colors for each category/region.

        Parameters
        ----------
        style : Style
            Style configuration (for colormap).

        Returns
        -------
        list
            List of colors (one per region).
        """
        ...

    @abstractmethod
    def get_x_range(self) -> tuple[float, float]:
        """Get the x-axis range for this scene.

        Returns
        -------
        tuple[float, float]
            (min, max) values for x-axis.
        """
        ...

    @abstractmethod
    def get_y_range(self) -> tuple[float, float]:
        """Get the y-axis range for this scene.

        Returns
        -------
        tuple[float, float]
            (min, max) values for y-axis.
        """
        ...

    def get_fixed_params(self) -> dict[str, Any]:
        """Get the fixed parameters for this scene.

        Returns
        -------
        dict[str, Any]
            Dictionary of fixed parameters. Empty dict if none.
        """
        return {}

    def get_fixed_params_text(self) -> str:
        """Format fixed parameters for display as annotation.

        Returns
        -------
        str
            Formatted string of fixed params, e.g., "tr: 25 | met: 1.2"
        """
        params = self.get_fixed_params()
        if not params:
            return ""
        parts = [f"{k}: {v}" for k, v in params.items()]
        return " | ".join(parts)
