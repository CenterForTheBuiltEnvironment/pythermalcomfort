"""Plot facade class for thermal comfort visualizations.

This module provides the Plot class, the primary interface for creating
thermal comfort plots in pythermalcomfort.
"""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.presets import get_preset
from pythermalcomfort.plots.ranges import Ranges
from pythermalcomfort.plots.regions import Regions
from pythermalcomfort.plots.style import Style

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure

    from pythermalcomfort.plots.presets import Preset


class Plot:
    """Main facade for creating thermal comfort visualizations.

    This class composes Ranges (data), Regions (context), and Style
    (aesthetics) to create plots. After creation:

    - Ranges and Regions are FROZEN (immutable)
    - Only Style can be modified via the `.style` property

    Use factory methods like `Plot.ranges()` to create instances.

    Examples
    --------
    Simple usage with predefined preset:

    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots import Plot
    >>> plot = Plot.ranges(
    ...     pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
    ... )
    >>> plot.style.title = "PMV Comfort Regions"
    >>> fig, ax = plot.render()

    Custom thresholds and styling:

    >>> from pythermalcomfort.plots import Style
    >>> custom_style = Style(cmap="RdYlBu_r", title="Custom Plot")
    >>> plot = Plot.ranges(
    ...     pmv_ppd_iso,
    ...     fixed_params={...},
    ...     thresholds=[-1, 0, 1],
    ...     style=custom_style,
    ... )
    """

    def __init__(
        self,
        ranges: Ranges | None,
        regions: Regions | None,
        style: Style,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
    ):
        """Initialize Plot. Prefer factory methods over direct construction."""
        self._ranges = ranges
        self._regions = regions
        self._style = style
        self._x_range = x_range
        self._y_range = y_range
        self._fig: Figure | None = None
        self._ax: plt.Axes | None = None
        self._artists: dict[str, Any] = {}

    @property
    def style(self) -> Style:
        """Access the (mutable) style configuration.

        This is the only component that can be modified after creation.
        """
        return self._style

    @style.setter
    def style(self, value: Style) -> None:
        """Replace the style configuration."""
        self._style = value

    @classmethod
    def ranges(
        cls,
        model_func: Callable[..., Any],
        fixed_params: dict[str, Any],
        *,
        thresholds: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        y_step: float | None = None,
        metric_attr: str | None = None,
        preset: Preset | None = None,
        style: Style | None = None,
        x_scan_step: float = 1.0,
        smooth_sigma: float = 0.8,
    ) -> Plot:
        """Create a ranges plot for threshold-based region visualization.

        This is the primary factory method for creating plots that show
        comfort/stress regions based on model thresholds.

        Parameters
        ----------
        model_func : Callable
            A pythermalcomfort model function (e.g., pmv_ppd_iso, utci).
        fixed_params : dict
            Model parameters held constant (e.g., {"tr": 25, "met": 1.2}).
        thresholds : Sequence[float], optional
            Threshold values for region boundaries. If None, uses preset.
        labels : Sequence[str], optional
            Labels for each region. If None, uses preset or auto-generates.
        x_range : tuple[float, float], optional
            (min, max) for x-axis. If None, uses preset default.
        y_range : tuple[float, float], optional
            (min, max) for y-axis. If None, uses preset default.
        y_step : float, optional
            Step size for y-axis grid. Defaults to ~50 steps.
        metric_attr : str, optional
            Attribute name to extract from model result.
        preset : Preset, optional
            Explicit preset to use. If None, auto-detected from model.
        style : Style, optional
            Custom style. If None, creates default with preset labels.
        x_scan_step : float, default 1.0
            Step size for x-axis threshold scanning.
        smooth_sigma : float, default 0.8
            Gaussian smoothing for curves. Set 0 to disable.

        Returns
        -------
        Plot
            A configured Plot ready to render.

        Raises
        ------
        ValueError
            If thresholds are not provided and no preset exists.

        Examples
        --------
        Using preset defaults:

        >>> from pythermalcomfort.models import pmv_ppd_iso
        >>> plot = Plot.ranges(
        ...     pmv_ppd_iso,
        ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
        ... )
        >>> fig, ax = plot.render()

        Custom thresholds:

        >>> plot = Plot.ranges(
        ...     pmv_ppd_iso,
        ...     fixed_params={...},
        ...     thresholds=[-1, 0, 1],
        ...     labels=["Cold", "Neutral", "Warm", "Hot"],
        ... )
        """
        # Get preset if not provided
        if preset is None:
            preset = get_preset(model_func)

        # Resolve thresholds
        if thresholds is None:
            if preset is None:
                msg = (
                    "thresholds required when no preset exists for this model. "
                    "Provide thresholds explicitly."
                )
                raise ValueError(msg)
            thresholds = preset.thresholds

        # Resolve labels
        if labels is None and preset is not None:
            labels = preset.labels

        # Resolve ranges
        if x_range is None:
            x_range = preset.x_range if preset else (10.0, 36.0)
        if y_range is None:
            y_range = preset.y_range if preset else (0.0, 100.0)

        # Resolve y_step
        if y_step is None:
            y_step = (y_range[1] - y_range[0]) / 50

        # Resolve metric_attr
        if metric_attr is None and preset is not None:
            metric_attr = preset.metric_attr

        # Get xy_mapper
        if preset is not None:
            xy_mapper = preset.get_xy_mapper()
        else:
            from pythermalcomfort.plots.utils import mapper_tdb_rh

            xy_mapper = mapper_tdb_rh

        # Build y values grid
        y_values = np.arange(y_range[0], y_range[1] + 1e-9, y_step)

        # Compute ranges data
        ranges_data = Ranges.from_model(
            model_func=model_func,
            xy_to_kwargs=xy_mapper,
            fixed_params=fixed_params,
            thresholds=list(thresholds),
            x_bounds=x_range,
            y_values=y_values,
            metric_attr=metric_attr,
            x_scan_step=x_scan_step,
            smooth_sigma=smooth_sigma,
        )

        # Build regions context
        regions = Regions(ranges=ranges_data, labels=labels)

        # Build style
        if style is None:
            style = Style()

        # Set default labels from preset
        if style.xlabel is None and preset is not None:
            style.xlabel = preset.xlabel
        if style.ylabel is None and preset is not None:
            style.ylabel = preset.ylabel

        return cls(
            ranges=ranges_data,
            regions=regions,
            style=style,
            x_range=x_range,
            y_range=y_range,
        )

    def render(self, ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        """Render the plot to a matplotlib figure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw on. If None, creates new figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure.
        ax : matplotlib.axes.Axes
            The matplotlib Axes with the rendered plot.

        Examples
        --------
        >>> fig, ax = plot.render()
        >>> plt.savefig("my_plot.png")
        >>> plt.show()
        """
        # Use base_style as context so our explicit settings can override
        style_context = (
            plt.style.context(self._style.base_style)
            if self._style.base_style
            else _nullcontext()
        )

        with style_context:
            # Create figure if needed
            if ax is None:
                fig, ax = plt.subplots(
                    figsize=self._style.figsize,
                    dpi=self._style.dpi,
                )
            else:
                fig = ax.get_figure()

            self._fig = fig
            self._ax = ax

            # Render regions if present
            if self._regions is not None:
                region_artists = self._regions.render(ax, self._style)
                self._artists.update(region_artists)

            # Apply explicit style settings (these override base_style)
            self._apply_style(ax)

        return fig, ax

    def _apply_style(self, ax: plt.Axes) -> None:
        """Apply style settings to the axes."""
        style = self._style

        # Labels
        if style.xlabel:
            ax.set_xlabel(style.xlabel, fontsize=style.font_sizes.get("label", 12))
        if style.ylabel:
            ax.set_ylabel(style.ylabel, fontsize=style.font_sizes.get("label", 12))
        if style.title:
            ax.set_title(style.title, fontsize=style.font_sizes.get("title", 14))

        # Tick labels
        ax.tick_params(labelsize=style.font_sizes.get("tick", 10))

        # Axis ranges
        if self._x_range:
            ax.set_xlim(self._x_range)
        if self._y_range:
            ax.set_ylim(self._y_range)

        # Grid
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha, zorder=-1)

    def show(self) -> None:
        """Render and display the plot."""
        if self._fig is None:
            self.render()
        plt.show()

    def save(self, filename: str, **kwargs) -> None:
        """Render and save the plot to a file.

        Parameters
        ----------
        filename : str
            Output filename (e.g., "plot.png", "plot.pdf").
        **kwargs
            Additional arguments passed to matplotlib.savefig().
        """
        if self._fig is None:
            self.render()
        self._fig.savefig(filename, bbox_inches="tight", **kwargs)
