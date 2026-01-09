"""Plot facade class for thermal comfort visualizations.

This module provides the Plot class, the primary interface for creating
thermal comfort plots in pythermalcomfort.
"""

from __future__ import annotations

from contextlib import nullcontext as _nullcontext
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.data_series import DataSeries
from pythermalcomfort.plots.scenes.adaptive_scene import AdaptiveScene
from pythermalcomfort.plots.scenes.range_scene import RangeScene
from pythermalcomfort.plots.style import Style
from pythermalcomfort.plots.summary import SummaryRenderer

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from matplotlib.figure import Figure

    from pythermalcomfort.plots.presets import Preset
    from pythermalcomfort.plots.scenes.base import BaseScene


class Plot:
    """Main facade for creating thermal comfort visualizations.

    Plot composes Scene (context), DataSeries (overlay), and Style (aesthetics)
    to create complete visualizations. After creation:

    - Scene and DataSeries are FROZEN (immutable)
    - Only Style can be modified via the `.style` property

    Use factory methods like `Plot.range()` or `Plot.adaptive()` to create instances.

    Examples
    --------
    Simple range plot:

    >>> from pythermalcomfort.models import utci
    >>> from pythermalcomfort.plots import Plot
    >>> plot = Plot.range(utci, fixed_params={"v": 1.0, "tr": 25})
    >>> plot.style.title = "UTCI Thermal Stress"
    >>> fig, ax = plot.render()

    Adaptive comfort with data overlay:

    >>> plot = Plot.adaptive()
    >>> plot = plot.add_data(x=t_outdoor, y=t_operative)
    >>> plot.style.show_summary = True
    >>> fig, axes = plot.render()

    Custom styling:

    >>> from pythermalcomfort.plots import Style
    >>> style = Style(cmap="RdYlBu_r", band_alpha=0.6)
    >>> plot = Plot.range(utci, style=style)
    """

    def __init__(
        self,
        scene: BaseScene,
        data_series: DataSeries | None = None,
        style: Style | None = None,
    ):
        """Initialize Plot.

        Prefer factory methods (Plot.range, Plot.adaptive) over direct construction.

        Parameters
        ----------
        scene : BaseScene
            The scene providing context/background.
        data_series : DataSeries or None
            Optional overlay data.
        style : Style or None
            Style configuration. Creates default if None.
        """
        self._scene = scene
        self._data_series = data_series
        self._style = style if style is not None else Style()
        self._fig: Figure | None = None
        self._axes: list[plt.Axes] = []
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

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def range(
        cls,
        model_func: Callable[..., Any],
        fixed_params: dict[str, Any] | None = None,
        *,
        thresholds: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        x_param: str = "tdb",
        y_param: str = "rh",
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        y_step: float | None = None,
        metric_attr: str | None = None,
        preset: Preset | None = None,
        style: Style | None = None,
    ) -> Plot:
        """Create a range plot for threshold-based region visualization.

        This factory creates plots showing comfort/stress regions based on
        model thresholds. Users can specify which model parameters map to
        x and y axes.

        Parameters
        ----------
        model_func : Callable
            A pythermalcomfort model function (e.g., utci, pmv_ppd_iso).
        fixed_params : dict or None
            Model parameters held constant.
        thresholds : Sequence[float] or None
            Threshold values for region boundaries. If None, uses preset.
        labels : Sequence[str] or None
            Labels for each region. If None, uses preset or auto-generates.
        x_param : str
            Model parameter name for x-axis (default: "tdb").
        y_param : str
            Model parameter name for y-axis (default: "rh").
        x_range : tuple or None
            (min, max) for x-axis. If None, uses preset default.
        y_range : tuple or None
            (min, max) for y-axis. If None, uses preset default.
        y_step : float or None
            Step size for y-axis grid. If None, auto-computed.
        metric_attr : str or None
            Attribute name to extract from model result.
        preset : Preset or None
            Explicit preset to use. If None, auto-detected from model.
        style : Style or None
            Custom style. Creates default if None.

        Returns
        -------
        Plot
            A configured Plot ready to render.

        Examples
        --------
        >>> from pythermalcomfort.models import utci
        >>> plot = Plot.range(utci, fixed_params={"v": 1.0, "tr": 25})
        >>> fig, ax = plot.render()

        Custom x/y parameters:

        >>> plot = Plot.range(
        ...     pmv_ppd_iso,
        ...     x_param="tdb",
        ...     y_param="vr",
        ...     fixed_params={"tr": 25, "rh": 50, "met": 1.2, "clo": 0.5},
        ...     y_range=(0.0, 2.0),
        ... )
        """
        scene = RangeScene.create(
            model_func=model_func,
            fixed_params=fixed_params,
            thresholds=thresholds,
            labels=labels,
            x_param=x_param,
            y_param=y_param,
            x_range=x_range,
            y_range=y_range,
            y_step=y_step,
            metric_attr=metric_attr,
            preset=preset,
        )

        plot_style = style if style is not None else Style()

        # Set axis labels from scene if not explicitly set
        if plot_style.xlabel is None:
            plot_style.xlabel = scene.xlabel
        if plot_style.ylabel is None:
            plot_style.ylabel = scene.ylabel

        return cls(scene=scene, style=plot_style)

    @classmethod
    def adaptive(
        cls,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        show_80_band: bool = True,
        show_90_band: bool = True,
        style: Style | None = None,
    ) -> Plot:
        """Create an adaptive comfort model plot.

        Creates a plot for the ASHRAE 55 Adaptive Comfort Model with
        fixed axes (outdoor temperature vs operative temperature).

        Parameters
        ----------
        x_range : tuple or None
            Outdoor temperature range. Default: (10, 33.5).
        y_range : tuple or None
            Operative temperature range. Default: (14, 36).
        show_80_band : bool
            Whether to show 80% acceptability band.
        show_90_band : bool
            Whether to show 90% acceptability band.
        style : Style or None
            Custom style. Creates default if None.

        Returns
        -------
        Plot
            A configured Plot ready to render.

        Examples
        --------
        >>> plot = Plot.adaptive()
        >>> plot = plot.add_data(x=t_outdoor, y=t_operative)
        >>> fig, axes = plot.render()
        """
        scene = AdaptiveScene.create(
            x_range=x_range,
            y_range=y_range,
            show_80_band=show_80_band,
            show_90_band=show_90_band,
        )

        plot_style = style if style is not None else Style()

        # Set axis labels
        if plot_style.xlabel is None:
            plot_style.xlabel = "Outdoor Temperature [°C]"
        if plot_style.ylabel is None:
            plot_style.ylabel = "Operative Temperature [°C]"

        return cls(scene=scene, style=plot_style)

    # =========================================================================
    # Data Methods
    # =========================================================================

    def add_data(
        self,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray | None = None,
    ) -> Plot:
        """Add data series overlay from arrays.

        Returns a new Plot instance with the data added (immutable pattern).

        Parameters
        ----------
        x : array-like
            X-coordinates of data points.
        y : array-like
            Y-coordinates of data points.
        values : array-like or None
            Optional metric values for coloring.

        Returns
        -------
        Plot
            New Plot instance with data added.

        Examples
        --------
        >>> plot = Plot.range(utci, fixed_params={...})
        >>> plot_with_data = plot.add_data(x=temps, y=rh_values)
        >>> fig, axes = plot_with_data.render()
        """
        data_series = DataSeries.from_arrays(x=x, y=y, values=values)
        return Plot(
            scene=self._scene,
            data_series=data_series,
            style=self._style.copy(),
        )

    def add_data_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        value_col: str | None = None,
    ) -> Plot:
        """Add data series overlay from DataFrame.

        Returns a new Plot instance with the data added (immutable pattern).

        Parameters
        ----------
        df : pandas.DataFrame
            Source DataFrame.
        x_col : str
            Column name for x-coordinates.
        y_col : str
            Column name for y-coordinates.
        value_col : str or None
            Optional column for values.

        Returns
        -------
        Plot
            New Plot instance with data added.

        Examples
        --------
        >>> plot = Plot.adaptive()
        >>> plot = plot.add_data_from_df(df, x_col="t_out", y_col="t_op")
        """
        data_series = DataSeries.from_dataframe(
            df=df, x_col=x_col, y_col=y_col, value_col=value_col
        )
        return Plot(
            scene=self._scene,
            data_series=data_series,
            style=self._style.copy(),
        )

    # =========================================================================
    # Rendering
    # =========================================================================

    def render(self, ax: plt.Axes | None = None) -> tuple[Figure, plt.Axes]:
        """Render the plot to a matplotlib figure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Existing axes for main plot. If None, creates new figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure.
        ax : matplotlib.axes.Axes
            The main axes.

        Examples
        --------
        >>> fig, ax = plot.render()
        >>> plt.savefig("my_plot.png")
        >>> plt.show()
        """
        # Determine if we need summary bar
        show_summary = (
            self._style.show_summary
            and self._data_series is not None
            and len(self._data_series) > 0
        )

        # Use base_style as context so our explicit settings can override
        style_context = (
            plt.style.context(self._style.base_style)
            if self._style.base_style
            else _nullcontext()
        )

        with style_context:
            # Create figure/axes
            if ax is None:
                fig, main_ax = plt.subplots(
                    figsize=self._style.figsize,
                    dpi=self._style.dpi,
                )
            else:
                fig = ax.get_figure()
                main_ax = ax
            self._axes = [main_ax]
            self._fig = fig

            # 1. Render scene
            scene_artists = self._scene.render(main_ax, self._style)
            self._artists["scene"] = scene_artists

            # 2. Render data series if present
            if self._data_series is not None:
                self._render_scatter(main_ax)

            # 3. Apply style to main axes
            self._apply_style(main_ax)

            # 4. Render summary bar if needed (as inset on main axes)
            if show_summary:
                renderer = SummaryRenderer()
                summary_artists = renderer.render(
                    main_ax, self._data_series, self._scene, self._style
                )
                self._artists["summary"] = summary_artists

            # 5. Add fixed params annotation if enabled
            if self._style.show_fixed_params:
                self._add_fixed_params_annotation(main_ax)

        return fig, main_ax

    def _render_scatter(self, ax: plt.Axes) -> None:
        """Render scatter points on the axes."""
        scatter = ax.scatter(
            self._data_series.x,
            self._data_series.y,
            s=self._style.scatter_size,
            c=self._style.scatter_color,
            alpha=self._style.scatter_alpha,
            edgecolors=self._style.scatter_edgecolor,
            linewidths=self._style.scatter_linewidth,
            marker=self._style.scatter_marker,
            zorder=10,  # On top of regions
            label="Observations",
        )
        self._artists["scatter"] = scatter

    def _apply_style(self, ax: plt.Axes) -> None:
        """Apply style settings to the axes."""
        style = self._style

        # Labels
        if style.xlabel:
            ax.set_xlabel(style.xlabel, fontsize=style.font_sizes.get("label", 12))
        if style.ylabel:
            ax.set_ylabel(style.ylabel, fontsize=style.font_sizes.get("label", 12))
        if style.title:
            ax.set_title(style.title, fontsize=style.font_sizes.get("title", 14), loc=style.title_alignment)
        if style.title_x_position is not None:
            ax.title.set_position((style.title_x_position, style.title_y_position))

        # Tick labels
        ax.tick_params(labelsize=style.font_sizes.get("tick", 10))

        # Axis ranges (scene already sets these, but style can override)
        ax.set_xlim(self._scene.get_x_range())
        ax.set_ylim(self._scene.get_y_range())

        # Grid
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha, zorder=-1)

    def _add_fixed_params_annotation(self, ax: plt.Axes) -> None:
        """Add fixed parameters annotation to the plot."""
        text = self._scene.get_fixed_params_text()
        if not text:
            return

        ax.text(
            1.04, 0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=self._style.font_sizes.get("tick", 10),
            color="gray",
        )

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
