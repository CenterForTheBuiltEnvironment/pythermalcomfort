"""Style configuration for plots.

This module provides the Style class for configuring plot appearance.
Style is the ONLY mutable component after plot creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.colors import Colormap


@dataclass
class Style:
    """User-adjustable plot styling.

    This is the only mutable component after plot creation. All fields have
    sensible defaults, allowing minimal configuration for common use cases.

    Parameters
    ----------
    figsize : tuple[float, float]
        Figure size in inches (width, height). Wider default for summary subplot.
    dpi : int
        Figure resolution in dots per inch.
    cmap : str or Colormap
        Colormap for region bands. Ignored if band_colors is provided.
    band_colors : Sequence[str] or None
        Explicit colors for each region. Overrides cmap if provided.
        Must have length equal to len(thresholds) + 1.
    band_alpha : float
        Opacity for filled region bands (0-1).
    line_color : str
        Color for threshold boundary lines.
    line_width : float
        Width of threshold boundary lines.
    xlabel : str or None
        X-axis label. If None, uses preset default.
    ylabel : str or None
        Y-axis label. If None, uses preset default.
    title : str or None
        Plot title.
    show_legend : bool
        Whether to display the legend.
    legend_loc : str
        Legend location (matplotlib location string).
    legend_ncol : int
        Number of columns in the legend.
    info_panel_width : float
        Width of the info panel as fraction of figure width (0-1).
    info_panel_background : str or None
        Background color for info panel. None for transparent.
    info_panel_alpha : float
        Opacity of the info panel background (0-1).
    show_grid : bool
        Whether to display grid lines.
    grid_alpha : float
        Opacity of grid lines (0-1).
    base_style : str
        Base matplotlib style to apply (e.g., 'seaborn-v0_8-whitegrid').
        Explicit settings in this Style instance will override the base style.
        Default is empty string (no base style). Note: Using base styles can
        cause display issues in Jupyter when rendering multiple plots.
    show_fixed_params : bool
        Whether to display fixed parameters as annotation/subtitle.
    fixed_params_loc : str
        Location for fixed params: "subtitle" or "annotation".
    scatter_size : float
        Size of scatter points.
    scatter_color : str
        Default color for scatter points (when not colored by value).
    scatter_alpha : float
        Opacity of scatter points (0-1).
    scatter_edgecolor : str
        Edge color of scatter points.
    scatter_linewidth : float
        Width of scatter point edges.
    scatter_marker : str
        Marker style for scatter points.
    show_summary : bool
        Whether to show summary distribution bar when data is added.
    summary_bar_y : float
        Vertical position of summary bar in info panel (0=bottom, 1=top).
    summary_bar_height : float
        Height of summary bar in axes coordinates.
    summary_bar_left : float
        Left margin of summary bar in axes coordinates.
    summary_bar_width : float
        Width of summary bar in axes coordinates (0-1).
    summary_min_pct_for_text : float
        Minimum percentage threshold to display text on a segment.

    Examples
    --------
    >>> style = Style(title="PMV Comfort Zones", band_alpha=0.6)
    >>> style.cmap = "viridis"  # Modify after creation
    >>> style.show_summary = True  # Enable summary subplot
    """

    # Figure settings
    figsize: tuple[float, float] = (10, 5)  # Wider for summary subplot
    dpi: int = 300

    # Region band settings
    cmap: str | Colormap = "coolwarm"
    band_colors: Sequence[str] | None = None
    band_alpha: float = 0.75

    # Threshold line settings
    line_color: str = "black"
    line_width: float = 1.0

    # Labels
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    title_x_position: float = 0.0  # Left-align by default
    title_y_position: float = 1.1  # Slightly above the plot
    title_alignment: str = "left"  # Align title to the left

    # Legend settings
    show_legend: bool = True
    legend_loc: str = "center left"  # Anchor point of the legend box
    legend_bbox: tuple[float, float] = (1.02, 0.5)  # Position outside right edge
    legend_ncol: int = 1
    legend_alpha: float = 0.0

    # Info panel settings
    info_panel_width: float = 0.25  # fraction of figure width (0-1)
    info_panel_background: str | None = "#f5f5f5"  # None for transparent
    info_panel_alpha: float = 0.3

    # Grid settings
    show_grid: bool = False
    grid_alpha: float = 0.3

    # Base matplotlib style (our explicit settings override this)
    # Default empty string for best Jupyter compatibility
    base_style: str = ""

    # Font settings
    font_family: str = "Arial"
    font_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "title": 14,
            "label": 12,
            "tick": 10,
            "legend": 10,
            "summary": 8,
        }
    )

    # Fixed params annotation settings
    show_fixed_params: bool = True

    # DataSeries scatter settings
    scatter_size: float = 30.0
    scatter_color: str = "#ea536e"  # Match earlier examples
    scatter_alpha: float = 0.8
    scatter_edgecolor: str = "white"
    scatter_linewidth: float = 0.5
    scatter_marker: str = "o"

    # Summary bar settings (stacked horizontal bar at bottom-right)
    show_summary: bool = False  # Off by default, user enables when adding data
    summary_bar_y: float = 0.08  # Vertical position (0=bottom, 1=top)
    summary_bar_height: float = 0.04  # Height in axes coordinates
    summary_bar_left: float = 0.05  # Left margin
    summary_bar_width: float = 0.9  # Width in axes coordinates (0-1)
    summary_min_pct_for_text: float = 8.0  # Min % to show text on segment

    def copy(self) -> Style:
        """Return a shallow copy of this Style instance.

        Returns
        -------
        Style
            A new Style instance with the same settings.
        """
        return Style(
            figsize=self.figsize,
            dpi=self.dpi,
            cmap=self.cmap,
            band_colors=self.band_colors,
            band_alpha=self.band_alpha,
            line_color=self.line_color,
            line_width=self.line_width,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            title=self.title,
            title_x_position=self.title_x_position,
            title_y_position=self.title_y_position,
            title_alignment=self.title_alignment,
            show_legend=self.show_legend,
            legend_loc=self.legend_loc,
            legend_bbox=self.legend_bbox,
            legend_ncol=self.legend_ncol,
            legend_alpha=self.legend_alpha,
            info_panel_width=self.info_panel_width,
            info_panel_background=self.info_panel_background,
            info_panel_alpha=self.info_panel_alpha,
            show_grid=self.show_grid,
            grid_alpha=self.grid_alpha,
            base_style=self.base_style,
            font_family=self.font_family,
            font_sizes=dict(self.font_sizes),
            show_fixed_params=self.show_fixed_params,
            scatter_size=self.scatter_size,
            scatter_color=self.scatter_color,
            scatter_alpha=self.scatter_alpha,
            scatter_edgecolor=self.scatter_edgecolor,
            scatter_linewidth=self.scatter_linewidth,
            scatter_marker=self.scatter_marker,
            show_summary=self.show_summary,
            summary_bar_y=self.summary_bar_y,
            summary_bar_height=self.summary_bar_height,
            summary_bar_left=self.summary_bar_left,
            summary_bar_width=self.summary_bar_width,
            summary_min_pct_for_text=self.summary_min_pct_for_text,
        )
