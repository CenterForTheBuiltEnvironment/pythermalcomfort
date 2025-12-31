"""SummaryRenderer - Category distribution visualization.

This module provides the SummaryRenderer class for rendering a stacked
horizontal bar showing the distribution of data points across categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

if TYPE_CHECKING:
    from pythermalcomfort.plots.data_series import DataSeries
    from pythermalcomfort.plots.scenes.base import BaseScene
    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class SummaryRenderer:
    """Renders category distribution as a stacked horizontal bar.

    Creates a compact stacked bar showing the percentage distribution
    of data points across categories, positioned below the legend.

    Examples
    --------
    >>> renderer = SummaryRenderer()
    >>> artists = renderer.render(ax, data_series, scene, style)
    """

    def render(
        self,
        ax: plt.Axes,
        data_series: DataSeries,
        scene: BaseScene,
        style: Style,
    ) -> dict[str, Any]:
        """Render stacked horizontal bar showing category distribution.

        Creates an inset axes positioned at the bottom-right of the plot
        (outside the main axes) showing a stacked bar with the same colors
        as the legend.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The main axes (used for positioning the inset).
        data_series : DataSeries
            Data points to summarize.
        scene : BaseScene
            Scene for category classification.
        style : Style
            Style configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'inset_ax', 'bars', and 'texts' artists.
        """
        # Get category labels and colors from scene
        labels = scene.get_labels()
        colors = scene.get_colors(style)

        # Compute percentages for each category
        percentages_dict = data_series.compute_category_percentages(scene)

        # Build ordered lists matching scene categories
        percentages = []
        for label in labels:
            percentages.append(percentages_dict.get(label, 0.0))

        # Create inset axes for the stacked bar
        # Position at bottom-right, outside the main plot
        inset_ax = inset_axes(
            ax,
            width=style.summary_bar_width,  # e.g., "30%" or "1.5"
            height=style.summary_bar_height,  # e.g., "3%" or "0.15"
            loc="lower right",
            bbox_to_anchor=(0.3, style.summary_y_position, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        # Render stacked horizontal bar
        bar_artists = []
        text_artists = []
        left = 0.0

        for i, (pct, color) in enumerate(zip(percentages, colors)):
            if pct > 0:
                # Draw bar segment
                bar = inset_ax.barh(
                    0,
                    pct,
                    left=left,
                    height=1.0,
                    color=color[:3] if len(color) == 4 else color,
                    alpha=style.band_alpha,
                    edgecolor="white",
                    linewidth=0.5,
                )
                bar_artists.extend(bar)

                # Add percentage text if segment is wide enough
                if pct >= style.summary_min_pct_for_text:
                    text = inset_ax.text(
                        left + pct / 2,
                        0,
                        f"{pct:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=style.font_sizes.get("summary", 8),
                        fontweight="bold",
                        color="white" if self._is_dark(color) else "black",
                    )
                    text_artists.append(text)

                left += pct

        # Style the inset axes
        inset_ax.set_xlim(0, 100)
        inset_ax.set_ylim(-0.5, 0.5)
        inset_ax.axis("off")

        # Add a subtle border
        for spine in inset_ax.spines.values():
            spine.set_visible(False)

        return {
            "inset_ax": inset_ax,
            "bars": bar_artists,
            "texts": text_artists,
        }

    def _is_dark(self, color) -> bool:
        """Check if a color is dark (for text contrast)."""
        # Handle RGBA tuple
        if hasattr(color, "__len__") and len(color) >= 3:
            r, g, b = color[0], color[1], color[2]
            # Perceived luminance formula
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            return luminance < 0.5
        return False

    def get_title(self, data_series: DataSeries) -> str:
        """Generate title for summary chart.

        Parameters
        ----------
        data_series : DataSeries
            Data being summarized.

        Returns
        -------
        str
            Title text.
        """
        n = len(data_series)
        return f"Distribution (n={n})"
