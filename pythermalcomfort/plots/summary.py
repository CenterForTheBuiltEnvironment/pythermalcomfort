"""SummaryRenderer - Category distribution visualization.

This module provides the SummaryRenderer class for rendering a stacked
horizontal bar showing the distribution of data points across categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

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

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on (typically the info panel).
        data_series : DataSeries
            Data points to summarize.
        scene : BaseScene
            Scene for category classification.
        style : Style
            Style configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'bars' and 'texts' artists.
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

        # Bar dimensions and position (from style settings)
        bar_y = style.summary_bar_y
        bar_height = style.summary_bar_height
        bar_left = style.summary_bar_left
        bar_width = style.summary_bar_width

        # Render stacked horizontal bar
        bar_artists = []
        text_artists = []
        x_pos = bar_left

        for i, (pct, color) in enumerate(zip(percentages, colors)):
            if pct > 0:
                segment_width = (pct / 100.0) * bar_width

                # Draw bar segment
                rect = plt.Rectangle(
                    (x_pos, bar_y),
                    segment_width,
                    bar_height,
                    facecolor=color[:3] if len(color) == 4 else color,
                    alpha=style.band_alpha,
                    edgecolor="white",
                    linewidth=0.5,
                    transform=ax.transAxes,
                    clip_on=False,
                )
                ax.add_patch(rect)
                bar_artists.append(rect)

                # Add percentage text if segment is wide enough
                if pct >= style.summary_min_pct_for_text:
                    text = ax.text(
                        x_pos + segment_width / 2,
                        bar_y + bar_height / 2,
                        f"{pct:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=style.font_sizes.get("summary", 8),
                        fontweight="bold",
                        color="white" if self._is_dark(color) else "black",
                        transform=ax.transAxes,
                    )
                    text_artists.append(text)

                x_pos += segment_width

        return {
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
