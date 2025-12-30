"""Regions context layer for rendering threshold-based regions.

This module provides the Regions class, a frozen dataclass that knows
how to render Ranges data as filled color bands on a matplotlib Axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pythermalcomfort.plots.ranges import Ranges
    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class Regions:
    """Context layer for rendering Ranges as filled color bands.

    This is a frozen (immutable) class that takes Ranges data and
    renders it as colored region bands with threshold boundary lines.

    Parameters
    ----------
    ranges : Ranges
        Computed threshold curves to render.
    labels : Sequence[str] or None, optional
        Labels for each region. If None, auto-generated from thresholds.
        Must have length equal to len(thresholds) + 1 if provided.

    Notes
    -----
    This class is frozen (immutable) after creation.

    Examples
    --------
    >>> regions = Regions(ranges=ranges, labels=["Cool", "Neutral", "Warm"])
    >>> artists = regions.render(ax, style)
    """

    ranges: Ranges
    labels: Sequence[str] | None = None

    def render(self, ax: plt.Axes, style: Style) -> dict[str, Any]:
        """Render regions as filled bands on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on.
        style : Style
            Style configuration for rendering.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'bands': list of PolyCollection artists
            - 'curves': list of Line2D artists
            - 'legend': Legend artist or None
        """
        curves = self.ranges.curves
        y_arr = self.ranges.y_values
        thresholds = self.ranges.thresholds
        x_lo, x_hi = self.ranges.x_bounds

        # Prepare colors
        n_regions = len(thresholds) + 1
        band_colors = self._get_band_colors(style, n_regions)

        # Constant x boundaries
        left_const = np.full_like(y_arr, x_lo, dtype=float)
        right_const = np.full_like(y_arr, x_hi, dtype=float)

        # Build region boundaries: (left_curve, right_curve) pairs
        if curves:
            regions_bounds = (
                [(left_const, curves[0])]
                + [(curves[i], curves[i + 1]) for i in range(len(curves) - 1)]
                + [(curves[-1], right_const)]
            )
        else:
            regions_bounds = [(left_const, right_const)]

        # Render filled bands
        band_artists = []
        for i, (left, right) in enumerate(regions_bounds):
            mask = np.isfinite(left) & np.isfinite(right)
            # Only fill where band has positive width
            width_mask = (right - left) > 1e-12
            mask = mask & width_mask

            if mask.any():
                coll = ax.fill_betweenx(
                    y_arr[mask],
                    left[mask],
                    right[mask],
                    color=band_colors[i],
                    alpha=style.band_alpha,
                    linewidth=0,
                    zorder=0,
                )
                band_artists.append(coll)

        # Render threshold curves
        curve_artists = []
        for curve in curves:
            mask = np.isfinite(curve)
            if mask.any():
                (ln,) = ax.plot(
                    curve[mask],
                    y_arr[mask],
                    color=style.line_color,
                    linewidth=style.line_width,
                    zorder=1,
                )
                curve_artists.append(ln)

        # Render legend
        legend_artist = None
        if style.show_legend:
            legend_artist = self._render_legend(ax, style, band_colors, thresholds)

        return {
            "bands": band_artists,
            "curves": curve_artists,
            "legend": legend_artist,
        }

    def _get_band_colors(self, style: Style, n_regions: int) -> list:
        """Get colors for each region band."""
        if style.band_colors is not None:
            if len(style.band_colors) != n_regions:
                msg = (
                    f"band_colors must have {n_regions} colors "
                    f"(got {len(style.band_colors)})"
                )
                raise ValueError(msg)
            return list(style.band_colors)

        # Sample from colormap
        cmap = plt.get_cmap(style.cmap)
        return [cmap(i / (n_regions - 1)) for i in range(n_regions)]

    def _render_legend(
        self,
        ax: plt.Axes,
        style: Style,
        band_colors: list,
        thresholds: list[float],
    ) -> plt.Legend:
        """Render legend for region bands."""
        n_regions = len(thresholds) + 1
        labels = self._get_labels(thresholds)

        legend_elements = []
        for i in range(n_regions):
            patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=band_colors[i],
                alpha=style.band_alpha,
                label=labels[i],
            )
            legend_elements.append(patch)

        return ax.legend(
            handles=legend_elements,
            loc=style.legend_loc,
            bbox_to_anchor=style.legend_bbox,
            ncol=min(style.legend_ncol, len(legend_elements)),
            framealpha=style.legend_alpha,
            fontsize=style.font_sizes.get("legend", 10),
        )

    def _get_labels(self, thresholds: list[float]) -> list[str]:
        """Get labels for each region."""
        if self.labels is not None:
            return list(self.labels)

        # Auto-generate labels from thresholds
        n_regions = len(thresholds) + 1
        labels = []

        for i in range(n_regions):
            if i == 0 and thresholds:
                labels.append(f"< {thresholds[0]:.1f}")
            elif i == n_regions - 1 and thresholds:
                labels.append(f"> {thresholds[-1]:.1f}")
            elif thresholds:
                labels.append(f"{thresholds[i - 1]:.1f} to {thresholds[i]:.1f}")
            else:
                labels.append("Region")

        return labels
