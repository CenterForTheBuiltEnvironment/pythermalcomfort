"""AdaptiveScene - ASHRAE 55 Adaptive Comfort Model visualization.

This module provides AdaptiveScene, a scene type that visualizes the
ASHRAE 55 Adaptive Comfort Model with fixed axes (outdoor temperature
vs operative temperature).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.scenes.base import BaseScene

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pythermalcomfort.plots.style import Style


# Default colors matching the earlier Plotly examples
ADAPTIVE_COLORS = [
    "rgba(255, 0, 0, 0.2)",    # Discomfort (outside 80%)
    "rgba(0, 0, 255, 0.2)",    # 80% Acceptability band
    "rgba(0, 128, 0, 0.2)",    # 90% Acceptability band
]

# Matplotlib-compatible colors
ADAPTIVE_COLORS_MPL = [
    (0.859, 0.447, 0.373, 1),      # Discomfort
    (0.8, 0.95, 0.7, 1),      # 80% band
    (0.35, 0.55, 0.3, 1),      # 90% band
]


@dataclass(frozen=True)
class AdaptiveScene(BaseScene):
    """ASHRAE 55 Adaptive Comfort Model scene.

    Visualizes the adaptive comfort model with fixed axes:
    - X-axis: Outdoor (prevailing mean) temperature
    - Y-axis: Indoor operative temperature

    Shows comfort bands for 80% and 90% acceptability based on:
    t_cmf = 0.31 * t_outdoor + 17.8
    - 90% acceptability: t_cmf ± 2.5°C
    - 80% acceptability: t_cmf ± 3.5°C

    Attributes
    ----------
    x_range : tuple[float, float]
        Range for outdoor temperature (default: 10-33.5°C).
    y_range : tuple[float, float]
        Range for operative temperature (default: 14-36°C).
    show_80_band : bool
        Whether to show 80% acceptability band.
    show_90_band : bool
        Whether to show 90% acceptability band.

    Examples
    --------
    >>> scene = AdaptiveScene.create()
    >>> artists = scene.render(ax, style)
    >>> category = scene.get_category(t_outdoor=25, t_operative=24)
    """

    x_range: tuple[float, float] = (10.0, 33.5)
    y_range: tuple[float, float] = (14.0, 36.0)
    show_80_band: bool = True
    show_90_band: bool = True
    x_padding: float = 2.0
    band_colors: tuple | None = None,

    # Override base class defaults
    thresholds: Sequence[float] = field(default_factory=lambda: [2.5, 3.5])
    labels: Sequence[str] | None = field(
        default_factory=lambda: [
            "Discomfort",
            "80% Acceptability",
            "90% Acceptability",
        ]
    )

    @classmethod
    def create(
        cls,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        show_80_band: bool = True,
        show_90_band: bool = True,
        x_padding: float = 2.0,
    ) -> AdaptiveScene:
        """Create an AdaptiveScene.

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

        Returns
        -------
        AdaptiveScene
            Configured scene ready to render.
        """
        return cls(
            x_range=x_range or (10.0, 33.5),
            y_range=y_range or (14.0, 36.0),
            show_80_band=show_80_band,
            show_90_band=show_90_band,
            x_padding=x_padding,
        )

    def _compute_comfort_temperature(self, t_outdoor: float) -> float:
        """Compute comfort temperature from outdoor temperature.

        Parameters
        ----------
        t_outdoor : float
            Outdoor (prevailing mean) temperature in °C.

        Returns
        -------
        float
            Comfort temperature in °C.
        """
        return 0.31 * t_outdoor + 17.8

    def render(self, ax: plt.Axes, style: Style) -> dict[str, Any]:
        """Render adaptive comfort bands on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on.
        style : Style
            Style configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'bands' and 'legend' artists.
        """
        t_outdoor = np.linspace(self.x_range[0], self.x_range[1], 100)
        t_cmf = self._compute_comfort_temperature(t_outdoor)

        # Compute band boundaries
        t_90_upper = t_cmf + 2.5
        t_90_lower = t_cmf - 2.5
        t_80_upper = t_cmf + 3.5
        t_80_lower = t_cmf - 3.5

        band_artists = []
        legend_elements = []

        # Draw 90% band (innermost, green)
        if self.show_90_band:
            band_90 = ax.fill_between(
                t_outdoor,
                t_90_lower,
                t_90_upper,
                color=ADAPTIVE_COLORS_MPL[2][:3],
                alpha=style.band_alpha * 0.8,
                linewidth=0,
                zorder=1,
            )
            band_artists.append(band_90)
            legend_elements.append(
                plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=ADAPTIVE_COLORS_MPL[2],
                    alpha=style.band_alpha,
                    label="90% Acceptability",
                )
            )

        # Draw 80% band (outer, blue) - only the parts outside 90%
        if self.show_80_band:
            # Upper part of 80% band (between 90% upper and 80% upper)
            band_80_upper = ax.fill_between(
                t_outdoor,
                t_90_upper,
                t_80_upper,
                color=ADAPTIVE_COLORS_MPL[1][:3],
                alpha=style.band_alpha * 0.8,
                linewidth=0,
                zorder=0,
            )
            # Lower part of 80% band (between 80% lower and 90% lower)
            band_80_lower = ax.fill_between(
                t_outdoor,
                t_80_lower,
                t_90_lower,
                color=ADAPTIVE_COLORS_MPL[1][:3],
                alpha=style.band_alpha * 0.8,
                linewidth=0,
                zorder=0,
            )
            band_artists.extend([band_80_upper, band_80_lower])
            legend_elements.append(
                plt.Rectangle(
                    (0, 0), 1, 1,
                    facecolor=ADAPTIVE_COLORS_MPL[1],
                    alpha=style.band_alpha,
                    label="80% Acceptability",
                )
            )

        # Set axis limits (extended by padding for display)
        ax.set_xlim(self.x_range[0] - self.x_padding, self.x_range[1] + self.x_padding)
        ax.set_ylim(self.y_range)

        # Add legend if enabled
        legend_artist = None
        if style.show_legend and legend_elements:
            legend_artist = ax.legend(
                handles=legend_elements,
                loc="upper left",
                framealpha=style.legend_alpha,
                fontsize=style.font_sizes.get("legend", 10),
            )

        return {
            "bands": band_artists,
            "curves": [],
            "legend": legend_artist,
        }

    def get_category(self, x: float, y: float) -> str:
        """Get category for a data point.

        Parameters
        ----------
        x : float
            Outdoor temperature.
        y : float
            Operative temperature.

        Returns
        -------
        str
            Category: "90% Acceptability", "80% Acceptability", or "Discomfort".
        """
        t_cmf = self._compute_comfort_temperature(x)
        deviation = abs(y - t_cmf)

        if deviation <= 2.5:
            return "90% Acceptability"
        elif deviation <= 3.5:
            return "80% Acceptability"
        else:
            return "Discomfort"

    def get_labels(self) -> list[str]:
        """Get category labels."""
        return [
            "90% Acceptability",
            "80% Acceptability",
            "Discomfort",
        ]

    def get_colors(self, style: Style) -> list:
        """Get colors for each category.

        Returns colors in same order as get_labels().
        """
        return [
            ADAPTIVE_COLORS_MPL[2],  # 90% (green)
            ADAPTIVE_COLORS_MPL[1],  # 80% (blue)
            ADAPTIVE_COLORS_MPL[0],  # Discomfort (red)
        ]

    def get_x_range(self) -> tuple[float, float]:
        """Get x-axis range (including padding)."""
        return (self.x_range[0] - self.x_padding, self.x_range[1] + self.x_padding)

    def get_y_range(self) -> tuple[float, float]:
        """Get y-axis range."""
        return self.y_range

    def get_xlabel(self, style: Style) -> str:
        """Get x-axis label."""
        if style.xlabel is not None:
            return style.xlabel
        return "Outdoor Temperature [°C]"

    def get_ylabel(self, style: Style) -> str:
        """Get y-axis label."""
        if style.ylabel is not None:
            return style.ylabel
        return "Operative Temperature [°C]"
