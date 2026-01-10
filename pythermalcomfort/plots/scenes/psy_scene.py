"""PsychrometricScene - Psychrometric chart visualization.

This module provides PsychrometricScene, a scene type that visualizes
a psychrometric chart with PMV-based comfort zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.utilities import psy_ta_rh, v_relative
from pythermalcomfort.plots.scenes.base import BaseScene

if TYPE_CHECKING:
    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class PsychrometricScene(BaseScene):
    """Psychrometric chart with PMV comfort zone.
    
    Visualizes a psychrometric chart with:
    - X-axis: Dry bulb temperature
    - Y-axis: Humidity ratio
    - Background: Relative humidity curves
    - Comfort zone: PMV-based region
    """

    # Fixed parameters for PMV calculation
    fixed_params: dict[str, Any] = field(default_factory=dict)

    # Axis ranges
    x_range: tuple[float, float] = (10.0, 36.0)
    y_range: tuple[float, float] = (0.0, 30.0)

    # RH curves to display (%)
    rh_lines: tuple[float, ...] = ( 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    # Whether to show all 7 PMV categories or just 3 (simple)
    extended_categories: bool = False

    # PMV thresholds for comfort zone
    thresholds: tuple[float, ...] = field(default_factory=lambda: (-0.5, 0.5))
    labels: tuple[str, ...] | None = field(default_factory=lambda: (
        "Neutral",
    ))

    @classmethod
    def create(
        cls,
        fixed_params: dict[str, Any] | None = None,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        rh_lines: tuple[float, ...] | None = None,
        extended_categories: bool = False,
    ) -> PsychrometricScene:
        """Create a PsychrometricScene.

        Parameters
        ----------
        fixed_params : dict or None
            Fixed parameters for PMV: tr, v, met, clo.
            Defaults: tr=25, v=0.1, met=1.2, clo=0.5
        x_range : tuple or None
            Dry bulb temperature range. Default: (10, 36).
        y_range : tuple or None
            Humidity ratio range. Default: (0, 30).
        rh_lines : tuple or None
            RH percentages to draw as curves.

        Returns
        -------
        PsychrometricScene
            Configured scene ready to render.
        """
        # Default fixed params for PMV
        default_params = {"tr": 25, "v": 0.1, "met": 1.2, "clo": 0.5}
        if fixed_params:
            default_params.update(fixed_params)

        return cls(
            fixed_params=default_params,
            x_range=x_range or (10, 36),
            y_range=y_range or (0, 30),
            rh_lines=rh_lines or (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
            extended_categories=extended_categories,
        )
    
    def _rh_to_humidity_ratio(self, tdb: float, rh: float) -> float:
        """Convert relative humidity to humidity ratio.

        Parameters
        ----------
        tdb : float
            Dry bulb temperature [°C].
        rh : float
            Relative humidity [%].

        Returns
        -------
        float
            Humidity ratio [g water / kg dry air].
        """
        # psy_ta_rh returns hr in kg/kg, multiply by 100000 for g/kg
        result = psy_ta_rh(tdb, rh / 100, 101325)
        return result["hr"] * 100000

    def _compute_pmv(self, tdb: float, rh: float) -> float:
        """Compute PMV at given temperature and RH.

        Parameters
        ----------
        tdb : float
            Dry bulb temperature [°C].
        rh : float
            Relative humidity [%].

        Returns
        -------
        float
            PMV value.
        """
        tr = self.fixed_params.get("tr", 25)
        v = self.fixed_params.get("v", 0.1)
        met = self.fixed_params.get("met", 1.2)
        clo = self.fixed_params.get("clo", 0.5)

        vr = v_relative(v=v, met=met)

        result = pmv_ppd_iso(
            tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, limit_inputs=False
        )
        return result.pmv

    def _find_comfort_temp(self, rh: float, target_pmv: float) -> float | None:
        """Find temperature where PMV equals target at given RH.

        Parameters
        ----------
        rh : float
            Relative humidity [%].
        target_pmv : float
            Target PMV value.

        Returns
        -------
        float or None
            Temperature [°C] where PMV = target, or None if not found.
        """
        def pmv_root(tdb):
            return self._compute_pmv(tdb, rh) - target_pmv

        try:
            return brentq(pmv_root, 0, 120)
        except ValueError:
            return None
        
    def render(self, ax: plt.Axes, style: Style) -> dict[str, Any]:
        """Render psychrometric chart on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on.
        style : Style
            Style configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'rh_curves', 'comfort_zone', 'legend' artists.
        """
        t_dry = np.linspace(0, 50, 500)

        # 1. Draw RH curves (background)
        rh_curve_artists = []
        for rh in self.rh_lines:
            hr_values = [self._rh_to_humidity_ratio(t, rh) for t in t_dry]
            line, = ax.plot(
                t_dry, hr_values,
                color="lightgrey",
                linewidth=0.8,
                zorder=0,
            )
            rh_curve_artists.append(line)

        # 2. Define thresholds and labels based on mode
        if self.extended_categories:
            pmv_thresholds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
            zone_labels = ["Cold", "Cool", "Slightly Cool", "Neutral", "Slightly Warm", "Warm", "Hot"]
            # Get colors from colormap
            cmap = plt.get_cmap(style.cmap)
            zone_colors = [cmap(i / 6) for i in range(7)]
        else:
            pmv_thresholds = [-0.5, 0.5]
            zone_labels = ["Comfortable"]
            zone_colors = [(0, 0.5, 0, 1)]  # Green

        # 3. Compute comfort zone boundaries for each threshold
        rh_levels = np.linspace(0, 100, 10)

        # Find temperatures for each PMV threshold
        threshold_curves = {}
        for pmv_val in pmv_thresholds:
            temps = []
            valid_rh = []
            for rh in rh_levels:
                t = self._find_comfort_temp(rh, pmv_val)
                if t is not None and self.x_range[0] <= t <= self.x_range[1]:
                    temps.append(t)
                    valid_rh.append(rh)
            if temps:
                # Convert to humidity ratio
                hr = [self._rh_to_humidity_ratio(t, rh) for t, rh in zip(temps, valid_rh)]
                threshold_curves[pmv_val] = (temps, hr)

        # 4. Draw comfort zones
        comfort_artists = []

        if self.extended_categories:
            pass # still to be implemented: extended categories
        else:
            # Simple mode: just draw neutral zone
            lower_curve = threshold_curves.get(-0.5)
            upper_curve = threshold_curves.get(0.5)

            if lower_curve and upper_curve:
                fill = ax.fill(
                    np.concatenate([upper_curve[0], lower_curve[0][::-1]]),
                    np.concatenate([upper_curve[1], lower_curve[1][::-1]]),
                    # color=zone_colors[0],
                    color=(0.8, 0.875, 0.7, 0.5),
                    alpha=style.band_alpha,
                    linewidth=0,
                    zorder=1,
                    label="Comfort Zone",
                )
                comfort_artists.append(fill)

        # 5. Set axis limits and labels
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 6. Legend
        legend_artist = None
        if style.show_legend:
            legend_artist = ax.legend(
                loc="upper left",
                framealpha=style.legend_alpha,
                fontsize=style.font_sizes.get("legend", 10),
            )

        return {
            "rh_curves": rh_curve_artists,
            "comfort_zone": comfort_artists,
            "legend": legend_artist,
        }
    
    def get_category(self, x: float, y: float) -> str:
        """Get category for a data point.

        Parameters
        ----------
        x : float
            Dry bulb temperature [°C].
        y : float
            Humidity ratio [g/kg].

        Returns
        -------
        str
            Category label.
        """
        try:
            from scipy.optimize import brentq

            def hr_diff(rh):
                return self._rh_to_humidity_ratio(x, rh) - y

            rh = brentq(hr_diff, 0, 100)
            pmv = self._compute_pmv(x, rh)

            if self.extended_categories:
                pass # still to be implemented: extended categories
            else:
                if pmv < -0.5:
                    return "Cooler than neutral"
                elif pmv > 0.5:
                    return "Warmer than neutral"
                else:
                    return "Neutral"
        except (ValueError, Exception):
            return "Out of Range"

    def get_labels(self) -> list[str]:
        """Get category labels."""
        if self.extended_categories:
            return None # still to be implemented: extended categories
        else:
            return ["Cooler than neutral", "Neutral", "Warmer than neutral"]


    def get_colors(self, style: Style) -> list:
        """Get colors for each category."""
        return [
            (0.569, 0.769, 0.914, 1),   # Cooler than Neutral (blue)
            (0.8, 0.875, 0.7, 1),   # Neutral (green)
            (0.859, 0.447, 0.373, 1),   # Warmer than Neutral (red)
            (0.5, 0.5, 0.5, 0.2),   # Out of range (gray)
        ]

    def get_x_range(self) -> tuple[float, float]:
        """Get x-axis range."""
        return self.x_range

    def get_y_range(self) -> tuple[float, float]:
        """Get y-axis range."""
        return self.y_range

    def get_fixed_params(self) -> dict[str, Any]:
        """Get fixed parameters."""
        return dict(self.fixed_params)

