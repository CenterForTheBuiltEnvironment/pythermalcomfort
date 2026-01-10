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
            x_range=x_range or (10.0, 36.0),
            y_range=y_range or (0, 30),
            rh_lines=rh_lines or (10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
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
        t_dry = np.linspace(self.x_range[0], self.x_range[1], 500)

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

        # 2. Compute comfort zone boundaries
        rh_levels = np.linspace(0, 100, 10)

        upper_temps = []
        lower_temps = []
        valid_rh = []

        for rh in rh_levels:
            t_upper = self._find_comfort_temp(rh, 0.5)
            t_lower = self._find_comfort_temp(rh, -0.5)
            if t_upper is not None and t_lower is not None:
                upper_temps.append(t_upper)
                lower_temps.append(t_lower)
                valid_rh.append(rh)

        # Convert to humidity ratio for plotting
        upper_hr = [self._rh_to_humidity_ratio(t, rh) for t, rh in zip(upper_temps, valid_rh)]
        lower_hr = [self._rh_to_humidity_ratio(t, rh) for t, rh in zip(lower_temps, valid_rh)]

        # 3. Draw comfort zone
        comfort_zone = ax.fill(
            np.concatenate([upper_temps, lower_temps[::-1]]),
            np.concatenate([upper_hr, lower_hr[::-1]]),
            color="green",
            alpha=style.band_alpha * 0.5,
            linewidth=0,
            zorder=1,
            label="Comfort Zone",
        )

        # 4. Set axis limits and labels
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 5. Legend
        legend_artist = None
        if style.show_legend:
            legend_artist = ax.legend(
                loc="upper left",
                framealpha=style.legend_alpha,
                fontsize=style.font_sizes.get("legend", 10),
            )

        return {
            "rh_curves": rh_curve_artists,
            "comfort_zone": comfort_zone,
            "legend": legend_artist,
        }
    
    def get_category(self, x: float, y: float) -> str:
        """Get category for a data point.

        Note: x is tdb, y is humidity ratio. We need to convert
        humidity ratio back to RH to compute PMV.

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
        # Approximate RH from humidity ratio (iterate to find it)
        # For simplicity, use a rough estimation
        try:
            # Search for RH that gives this humidity ratio at this temp
            from scipy.optimize import brentq

            def hr_diff(rh):
                return self._rh_to_humidity_ratio(x, rh) - y

            rh = brentq(hr_diff, 0, 120)
            pmv = self._compute_pmv(x, rh)

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
        return [
            "Neutral",
            "Cooler than neutral",
            "Warmer than neutral",
            "Out of Range",
        ]

    def get_colors(self, style: Style) -> list:
        """Get colors for each category."""
        return [
            (0.0, 0.5, 0.0, 0.5),   # Neutral (green)
            (0.0, 0.0, 1.0, 0.5),   # Cooler than Neutral (blue)
            (1.0, 0.0, 0.0, 0.5),   # Warmer than Neutral (red)
            (0.5, 0.5, 0.5, 0.5),   # Out of range (gray)
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

