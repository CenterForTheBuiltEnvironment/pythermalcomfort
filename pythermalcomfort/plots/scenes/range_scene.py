"""RangeScene - Generic threshold-based range visualization.

This module provides RangeScene, a flexible scene type that can visualize
threshold-based regions for any thermal comfort model with configurable
x and y axis parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.presets import get_preset
from pythermalcomfort.plots.scenes.base import BaseScene
from pythermalcomfort.plots.utils import make_metric_eval, solve_threshold_curves

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pythermalcomfort.plots.presets import Preset
    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class RangeScene(BaseScene):
    """Generic threshold-based range scene with flexible x/y selection.

    This scene visualizes threshold-based regions computed from a thermal
    comfort model. Users can choose which model parameters map to the x
    and y axes; remaining parameters are fixed.

    Attributes
    ----------
    model_func : Callable
        The thermal comfort model function to evaluate.
    fixed_params : dict[str, Any]
        Parameters held constant (not on x or y axes).
    thresholds : Sequence[float]
        Threshold values defining region boundaries.
    labels : Sequence[str] or None
        Labels for each region.
    x_param : str
        Model parameter name for x-axis (default: "tdb").
    y_param : str
        Model parameter name for y-axis (default: "rh").
    x_range : tuple[float, float]
        (min, max) range for x-axis.
    y_range : tuple[float, float]
        (min, max) range for y-axis.
    y_step : float
        Step size for y-axis grid computation.
    metric_attr : str or None
        Attribute name to extract from model result.
    xlabel : str or None
        Custom x-axis label.
    ylabel : str or None
        Custom y-axis label.

    Examples
    --------
    >>> from pythermalcomfort.models import utci
    >>> scene = RangeScene.create(utci, fixed_params={"v": 1.0, "tr": 25})
    >>> artists = scene.render(ax, style)
    """

    # Note: All fields must have defaults due to dataclass inheritance rules
    model_func: Callable[..., Any] | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)
    x_param: str = "tdb"
    y_param: str = "rh"
    x_range: tuple[float, float] = (10.0, 36.0)
    y_range: tuple[float, float] = (0.0, 100.0)
    y_step: float = 2.0
    metric_attr: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None

    @classmethod
    def create(
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
        xlabel: str | None = None,
        ylabel: str | None = None,
        preset: Preset | None = None,
    ) -> RangeScene:
        """Create a RangeScene with preset defaults.

        Parameters
        ----------
        model_func : Callable
            A pythermalcomfort model function (e.g., utci, pmv_ppd_iso).
        fixed_params : dict or None
            Model parameters held constant.
        thresholds : Sequence[float] or None
            Threshold values. If None, uses preset defaults.
        labels : Sequence[str] or None
            Region labels. If None, uses preset or auto-generates.
        x_param : str
            Model parameter for x-axis.
        y_param : str
            Model parameter for y-axis.
        x_range : tuple or None
            X-axis range. If None, uses preset default.
        y_range : tuple or None
            Y-axis range. If None, uses preset default.
        y_step : float or None
            Y-axis step size. If None, auto-computed.
        metric_attr : str or None
            Attribute to extract from model result.
        xlabel : str or None
            Custom x-axis label.
        ylabel : str or None
            Custom y-axis label.
        preset : Preset or None
            Explicit preset. If None, auto-detected from model.

        Returns
        -------
        RangeScene
            Configured scene ready to render.
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

        # Resolve axis labels
        if xlabel is None and preset is not None:
            xlabel = preset.xlabel
        if ylabel is None and preset is not None:
            ylabel = preset.ylabel

        return cls(
            model_func=model_func,
            fixed_params=dict(fixed_params or {}),
            thresholds=list(thresholds),
            labels=list(labels) if labels else None,
            x_param=x_param,
            y_param=y_param,
            x_range=x_range,
            y_range=y_range,
            y_step=y_step,
            metric_attr=metric_attr,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    def _get_xy_mapper(self) -> Callable[[float, float, dict], dict]:
        """Build a mapper function for (x, y) -> model kwargs."""
        x_param = self.x_param
        y_param = self.y_param

        def mapper(x: float, y: float, fixed: dict) -> dict:
            kwargs = {x_param: float(x), y_param: float(y)}
            kwargs.update(fixed)
            return kwargs

        return mapper

    def _compute_ranges(self) -> dict:
        """Compute threshold curves using the solver."""
        y_values = np.arange(
            self.y_range[0], self.y_range[1] + 1e-9, float(self.y_step)
        )

        # Build metric evaluator
        metric_xy = make_metric_eval(
            model_func=self.model_func,
            xy_to_kwargs=self._get_xy_mapper(),
            fixed_params=self.fixed_params,
            metric_attr=self.metric_attr,
        )

        # Solve for threshold curves
        result = solve_threshold_curves(
            metric_xy=metric_xy,
            thresholds=list(self.thresholds),
            y_values=y_values,
            x_bounds=self.x_range,
        )

        return {
            "curves": result["curves"],
            "y_values": result["y_values"],
            "thresholds": result["thresholds"],
            "x_bounds": self.x_range,
        }

    def render(self, ax: plt.Axes, style: Style) -> dict[str, Any]:
        """Render threshold regions on the axes."""
        # Compute ranges
        data = self._compute_ranges()
        curves = data["curves"]
        y_arr = data["y_values"]
        x_lo, x_hi = data["x_bounds"]
        n_regions = len(data["thresholds"]) + 1

        # Prepare colors
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
            legend_artist = self._render_legend(ax, style, band_colors)

        # Set axis limits
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)

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
    ) -> plt.Legend:
        """Render legend for region bands."""
        label_list = self._get_region_labels()

        legend_elements = []
        for i, label in enumerate(label_list):
            patch = plt.Rectangle(
                (0, 0), 1, 1,
                facecolor=band_colors[i],
                alpha=style.band_alpha,
                label=label,
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

    def _get_region_labels(self) -> list[str]:
        """Get labels for each region (for legend)."""
        if self.labels is not None:
            return list(self.labels)

        # Auto-generate labels from thresholds
        n_regions = len(self.thresholds) + 1
        result = []
        for i in range(n_regions):
            if i == 0 and self.thresholds:
                result.append(f"< {self.thresholds[0]:.1f}")
            elif i == n_regions - 1 and self.thresholds:
                result.append(f"> {self.thresholds[-1]:.1f}")
            elif self.thresholds:
                result.append(f"{self.thresholds[i-1]:.1f} to {self.thresholds[i]:.1f}")
            else:
                result.append("Region")
        return result

    def get_category(self, x: float, y: float) -> str:
        """Get the category label for a data point.

        Evaluates the model at (x, y) and returns the corresponding
        category label based on thresholds.

        Parameters
        ----------
        x : float
            X-coordinate (value of x_param).
        y : float
            Y-coordinate (value of y_param).

        Returns
        -------
        str
            Category label for this point.
        """
        from pythermalcomfort.plots.utils import extract_metric

        # Build kwargs and evaluate model
        mapper = self._get_xy_mapper()
        kwargs = mapper(x, y, self.fixed_params)

        try:
            result = self.model_func(**kwargs, limit_inputs=False)
            value = extract_metric(result, self.metric_attr)
        except Exception:
            # Return "Out of Range" if model fails
            return self.get_labels()[0] if self.labels else "Out of Range"

        # Classify based on thresholds
        labels = self.get_labels()
        for i, threshold in enumerate(self.thresholds):
            if value < threshold:
                return labels[i]
        return labels[-1]

    def get_labels(self) -> list[str]:
        """Get category labels for this scene.

        Returns
        -------
        list[str]
            List of category labels.
        """
        if self.labels is not None:
            return list(self.labels)

        # Auto-generate from thresholds
        n_regions = len(self.thresholds) + 1
        labels = []
        for i in range(n_regions):
            if i == 0 and self.thresholds:
                labels.append(f"< {self.thresholds[0]:.1f}")
            elif i == n_regions - 1 and self.thresholds:
                labels.append(f"> {self.thresholds[-1]:.1f}")
            elif self.thresholds:
                labels.append(f"{self.thresholds[i-1]:.1f} to {self.thresholds[i]:.1f}")
            else:
                labels.append("Region")
        return labels

    def get_colors(self, style: Style) -> list:
        """Get colors for each region.

        Parameters
        ----------
        style : Style
            Style configuration.

        Returns
        -------
        list
            List of colors (one per region).
        """
        n_regions = len(self.thresholds) + 1

        if style.band_colors is not None:
            return list(style.band_colors)

        cmap = plt.get_cmap(style.cmap)
        return [cmap(i / (n_regions - 1)) for i in range(n_regions)]

    def get_x_range(self) -> tuple[float, float]:
        """Get x-axis range."""
        return self.x_range

    def get_y_range(self) -> tuple[float, float]:
        """Get y-axis range."""
        return self.y_range

    def get_fixed_params(self) -> dict[str, Any]:
        """Get fixed parameters."""
        return dict(self.fixed_params)

    def get_xlabel(self, style: Style) -> str:
        """Get x-axis label."""
        if style.xlabel is not None:
            return style.xlabel
        if self.xlabel is not None:
            return self.xlabel
        return f"{self.x_param}"

    def get_ylabel(self, style: Style) -> str:
        """Get y-axis label."""
        if style.ylabel is not None:
            return style.ylabel
        if self.ylabel is not None:
            return self.ylabel
        return f"{self.y_param}"
