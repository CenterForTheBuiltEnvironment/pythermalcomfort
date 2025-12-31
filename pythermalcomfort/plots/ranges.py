"""Ranges data representation for threshold-based plots.

This module provides the Ranges class, a frozen dataclass that holds
computed threshold curves from model functions and can render them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from pythermalcomfort.plots.style import Style


@dataclass(frozen=True)
class Ranges:
    """Computed threshold curves for region-based plotting.

    This is a frozen (immutable) data representation holding the results
    of threshold curve computation. Use the `from_model` factory method
    to compute ranges from a pythermalcomfort model function.

    Parameters
    ----------
    curves : list[np.ndarray]
        List of x(y) curves, one per threshold. Each array has the same
        shape as y_values. NaN indicates no solution at that y value.
    y_values : np.ndarray
        Y-axis values at which curves were computed.
    thresholds : list[float]
        Threshold values that define region boundaries.
    x_bounds : tuple[float, float]
        (min, max) bounds for the x-axis.

    Notes
    -----
    This class is frozen (immutable) after creation. To modify ranges,
    create a new Ranges instance.

    Examples
    --------
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.utils import mapper_tdb_rh
    >>> ranges = Ranges.from_model(
    ...     model_func=pmv_ppd_iso,
    ...     xy_to_kwargs=mapper_tdb_rh,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
    ...     thresholds=[-0.5, 0.5],
    ...     x_bounds=(18, 30),
    ...     y_values=np.arange(20, 80, 2),
    ... )
    """

    curves: list[npt.NDArray[np.floating[Any]]]
    y_values: npt.NDArray[np.floating[Any]]
    thresholds: list[float]
    x_bounds: tuple[float, float]

    @classmethod
    def from_model(
        cls,
        model_func: Callable[..., Any],
        xy_to_kwargs: Callable[[float, float, dict[str, Any]], dict[str, Any]],
        fixed_params: dict[str, Any],
        thresholds: Sequence[float],
        x_bounds: tuple[float, float],
        y_values: Sequence[float] | npt.NDArray[np.floating[Any]],
        *,
        metric_attr: str | None = None,
        x_scan_step: float = 1.0,
        smooth_sigma: float = 0.8,
    ) -> Ranges:
        """Compute ranges from a model function.

        This factory method uses the existing solve_threshold_curves
        infrastructure to compute where metric(x, y) == threshold.

        Parameters
        ----------
        model_func : Callable
            A pythermalcomfort model function (e.g., pmv_ppd_iso, utci).
        xy_to_kwargs : Callable
            Function mapping (x, y, fixed_params) to model kwargs.
            See pythermalcomfort.plots.utils for predefined mappers.
        fixed_params : dict
            Model parameters held constant for all evaluations.
        thresholds : Sequence[float]
            Threshold values to solve for.
        x_bounds : tuple[float, float]
            (min, max) bounds for the x-axis search.
        y_values : Sequence[float] or np.ndarray
            Y-axis values at which to compute curves.
        metric_attr : str or None, optional
            Attribute name to extract from model result (e.g., "pmv").
            If None, attempts to infer or coerce to float.
        x_scan_step : float, default 1.0
            Step size for scanning x-axis when finding thresholds.
        smooth_sigma : float, default 0.8
            Gaussian smoothing sigma for curves. Set 0 to disable.

        Returns
        -------
        Ranges
            Computed threshold curves ready for rendering.

        Examples
        --------
        >>> from pythermalcomfort.models import pmv_ppd_iso
        >>> from pythermalcomfort.plots.utils import mapper_tdb_rh
        >>> ranges = Ranges.from_model(
        ...     model_func=pmv_ppd_iso,
        ...     xy_to_kwargs=mapper_tdb_rh,
        ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
        ...     thresholds=[-0.5, 0.5],
        ...     x_bounds=(18, 30),
        ...     y_values=np.arange(20, 80, 2),
        ...     metric_attr="pmv",
        ... )
        """
        from pythermalcomfort.plots.utils import make_metric_eval, solve_threshold_curves

        # Build metric evaluator
        metric_xy = make_metric_eval(
            model_func=model_func,
            xy_to_kwargs=xy_to_kwargs,
            fixed_params=fixed_params,
            metric_attr=metric_attr,
        )

        # Solve for threshold curves
        result = solve_threshold_curves(
            metric_xy=metric_xy,
            thresholds=list(thresholds),
            y_values=y_values,
            x_bounds=x_bounds,
            x_scan_step=x_scan_step,
            smooth_sigma=smooth_sigma,
        )

        return cls(
            curves=result["curves"],
            y_values=result["y_values"],
            thresholds=result["thresholds"],
            x_bounds=x_bounds,
        )

    @property
    def n_regions(self) -> int:
        """Number of regions (len(thresholds) + 1)."""
        return len(self.thresholds) + 1

    def render(
        self,
        ax: plt.Axes,
        style: Style,
        labels: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Render ranges as filled color bands on the axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to render on.
        style : Style
            Style configuration for rendering.
        labels : Sequence[str] or None, optional
            Labels for each region. If None, auto-generated from thresholds.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - 'bands': list of PolyCollection artists
            - 'curves': list of Line2D artists
            - 'legend': Legend artist or None
        """
        y_arr = self.y_values
        x_lo, x_hi = self.x_bounds

        # Prepare colors
        band_colors = self._get_band_colors(style)

        # Constant x boundaries
        left_const = np.full_like(y_arr, x_lo, dtype=float)
        right_const = np.full_like(y_arr, x_hi, dtype=float)

        # Build region boundaries: (left_curve, right_curve) pairs
        if self.curves:
            regions_bounds = (
                [(left_const, self.curves[0])]
                + [
                    (self.curves[i], self.curves[i + 1])
                    for i in range(len(self.curves) - 1)
                ]
                + [(self.curves[-1], right_const)]
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
        for curve in self.curves:
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
            legend_artist = self._render_legend(ax, style, band_colors, labels)

        return {
            "bands": band_artists,
            "curves": curve_artists,
            "legend": legend_artist,
        }

    def _get_band_colors(self, style: Style) -> list:
        """Get colors for each region band."""
        if style.band_colors is not None:
            if len(style.band_colors) != self.n_regions:
                msg = (
                    f"band_colors must have {self.n_regions} colors "
                    f"(got {len(style.band_colors)})"
                )
                raise ValueError(msg)
            return list(style.band_colors)

        # Sample from colormap
        cmap = plt.get_cmap(style.cmap)
        return [cmap(i / (self.n_regions - 1)) for i in range(self.n_regions)]

    def _render_legend(
        self,
        ax: plt.Axes,
        style: Style,
        band_colors: list,
        labels: Sequence[str] | None,
    ) -> plt.Legend:
        """Render legend for region bands."""
        label_list = self._get_labels(labels)

        legend_elements = []
        for i in range(self.n_regions):
            patch = plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=band_colors[i],
                alpha=style.band_alpha,
                label=label_list[i],
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

    def _get_labels(self, labels: Sequence[str] | None) -> list[str]:
        """Get labels for each region."""
        if labels is not None:
            return list(labels)

        # Auto-generate labels from thresholds
        result = []
        for i in range(self.n_regions):
            if i == 0 and self.thresholds:
                result.append(f"< {self.thresholds[0]:.1f}")
            elif i == self.n_regions - 1 and self.thresholds:
                result.append(f"> {self.thresholds[-1]:.1f}")
            elif self.thresholds:
                result.append(
                    f"{self.thresholds[i - 1]:.1f} to {self.thresholds[i]:.1f}"
                )
            else:
                result.append("Region")

        return result
