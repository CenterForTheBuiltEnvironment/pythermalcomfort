from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from .utils import (
    make_metric_eval,
    solve_threshold_curves,
)


# Default plot configuration constants
DEFAULT_FIGSIZE = (7, 4)
DEFAULT_DPI = 300
DEFAULT_STYLE = "seaborn-v0_8-whitegrid"
MIN_REGION_WIDTH = 1e-12  # Minimum width for valid regions
DEFAULT_LEGEND_LOC = "lower center"  # Default legend location
DEFAULT_LEGEND_ANCHOR = (0.5, 1)  # Default legend anchor position

def calc_plot_ranges(
    *,
    model_func: Callable[..., Any],
    xy_to_kwargs: Callable[[float, float, dict[str, Any]], dict[str, Any]],
    fixed_params: dict[str, Any] | None,
    thresholds: Sequence[float],
    x_bounds: tuple[float, float],
    y_values: Sequence[float],
    metric_attr: str | None = None,
    ax: plt.Axes | None = None,
    # Visual controls
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    cmap: Colormap | str = "coolwarm",
    band_colors: Sequence[str] | None = None,
    band_alpha: float = 0.85,
    line_color: str = "black",
    line_width: float = 1.0,
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Optional per-y left boundary (e.g., psychrometric saturation clip)
    x_left_clip: Sequence[float] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot threshold regions for a generic (x, y) mapping to a model.

    This function visualizes regions defined by one or more threshold values
    for a metric computed from a model function, over a 2D domain defined by
    x and y axes. It is generic and can be used for any model where two
    variables are varied and the rest are fixed. The function returns the
    matplotlib Axes and a dictionary of artists for further customization.

    Parameters
    ----------
    model_func : Callable[..., Any]
        The model function to evaluate. Must accept keyword arguments for all
        required variables and return a result with the desired metric.
    xy_to_kwargs : Callable[[float, float, dict[str, Any]], dict[str, Any]]
        Function mapping (x, y, fixed_params) to the model_func kwargs dict.
        This allows flexible assignment of x/y to model variables.
    fixed_params : dict[str, Any] or None
        Dictionary of model parameters to keep fixed for all evaluations.
    thresholds : Sequence[float]
        List of threshold values to define the region boundaries. Must be
        sorted in increasing order.
    x_bounds : tuple[float, float]
        The (min, max) range for the x-axis variable.
    y_values : Sequence[float]
        Sequence of y-axis values at which to compute the threshold curves.
    metric_attr : str or None, optional
        Name of the attribute or key in the model_func result to use as the
        metric for thresholding. If None, the model_func result is used directly.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.
    xlabel : str or None, optional
        Label for the x-axis. If None, no label is set.
    ylabel : str or None, optional
        Label for the y-axis. If None, no label is set.
    legend : bool, default True
        Whether to add a default legend for the regions.
    cmap : matplotlib.colors.Colormap or str, default "coolwarm"
        Colormap to use for the regions. Ignored if band_colors is provided.
    band_colors : Sequence[str] or None, optional
        List of colors for each region between thresholds. If provided,
        overrides cmap. Must have length len(thresholds) + 1.
    band_alpha : float, default 0.85
        Alpha (opacity) for the filled regions.
    line_color : str, default "black"
        Color for the threshold curves.
    line_width : float, default 1.0
        Line width for the threshold curves.
    x_scan_step : float, default 1.0
        Step size for scanning the x-axis when solving for thresholds.
    smooth_sigma : float, default 0.8
        Sigma for optional smoothing of the threshold curves.
    x_left_clip : Sequence[float] or None, optional
        Optional per-y left boundary (e.g., for psychrometric saturation).
        Must be the same length as y_values if provided.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes with the plot.
    artists : dict[str, Any]
        Dictionary with keys 'bands', 'curves', and 'legend' containing the
        corresponding matplotlib artists for further customization.

    Raises
    ------
    ValueError
        If both cmap and band_colors are provided, or if band_colors has the
        wrong length, or if x_left_clip has the wrong shape.

    Examples
    --------
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.generic import calc_plot_ranges
    >>> def xy_to_kwargs(t, rh, fixed):
    ...     return {**fixed, "tdb": t, "rh": rh}
    >>> ax, artists = calc_plot_ranges(
    ...     model_func=pmv_ppd_iso,
    ...     xy_to_kwargs=xy_to_kwargs,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1},
    ...     thresholds=[-0.5, 0.5],
    ...     x_bounds=(18, 30),
    ...     y_values=np.linspace(10, 90, 100),
    ...     metric_attr="pmv",
    ...     xlabel="Air temperature [°C]",
    ...     ylabel="Relative humidity [%]",
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    if ax is None:
        plt.style.use(DEFAULT_STYLE)
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI, constrained_layout=True)

    # Build evaluator and compute curves
    metric_xy = make_metric_eval(
        model_func=model_func,
        xy_to_kwargs=xy_to_kwargs,
        fixed_params=fixed_params,
        metric_attr=metric_attr,
    )

    res = solve_threshold_curves(
        metric_xy=metric_xy,
        thresholds=thresholds,
        y_values=y_values,
        x_bounds=x_bounds,
        x_scan_step=x_scan_step,
        smooth_sigma=smooth_sigma,
    )

    curves: list[np.ndarray] = res["curves"]
    y_values_array: np.ndarray = res["y_values"]
    thresholds_list: list[float] = res["thresholds"]

    # Prepare color bands
    # Validate that only one color specification method is used
    if band_colors is not None and cmap != "coolwarm":
        raise ValueError("Cannot specify both band_colors and custom cmap. Use either band_colors or the default cmap.")

    needed = len(thresholds) + 1
    if band_colors is not None:
        if len(band_colors) != needed:
            raise ValueError("band_colors must have length equal to number of regions")
    else:
        cmap_obj = plt.get_cmap(cmap)
        band_colors = [cmap_obj(i / (needed - 1)) for i in range(needed)]

    # Optional left clip per y (same length as y_values_array)
    clip_arr = None
    if x_left_clip is not None:
        clip_arr = np.asarray(list(x_left_clip), dtype=float)
        if clip_arr.shape != y_values_array.shape:
            raise ValueError("x_left_clip must have same shape as y_values")

    # Constant x boundaries
    x_lo, x_hi = float(x_bounds[0]), float(x_bounds[1])
    left_const = np.full_like(y_values_array, x_lo, dtype=float)
    right_const = np.full_like(y_values_array, x_hi, dtype=float)

    # Build regions between curves
    regions = []
    
    if curves:
        # First region: from left boundary to first curve
        regions.append((left_const, curves[0]))
        
        # Middle regions: between consecutive curves
        for i in range(len(curves) - 1):
            regions.append((curves[i], curves[i + 1]))
        
        # Last region: from last curve to right boundary
        regions.append((curves[-1], right_const))
    else:
        # No curves: single region from left to right boundary
        regions.append((left_const, right_const))

    band_artists = []
    for i, (left, right) in enumerate(regions):
        # Apply left clip if provided to ensure valid domain (e.g., RH <= 100%)
        left_plot = np.maximum(left, clip_arr) if clip_arr is not None else left
        
        # Combined mask for finite values and positive width
        finite_mask = np.isfinite(left_plot) & np.isfinite(right)
        width_mask = (right - left_plot) > MIN_REGION_WIDTH
        valid_mask = finite_mask & width_mask
        
        if valid_mask.any():
                coll = ax.fill_betweenx(
                    y_values_array[valid_mask],
                    left_plot[valid_mask],
                    right[valid_mask],
                    color=band_colors[i],
                    alpha=band_alpha,
                    linewidth=0,
                )
                band_artists.append(coll)

    # Draw threshold curves (clip to valid domain if clip_arr provided)
    curve_artists = []
    for curve in curves:
        m = np.isfinite(curve)
        if clip_arr is not None:
            m = m & (curve >= clip_arr)
        if m.any():
            (ln,) = ax.plot(curve[m], y_values_array[m], color=line_color, linewidth=line_width)
            curve_artists.append(ln)

    # Minimal axis labels
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Default legend with band labels
    legend_artist = None
    if legend:
        legend_elements = []
        for i in range(needed):
            if i == 0 and len(thresholds_list) > 0:
                label = f"< {thresholds_list[0]:.1f}"
            elif i == needed - 1 and len(thresholds_list) > 0:
                label = f"> {thresholds_list[-1]:.1f}"
            elif len(thresholds_list) == 0:
                label = "Region"
            else:
                label = f"{thresholds_list[i - 1]:.1f} to {thresholds_list[i]:.1f}"
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=band_colors[i],
                    alpha=band_alpha,
                    label=label,
                )
            )
        legend_artist = ax.legend(
            handles=legend_elements,
            loc=DEFAULT_LEGEND_LOC,
            bbox_to_anchor=DEFAULT_LEGEND_ANCHOR,
            ncol=min(6, len(legend_elements)),
            framealpha=0.8,
            markerscale=0.6,
        )

    artists = {"bands": band_artists, "curves": curve_artists, "legend": legend_artist}
    return ax, artists
