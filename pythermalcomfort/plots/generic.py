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
    """Plot threshold regions for a generic (x,y) mapping to a model.

    This is a generic utility to visualize comfort/risk regions defined by
    metric(x,y) thresholds. Only minimal formatting is applied. The Axes is
    returned for further customization by the caller.

    Returns
    -------
    ax, artists
        The Matplotlib Axes and a dict with 'bands', 'curves', 'legend'.
    """
    if ax is None:
        plt.style.use("seaborn-v0_8-whitegrid")
        _, ax = plt.subplots(figsize=(7, 5), dpi=300, constrained_layout=True)

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
    y_arr: np.ndarray = res["y_values"]
    thr_list: list[float] = res["thresholds"]

    # Prepare color bands
    if band_colors is not None and cmap != "coolwarm":
        raise ValueError("Provide only one of cmap or band_colors, not both.")

    needed = len(thresholds) + 1
    if band_colors is not None:
        if len(band_colors) != needed:
            raise ValueError("band_colors must have length equal to number of regions")
        band_colors = band_colors
    else:
        cmap_obj = plt.get_cmap(cmap)
        band_colors = [cmap_obj(i / (needed - 1)) for i in range(needed)]

    # Optional left clip per y (same length as y_arr)
    clip_arr = None
    if x_left_clip is not None:
        clip_arr = np.asarray(list(x_left_clip), dtype=float)
        if clip_arr.shape != y_arr.shape:
            raise ValueError("x_left_clip must have same shape as y_values")

    # Constant x boundaries
    x_lo, x_hi = float(x_bounds[0]), float(x_bounds[1])
    left_const = np.full_like(y_arr, x_lo, dtype=float)
    right_const = np.full_like(y_arr, x_hi, dtype=float)

    # Fill regions between curves
    regions = (
        ([(left_const, curves[0])] if curves else [])
        + [(curves[i], curves[i + 1]) for i in range(len(curves) - 1)]
        + ([(curves[-1], right_const)] if curves else [(left_const, right_const)])
    )

    band_artists = []
    for i, (left, right) in enumerate(regions):
        # Apply left clip if provided to ensure valid domain (e.g., RH <= 100%)
        left_plot = np.maximum(left, clip_arr) if clip_arr is not None else left
        m = np.isfinite(left_plot) & np.isfinite(right)
        if m.any():
            # Only fill where the band has positive width after clipping
            width_mask = (right - left_plot) > 1e-12
            m = m & width_mask
            if m.any():
                coll = ax.fill_betweenx(
                    y_arr[m],
                    left_plot[m],
                    right[m],
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
            (ln,) = ax.plot(curve[m], y_arr[m], color=line_color, linewidth=line_width)
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
            if i == 0 and len(thr_list) > 0:
                label = f"< {thr_list[0]:.1f}"
            elif i == needed - 1 and len(thr_list) > 0:
                label = f"> {thr_list[-1]:.1f}"
            elif len(thr_list) == 0:
                label = "Region"
            else:
                label = f"{thr_list[i - 1]:.1f} to {thr_list[i]:.1f}"
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
            loc="lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=min(6, len(legend_elements)),
            framealpha=0.8,
            markerscale=0.6,
        )

    artists = {"bands": band_artists, "curves": curve_artists, "legend": legend_artist}
    return ax, artists
