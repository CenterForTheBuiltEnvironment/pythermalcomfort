from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import plot_threshold_region
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    humidity_ratio_from_t_rh,
    mapper_tdb_w,
)

__all__ = ["plot_psychrometric_regions"]


def plot_psychrometric_regions(
    model_func: Callable[..., Any],
    *,
    fixed_params: dict[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
    t_range: tuple[float, float] = (10.0, 36.0),
    w_range: tuple[float, float] = (0.0, 0.03),
    w_step: float = 5e-4,  # 0.5 g/kg
    rh_isolines: Sequence[float] = tuple(range(10, 100, 10)) + (100,),
    draw_background: bool = True,
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Forwarded plot customizations (visual + solver) to plot_threshold_region
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort regions over a simple psychrometric chart.

    The x-axis is dry-bulb temperature (°C). The y-axis is humidity ratio
    (kg water/kg dry air). Background shows saturation and RH isolines.
    Pressure is taken from fixed_params['p_atm'] if provided (Pa), else
    101325 Pa. Use plot_kwargs to customize colors/lines like other wrappers.
    """
    fixed_params = dict(fixed_params or {})
    # Validate ranges and steps
    t_lo, t_hi = _validate_range("t_range", t_range)
    w_lo, w_hi = _validate_range("w_range", w_range)
    if w_step <= 0:
        raise ValueError("w_step must be positive")
    if x_scan_step <= 0:
        raise ValueError("x_scan_step must be positive")

    # Determine thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(model_func)
        if thresholds is None:
            raise ValueError(
                "No thresholds provided and no defaults registered for this model."
            )

    # Init axes
    if ax is None:
        plt.style.use("seaborn-v0_8-whitegrid")
        _, ax = plt.subplots(figsize=(8, 5.5), dpi=300, constrained_layout=True)

    # Background chart (saturation curve and RH isolines)
    p_pa = float(fixed_params.get("p_atm", 101325.0))
    t_grid_bg = np.linspace(t_lo, t_hi, 300)
    if draw_background:
        # RH isolines (10..100%)
        for rh in rh_isolines:
            w_curve = np.array(
                [humidity_ratio_from_t_rh(t, rh, p_pa=p_pa) for t in t_grid_bg]
            )
            if int(rh) == 100:
                ax.plot(
                    t_grid_bg, w_curve, color="dimgray", lw=1.2, label="100% RH (sat.)"
                )
            else:
                ax.plot(t_grid_bg, w_curve, color="lightgray", lw=0.6, ls="--")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                loc="lower right",
                framealpha=0.9,
                fontsize=8,
            )

    # Build y (humidity ratio) grid for solver
    y_values = np.arange(w_lo, w_hi + 1e-12, float(w_step))

    # Compute left clip per y to enforce RH <= 100% (to the right of saturation)
    # For each W=y, find the minimum T where W_sat(T) >= y
    t_grid_clip = np.linspace(t_lo, t_hi, 600)
    w_sat_clip = np.array(
        [humidity_ratio_from_t_rh(t, 100.0, p_pa=p_pa) for t in t_grid_clip]
    )
    x_left_clip = np.full_like(y_values, t_hi, dtype=float)
    for i, w in enumerate(y_values):
        mask = w_sat_clip >= float(w)
        if mask.any():
            # first temperature where saturation curve is above this W
            x_left_clip[i] = float(t_grid_clip[np.argmax(mask)])
        else:
            # if never achievable within [t_lo, t_hi], leave at t_hi (no area)
            x_left_clip[i] = t_hi

    # Delegate region plotting (mapper converts W->RH internally)
    kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_tdb_w,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Dry-bulb air temperature [°C]",
        "ylabel": "Humidity ratio [kg/kg]",
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
        "x_left_clip": x_left_clip,
    }
    if plot_kwargs:
        kwargs.update(plot_kwargs)

    ax, artists = plot_threshold_region(**kwargs)

    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(w_lo, w_hi)

    return ax, artists


if __name__ == "__main__":
    # Minimal example (ensure model_func can accept tdb/rh and other params)
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = plot_psychrometric_regions(
        pmv_ppd_iso,
        fixed_params={
            "tr": 25.0,
            "met": 1.0,
            "clo": 0.61,
            "vr": 0.1,
            "wme": 0.0,
        },
        thresholds=[-0.5, 0.5],
        t_range=(10, 36),
        w_range=(0.0, 0.03),
        w_step=5e-4,
        plot_kwargs={"band_alpha": 0.35, "line_color": "k", "cmap": "viridis"},
    )
    plt.show()
