from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import calc_plot_ranges
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    humidity_ratio_from_t_rh,
    mapper_tdb_w,
)

__all__ = ["ranges_tdb_psychrometric"]


def ranges_tdb_psychrometric(
    model_func: Callable[..., Any],
    *,
    fixed_params: dict[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
    t_range: tuple[float, float] = (10.0, 36.0),
    w_range: tuple[float, float] = (0.0, 0.03),
    w_step: float = 5e-4,  # 0.5 g/kg
    rh_isolines: Sequence[float] = tuple(range(10, 100, 10)) + (100,),
    draw_background: bool = True,
    # Visual controls

    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Additional matplotlib parameters
    title: str | None = None,             
    fontsize: float = 12.0,                
    plot_kwargs: dict[str, Any] | None = None,  # The previous version allowed direct transmission of any parameters, which may have affected the core logic.
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk ranges over a psychrometric chart (Tdb vs humidity ratio).

    Visual appearance can be overridden via ``plot_kwargs`` (e.g., cmap, labels, etc.).
    Core solving logic remains unchanged.
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
     # The style is not fixed; all settings will be based on the Figure level from now on.
    if ax is None:
        _, ax = plt.subplots()  

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
        if handles and legend:  
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
    # Explicit visual controls
    if plot_kwargs: kwargs.update({k: v for k, v in plot_kwargs.items() if k not in ("model_func", "xy_to_kwargs")})
    #Controllable appearance and reliable computing
    ax, artists = calc_plot_ranges(**kwargs)  


    ax.set_xlim(t_lo, t_hi)  
    ax.set_ylim(w_lo, w_hi)   


    fig = ax.figure                       
    fig.set_size_inches(7, 5)             
    fig.set_dpi(150)                      # Size and clarity modification
    fig.suptitle(                         # Place the title on the Figure layer to avoid overlapping with the legend.
        title or "Psychrometric Chart (Tdb vs Humidity Ratio)",
        fontsize=fontsize,
        y=0.96,
    )

    try:
        pos = ax.get_position()
        ax.set_position([pos.x0, max(0.06, pos.y0 + 0.02), pos.width, pos.height * 0.92])  # Move the text upwards and compress it to leave space for the title/legend.
    except Exception:
        pass  # To prevent individual backend errors

    return ax, artists


if __name__ == "__main__":
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = ranges_tdb_psychrometric(
        model_func=pmv_ppd_iso,
        fixed_params={
            "tr": 25.0,
            "met": 1.2,
            "clo": 0.5,
            "vr": 0.1,
            "wme": 0.0,
          
        },
        thresholds=[-0.5, 0.5],
        t_range=(18, 30),
        w_range=(0.002, 0.018),
        w_step=0.001,

        title="Psychrometric comfort ranges",  
        fontsize=12.0,                          
    )

    import matplotlib.pyplot as plt
    plt.show()
