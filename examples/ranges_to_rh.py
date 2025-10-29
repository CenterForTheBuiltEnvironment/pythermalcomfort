from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "Plotting requires the optional 'plots' extra. "
        "Install with: pip install pythermalcomfort[plots]"
    ) from exc
import numpy as np

from pythermalcomfort.plots.generic import calc_plot_ranges
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    mapper_top_rh,
)

__all__ = ["ranges_to_rh"]


def ranges_to_rh(
    model_func: Callable[..., Any],
    *,
    fixed_params: dict[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
    t_range: tuple[float, float] = (10.0, 36.0),
    rh_range: tuple[float, float] = (0.0, 100.0),
    rh_step: float = 2.0,

    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Additional matplotlib parameters
    
    title: str | None = None,                    
    fontsize: float = 12.0,                    
    plot_kwargs: dict[str, Any] | None = None,   
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk ranges on an operative temperature vs relative humidity chart.

    This function visualizes regions defined by one or more threshold values for a
    comfort metric (e.g., PMV, SET) as a function of operative temperature (x-axis)
    and relative humidity (y-axis). Visual options can be overridden via plot_kwargs.
    """

    # Validate ranges and steps
    t_lo, t_hi = _validate_range("t_range", t_range)
    rh_lo, rh_hi = _validate_range("rh_range", rh_range)
    if rh_step <= 0:
        raise ValueError("rh_step must be positive")
    if x_scan_step <= 0:
        raise ValueError("x_scan_step must be positive")

    # Determine thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(model_func)
        if thresholds is None:
            raise ValueError(
                "No thresholds provided and no defaults registered for this model."
            )

    # Build y (RH) grid
    y_values = np.arange(rh_lo, rh_hi + 1e-9, float(rh_step))
    
    kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_top_rh,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Operative temperature [°C]",    
        "ylabel": "Relative humidity [%]",         
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
    }  

    if plot_kwargs: kwargs.update({k: v for k, v in plot_kwargs.items() if k not in ("model_func", "xy_to_kwargs")})
    ax, artists = calc_plot_ranges(**kwargs)       # The call remains unchanged, only the source of the parameters changes

    
    fig = ax.figure                                  
    fig.set_size_inches(7, 5)                        
    fig.set_dpi(150)                                 
    fig.suptitle(                                    
        title or "Operative Temperature vs Relative Humidity",
        fontsize=fontsize,
        y=0.96,
    )

  
    try:
        pos = ax.get_position()
        ax.set_position([pos.x0, max(0.06, pos.y0 + 0.02), pos.width, pos.height * 0.92])  # 新：留白
    except Exception:
        pass 

    return ax, artists


if __name__ == "__main__":
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = ranges_to_rh(
        model_func=pmv_ppd_iso,
        fixed_params={"met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(18, 30),
        rh_range=(10, 90),
        rh_step=2.0,
        title="Operative Temperature vs RH",            
        fontsize=12.0,                                 
    )

    import matplotlib.pyplot as plt
    plt.show()
