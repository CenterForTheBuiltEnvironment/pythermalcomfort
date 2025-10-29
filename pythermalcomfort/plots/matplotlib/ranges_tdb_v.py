from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import calc_plot_ranges
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    mapper_tdb_vr,
)

__all__ = ["ranges_tdb_v"]


def ranges_tdb_v(
    model_func: Callable[..., Any],
    *,
    fixed_params: dict[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
    t_range: tuple[float, float] = (10.0, 36.0),
    v_range: tuple[float, float] = (0.0, 1.5),
    v_step: float = 0.05,
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Title parameter
    title: str | None = None,  
    # Font size parameter 
    fontsize: float = 12,     
    # Forwarded plot customizations (visual + solver) to plot_threshold_region
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort metric regions on an air temperature vs air speed (vr) chart.

    This function visualizes regions defined by one or more threshold values for a
    comfort metric (e.g., PMV, SET) as a function of air temperature (x-axis)
    and air speed (y-axis). It is a convenience wrapper around ``calc_plot_ranges``
    with sensible defaults for temperature and air speed, and is suitable for most
    comfort models in pythermalcomfort.

    Parameters
    ----------
    model_func : Callable[..., Any]
        The comfort model function to evaluate. Must accept keyword arguments for all
        required variables and return a result with the desired metric.
    fixed_params : dict[str, Any] or None, optional
        Dictionary of model parameters to keep fixed for all evaluations (e.g.,
        tr, met, clo, rh, wme). If None, all non-x/y model arguments must be
        provided by the user.
    thresholds : Sequence[float] or None, optional
        List of threshold values to define the region boundaries. If None, uses
        the default thresholds registered for the model.
    t_range : tuple[float, float], default (10.0, 36.0)
        The (min, max) range for air temperature [°C] on the x-axis.
    v_range : tuple[float, float], default (0.0, 1.5)
        The (min, max) range for air speed [m/s] on the y-axis.
    v_step : float, default 0.05
        Step size for the air speed grid.
    x_scan_step : float, default 1.0
        Step size for scanning the temperature axis when solving for thresholds.
    smooth_sigma : float, default 0.8
        Sigma for optional smoothing of the threshold curves.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.
    legend : bool, default True
        Whether to add a default legend for the regions.
    plot_kwargs : dict[str, Any] or None, optional
        Additional keyword arguments forwarded to ``calc_plot_ranges`` for further
        customization (e.g., cmap, band_colors, xlabel, ylabel, etc.).

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
        If v_step or x_scan_step is not positive, or if no thresholds are provided
        and no defaults are registered for the model.

    Examples
    --------
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.matplotlib import ranges_tdb_v
    >>> ax, artists = ranges_tdb_v(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "rh": 50, "met": 1.2, "clo": 0.5},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     v_range=(0.0, 1.0),
    ...     v_step=0.05,
    ...     plot_kwargs={"cmap": "viridis"},
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    """
    # Validate ranges and steps
    t_lo, t_hi = _validate_range("t_range", t_range)
    v_lo, v_hi = _validate_range("v_range", v_range)
    if v_step <= 0:
        raise ValueError("v_step must be positive")
    if x_scan_step <= 0:
        raise ValueError("x_scan_step must be positive")

    # Determine thresholds
    if thresholds is None:
        thresholds = get_default_thresholds(model_func)
        if thresholds is None:
            raise ValueError(
                "No thresholds provided and no defaults registered for this model."
            )

    # Build y (air speed) grid
    y_values = np.arange(v_lo, v_hi + 1e-12, float(v_step))

    # Prepare call with sane defaults then let plot_kwargs override
    kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_tdb_vr,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Air temperature [°C]",
        "ylabel": "Air speed [m/s]",
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
    }
    
    if plot_kwargs:
        # only prevent logic parameters from being overridden
        kwargs.update({k: v for k, v in plot_kwargs.items() if k not in ("model_func", "xy_to_kwargs")})

    ax, artists = calc_plot_ranges(**kwargs)

    # Automatically control figure size (ensure the chart is properly scaled)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    fig.set_dpi(150)
    # Place title at the Figure level (ensure it doesn’t overlap with legend)
    fig.suptitle(title or "Air Temperature vs Air Speed", fontsize=fontsize, y=0.96) 

    # Adjust plot area position (to leave space for the title and legend）
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - 0.03, pos.width, pos.height])

    return ax, artists


if __name__ == "__main__":
    # Small smoke test (user must supply appropriate fixed params)
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = ranges_tdb_v(
        model_func=pmv_ppd_iso,
        fixed_params={"tr": 30, "rh": 50, "met": 1.2, "clo": 0.5, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(10, 36),
        v_range=(0.0, 1.5),
        v_step=0.05,
    )
    plt.show()
