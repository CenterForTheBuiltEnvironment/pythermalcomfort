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
    # Visual controls (most commonly used)
    cmap: str = "coolwarm",
    band_alpha: float = 0.85,
    line_color: str = "black",
    line_width: float = 1.0,
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Additional matplotlib parameters
    **kwargs: Any,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk ranges over a psychrometric chart (Tdb vs humidity ratio).

    This function visualizes regions defined by one or more threshold values for a
    comfort metric (e.g., PMV, SET) as a function of dry-bulb temperature (x-axis)
    and humidity ratio (y-axis). The background shows the saturation curve and
    relative humidity isolines. The most commonly used visual parameters
    are exposed directly in the function signature, while additional customization
    options are available through **kwargs.

    Parameters
    ----------
    model_func : Callable[..., Any]
        The comfort model function to evaluate. Must accept keyword arguments for all
        required variables and return a result with the desired metric.
    fixed_params : dict[str, Any] or None, optional
        Dictionary of model parameters to keep fixed for all evaluations (e.g.,
        tr, met, clo, vr, wme, p_atm). If None, all non-x/y model arguments must be
        provided by the user.
    thresholds : Sequence[float] or None, optional
        List of threshold values to define the region boundaries. If None, uses
        the default thresholds registered for the model.
    t_range : tuple[float, float], default (10.0, 36.0)
        The (min, max) range for dry-bulb temperature [°C] on the x-axis.
    w_range : tuple[float, float], default (0.0, 0.03)
        The (min, max) range for humidity ratio [kg/kg] on the y-axis.
    w_step : float, default 5e-4
        Step size for the humidity ratio grid.
    rh_isolines : Sequence[float], default (10, 20, ..., 100)
        Relative humidity values (%) for background isolines.
    draw_background : bool, default True
        Whether to draw the saturation curve and RH isolines in the background.
    cmap : str, default "coolwarm"
        Colormap name for the regions. Common options: "coolwarm", "viridis",
        "plasma", "RdYlBu". See matplotlib colormaps for full list.
    band_alpha : float, default 0.85
        Transparency (0-1) for the filled regions.
    line_color : str, default "black"
        Color for the threshold curves.
    line_width : float, default 1.0
        Line width for the threshold curves.
    x_scan_step : float, default 1.0
        Step size for scanning the temperature axis when solving for thresholds.
    smooth_sigma : float, default 0.8
        Sigma for optional smoothing of the threshold curves.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If None, a new figure and axes are created.
    legend : bool, default True
        Whether to add a default legend for the regions.
    **kwargs : Any
        Additional keyword arguments passed to the underlying plotting function.
        Common options include:
        - band_colors: list of colors for each region (overrides cmap)
        - xlabel, ylabel: axis labels
        - title: plot title
        - figsize: figure size tuple
        For more options, see matplotlib.pyplot documentation.

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
        If w_step or x_scan_step is not positive, or if no thresholds are provided
        and no defaults are registered for the model.

    Examples
    --------
    Basic usage:
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.matplotlib import ranges_tdb_psychrometric
    >>> ax, artists = ranges_tdb_psychrometric(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     w_range=(0.002, 0.018),
    ...     w_step=0.001,
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    
    With visual customization:
    >>> ax, artists = ranges_tdb_psychrometric(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     w_range=(0.002, 0.018),
    ...     w_step=0.001,
    ...     cmap="viridis",
    ...     band_alpha=0.6,
    ...     line_color="darkred",
    ...     line_width=2.0,
    ...     draw_background=True,
    ... )
    >>> plt.show()
    
    With advanced customization via **kwargs:
    >>> ax, artists = ranges_tdb_psychrometric(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     w_range=(0.002, 0.018),
    ...     w_step=0.001,
    ...     cmap="viridis",
    ...     band_alpha=0.6,
    ...     band_colors=["lightblue", "lightgreen", "lightcoral"],
    ...     title="Custom Psychrometric Chart",
    ...     figsize=(12, 8),
    ... )
    >>> plt.show()
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

    if ax is None:
        plt.style.use("seaborn-v0_8-whitegrid")
        _, ax = plt.subplots(figsize=(7, 3), dpi=300, constrained_layout=True)

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
    calc_kwargs: dict[str, Any] = {
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
        # Explicit visual controls
        "cmap": cmap,
        "band_alpha": band_alpha,
        "line_color": line_color,
        "line_width": line_width,
    }
    # Allow additional matplotlib parameters via **kwargs
    calc_kwargs.update(kwargs)

    ax, artists = calc_plot_ranges(**calc_kwargs)

    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(w_lo, w_hi)

    return ax, artists
