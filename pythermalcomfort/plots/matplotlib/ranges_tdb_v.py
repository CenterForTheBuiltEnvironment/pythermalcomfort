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
    """Plot comfort/risk ranges on an air temperature vs air speed chart.

    This function visualizes regions defined by one or more threshold values for a
    comfort metric (e.g., PMV, SET) as a function of air temperature (x-axis)
    and air speed (y-axis). The most commonly used visual parameters
    are exposed directly in the function signature, while additional customization
    options are available through **kwargs.

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
        Additional keyword arguments passed to ``calc_plot_ranges``.
        Common options include:
        - band_colors: list of colors for each region (overrides cmap)
        Note: To set axis labels, title, or figure size, use the returned Axes object:
        ``ax.set_xlabel(...)``, ``ax.set_title(...)``, ``ax.figure.set_size_inches(...)``

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes with the plot. Use this to customize labels, title,
        and figure size: ``ax.set_xlabel(...)``, ``ax.set_ylabel(...)``,
        ``ax.set_title(...)``, ``ax.figure.set_size_inches(...)``
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
    Basic usage:
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.matplotlib import ranges_tdb_v
    >>> ax, artists = ranges_tdb_v(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "rh": 50, "met": 1.2, "clo": 0.5},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     v_range=(0.0, 1.0),
    ...     v_step=0.05,
    ... )
    >>> import matplotlib.pyplot as plt
    >>> plt.show()
    
    With visual customization:
    >>> ax, artists = ranges_tdb_v(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "rh": 50, "met": 1.2, "clo": 0.5},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     v_range=(0.0, 1.0),
    ...     v_step=0.05,
    ...     cmap="viridis",
    ...     band_alpha=0.6,
    ...     line_color="darkred",
    ...     line_width=2.0,
    ... )
    >>> plt.show()
    
    With advanced customization via **kwargs and returned Axes:
    >>> ax, artists = ranges_tdb_v(
    ...     model_func=pmv_ppd_iso,
    ...     fixed_params={"tr": 25, "rh": 50, "met": 1.2, "clo": 0.5},
    ...     thresholds=[-0.5, 0.5],
    ...     t_range=(18, 30),
    ...     v_range=(0.0, 1.0),
    ...     v_step=0.05,
    ...     cmap="viridis",
    ...     band_alpha=0.6,
    ...     band_colors=["lightblue", "lightgreen", "lightcoral"],
    ... )
    >>> ax.set_xlabel("Air temperature [°C]")
    >>> ax.set_ylabel("Air speed [m/s]")
    >>> ax.set_title("Custom Temperature vs Air Speed Chart")
    >>> ax.figure.set_size_inches(10, 6)
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

    # Prepare call with explicit visual parameters and **kwargs for additional customization
    calc_kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_tdb_vr,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
        # Explicit visual controls
        "cmap": cmap,
        "band_alpha": band_alpha,
        "line_color": line_color,
        "line_width": line_width,
    }
    # Allow additional parameters via **kwargs (overrides defaults)
    if kwargs:
        calc_kwargs.update(kwargs)

    ax, artists = calc_plot_ranges(**calc_kwargs)

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
