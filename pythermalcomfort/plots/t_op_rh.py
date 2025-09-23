from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import plot_threshold_region
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    mapper_top_rh,
)

__all__ = ["plot_top_rh"]


def plot_top_rh(
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
    # Forwarded plot customizations to plot_threshold_region
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot regions on an operative temperature vs relative humidity chart.

    Enforces tr == tdb by construction. Use plot_kwargs to override visuals.
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

    # Prepare call with sane defaults then let plot_kwargs override
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
    if plot_kwargs:
        kwargs.update(plot_kwargs)

    ax, artists = plot_threshold_region(**kwargs)
    return ax, artists


if __name__ == "__main__":
    # Example (user must pass appropriate fixed params for their model)
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = plot_top_rh(
        model_func=pmv_ppd_iso,
        fixed_params={"met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(10, 36),
        rh_range=(0, 100),
        # plot_kwargs={"cmap": "viridis", "band_alpha": 0.6},
    )
    plt.show()
