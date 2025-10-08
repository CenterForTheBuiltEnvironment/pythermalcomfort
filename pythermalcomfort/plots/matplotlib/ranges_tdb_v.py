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
    # Forwarded plot customizations (visual + solver) to plot_threshold_region
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot metric regions on a Temperature vs air-speed (vr) chart.

    Use plot_kwargs to override plotting defaults (cmap, band_alpha, etc.)
    without expanding the API (e.g., plot_kwargs={'cmap': 'viridis'}).
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
        kwargs.update(plot_kwargs)

    ax, artists = calc_plot_ranges(**kwargs)

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
