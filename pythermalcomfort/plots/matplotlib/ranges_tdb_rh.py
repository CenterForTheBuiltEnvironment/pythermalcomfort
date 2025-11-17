from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import calc_plot_ranges
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    mapper_tdb_rh,
)
import seaborn as sns

__all__ = ["ranges_tdb_rh"]


def ranges_tdb_rh(
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
    # Forwarded plot customizations (visual + solver) to plot_threshold_region
    plot_kwargs: dict[str, Any] | None = None,
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk region on a Temperature Relative Humidity chart.

    Minimal formatting is applied; the function returns the Matplotlib Axes
    so callers can further customize. It can also return the created artists
    for advanced styling if needed.

    Args
    ----
    model_func:
        A pythermalcomfort model callable, e.g., pmv_ppd_iso, set_tmp,
        heat_index_rothfusz, utci. The function must accept tdb and rh among
        its parameters. Non-(tdb,rh) required parameters must be provided via
        ``fixed_params``. No hidden defaults are applied.
    fixed_params:
        Dict of model parameters held constant (e.g., met, clo, v). Required
        by many models.
    thresholds:
        Sequence of threshold values for the metric returned by ``model_func``.
        If None, a registered default will be used if available; otherwise a
        ValueError is raised.
    t_range:
        (min, max) air temperature bounds in °C.
    rh_range:
        (min, max) relative humidity bounds in %.
    rh_step:
        Spacing in relative humidity used to compute curves (default 2 %).
    x_scan_step:
        Step in °C for bracketing across temperature (default 1 °C).
    smooth_sigma:
        Gaussian smoothing sigma (in RH index units). Set 0 to disable.
    ax:
        Optional Matplotlib Axes. If None, a new figure/axes is created.
    legend:
        Whether to add a default legend for the filled bands.
    plot_kwargs:
        Optional dict of keyword overrides passed to ``plot_threshold_region``.
        Use this to customize visual/solver defaults without expanding this API.
        Examples: {'cmap': 'viridis', 'band_alpha': 0.6, 'line_color': 'k',
        'x_scan_step': 0.5, 'smooth_sigma': 0.0}.

    Returns
    -------
    ax, artists
        The Matplotlib Axes and a dict with 'bands', 'curves', 'legend'.
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
        "xy_to_kwargs": mapper_tdb_rh,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Air temperature [°C]",
        "ylabel": "Relative humidity [%]",
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
    }
    if plot_kwargs:
        # Let user-provided overrides take precedence (e.g., cmap/band_alpha/etc.)
        kwargs.update(plot_kwargs)

    # Delegate to generic plotter
    ax, artists = calc_plot_ranges(**kwargs)

    return ax, artists


if __name__ == "__main__":
    from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso
    import matplotlib.pyplot as plt

    fixed_params = {
        "tr": 25,
        "vr": 0.3,
        "met": 1.2,
        "clo": 0.5,
    }

    ax, artists = ranges_tdb_rh(
        model_func=pmv_ppd_iso,
        fixed_params=fixed_params,
        t_range=(10, 40),
        rh_range=(0, 100),
        thresholds=[-3, -2, -1, 0, 1, 2, 3],
    )

    fig = plt.gcf()
    fig.set_size_inches(4, 3)         
    plt.title("PMV Comfort Zones vs. Air Temp and Humidity (ISO 7730)", fontsize=8, pad=15)
    plt.xlabel("Air Temperature [°C]", fontsize=7)
    plt.ylabel("Relative Humidity [%]", fontsize=7)

    plt.xlim(10, 40)
    plt.ylim(0, 100)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    leg = plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
                     ncol=5, fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig("pmv_iso_comfort_zones.png", dpi=300, bbox_inches="tight")
    plt.show()



