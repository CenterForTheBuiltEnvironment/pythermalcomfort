from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.plots.generic import plot_threshold_region
from pythermalcomfort.plots.utils import get_default_thresholds, mapper_tdb_rh

__all__ = ["plot_t_rh"]


def _validate_range(name: str, rng: tuple[float, float]) -> tuple[float, float]:
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        raise ValueError(f"{name} must be a (min, max) tuple")
    lo, hi = float(rng[0]), float(rng[1])
    if lo >= hi:
        raise ValueError(f"{name} must be strictly increasing (min < max)")
    return lo, hi


def plot_t_rh(
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
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk region on a Temperature–Relative Humidity chart.

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

    # Delegate to generic plotter
    ax, artists = plot_threshold_region(
        model_func=model_func,
        xy_to_kwargs=mapper_tdb_rh,
        fixed_params=fixed_params,
        thresholds=thresholds,
        x_bounds=(t_lo, t_hi),
        y_values=y_values,
        metric_attr=None,
        ax=ax,
        xlabel="Air temperature [°C]",
        ylabel="Relative humidity [%]",
        legend=legend,
        x_scan_step=float(x_scan_step),
        smooth_sigma=float(smooth_sigma),
    )

    return ax, artists


if __name__ == "__main__":
    # Tiny smoke example (requires user to supply parameters as needed)
    from pythermalcomfort.models import pmv_ppd_iso

    ax, _ = plot_t_rh(
        model_func=pmv_ppd_iso,
        fixed_params={"tr": 30, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(10, 36),
        rh_range=(0, 100),
    )
    ax.set_title("PMV comfort region (example)")
    plt.show()

    # Tiny smoke example (requires user to supply parameters as needed)
    from pythermalcomfort.models import set_tmp

    ax, _ = plot_t_rh(
        model_func=set_tmp,
        fixed_params={"tr": 30, "met": 1.2, "clo": 0.5, "v": 0.1, "wme": 0.0},
        thresholds=[26, 28, 30],
        t_range=(10, 36),
        rh_range=(0, 100),
    )
    ax.set_title("PMV comfort region (example)")
    plt.show()
