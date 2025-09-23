"""Plotting utilities for threshold-based comfort regions.

This module provides a generic, reusable solver to compute threshold
curves for any metric derived from a pythermalcomfort model. It avoids
hidden defaults: callers must pass all non-(x,y) model parameters.

Key components
--------------
- DEFAULT_THRESHOLDS: internal registry of sensible default thresholds
  per model function name. Only thresholds are provided by default.
- make_metric_eval: builds a metric(x, y) function from a model function
  and a small "mapper" that turns (x, y) into model kwargs.
- solve_threshold_curves: generic solver returning x(y) arrays for one or
  multiple thresholds, with optional smoothing and warnings for unsolved
  points.

Notes
-----
- Root finding uses a uniform scan across x-bounds to locate a sign
  change, then Brent's method (brentq) on the first bracketing interval
  found (left-to-right). This is necessary because brentq requires a
  bracket; scanning finds it robustly.
- Smoothing is applied along the y-axis on contiguous valid segments
  only; NaN segments are preserved.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

# --------------------------- Defaults registry ----------------------------

DEFAULT_THRESHOLDS: dict[str, list[float]] = {
    # PMV common comfort band (can be overridden by caller)
    "pmv_ppd_iso": [-0.5, 0.5],
    "pmv_ppd_ashrae": [-0.5, 0.5],
    # SET commonly used breakpoints
    "set_tmp": [22.0, 24.0, 26.0, 28.0, 32.0],
    # Heat Index categories (approximate)
    "heat_index_rothfusz": [30.0, 35.0, 40.0, 55.0],
    # UTCI stress categories (approximate edges)
    "utci": [-40.0, -27.0, -13.0, -1.0, 9.0, 26.0, 32.0, 38.0, 46.0],
}


def get_default_thresholds(model_func: Callable[..., Any]) -> list[float] | None:
    """Return default thresholds for a model function if known.

    Args:
        model_func: The model function from pythermalcomfort.models.

    Returns:
        A list of float thresholds or None if no defaults are registered.
    """
    return DEFAULT_THRESHOLDS.get(getattr(model_func, "__name__", ""))


# ------------------------ Metric extraction helpers -----------------------


def extract_metric(result: Any, metric_attr: str | None = None) -> float:
    """Extract a scalar metric from a model result.

    Args:
        result: Object returned by the model function.
        metric_attr: If provided, the attribute name to read (e.g., "pmv",
            "set", "hi", "utci"). If None, attempt to infer a single
            numeric attribute among common names.

    Returns:
        Scalar metric value.

    Raises:
        ValueError: If a metric cannot be determined.
    """
    if metric_attr is not None:
        try:
            return float(getattr(result, metric_attr))
        except Exception as exc:  # noqa: BLE001
            msg = f"Result object has no readable attribute {metric_attr!r}"
            raise ValueError(msg) from exc

    # Fallback inference for common names
    for name in ("pmv", "set", "hi", "utci"):
        if hasattr(result, name):
            return float(getattr(result, name))

    # If it's already a float-like
    try:
        return float(result)
    except Exception as exc:  # noqa: BLE001
        msg = f"Cannot extract metric from result of type {type(result)}"
        raise ValueError(msg) from exc


# ------------------------ Mapper & evaluator builder ----------------------

# Mapper type: given (x, y, fixed_params) → kwargs for model_func
XYToKwargs = Callable[[float, float, dict[str, Any]], dict[str, Any]]


def make_metric_eval(
    model_func: Callable[..., Any],
    xy_to_kwargs: XYToKwargs,
    fixed_params: dict[str, Any] | None = None,
    metric_attr: str | None = None,
) -> Callable[[float, float], float]:
    """Build a metric(x, y) evaluator from a model and a mapper.

    Args:
        model_func: The pythermalcomfort model function to call.
        xy_to_kwargs: A small function that maps (x, y, fixed_params) to the
            keyword arguments for the model function (e.g., {'tdb': x, 'rh': y}).
        fixed_params: Model parameters that are constant across the grid.
            This must include any required parameters other than x and y.
        metric_attr: The attribute name to extract from the result; if None,
            a common name will be inferred (pmv, set, hi, utci) or the result
            coerced to float.

    Returns:
        A function metric(x, y) -> float.

    Notes:
        No hidden defaults are applied. If the model requires parameters
        beyond those produced by xy_to_kwargs, they must be provided via
        fixed_params.
    """
    fixed = dict(fixed_params or {})

    def metric_xy(x: float, y: float) -> float:
        kwargs = xy_to_kwargs(float(x), float(y), fixed)
        res = model_func(**kwargs)
        return extract_metric(res, metric_attr=metric_attr)

    return metric_xy


# ------------------------------ Smoothing ---------------------------------


def _smooth_single_curve(x_of_y: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing along y on contiguous valid segments.

    Args:
        x_of_y: Array of x values (may contain NaNs) indexed by y.
        sigma: Standard deviation (in index units) for Gaussian filter.

    Returns:
        Smoothed array with NaNs preserved where values were invalid.
    """
    if sigma <= 0 or not np.isfinite(x_of_y).any():
        return x_of_y

    out = x_of_y.copy()
    valid = np.isfinite(x_of_y)
    if not valid.any():
        return out

    # Find contiguous valid segments
    idx = np.arange(len(x_of_y))
    # Split where validity changes
    splits = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
    for seg in np.split(idx, splits):
        seg_mask = valid[seg]
        if seg_mask.size == 0 or not seg_mask.any():
            continue
        seg_idx = seg[seg_mask]
        vals = out[seg_idx]
        if vals.size >= 3:
            out[seg_idx] = gaussian_filter1d(vals, sigma=sigma, mode="nearest")
    return out


# --------------------------- Threshold solver -----------------------------


def solve_threshold_curves(
    metric_xy: Callable[[float, float], float],
    thresholds: Sequence[float],
    y_values: Sequence[float],
    x_bounds: tuple[float, float],
    *,
    x_scan_step: float = 1.0,
    brent_tol: float = 1e-6,
    brent_maxiter: int = 100,
    smooth_sigma: float = 0.8,
    warn_on_unsolved: bool = True,
) -> dict[str, Any]:
    """Compute x(y) curves such that metric(x, y) == threshold for each threshold.

    Args:
        metric_xy: Callable returning the metric value given (x, y).
        thresholds: Threshold values to solve for.
        y_values: Sequence of y values at which to find the boundary.
        x_bounds: (x_min, x_max) bounds for searching.
        x_scan_step: Step size for the uniform scan used to find brackets.
        brent_tol: Absolute tolerance for Brent's method (xtol and rtol).
        brent_maxiter: Maximum iterations for Brent's method.
        smooth_sigma: Gaussian smoothing sigma (in y-index units). Set 0 to
            disable smoothing.
        warn_on_unsolved: If True, warn when no root is found for a y.

    Returns:
        A dict with:
          - 'curves': list[np.ndarray] with shape (len(y_values),) per threshold
          - 'y_values': np.ndarray of y positions
          - 'thresholds': list of thresholds
          - 'unsolved_counts': list[int] per threshold
    """
    y_arr = np.asarray(list(y_values), dtype=float)
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    if x_min >= x_max:
        raise ValueError("x_bounds must be strictly increasing (min < max)")
    if x_scan_step <= 0:
        raise ValueError("x_scan_step must be positive")

    # Precompute scan grid
    # Ensure inclusive of x_max considering floating rounding
    n_steps = int(np.floor((x_max - x_min) / x_scan_step))
    x_scan = x_min + np.arange(n_steps + 1) * x_scan_step
    if x_scan[-1] < x_max:
        x_scan = np.append(x_scan, x_max)

    curves: list[np.ndarray] = []
    unsolved_counts: list[int] = []

    for thr in thresholds:
        xs = np.full_like(y_arr, np.nan, dtype=float)
        unsolved = 0
        for i, y in enumerate(y_arr):
            # bind loop variables to locals so closure does not capture loop vars
            y_local = float(y)
            thr_local = float(thr)

            # Define root function f(x) = metric(x, y_local) - thr_local
            def f(x: float, y_local=y_local, thr_local=thr_local) -> float:
                return metric_xy(float(x), float(y_local)) - thr_local

            # Scan to find first sign change
            vals = np.array([f(x) for x in x_scan])
            sgn = np.sign(vals)
            idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]
            if idx.size == 0:
                unsolved += 1
                continue
            j = int(idx[0])
            a, b = float(x_scan[j]), float(x_scan[j + 1])
            try:
                xs[i] = brentq(
                    f,
                    a,
                    b,
                    xtol=brent_tol,
                    rtol=brent_tol,
                    maxiter=brent_maxiter,
                )
            except Exception:
                unsolved += 1
                # leave NaN

        # Smoothing on contiguous segments
        xs = _smooth_single_curve(xs, sigma=smooth_sigma)

        curves.append(xs)
        unsolved_counts.append(unsolved)

    if warn_on_unsolved and any(c > 0 for c in unsolved_counts):
        msg = ", ".join(
            f"thr={thr}: {cnt} point(s) unsolved"
            for thr, cnt in zip(thresholds, unsolved_counts, strict=False)
        )
        warnings.warn(
            f"Threshold solver: some y-values had no bracket within x_bounds ({msg}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "curves": curves,
        "y_values": y_arr,
        "thresholds": list(thresholds),
        "unsolved_counts": unsolved_counts,
    }


# ---------------------------- Convenience mappers -------------------------


def mapper_tdb_rh(x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
    """Map T-RH chart inputs to model kwargs.

    Map x -> tdb (air temperature in °C) and y -> rh (%). Other model params
    must be supplied via "fixed".
    """
    kwargs = {"tdb": float(x), "rh": float(y)}
    kwargs.update(fixed)
    return kwargs


# Add mapper for temperature vs air-speed (vr)
def mapper_tdb_vr(x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
    """Map T-v (air speed) chart inputs to model kwargs.

    Map x -> tdb (air temperature in °C) and y -> vr (air speed in m/s).
    Other model params must be supplied via "fixed".
    """
    kwargs = {"tdb": float(x), "vr": float(y)}
    kwargs.update(fixed)
    return kwargs


def mapper_top_rh(x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
    """Map operative temperature vs RH to model kwargs with tr == tdb.

    Map x -> tdb and tr (both set to operative temperature) and y -> rh.
    """
    kwargs = {"tdb": float(x), "tr": float(x), "rh": float(y)}
    kwargs.update(fixed)
    return kwargs


def _validate_range(name: str, rng: tuple[float, float]) -> tuple[float, float]:
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        msg = f"{name} must be a (min, max) tuple"
        raise ValueError(msg)
    lo, hi = float(rng[0]), float(rng[1])
    if lo >= hi:
        msg = f"{name} must be strictly increasing (min < max)"
        raise ValueError(msg)
    return lo, hi
