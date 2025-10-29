"""Plotting utilities for threshold-based comfort regions.

This module provides a generic, reusable solver to compute threshold
curves for any metric derived from a pythermalcomfort model. It avoids
hidden defaults: callers must pass all non-(x,y) model parameters.

Key components
--------------
- DEFAULT_THRESHOLDS: internal registry of sensible default thresholds
  per model function name. Only thresholds are provided by default.
- MetricMapper: Abstract base class for defining how (x, y) map to model kwargs.
- MAPPER_REGISTRY: Global registry of convenience MetricMappers.
- make_metric_eval: builds a metric(x, y) function from a model function
  and a MetricMapper instance.
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
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq

# Defaults registry

DEFAULT_THRESHOLDS: dict[str, list[float]] = { # Dictionary holding default threshold values
    # PMV common comfort band (can be overridden by caller)
    "pmv_ppd_iso": [-0.5, 0.5], # PMV comfort thresholds for ISO standard
    "pmv_ppd_ashrae": [-0.5, 0.5], # PMV comfort thresholds for ASHRAE standard
    # SET commonly used breakpoints
    "set_tmp": [22.0, 24.0, 26.0, 28.0, 32.0], # SET temperature breakpoints
    # Heat Index categories (approximate)
    "heat_index_rothfusz": [30.0, 35.0, 40.0, 55.0], # Heat Index category thresholds
    # UTCI stress categories (approximate edges)
    "utci": [-40.0, -27.0, -13.0, -1.0, 9.0, 26.0, 32.0, 38.0, 46.0], # UTCI stress category thresholds
}


def get_default_thresholds(model_func: Callable[..., Any]) -> list[float] | None:
    """Return default thresholds for a model function if known.

    Args:
        model_func: The model function from pythermalcomfort.models.

    Returns:
        A list of float thresholds or None if no defaults are registered.
    """
    # Attempt to retrieve thresholds using the function's name, returning None if not found
    return DEFAULT_THRESHOLDS.get(getattr(model_func, "__name__", ""))


def _validate_range(name: str, range_tuple: tuple[float, float]) -> tuple[float, float]:
    """Validate that a range tuple has min < max.
    
    Args:
        name: Name of the parameter for error messages.
        range_tuple: Tuple of (min, max) values.
        
    Returns:
        The validated range tuple.
        
    Raises:
        ValueError: If min >= max.
    """
    min_val, max_val = range_tuple
    if min_val >= max_val:
        raise ValueError(f"{name} must have min < max, got {min_val} >= {max_val}")
    return (float(min_val), float(max_val))


# Psychrometric conversions 


def _svp_water_pa(t_c: float) -> float:
    """Compute saturation vapor pressure over liquid water (Pa).

    Uses Magnus-Tetens approximation; adequate for plotting purposes.
    """
    t = float(t_c) # Ensure temperature is a float
    # Calculate saturation vapor pressure using Magnus-Tetens formula
    return 610.94 * np.exp((17.625 * t) / (t + 243.04))


def humidity_ratio_from_t_rh(
    t_c: float,
    rh_percent: float,
    p_pa: float = 101325.0, # Default pressure is standard atmospheric pressure
) -> float:
    """Convert (T, RH) to humidity ratio W (kg/kg dry air)."""
    psat = _svp_water_pa(t_c) # Get saturation vapor pressure at T
    # Calculate partial vapor pressure (pv), ensuring RH is between 0 and 1
    pv = max(0.0, min(1.0, float(rh_percent) / 100.0)) * psat
    denom = float(p_pa) - pv # Calculate the denominator: (Total Pressure - Vapor Pressure)
    if denom <= 0: # Handle edge case where vapor pressure equals or exceeds total pressure
        denom = np.finfo(float).tiny # Use a very small positive number
    # Calculate humidity ratio (W)
    return 0.62198 * pv / denom


def rh_from_t_w(t_c: float, w: float, p_pa: float = 101325.0) -> float:
    """Convert (T, W) to RH (%)."""
    w = max(0.0, float(w)) # Ensure humidity ratio is not negative
    # Calculate partial vapor pressure (pv) from W and total pressure
    pv = (w * float(p_pa)) / (0.62198 + w)
    psat = _svp_water_pa(t_c) # Get saturation vapor pressure at T
    # Calculate relative humidity (%)
    rh = 100.0 * pv / psat if psat > 0 else 0.0
    # Ensure RH is between 0% and 100%
    return float(np.clip(rh, 0.0, 100.0))


#  Metric extraction helpers


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
    if metric_attr is not None: # If a specific attribute name is given
        try:
            # Try to get the attribute from the result object and convert to float
            return float(getattr(result, metric_attr))
        except Exception as exc:   # Catch any error during extraction
            msg = f"Result object has no readable attribute {metric_attr!r}"
            raise ValueError(msg) from exc # Raise an error if extraction failed

    # Fallback inference for common names
    for name in ("pmv", "set", "hi", "utci"): # Loop through common metric names
        if hasattr(result, name): # Check if the result object has this attribute
            return float(getattr(result, name)) # Extract and return the float value

    # If it's already a float-like (e.g., a simple number return value)
    try:
        return float(result) # Coerce the result directly to float
    except Exception as exc:  # Catch if coercion fails
        msg = f"Cannot extract metric from result of type {type(result)}"
        raise ValueError(msg) from exc # Raise an error


#  Mapper & evaluator builder 

class MetricMapper(ABC):
    """Abstract base class for defining how (x, y) map to model kwargs.

    Subclasses implement the .map() method to transform the (x, y) coordinates
    and fixed parameters into the keyword arguments required by the
    pythermalcomfort model function.
    """

    @abstractmethod # Decorator enforces that subclasses must implement this method
    def map(self, x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
        """Map (x, y) and fixed parameters to model function kwargs.

        Args:
            x: The x-coordinate (e.g., dry-bulb temperature).
            y: The y-coordinate (e.g., relative humidity).
            fixed: A dictionary of fixed model parameters.

        Returns:
            A dictionary of keyword arguments for the model function.
        """
        pass # Must be implemented by subclasses


MAPPER_REGISTRY: dict[str, MetricMapper] = {} # Global dictionary to store named MetricMapper instances


def register_mapper(name: str) -> Callable[[type[MetricMapper]], type[MetricMapper]]:
    """Decorator to register a MetricMapper subclass in MAPPER_REGISTRY."""
    def decorator(cls: type[MetricMapper]) -> type[MetricMapper]: # Inner decorator function
        if not issubclass(cls, MetricMapper): # Check if the decorated class inherits from MetricMapper
            raise TypeError("Only subclasses of MetricMapper can be registered.")
        MAPPER_REGISTRY[name] = cls() # Instantiate the class and register it under the given name
        return cls # Return the original class
    return decorator # Return the decorator function


def make_metric_eval(
    model_func: Callable[..., Any],
    mapper: MetricMapper,
    fixed_params: dict[str, Any] | None = None,
    metric_attr: str | None = None,
) -> Callable[[float, float], float]:
    """Build a metric(x, y) evaluator from a model and a mapper.

    Args:
        model_func: The pythermalcomfort model function to call.
        mapper: An instance of MetricMapper that maps (x, y) to kwargs.
        fixed_params: Model parameters that are constant across the grid.
            This must include any required parameters other than x and y.
        metric_attr: The attribute name to extract from the result; if None,
            a common name will be inferred (pmv, set, hi, utci) or the result
            coerced to float.

    Returns:
        A function metric(x, y) -> float.

    Raises:
        ValueError: If model_func is not a callable function.

    Notes:
        No hidden defaults are applied. If the model requires parameters
        beyond those produced by mapper.map(), they must be provided via
        fixed_params.
    """
    if not callable(model_func): # Check if the model function is callable
        raise ValueError(f"model_func must be a callable function, got {type(model_func)}")

    fixed = dict(fixed_params or {}) # Create a local copy of fixed parameters, default to empty dict

    def metric_xy(x: float, y: float) -> float: # The metric evaluation function to be returned
        kwargs = mapper.map(float(x), float(y), fixed) # Use the mapper to get all model keyword arguments
        # limit_inputs=False is passed to ensure pythermalcomfort doesn't clip
        # inputs which might be necessary to locate the threshold curve.
        res = model_func(**kwargs, limit_inputs=False) # Call the thermal comfort model function
        return extract_metric(res, metric_attr=metric_attr) # Extract and return the final scalar metric value

    return metric_xy # Return the specialized evaluator function


#  Smoothing 

def _smooth_single_curve(x_of_y: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing along y on contiguous valid segments.

    Args:
        x_of_y: Array of x values (may contain NaNs) indexed by y.
        sigma: Standard deviation (in index units) for Gaussian filter.

    Returns:
        Smoothed array with NaNs preserved where values were invalid.
    """
    if sigma <= 0 or not np.isfinite(x_of_y).any(): # Check if smoothing is needed or possible
        return x_of_y # Return original array if not

    out = x_of_y.copy() # Create a copy for the output
    valid = np.isfinite(x_of_y) # Boolean array: True for non-NaN/non-inf values
    if not valid.any(): # Check if there are any valid points at all
        return out # Return the array if all points are invalid (NaN)

    # Find contiguous valid segments
    idx = np.arange(len(x_of_y)) # Array of indices [0, 1, 2, ...]
    # Find indices where validity changes (start/end of NaN segments)
    splits = np.where(np.diff(valid.astype(int)) != 0)[0] + 1
    for seg in np.split(idx, splits): # Iterate over index segments
        seg_mask = valid[seg] # Validity mask for the current segment
        if seg_mask.size == 0 or not seg_mask.any(): # Skip empty or fully invalid segments
            continue
        seg_idx = seg[seg_mask] # Indices of the valid points in the current segment
        vals = out[seg_idx] # The x-values for this contiguous valid segment
        if vals.size >= 3: # Smoothing requires at least 3 points
            # Apply Gaussian filter, 'nearest' mode handles segment edges
            out[seg_idx] = gaussian_filter1d(vals, sigma=sigma, mode="nearest")
    return out # Return the array with smoothed valid segments


#  Threshold solver 

def _solve_single_x_for_y(
    metric_xy: Callable[[float, float], float],
    y_local: float,
    thr_local: float,
    x_scan: np.ndarray,
    brent_tol: float,
    brent_maxiter: int,
) -> float | None:
    """Finds the single x value such that metric(x, y_local) == thr_local."""

    # Define root function f(x) = metric(x, y_local) - thr_local
    def f(x: float) -> float:
        return metric_xy(float(x), float(y_local)) - thr_local

    # Scan to find first sign change (bracket for brentq)
    vals = np.array([f(x) for x in x_scan]) # Evaluate f(x) across the x_scan grid
    sgn = np.sign(vals) # Get the sign of the evaluated values
    # Find the first index where the sign changes (sgn[i] * sgn[i+1] <= 0)
    idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]

    if idx.size == 0:
        return None  # No bracket found
    
    j = int(idx[0]) # Index of the start of the first sign-change interval
    a, b = float(x_scan[j]), float(x_scan[j + 1]) # The bracketing interval [a, b]
    
    try:
        # Brent's method finds the root within the bracket (a, b)
        return brentq(
            f,
            a,
            b,
            xtol=brent_tol, # Absolute tolerance for the root
            rtol=brent_tol, # Relative tolerance for the root
            maxiter=brent_maxiter, # Maximum number of iterations
        )
    except Exception: # Catch any exception from brentq (e.g., failure to converge)
        return None # Root finding failed


def _process_threshold_curve(
    metric_xy: Callable[[float, float], float],
    thr: float,
    y_arr: np.ndarray,
    x_scan: np.ndarray,
    brent_tol: float,
    brent_maxiter: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, int]:
    """Computes x(y) curve for a single threshold, including smoothing and counting unsolved points."""

    xs = np.full_like(y_arr, np.nan, dtype=float) # Initialize array for x-roots with NaNs
    unsolved = 0 # Counter for points where a root could not be found

    for i, y in enumerate(y_arr): # Loop over each y-value
        x_root = _solve_single_x_for_y( # Find the x-value for the current y and threshold
            metric_xy,
            y,
            thr,
            x_scan,
            brent_tol,
            brent_maxiter,
        )
        if x_root is not None:
            xs[i] = x_root # Store the found root
        else:
            unsolved += 1 # Increment the unsolved counter

    # Smoothing on contiguous segments
    xs = _smooth_single_curve(xs, sigma=smooth_sigma) # Apply smoothing to the curve
    
    return xs, unsolved # Return the curve array and the count of unsolved points


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

    ... (Args and Returns remain the same)
    """
    y_arr = np.asarray(list(y_values), dtype=float) # Convert y-values to a numpy array
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1]) # Unpack and cast x-bounds
    
    # 1. Parameter Validation
    if x_min >= x_max: # Check for valid x-bounds
        raise ValueError("x_bounds must be strictly increasing (min < max)")
    if x_scan_step <= 0: # Check for valid step size
        raise ValueError("x_scan_step must be positive")

    # 2. Precompute scan grid
    # Calculate number of full steps
    n_steps = int(np.floor((x_max - x_min) / x_scan_step))
    # Create array of scan points (x_min, x_min+step, ...)
    x_scan = x_min + np.arange(n_steps + 1) * x_scan_step
    if x_scan[-1] < x_max:
        x_scan = np.append(x_scan, x_max) # Ensure the x_max boundary is included

    curves: list[np.ndarray] = [] # List to store the computed curve arrays
    unsolved_counts: list[int] = [] # List to store the unsolved point count for each curve

    # 3. Process each threshold using the helper function
    for thr in thresholds: # Loop over all specified thresholds
        xs, unsolved = _process_threshold_curve( # Compute the curve for this threshold
            metric_xy,
            float(thr),
            y_arr,
            x_scan,
            brent_tol,
            brent_maxiter,
            smooth_sigma,
        )
        curves.append(xs) # Add the computed curve to the list
        unsolved_counts.append(unsolved) # Add the count of unsolved points

    # 4. Handle warnings
    if warn_on_unsolved and any(c > 0 for c in unsolved_counts): # Check if any curve has unsolved points
        # Create a message detailing the unsolved points per threshold
        msg = ", ".join(
            f"thr={thr}: {cnt} point(s) unsolved"
            for thr, cnt in zip(thresholds, unsolved_counts, strict=False)
        )
        warnings.warn( # Issue a warning to the user
            f"Threshold solver: some y-values had no bracket within x_bounds ({msg}).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Return results in a structured dictionary
    return {
        "curves": curves,
        "y_values": y_arr,
        "thresholds": list(thresholds),
        "unsolved_counts": unsolved_counts,
    }


#  Convenience mappers 

@register_mapper("tdb_rh") # Register this class under the name "tdb_rh"
class TemperatureVsRelativeHumidityMapper(MetricMapper):
    """Map x -> tdb (°C) and y -> rh (%).

    Other model params must be supplied via "fixed".
    """
    def map(self, x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
        kwargs = {"tdb": float(x), "rh": float(y)} # Map x to dry-bulb temperature, y to relative humidity
        kwargs.update(fixed) # Add the fixed parameters
        return kwargs # Return the final keyword arguments


@register_mapper("tdb_vr") # Register this class under the name "tdb_vr"
class TemperatureVsAirSpeedMapper(MetricMapper):
    """Map x -> tdb (°C) and y -> vr (air speed in m/s).

    Other model params must be supplied via "fixed".
    """
    def map(self, x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
        kwargs = {"tdb": float(x), "vr": float(y)} # Map x to dry-bulb temperature, y to air speed
        kwargs.update(fixed) # Add the fixed parameters
        return kwargs # Return the final keyword arguments


@register_mapper("top_rh") # Register this class under the name "top_rh"
class OperativeTemperatureVsRelativeHumidityMapper(MetricMapper):
    """Map x -> top (°C) and y -> rh (%) assuming tr == tdb == top.

    Map x -> tdb and tr (both set to operative temperature) and y -> rh.
    Other model params must be supplied via "fixed".
    """
    def map(self, x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
        # Map x to both dry-bulb (tdb) and radiant (tr) temperature, y to relative humidity
        kwargs = {"tdb": float(x), "tr": float(x), "rh": float(y)}
        kwargs.update(fixed) # Add the fixed parameters
        return kwargs # Return the final keyword arguments


@register_mapper("tdb_w") # Register this class under the name "tdb_w"
class TemperatureVsHumidityRatioMapper(MetricMapper):
    """Map x -> tdb (°C) and y -> W (humidity ratio, kg/kg dry air).

    Computes rh from (tdb=x, W=y) at pressure p_atm (Pa) in fixed; default 101325.
    Passes tdb and computed rh to model. Other params are forwarded unchanged.
    """
    def map(self, x: float, y: float, fixed: dict[str, Any]) -> dict[str, Any]:
        # Get atmospheric pressure, defaulting to 101325 Pa
        p_pa = float(fixed.get("p_atm", 101325.0))
        # Ensure 'rh' is computed and used, overriding any 'rh' in fixed
        # Compute RH from TDB (x) and Humidity Ratio (y)
        rh = rh_from_t_w(float(x), float(y), p_pa=p_pa)
        kwargs = {"tdb": float(x), "rh": float(rh)} # Start with TDB and computed RH
        kwargs.update(fixed) # Add fixed parameters (like clothing, met, etc.)
        # Ensure mapper-controlled rh overrides any provided in fixed
        kwargs["rh"] = float(rh) # Explicitly set RH to the computed value
        return kwargs # Return the final keyword arguments


# Convenience aliases for the mappers
mapper_tdb_rh = MAPPER_REGISTRY["tdb_rh"]
mapper_tdb_vr = MAPPER_REGISTRY["tdb_vr"] 
mapper_top_rh = MAPPER_REGISTRY["top_rh"]
mapper_tdb_w = MAPPER_REGISTRY["tdb_w"]