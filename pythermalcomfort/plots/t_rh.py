from collections.abc import Callable
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq
from typing import Any, Sequence

# ---------- Threshold Validation ----------


def validate_thresholds(thresholds: Any) -> list[float]:
    """Minimal threshold validation: tuple(lo,hi) or list of numbers."""
    if thresholds is None:
        raise ValueError("`thresholds` is required.")
    if isinstance(thresholds, tuple):
        if len(thresholds) != 2:
            raise ValueError("Tuple thresholds must be (lo, hi).")
        seq = thresholds
    elif isinstance(thresholds, Sequence) and not isinstance(thresholds, (str, bytes)):
        if not thresholds:
            raise ValueError("Threshold list cannot be empty.")
        seq = thresholds
    else:
        raise ValueError("Thresholds must be tuple(lo,hi) or list of numbers.")
    return [float(v) for v in seq]


# ---------- Range Validation ----------


def validate_range(rng: Any, field_name: str) -> tuple[float, float]:
    """Minimal range validation: must be strictly increasing."""
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        raise ValueError(f"`{field_name}` must be (lo, hi).")
    lo, hi = float(rng[0]), float(rng[1])
    if lo >= hi:
        raise ValueError(f"`{field_name}` must be strictly increasing.")
    return (lo, hi)


# ---------- FillMode Validation ----------


def coerce_fill_mode(v):
    """Coerce value to 'bands' or 'contour'."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"bands", "band"}:
            return "bands"
        if s in {"contour", "contours"}:
            return "contour"
    if v in {"bands", "contour"}:
        return v
    raise ValueError(f"Invalid fill_mode: {v!r}. Use 'bands' or 'contour'.")

ModelFunc = Callable[..., object]

# ---------- Enums and Constants ----------


class FillMode(Enum):
    """Fill mode for chart visualization.

    BANDS: Fill areas between threshold curves
    CONTOUR: Use contour lines for visualization
    """

    BANDS = "bands"
    CONTOUR = "contour"


# ---------- Centralized defaults ----------


class ModelDefaults:
    """Unified model default configuration class.

    Provides default parameters, thresholds, and ranges for different
    thermal comfort models (PMV, SET, Heat Index, UTCI).
    """

    # Use mapping dictionary directly to avoid duplicate mappings
    CONFIGS = {
        "pmv_ppd_iso": {
            "thresholds": [-0.5, 0.5],
            "t_range": (10, 36),
            "rh_range": (0, 100),
            "model_params": {
                "vr": 0.1,  # Relative air velocity
                "met": 1.2,  # Metabolic rate
                "clo": 0.5,  # Clothing insulation
                "wme": 0.0,  # External mechanical work
                "model": "7730-2005",  # Model standard
                "units": "SI",  # Units
                "limit_inputs": False,  # Don't limit input range
                "round_output": True,  # Round output
            },
        },
        "pmv_ppd_ashrae": {
            "thresholds": [-0.5, 0.5],
            "t_range": (10, 36),
            "rh_range": (0, 100),
            "model_params": {
                "vr": 0.1,
                "met": 1.2,
                "clo": 0.5,
                "wme": 0.0,
                "model": "55-2020",
                "units": "SI",
                "limit_inputs": False,
                "round_output": True,
            },
        },
        "set_tmp": {
            "thresholds": [22, 24, 26, 28, 32],
            "t_range": (10, 40),
            "rh_range": (40, 90),
            "model_params": {
                "tr": 25.0,
                "v": 0.15,
                "met": 1.2,
                "clo": 0.5,
                "wme": 0.0,
                "round_output": False,
            },
        },
        "heat_index_rothfusz": {
            "thresholds": [30, 35, 40, 55],
            "t_range": (25, 60),
            "rh_range": (20, 90),
            "model_params": {
                "round_output": True,
                "limit_inputs": True,
            },
        },
        "utci": {
            "thresholds": [-40, -27, -13, -1, 9, 26, 32, 38, 46],
            "t_range": (-40, 45),
            "rh_range": (0, 100),
            "model_params": {
                "v": 2.0,
                "units": "SI",
                "limit_inputs": False,
                "round_output": True,
            },
        },
    }

    @classmethod
    def get_config(cls, func_name: str):
        """Get model configuration"""
        return cls.CONFIGS.get(func_name)


# ---------- ChartConfig ----------


class ChartConfig(BaseModel):
    """Configuration for thermal comfort chart generation.

    Core parameters: model function and model_params.
    Auto-applies model-specific defaults for thresholds and ranges.
    """

    model: ModelFunc
    model_params: dict[str, object]

    thresholds: list[float] | None = None
    t_range: tuple[float, float] | None = None
    rh_range: tuple[float, float] | None = None

    fill_mode: FillMode | str = FillMode.BANDS

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def _apply_model_defaults(cls, values: dict[str, object]) -> dict[str, object]:
        # Apply model defaults
        func_name = values["model"].__name__
        config = ModelDefaults.get_config(func_name)
        values.setdefault("thresholds", config["thresholds"])
        values.setdefault("t_range", config["t_range"])
        values.setdefault("rh_range", config["rh_range"])

        default_params = config["model_params"]
        user_params = values.get("model_params") or {}
        values["model_params"] = {**default_params, **user_params}

        if values.get("fill_mode") and isinstance(values["fill_mode"], str):
            values["fill_mode"] = (
                FillMode.BANDS
                if coerce_fill_mode(values["fill_mode"]) == "bands"
                else FillMode.CONTOUR
            )

        values["t_range"] = validate_range(values["t_range"], "t_range")
        values["rh_range"] = validate_range(values["rh_range"], "rh_range")
        values["thresholds"] = validate_thresholds(values["thresholds"])

        return values


# ---------- Utilities ----------


def _bracket_root(f, grid: np.ndarray) -> tuple[float | None, float | None]:
    """Find bracketing interval for root finding.

    Returns (a, b) where f(a) and f(b) have opposite signs, or (None, None) if
    no root found.
    """
    vals = np.array([f(x) for x in grid])
    sgn = np.sign(vals)
    idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]
    if idx.size == 0:
        return None, None
    j = int(idx[0])
    return float(grid[j]), float(grid[j + 1])


def apply_smooth_curves(curves, rh_vec, smooth_factor: float = 1.0):
    """Apply smoothing effects to ensure smooth and beautiful charts.

    Uses spline interpolation and Gaussian filtering for curve smoothing.
    Returns smoothed curves and denser humidity vector.
    """
    if smooth_factor <= 0:
        return curves, rh_vec

    # Increase density factor to generate denser humidity points for interpolation
    density_factor = max(1, smooth_factor * 1)
    dense_rh = np.linspace(rh_vec[0], rh_vec[-1], int(len(rh_vec) * density_factor))
    smoothed = []

    for curve in curves:
        # Create valid data mask: mark which points are valid values (non-NaN, non-infinite)
        valid_data_mask = np.isfinite(curve)

        # Check if there are enough valid data points for smoothing (at least 3 points)
        if valid_data_mask.sum() >= 3:
            # Simplified smoothing: only use spline interpolation
            from scipy.interpolate import UnivariateSpline

            spline = UnivariateSpline(
                rh_vec[valid_data_mask], curve[valid_data_mask], s=0.5
            )  # Increase smoothing parameter
            dense_curve = spline(dense_rh)

            # Light Gaussian filtering
            dense_curve = gaussian_filter1d(dense_curve, sigma=0.3)
            smoothed.append(dense_curve)
        else:
            # If there are too few valid data points, fill with NaN directly
            smoothed.append(np.full_like(dense_rh, np.nan))

    return smoothed, dense_rh


# ---------- Metric extraction (simplified) ----------


def _extract_metric(res: Any) -> float:
    """Extract numerical metrics from thermal comfort model results.

    Auto-detects metric attribute (pmv, set, hi, utci) from model result object.
    """
    # Extract corresponding attributes directly based on model type
    if hasattr(res, "pmv"):
        return float(res.pmv)
    elif hasattr(res, "set"):
        return float(res.set)
    elif hasattr(res, "hi"):  # Heat Index model
        return float(res.hi)
    elif hasattr(res, "utci"):
        return float(res.utci)
    else:
        raise ValueError(f"Unknown model result type: {type(res)}")


# ---------- Evaluator ----------


def _build_metric_eval(model_func: ModelFunc, model_params: dict[str, object]):
    """Build model evaluation function, model_params already contains all necessary default parameters.

    Creates cached evaluation function for thermal comfort models.
    Handles model-specific parameter requirements (e.g., UTCI tr calculation).
    """
    func_name = model_func.__name__
    params = dict(model_params or {})

    def eval_function(temperature: float, humidity: float) -> float:
        # Prepare parameters
        eval_params = dict(params)

        # Special handling for different models
        if func_name == "heat_index_rothfusz":
            # Only pass parameters needed by heat_index_rothfusz
            eval_params = {
                k: v
                for k, v in eval_params.items()
                if k in ["round_output", "limit_inputs"]
            }
        elif func_name == "utci":
            # UTCI model: tr = temperature + 5
            eval_params["tr"] = float(temperature) + 5.0
        else:
            # Other models: only set to temperature when tr is None
            if eval_params.get("tr") is None:
                eval_params["tr"] = float(temperature)

        # Call model function
        result = model_func(tdb=float(temperature), rh=float(humidity), **eval_params)
        return _extract_metric(result)

    return eval_function


# ---------- Core computation & plotting ----------


def _compute_comfort_curves(config: ChartConfig):
    """Compute comfort zone curves for given configuration.

    Uses root finding to locate threshold boundaries at each humidity level.
    Returns smoothed curves, humidity vector, and threshold list.
    """
    # Create humidity and temperature grids
    rh_vec = np.linspace(config.rh_range[0], config.rh_range[1], 50)
    scan_T = np.linspace(config.t_range[0], config.t_range[1], 100)

    # Build evaluation function and get thresholds
    eval_fn = _build_metric_eval(config.model, config.model_params)
    thr_list = list(config.thresholds)

    # Compute curves for each threshold
    curves = []
    for thr in thr_list:
        Ts = np.full_like(rh_vec, np.nan, dtype=float)
        for i, rh in enumerate(rh_vec):
            # Calculate target function value directly
            def root_func(temperature):
                return eval_fn(float(temperature), float(rh)) - thr

            # Find root using bracketing and Brent's method
            a, b = _bracket_root(root_func, scan_T)
            if a is not None:
                Ts[i] = brentq(root_func, a, b, xtol=1e-6, rtol=1e-6, maxiter=100)
        curves.append(Ts)

    # Apply smoothing and return results
    curves, rh_vec = apply_smooth_curves(curves, rh_vec)
    return curves, rh_vec, thr_list


def plot_t_rh_chart(config: ChartConfig | dict[str, object]):
    """Generate thermal comfort chart showing comfort zones.

    Main plotting function. Creates matplotlib figure with comfort zones
    as colored bands between threshold curves.
    """
    # Convert dict to ChartConfig if needed
    if isinstance(config, dict):
        config = ChartConfig(**config)

    # Create figure and compute comfort curves
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    curves, rh_vec, thr_list = _compute_comfort_curves(config)

    # Generate color bands for visualization
    needed = len(thr_list) + 1
    cmap = plt.get_cmap("coolwarm")
    band_colors = [cmap(i / (needed - 1)) for i in range(needed)]

    # Create constant temperature boundaries
    t_lo, t_hi = config.t_range
    left_const = np.full_like(rh_vec, t_lo)
    right_const = np.full_like(rh_vec, t_hi)

    # Draw filled regions
    # Define regions between curves and boundaries
    regions = (
        [(left_const, curves[0])]
        + [(curves[i], curves[i + 1]) for i in range(len(curves) - 1)]
        + [(curves[-1], right_const)]
    )
    # Fill regions with colors
    for i, (left, right) in enumerate(regions):
        m = np.isfinite(left) & np.isfinite(right)
        if m.any():
            ax.fill_betweenx(
                rh_vec[m],
                left[m],
                right[m],
                color=band_colors[i],
                alpha=0.85,
                linewidth=0,
            )

    # Draw threshold curves
    for curve in curves:
        m = np.isfinite(curve)
        if m.any():
            ax.plot(curve[m], rh_vec[m], color="black", linewidth=1)

    # Add legend
    # Create legend elements for each band
    legend_elements = []
    for i in range(needed):
        if i == 0:
            label = f"< {thr_list[0]:.1f}"
        elif i == needed - 1:
            label = f"> {thr_list[-1]:.1f}"
        else:
            label = f"{thr_list[i - 1]:.1f} to {thr_list[i]:.1f}"
        legend_elements.append(
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=band_colors[i], alpha=0.85, label=label
            )
        )

    # Display legend
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=6,
        framealpha=0.8,
        markerscale=0.6,
    )

    # Set default axis labels and title
    ax.set_xlabel("Air temperature [°C]")
    ax.set_ylabel("Relative humidity [%]")

    return fig, ax


# ---------- Run Examples ----------

if __name__ == "__main__":
    from pythermalcomfort.models import heat_index_rothfusz, pmv_ppd_iso, set_tmp, utci

    pmv_config = ChartConfig(
        model=pmv_ppd_iso,
        model_params={
            "met": 2.0,
            "clo": 0.3,
            "vr": 0.2,
        },
        thresholds=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
        t_range=(10, 36),
        rh_range=(0, 100),
        fill_mode=FillMode.BANDS,
    )

    fig1, ax1 = plot_t_rh_chart(pmv_config)
    ax1.set_title("PMV Comfort Zones")
    plt.show()

    set_config = ChartConfig(
        model=set_tmp,
        thresholds=[22, 24, 26, 28, 32],
        t_range=(10, 40),
        rh_range=(40, 90),
        fill_mode=FillMode.BANDS,
    )

    fig2, ax2 = plot_t_rh_chart(set_config)
    ax2.set_title("SET Comfort Zones")
    plt.show()

    heat_index_config = ChartConfig(
        model=heat_index_rothfusz,
        thresholds=[30, 35, 40, 55],
        t_range=(25, 60),
        rh_range=(20, 90),
        fill_mode=FillMode.BANDS,
    )

    fig3, ax3 = plot_t_rh_chart(heat_index_config)
    ax3.set_title("Heat Index Risk Zones")
    plt.show()

    utci_config = ChartConfig(
        model=utci,
        thresholds=[-40, -27, -13, -1, 9, 26, 32, 38, 46],
        t_range=(-40, 45),
        rh_range=(0, 100),
        fill_mode=FillMode.BANDS,
    )

    fig4, ax4 = plot_t_rh_chart(utci_config)
    ax4.set_title("UTCI Comfort Zones")
    ax4.set_xlabel("Temperature (°C)")
    ax4.set_ylabel("Relative Humidity (%)")
    plt.show()
