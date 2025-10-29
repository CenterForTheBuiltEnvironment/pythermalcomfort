"""
pmv_contours.py
----------------
Unified solver-based plotting for thermal comfort charts (PMV, SET, etc.).
- thresholds=(lo,hi) 或 thresholds=[t1,t2,...] → 相邻等值线之间色带填充 + 黑色等值线 + 标签
- 优先用 scipy.optimize.brentq；失败回退到稳健二分法
- 仅使用 Matplotlib
"""

from typing import Any, Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass

# SciPy brentq（优先）；不可用则回退
try:
    from scipy.optimize import brentq as _brentq
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# 内部常量（不再暴露在API中）
_RH_POINTS = 601
_BRACKET_SCAN_POINTS = 800

ModelFunc = Callable[..., Any]


@dataclass
class ChartConfig:
    """配置参数的数据类"""
    # 必需参数
    model_params: Dict[str, Any]  # 模型参数：met, clo, v, tr, wme, limit_inputs 等
    
    # 可选参数
    t_range: Tuple[float, float] = (15.0, 40.0)
    rh_range: Tuple[float, float] = (20.0, 90.0)
    thresholds: Optional[Union[Tuple[float, float], List[float]]] = None
    threshold_labels: Optional[List[str]] = None
    palette: Optional[List[str]] = None
    dpi: Optional[int] = None
    figsize: Tuple[int, int] = (8, 6)
    alpha: float = 0.85
    metric: Optional[str] = None
    title: Optional[str] = None
    save_path: Optional[str] = None
    show_legend: bool = True
    show_iso_labels: bool = True


# =========================
# Public API (single entry)
# =========================
def plot_t_rh_chart(
    model_func: ModelFunc,
    config: Union[ChartConfig, Dict[str, Any]],
) -> Tuple[plt.Figure, plt.Axes]:
    """
    在 T–RH 平面绘制基于等值线求解的舒适色带图。
    
    Args:
        model_func: 模型函数 (pmv_ppd_iso 等)
        config: ChartConfig 实例或配置字典，必须包含模型参数
    
    Returns:
        matplotlib 图和轴对象
    """
    # 处理配置参数
    if isinstance(config, dict):
        config = ChartConfig(**config)
    elif not isinstance(config, ChartConfig):
        raise TypeError("config must be ChartConfig instance or dictionary")
    
    # 验证必需的模型参数
    if not config.model_params:
        raise ValueError("model_params must be provided in config")
    
    # 1) 指标类型
    key = _infer_metric_key(model_func, config.metric)

    # 2) 默认阈值
    if config.thresholds is None:
        config.thresholds = (-0.5, 0.5) if key == "pmv" else [22, 24, 26, 28, 30]

    # 3) 规范化阈值为严格递增列表
    thr_list = _normalize_thresholds(config.thresholds)

    # 4) 轴向采样（使用内部常量）
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    rh_vec = np.linspace(float(config.rh_range[0]), float(config.rh_range[1]), 
                        int(max(3, _RH_POINTS)))
    t_lo, t_hi = float(config.t_range[0]), float(config.t_range[1])
    scan_T = np.linspace(t_lo, t_hi, int(max(50, _BRACKET_SCAN_POINTS)))

    # 5) 构建 f(T,RH) 评估函数
    eval_fn = _build_metric_eval(model_func, key, config.model_params)

    # 6) 求解每条等值线 T(RH)
    def _solve_curve(target: float) -> np.ndarray:
        Ts = np.full_like(rh_vec, np.nan, dtype=float)
        for i, rh in enumerate(rh_vec):
            def g(t): return eval_fn(float(t), float(rh)) - float(target)
            a, b = _bracket_root(g, scan_T)
            if a is not None:
                Ts[i] = _root_solve(g, a, b)
        return Ts
    

    curves = [_solve_curve(thr) for thr in thr_list]  # list of 1D arrays: T(rh)

    # 7) 色带与图例
    band_colors = _make_colors(len(thr_list) + 1, config.palette)
    band_names = _make_band_labels(thr_list, config.threshold_labels, key)

    # 8) 填充左侧带： [t_lo, 第一条曲线]
    left_const = np.full_like(rh_vec, t_lo)
    first = curves[0]
    mask = np.isfinite(first)
    if mask.any():
        ax.fill_betweenx(rh_vec[mask], left_const[mask], first[mask],
                         color=band_colors[0], alpha=config.alpha, linewidth=0)

    # 9) 填充中间带：相邻曲线之间
    for i in range(len(curves) - 1):
        a, b = curves[i], curves[i + 1]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.any():
            ax.fill_betweenx(rh_vec[mask], a[mask], b[mask],
                             color=band_colors[i + 1], alpha=config.alpha, linewidth=0)

    # 10) 填充右侧带： [最后一条曲线, t_hi]
    last = curves[-1]
    mask = np.isfinite(last)
    if mask.any():
        right_const = np.full_like(rh_vec, t_hi)
        ax.fill_betweenx(rh_vec[mask], last[mask], right_const[mask],
                         color=band_colors[-1], alpha=config.alpha, linewidth=0)

    # 11) 叠加黑色等值线 + 标签
    for thr, curve in zip(thr_list, curves):
        mask = np.isfinite(curve)
        ax.plot(curve[mask], rh_vec[mask], "-", lw=1.6, color="black")
        if config.show_iso_labels and mask.any():
            x_med, y_med = _median_point(curve[mask], rh_vec[mask])
            if not (np.isnan(x_med) or np.isnan(y_med)):
                ax.text(x_med, y_med, f"{key.upper()}={thr:g}",
                        fontsize=9, ha="left", va="center", color="black",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.4, pad=1.5))

    # 12) 图例
    if config.show_legend:
        handles = [Patch(facecolor=band_colors[i], edgecolor="none", label=band_names[i])
                   for i in range(len(band_colors))]
        leg = ax.legend(handles=handles, loc="upper left", frameon=True, title="Thresholds",
                        ncol=1, title_fontsize=12, fontsize=11)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.92)
        leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_linewidth(1.0)




    # 13) 坐标与样式
    ax.set_xlim(config.t_range); ax.set_ylim(config.rh_range)
    ax.set_xlabel("Air temperature [°C]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative humidity [%]", fontsize=14, fontweight="bold")
    if config.title is None:
        config.title = f"{key.upper()} — Bands from solver-based iso-curves"
    ax.set_title(config.title, fontsize=16, fontweight="bold", pad=18)
    ax.grid(alpha=0.3, linewidth=0.8)
    ax.set_facecolor("#f6fbff")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)

    if config.save_path:
        fig.savefig(config.save_path, dpi=config.dpi or 300, bbox_inches="tight")
    plt.tight_layout()
    return fig, ax


# =========================
# ---- Helper functions ---
# =========================
def _infer_metric_key(model_func: ModelFunc, metric: Optional[str]) -> str:
    if isinstance(metric, str) and metric.lower() in {"pmv", "set"}:
        return metric.lower()
    name = model_func.__name__.lower()
    if "set" in name:
        return "set"
    if "pmv" in name or "ppd" in name:
        return "pmv"
    return "metric"


def _normalize_thresholds(thresholds: Union[Tuple[float, float], List[float]]) -> List[float]:
    if isinstance(thresholds, tuple) and len(thresholds) == 2:
        lst = [float(thresholds[0]), float(thresholds[1])]
    elif isinstance(thresholds, (list, np.ndarray)):
        lst = [float(x) for x in thresholds]
    else:
        raise ValueError("`thresholds` must be a tuple(lo,hi) or a list of floats.")
    lst = sorted(set(lst))
    if len(lst) < 1:
        raise ValueError("`thresholds` cannot be empty.")
    return lst


def _build_metric_eval(model_func: ModelFunc, key: str, model_params: Dict[str, Any]):
    """
    返回 eval_fn(ta, rh) -> 标量：
    - PMV: 取 'pmv'；支持 tr=None→ta
    - SET: 取 'set'；若缺失再退回 'pmv'
    """
    if key == "pmv":
        vr = model_params.get("vr", model_params.get("v", 0.1))
        met = model_params.get("met", 1.2)
        clo = model_params.get("clo", 0.5)
        wme = model_params.get("wme", 0.0)
        limit_inputs = model_params.get("limit_inputs", False)

        def eval_fn(ta: float, rh: float) -> float:
            tr_val = ta if model_params.get("tr", None) is None else float(model_params["tr"])
            res = model_func(
                tdb=float(ta), tr=tr_val, vr=float(vr), rh=float(rh),
                met=float(met), clo=float(clo), wme=float(wme),
                limit_inputs=bool(limit_inputs),
            )
            if isinstance(res, dict):
                return float(res["pmv"])
            return float(getattr(res, "pmv"))
        return eval_fn

    # SET or generic scalar exposing "set"
    def eval_fn(ta: float, rh: float) -> float:
        res = model_func(
            tdb=float(ta), rh=float(rh),
            tr=float(model_params.get("tr", 25.0)),
            v=float(model_params.get("v", model_params.get("vr", 0.15))),
            met=float(model_params.get("met", 1.2)),
            clo=float(model_params.get("clo", 0.5)),
            wme=float(model_params.get("wme", 0.0)),
            round_output=False,
        )
        # 先取 set；没有再回退 pmv；兼容 dict/对象
        if hasattr(res, "set"):
            return float(res.set)
        if isinstance(res, dict) and "set" in res:
            return float(res["set"])
        if hasattr(res, "pmv"):
            return float(res.pmv)
        if isinstance(res, dict) and "pmv" in res:
            return float(res["pmv"])
        raise ValueError("Model output has neither 'set' nor 'pmv'.")
    return eval_fn


def _bracket_root(f, grid: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    vals = np.array([f(x) for x in grid])
    sgn = np.sign(vals)
    idx = np.where(sgn[:-1] * sgn[1:] <= 0)[0]
    if idx.size == 0:
        return None, None
    j = int(idx[0])
    return float(grid[j]), float(grid[j + 1])


def _root_solve(f, a: float, b: float, tol: float = 1e-3) -> float:
    if _HAVE_SCIPY:
        try:
            return float(_brentq(f, a, b, xtol=tol, rtol=tol, maxiter=100))
        except Exception:
            pass
    # 回退：二分法
    fa, fb = f(a), f(b)
    if np.isnan(fa) or np.isnan(fb) or fa * fb > 0:
        return np.nan
    lo, hi = a, b
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.isnan(fm) or abs(fm) < tol or abs(hi - lo) < tol:
            return float(mid)
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
    return float(0.5 * (lo + hi))


def _make_colors(n: int, palette: Optional[List[str]] = None) -> List[str]:
    default = ["#08519c", "#3182bd", "#74c476", "#feb24c", "#fd8d3c", "#d7301f"]
    pal = default if palette is None else palette
    return (pal * ((n + len(pal) - 1) // len(pal)))[:n]


def _make_band_labels(thr_list: List[float], labels: Optional[List[str]], metric: str) -> List[str]:
    """
    Build legend labels for bands. There are N = len(thr_list) thresholds and
    N+1 bands: (-inf, t1], (t1, t2], ..., (tN-1, tN], (tN, +inf).
    - If `labels` has exactly N+1 items, use them.
    - If `labels` has exactly N-1 items, treat them as the middle ranges and synthesize the two edge labels.
    - Otherwise, auto-generate: ["< t1", "t1–t2", ..., ">= tN"].
    """
    n = len(thr_list)
    n_bands = n + 1

    def fmt(x: float) -> str:
        return f"{x:g}"

    # Case 1: full set provided
    if labels is not None and len(labels) == n_bands:
        return list(labels)

    # Case 2: only interior bands provided
    if labels is not None and len(labels) == max(0, n - 1):
        left = f"< {fmt(thr_list[0])}"
        right = f"≥ {fmt(thr_list[-1])}"
        return [left] + list(labels) + [right]

    # Case 3: auto-generate all
    out: List[str] = []
    if n == 0:
        out = ["All"]
    else:
        out.append(f"< {fmt(thr_list[0])}")
        for i in range(n - 1):
            out.append(f"{fmt(thr_list[i])} – {fmt(thr_list[i+1])}")
        out.append(f"≥ {fmt(thr_list[-1])}")
    return out[:n_bands]


def _median_point(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return np.nan, np.nan
    dx = np.diff(x); dy = np.diff(y)
    seg_len = np.hypot(dx, dy)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    if cum[-1] <= 0:
        return float(np.nanmedian(x)), float(np.nanmedian(y))
    target = 0.5 * cum[-1]
    j = np.searchsorted(cum, target) - 1
    j = max(0, min(j, x.size - 2))
    w = (target - cum[j]) / max(seg_len[j], 1e-9)
    xm = x[j] + w * (x[j + 1] - x[j])
    ym = y[j] + w * (y[j + 1] - y[j])
    return float(xm), float(ym)




