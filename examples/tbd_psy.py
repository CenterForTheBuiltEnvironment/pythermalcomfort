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
    # cmap: str = "coolwarm",               # 原：在签名中直接暴露颜色等外观参数
    # band_alpha: float = 0.85,             # 原：同上
    # line_color: str = "black",            # 原：同上
    # line_width: float = 1.0,              # 原：同上
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Additional matplotlib parameters
    # **kwargs: Any,                       # 原：任意参数直传，可能影响核心逻辑
    # ---------------- 新增，与队友风格保持一致 ----------------
    title: str | None = None,              # 新：标题（放到 Figure 层）
    fontsize: float = 12.0,                # 新：字体大小
    plot_kwargs: dict[str, Any] | None = None,  # 新：仅用于外观覆盖（白名单）
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk ranges over a psychrometric chart (Tdb vs humidity ratio).

    Visual appearance can be overridden via ``plot_kwargs`` (e.g., cmap, labels, etc.).
    Core solving logic remains unchanged.
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

    # if ax is None:
    #     plt.style.use("seaborn-v0_8-whitegrid")
    #     _, ax = plt.subplots(figsize=(7, 3), dpi=300, constrained_layout=True)
    # 上面原逻辑：固定 seaborn 样式/尺寸/DPI；不利于全家族一致控制       # 新：不固定样式，后面统一用 Figure 级设置
    if ax is None:
        _, ax = plt.subplots()  # 新：先创建默认 Axes，尺寸/DPI 由下方统一设置

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
        if handles and legend:  # 原：任何情况下都加图例；新：仅当 legend=True 时添加
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
    # calc_kwargs: dict[str, Any] = {
    #     "model_func": model_func,
    #     "xy_to_kwargs": mapper_tdb_w,
    #     "fixed_params": fixed_params,
    #     "thresholds": thresholds,
    #     "x_bounds": (t_lo, t_hi),
    #     "y_values": y_values,
    #     "metric_attr": None,
    #     "ax": ax,
    #     "xlabel": "Dry-bulb air temperature [°C]",
    #     "ylabel": "Humidity ratio [kg/kg]",
    #     "legend": legend,
    #     "x_scan_step": float(x_scan_step),
    #     "smooth_sigma": float(smooth_sigma),
    #     "x_left_clip": x_left_clip,
    #     # Explicit visual controls
    #     "cmap": cmap,
    #     "band_alpha": band_alpha,
    #     "line_color": line_color,
    #     "line_width": line_width,
    # }
    # calc_kwargs.update(kwargs)
    # 上面原逻辑：在此处显式设置颜色等外观，并将 **kwargs 全量透传        # 新：仅允许外观从 plot_kwargs 白名单覆盖，防止误改核心参数

    kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_tdb_w,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Dry-bulb air temperature [°C]",  # 外观项：可被 plot_kwargs 覆盖
        "ylabel": "Humidity ratio [kg/kg]",         # 外观项：可被 plot_kwargs 覆盖
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
        "x_left_clip": x_left_clip,
    }  # 新：不在这里强制配色，保持默认

    if plot_kwargs: kwargs.update({k: v for k, v in plot_kwargs.items() if k not in ("model_func", "xy_to_kwargs")})

    ax, artists = calc_plot_ranges(**kwargs)  # 新：调用保持不变

    # ax.set_xlim(t_lo, t_hi)
    # ax.set_ylim(w_lo, w_hi)
    # 原：直接设置轴范围                                            # 新：保持但下方统一做版式
    ax.set_xlim(t_lo, t_hi)   # 新：与原一致
    ax.set_ylim(w_lo, w_hi)   # 新：与原一致

    # ---------------- 统一图幅/DPI与标题，给标题图例留空间 ----------------
    fig = ax.figure                       # 新：拿到当前 Figure，避免多图场景拿错
    fig.set_size_inches(7, 5)             # 新：统一尺寸（与队友一致）
    fig.set_dpi(150)                      # 新：统一清晰度
    fig.suptitle(                         # 新：标题放 Figure 层，避免与图例重叠
        title or "Psychrometric Chart (Tdb vs Humidity Ratio)",
        fontsize=fontsize,
        y=0.96,
    )

    try:
        pos = ax.get_position()
        ax.set_position([pos.x0, max(0.06, pos.y0 + 0.02), pos.width, pos.height * 0.92])  # 新：上移压缩，为标题/图例留空间
    except Exception:
        pass  # 新：防止个别后端报错

    return ax, artists


if __name__ == "__main__":
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = ranges_tdb_psychrometric(
        model_func=pmv_ppd_iso,
        fixed_params={
            "tr": 25.0,
            "met": 1.2,
            "clo": 0.5,
            "vr": 0.1,
            "wme": 0.0,
            # "p_atm": 101325.0,   # 可选：大气压，若不填使用默认
        },
        thresholds=[-0.5, 0.5],
        t_range=(18, 30),
        w_range=(0.002, 0.018),
        w_step=0.001,
        # 不更改颜色，如需外观覆盖仅通过 plot_kwargs 提供（可选）：
        # plot_kwargs={"xlabel": "Dry-bulb T [°C]"},
        title="Psychrometric comfort ranges",  # 新：与队友风格一致
        fontsize=12.0,                          # 新：统一字体
    )

    import matplotlib.pyplot as plt
    plt.show()
