from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "Plotting requires the optional 'plots' extra. "
        "Install with: pip install pythermalcomfort[plots]"
    ) from exc
import numpy as np

from pythermalcomfort.plots.generic import calc_plot_ranges
from pythermalcomfort.plots.utils import (
    _validate_range,
    get_default_thresholds,
    mapper_top_rh,
)

__all__ = ["ranges_to_rh"]


def ranges_to_rh(
    model_func: Callable[..., Any],
    *,
    fixed_params: dict[str, Any] | None = None,
    thresholds: Sequence[float] | None = None,
    t_range: tuple[float, float] = (10.0, 36.0),
    rh_range: tuple[float, float] = (0.0, 100.0),
    rh_step: float = 2.0,
    # Visual controls (most commonly used)
    # cmap: str = "coolwarm",                     # 原：在签名中直接暴露颜色等外观参数
    # band_alpha: float = 0.85,                   # 原：同上
    # line_color: str = "black",                  # 原：同上
    # line_width: float = 1.0,                    # 原：同上
    # Solver controls
    x_scan_step: float = 1.0,
    smooth_sigma: float = 0.8,
    # Plot controls
    ax: plt.Axes | None = None,
    legend: bool = True,
    # Additional matplotlib parameters
    # **kwargs: Any,                             # 原：任意参数直接透传，可能误改核心逻辑
    # ---------------- 新增，与队友风格保持一致 ----------------
    title: str | None = None,                     # 新：标题参数（放到 Figure 层）
    fontsize: float = 12.0,                       # 新：全局字体大小（标题等）
    plot_kwargs: dict[str, Any] | None = None,    # 新：仅用于外观的覆盖字典（白名单）
) -> tuple[plt.Axes, dict[str, Any]]:
    """Plot comfort/risk ranges on an operative temperature vs relative humidity chart.

    This function visualizes regions defined by one or more threshold values for a
    comfort metric (e.g., PMV, SET) as a function of operative temperature (x-axis)
    and relative humidity (y-axis). Visual options can be overridden via plot_kwargs.
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

    # ---------------- 原实现：显式传入颜色等外观 + **kwargs 直传 ----------------
    # calc_kwargs: dict[str, Any] = {
    #     "model_func": model_func,
    #     "xy_to_kwargs": mapper_top_rh,
    #     "fixed_params": fixed_params,
    #     "thresholds": thresholds,
    #     "x_bounds": (t_lo, t_hi),
    #     "y_values": y_values,
    #     "metric_attr": None,
    #     "ax": ax,
    #     "xlabel": "Operative temperature [°C]",
    #     "ylabel": "Relative humidity [%]",
    #     "legend": legend,
    #     "x_scan_step": float(x_scan_step),
    #     "smooth_sigma": float(smooth_sigma),
    #     "cmap": cmap,
    #     "band_alpha": band_alpha,
    #     "line_color": line_color,
    #     "line_width": line_width,
    # }
    # calc_kwargs.update(kwargs)
    # ax, artists = calc_plot_ranges(**calc_kwargs)

    # ---------------- 新实现：基础参数 + 外观仅通过 plot_kwargs（白名单）覆盖 ----------------
    kwargs: dict[str, Any] = {
        "model_func": model_func,
        "xy_to_kwargs": mapper_top_rh,
        "fixed_params": fixed_params,
        "thresholds": thresholds,
        "x_bounds": (t_lo, t_hi),
        "y_values": y_values,
        "metric_attr": None,
        "ax": ax,
        "xlabel": "Operative temperature [°C]",    # 外观项：可被 plot_kwargs 覆盖
        "ylabel": "Relative humidity [%]",         # 外观项：可被 plot_kwargs 覆盖
        "legend": legend,
        "x_scan_step": float(x_scan_step),
        "smooth_sigma": float(smooth_sigma),
    }  # 新：不直接在签名层设定颜色，保持默认配色不变

    if plot_kwargs: kwargs.update({k: v for k, v in plot_kwargs.items() if k not in ("model_func", "xy_to_kwargs")})
    ax, artists = calc_plot_ranges(**kwargs)       # The call remains unchanged, only the source of the parameters changes

    # ---------------- Figure 层统一外观（不改变配色） ----------------
    fig = ax.figure                                  # 新：使用当前 Axes 的 Figure，避免多图拿错
    fig.set_size_inches(7, 5)                        # 新：统一图幅大小，适配论文/幻灯片
    fig.set_dpi(150)                                 # 新：统一清晰度
    fig.suptitle(                                    # 新：将标题放在 Figure 层，避免与图例重叠
        title or "Operative Temperature vs Relative Humidity",
        fontsize=fontsize,
        y=0.96,
    )

    # ---------------- 上移并压缩绘图区，为标题与图例留空间 ----------------
    try:
        pos = ax.get_position()
        ax.set_position([pos.x0, max(0.06, pos.y0 + 0.02), pos.width, pos.height * 0.92])  # 新：留白
    except Exception:
        pass  # 新：防守式，确保在特定后端下不报错

    return ax, artists


if __name__ == "__main__":
    from pythermalcomfort.models import pmv_ppd_iso  # type: ignore

    ax, _ = ranges_to_rh(
        model_func=pmv_ppd_iso,
        fixed_params={"met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0.0},
        thresholds=[-0.5, 0.5],
        t_range=(18, 30),
        rh_range=(10, 90),
        rh_step=2.0,
        # 不修改颜色；如需外观覆盖，仅用于视觉项：
        # plot_kwargs={"xlabel": "Operative Temp [°C]"}  # 示例：可选
        title="Operative Temperature vs RH",            # 新：与队友风格一致
        fontsize=12.0,                                  # 新：统一字体
    )

    import matplotlib.pyplot as plt
    plt.show()
