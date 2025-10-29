from pythermalcomfort.models import pmv_ppd_iso
from pmv_contours import plot_t_rh_chart, ChartConfig
import matplotlib.pyplot as plt

#使用 ChartConfig 类
config = ChartConfig(
    model_params={
        "met": 1.2,
        "clo": 0.5,
        "v": 0.1,
        "wme": 0.0,
        "limit_inputs": False
    },
    t_range=(18, 36), 
    rh_range=(0, 100),
    thresholds=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    # thresholds=[-0.5, 0.5],
    title="PMV (ISO 7730) — Bands from solver-based iso-curves",
    figsize=(7, 5), 
    dpi=120, 
    alpha=0.85,
)

# config = ChartConfig(
#     model_params={
#         "met": 2.0,      # 较高活动水平
#         "clo": 1.0,      # 厚重服装
#         "v": 0.2,        # 较高风速
#         "tr": 28,        # 高温辐射
#         "limit_inputs": False
#     },
#     t_range=(5, 40),     # 宽温度范围
#     rh_range=(5, 95),    # 宽湿度范围
#     thresholds=[-3, -2, -1, 0, 1, 2, 3],  # 宽阈值范围
#     title="Extreme Conditions Test",
#     figsize=(10, 8),
#     dpi=150,
#     alpha=0.6,
# )

fig, ax = plot_t_rh_chart(pmv_ppd_iso, config)
plt.show()

