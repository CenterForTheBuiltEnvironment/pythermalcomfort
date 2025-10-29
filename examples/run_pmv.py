from pythermalcomfort.models import pmv_ppd_iso
from pmv_contours import plot_t_rh_chart, ChartConfig, FillMode, ModelType
import matplotlib.pyplot as plt

# Using dict
config = {
    "model_func": pmv_ppd_iso,
    "model_params": {"met": 1.2, "clo": 0.5, "v": 0.1, "wme": 0.0, "limit_inputs": False},
    "t_range": (18, 36),
    "rh_range": (0, 100),
    "thresholds": [-0.5, 0.5],
    "fill_mode": FillMode.BANDS,
    "model_type": ModelType.PMV,   
}
fig, ax = plot_t_rh_chart(config)
ax.set_xlabel("Air temperature [°C]")
ax.set_ylabel("Relative humidity [%]")
ax.set_xlim(18, 36)
ax.set_ylim(0, 100)
plt.show()

# Using ChartConfig
config = ChartConfig(
    model_func=pmv_ppd_iso,
    model_params={"met": 1.2, "clo": 0.5, "v": 0.1, "wme": 0.0, "limit_inputs": False},
    t_range=(18, 36),
    rh_range=(0, 100),
    thresholds=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    fill_mode=FillMode.BANDS,
    model_type=ModelType.PMV,   
)
fig, ax = plot_t_rh_chart(config)
ax.set_xlabel("Air temperature [°C]")
ax.set_ylabel("Relative humidity [%]")
ax.set_xlim(18, 36)
ax.set_ylim(0, 100)
plt.show()