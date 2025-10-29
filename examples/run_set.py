from pythermalcomfort.models import set_tmp
from pmv_contours import plot_t_rh_chart, ChartConfig, FillMode, ModelType
import matplotlib.pyplot as plt

# Using dict
config = {
    "model_func": set_tmp,
    "model_params": {"met": 1.2, "clo": 0.5, "v": 0.1, "tr": 25.0},
    "t_range": (15, 40),
    "rh_range": (40, 90),
    "thresholds": [22, 24, 26, 28, 32],
    "fill_mode": FillMode.BANDS,
    "model_type": ModelType.SET,   
}
fig, ax = plot_t_rh_chart(config)
ax.set_xlabel("Air temperature [°C]")
ax.set_ylabel("Relative humidity [%]")
ax.set_xlim(15, 40)
ax.set_ylim(40, 90)
plt.show()

# Using ChartConfig
config = ChartConfig(
    model_func=set_tmp,
    model_params={"met": 1.2, "clo": 0.5, "v": 0.1, "tr": 25.0},
    t_range=(15, 40),
    rh_range=(40, 90),
    thresholds=[22, 24, 26, 28, 32],
    fill_mode=FillMode.BANDS,
    model_type=ModelType.SET,   
)
fig, ax = plot_t_rh_chart(config)
ax.set_xlabel("Air temperature [°C]")
ax.set_ylabel("Relative humidity [%]")
ax.set_xlim(15, 40)
ax.set_ylim(40, 90)
plt.show()