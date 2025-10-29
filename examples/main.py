import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import polyfit

# hss 可选导入（没有就跳过 HSS 例子）
try:
    from hss import hss_world_rugby
    HAS_HSS = True
except ModuleNotFoundError:
    HAS_HSS = False
    print("[INFO] 'hss' not installed; skipping HSS section.")

# 兼容导入 pmv 函数：优先 pmv_ppd，回退 pmv_ppd_iso
try:
    from pythermalcomfort.models import pmv_ppd as _pmv_func
except ImportError:
    from pythermalcomfort.models import pmv_ppd_iso as _pmv_func

def _pmv_value(params: dict) -> float:
    if "v" in params and "vr" not in params:
        params["vr"] = params.pop("v")
    res = _pmv_func(**params)
    return res.pmv if hasattr(res, "pmv") else res["pmv"]

# =============================== HSS 原有代码（加一个保护） ===============================
if HAS_HSS:
    # inputs
    v = 1.5
    tg_delta = 8

    thresholds = [0, 100, 150, 200, 250, 300]
    risk_name = ['Low', 'Moderate', 'High', 'Very High', 'Extreme']
    colors = ["#0096FF", "#72c66e", "#f6ee54", "#fabd57", "#ee4d55"]

    results = []
    for t in np.arange(15, 45., .1):
        for rh in range(0, 101, 1):
            try:
                result = hss_world_rugby(tdb=t, rh=rh, v=v, tg=t+tg_delta)
                results.append({
                    "t": t,
                    "rh": rh,
                    "hss": result,
                })
            except ZeroDivisionError as e:
                print(f"Error for t={t}, rh={rh}: {e}")
                continue

    df = pd.DataFrame(results)
    df["hss_bin"] = pd.cut(df["hss"], bins=thresholds, labels=risk_name)
    df["hss_bin_value"] = df["hss_bin"].cat.codes
    df_pivot = df.pivot(index="rh", columns="t", values="hss_bin_value")
    df_pivot.sort_index(ascending=False, inplace=True)

    sns.heatmap(df_pivot, cmap="YlOrRd")
    plt.tight_layout()
    plt.show()

    # 后面这段基于 df_pivot 的“分段填色曲线”也要一起包在 if HAS_HSS: 里面
    results = []
    for ix, row in df_pivot.iterrows():
        diff = row[row.diff() == 1]
        temps = diff.index.tolist()
        risks = diff.values.tolist()
        print(ix, temps, risks)
        for risk, temp in zip(risks, temps):
            results.append({"t": temp, "rh": ix, "risk": risk})
    df_risk = pd.DataFrame(results)

    f, ax = plt.subplots()
    y_before = 0
    for ix, risk in enumerate(df_risk["risk"].unique()):
        data = df_risk[df_risk["risk"] == risk]
        data.drop_duplicates("t", inplace=True)
        x = np.linspace(0, 100, num=100)
        z = polyfit(data.t.values, data.rh.values, 3)
        y = np.polyval(z, x)
        ax.fill_between(x, y_before, y, alpha=0.9, color=colors[ix])
        y_before = y
        plt.plot(x, y, label=risk, color=colors[ix], linewidth=2)

    ax.fill_between(x, y_before, 100, alpha=0.9, color="#ee4d55")
    ax.set(xlabel="Temperature (°C)", ylabel="Relative Humidity (%)",
           xlim=(15, 45), ylim=(0, 100))
    sns.despine()
    plt.title(f"v_{v}_tg_{tg_delta}")
    plt.tight_layout()
    plt.savefig(f"hss_risk_world_rugby_v_{v}_tg_{tg_delta}.png", dpi=300)
    plt.show()
# =============================  HSS 保护结束  =============================



# =============================== PMV 新增块（沿用你原先风格） ===============================
# 说明：
# - 保留你原来的 seaborn heatmap 风格与流程，只是把 PMV 计算改成用 _pmv_value(...)（兼容新旧）
# - 默认参数 met/clo/tr/vr 可自由改；阈值也可换成你注释里那组（含 -0.5/0/+0.5）

pmv_params = dict(tr=25, vr=0.1, met=1.2, clo=0.5)  # 可按需修改
pmv_thresholds = [-100, -1.5, -0.5, 0.5, 1.5, 2.5, 100]  # 含舒适带 -0.5~0.5
pmv_risk_name = ['Very Cold', 'Cold', 'Neutral', 'Warm', 'Hot', 'Very Hot']
pmv_cmap = "rainbow"  # 与你注释一致

pmv_results = []
for t in np.arange(10, 36., .1):
    for rh in range(0, 101, 1):
        try:
            pmv = _pmv_value({
                "tdb": t,
                "tr": t,
                "rh": rh,
                "vr": pmv_params["vr"],
                "met": pmv_params["met"],
                "clo": pmv_params["clo"],
                "limit_inputs": False,
            })
            pmv_results.append({
                "t": t,
                "rh": rh,
                "pmv": pmv,
            })
        except ZeroDivisionError as e:
            print(f"PMV Error for t={t}, rh={rh}: {e}")
            continue

pmv_df = pd.DataFrame(pmv_results)
pmv_df["pmv_bin"] = pd.cut(
    pmv_df["pmv"],
    bins=pmv_thresholds,
    labels=pmv_risk_name,
)
pmv_df["pmv_bin_value"] = pmv_df["pmv_bin"].cat.codes
pmv_pivot = pmv_df.pivot(index="rh", columns="t", values="pmv_bin_value")
pmv_pivot.sort_index(ascending=False, inplace=True)

# 出图（延续你的风格）
sns.heatmap(pmv_pivot, cmap=pmv_cmap)
plt.tight_layout()
plt.title(f"PMV map (met={pmv_params['met']}, clo={pmv_params['clo']}, tr={pmv_params['tr']}, vr={pmv_params['vr']})")
plt.savefig("pmv_map.png", dpi=300)
plt.show()

# 导出网格数据，便于复现/校验（老师要求可复现）
pmv_df.to_csv("pmv_grid.csv", index=False)
