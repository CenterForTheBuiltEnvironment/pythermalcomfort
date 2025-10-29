# -*- coding: utf-8 -*-
# 运行你主文件 phstest.py 中的底层核心函数（绕开 np.vectorize 的 32 操作数限制）
# 与 MATLAB PHS79332023 测试脚本 1:1 对应的 5 个工况

import math
import pandas as pd
import numpy as np
import phstest  # 你的主文件（内含 phs 和 _phs_optimized）
from pythermalcomfort.utilities import met_to_w_m2

# 通过 .pyfunc 绕开 numpy.vectorize 的限制（不是调用 phs()）
phs_core = phstest._phs_optimized.pyfunc

# ---------- 公共输入 ----------
weight = 75.0
height = 1.80
Adu = 0.202 * (height ** 0.725) * (weight ** 0.425)  # m²
D = 15.0
eps_g = 0.95
duration = 480  # min

def tg2tr(Tg, Ta, Va, D_cm=15.0, eps_g=0.95):
    return ((Tg + 273.0) ** 4
            + (1.111e8 / (eps_g * (0.01 * D_cm) ** 0.4)) * (Va ** 0.6) * (Tg - Ta)
            ) ** 0.25 - 273.0

def posture_str(code: int) -> str:
    return {1: "standing", 2: "sitting", 3: "crouching"}[code]

def accl_to_flag(accl_matlab: int) -> int:
    # 与你之前约定保持一致：MATLAB 里 accl=1 记为“已适应”，→ 100；其他（含2）→ 0
    return 100 if accl_matlab == 1 else 0

def safe_int(x, cap=None):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if cap is not None and x >= cap:
            return None
        return int(x)
    except Exception:
        return None


# ---------- 五个工况（与 MATLAB 一致） ----------
cases = [
    # accl, posture, Ta, Tg, Va, RH, M,   Icl, Ap,   Fr
    (1,    1,       40,  40, 0.30, 35, 300, 0.5, 0.0, 0.0),
    (2,    1,       35,  35, 0.10, 60, 300, 0.5, 0.0, 0.0),
    (2,    1,       30,  45, 0.10, 35, 300, 0.8, 0.30, 0.85),
    (2,    1,       30,  30, 1.00, 45, 450, 0.5, 0.0, 0.0),
    (1,    2,       35,  50, 1.00, 30, 250, 1.0, 0.20, 0.85),
]

rows = []

for idx, (accl, posture, Ta, Tg, Va, RH, M, Icl, Ap, Fr) in enumerate(cases, start=1):
    Tr = tg2tr(Tg, Ta, Va, D_cm=D, eps_g=eps_g)

    # 2023 口径：p_a[kPa] = 0.6105 * exp(17.27*Ta/(Ta+237.3)) * RH/100
    p_a = 0.6105 * math.exp(17.27 * Ta / (Ta + 237.3)) * RH / 100.0

    # 注意：_phs_optimized 期望 met 的单位是 W/m²（你的 phs() 才用 met 单位）
    met_wm2 = (M / Adu)             # W 总除以体表面积 = W/m²
    wme_wm2 = 0.0

    # 直接调你主文件里的底层核心：返回 10 个量
    (t_re, t_sk, t_cr, t_cr_eq, t_sk_t_cr_wg,
     sweat_rate, sw_tot_g, d_lim_loss_50, d_lim_loss_95, d_lim_t_re) = phs_core(
        Ta,                 # tdb
        Tr,                 # tr
        Va,                 # v
        p_a,                # p_a [kPa]
        met_wm2,            # met [W/m²]
        Icl,                # clo [clo]
        posture_str(posture),
        1,                  # drink (1=可饮水)
        accl_to_flag(accl), # 0/100
        weight,             # kg
        wme_wm2,            # 外功 [W/m²]
        0.38,               # i_mst
        Ap,                 # a_p
        height,             # m
        0.0,                # walk_sp
        0.0,                # theta
        duration,           # min
        Fr,                 # f_r
        34.1,               # t_sk
        36.8,               # t_cr
        36.8,               # t_re
        36.8,               # t_cr_eq
        0.3,                # t_sk_t_cr_wg
        0.0,                # sw_tot (初始)
        "7933-2023",        # model
    )

    # 与 MATLAB OUT 对齐的 6 列
    # SWp(g/h) = 3600 * sweat_rate(W/m²) * Adu / 2540
    SWp_gph = 3600.0 * float(sweat_rate) * Adu / 2540.0

    rows.append([
        idx,
        round(Tr, 2),
        round(SWp_gph, 1) if not math.isnan(SWp_gph) else None,
        round(float(sw_tot_g), 1) if not math.isnan(sw_tot_g) else None,
        round(float(t_re), 2) if not math.isnan(t_re) else None,
        safe_int(float(d_lim_loss_95)),
        safe_int(float(d_lim_t_re), cap=duration),
    ])

# ---------- 输出 ----------
df = pd.DataFrame(rows, columns=[
    "Case", "Tr (°C)", "SWp (g/h)", "SWtotg (g)", "Tre (°C)", "Dlimloss (min)", "Dlimtcr (min)"
])

print("\n==== PHS7933-2023 测试结果（Python，直接调用 phstest._phs_optimized.pyfunc） ====\n")
print(df.to_string(index=False))

df.to_csv("phs_test_results.csv", index=False)
print("\n结果已保存：phs_test_results.csv")
