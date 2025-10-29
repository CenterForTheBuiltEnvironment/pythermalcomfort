import os
import numpy as np
import matplotlib.pyplot as plt
from pmv_contours import compute_pmv_field, plot_pmv_contours

def _save(fig, name):
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(f"outputs/{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"outputs/{name}.svg", bbox_inches="tight")
    print("Saved:", name)

def main():
    x = np.linspace(18, 32, 401)    # 温度 (°C)
    y = np.linspace(20, 80, 401)    # 相对湿度 (%)

    # ===== 图1：基准条件 (clo=0.5) =====
    base_params = {"v": 0.2, "met": 1.0, "clo": 0.5, "tr": 28}
    field_base = compute_pmv_field(
        "tdb", "rh", x, y,
        fixed=base_params, model="ISO",
        tr_match_tdb=False, limit_inputs=False
    )
    fig1 = plot_pmv_contours(field_base,
                             title="PMV (ISO 7730) — Baseline (clo=0.5)")
    _save(fig1, "pmv_iso_baseline")
    fig1.show()

    # ===== 图2：参数灵敏度 (clo=0.7) =====
    new_params = {**base_params, "clo": 0.7}
    field_clo = compute_pmv_field(
        "tdb", "rh", x, y,
        fixed=new_params, model="ISO",
        tr_match_tdb=False, limit_inputs=False
    )
    fig2 = plot_pmv_contours(field_clo,
                             title="PMV (ISO 7730) — Clothing Effect (clo=0.7)")
    _save(fig2, "pmv_iso_clo0.7")
    fig2.show()

if __name__ == "__main__":
    main()
