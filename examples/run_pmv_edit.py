
import numpy as np
import matplotlib.pyplot as plt
from pmv_contours_edit import compute_pmv_field, plot_pmv_contours

def main():
    x = np.linspace(18, 32, 401)
    y = np.linspace(20, 80, 401)

   
    base_params = {"v": 0.2, "met": 1.0, "clo": 0.5, "tr": 28}
    field_base = compute_pmv_field("tdb", "rh", x, y, fixed=base_params, model="ISO",
                                   tr_match_tdb=False, limit_inputs=False)
    fig1 = plot_pmv_contours(field_base, title="PMV (ISO 7730) — Baseline (clo=0.5)")
    fig1.show()


if __name__ == "__main__":
    main()
    plt.show()
