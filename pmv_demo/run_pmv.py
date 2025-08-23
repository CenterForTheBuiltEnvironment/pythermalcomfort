# run_pmv.py
import matplotlib.pyplot as plt
import numpy as np
from pmv_contours import (
    DEFAULT_COMFORT_BAND,  # can be overridden with custom values
    DEFAULT_PMV_LEVELS,  # can be overridden with custom values
    compute_pmv_field,
    plot_pmv_contours,
)


def main():
    # —— Unified entry point: only modify here ——
    PARAMS = {"v": 0.2, "met": 1.0, "clo": 0.5, "tr": 28}
    PMV_LEVELS = DEFAULT_PMV_LEVELS  # e.g. change to (-3,-2,-1,-0.7,-0.3,0.3,0.7,2,3)
    COMFORT_BAND = DEFAULT_COMFORT_BAND  # set None to disable; or change to (-0.7, 0.7)

    # High-resolution grid (smooth contour curves)
    x = np.linspace(18, 32, 401)  # tdb °C
    y = np.linspace(20, 80, 401)  # rh %

    # Compute PMV field (choose ISO or ASHRAE)
    field = compute_pmv_field(
        "tdb",
        "rh",
        x,
        y,
        fixed=PARAMS,
        model="ISO",
        tr_match_tdb=False,
        limit_inputs=False,
    )

    # Plot PMV contours
    fig = plot_pmv_contours(
        field,
        levels=PMV_LEVELS,
        comfort_band=COMFORT_BAND,
        title="PMV (ISO 7730) — Thermal Comfort Zones",
        show_colorbar=True,
        show_lines=True,
        font_size=11,
    )

    fig.show()


if __name__ == "__main__":
    main()
    plt.show()
