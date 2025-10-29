# pmv_contours_edit.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass
from pythermalcomfort.models import pmv_ppd_iso, pmv_ppd_ashrae

@dataclass
class PMVField:
    X: np.ndarray
    Y: np.ndarray
    PMV: np.ndarray
    PPD: np.ndarray
    xname: str
    yname: str
    meta: dict

DEFAULT_PMV_LEVELS = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)

def compute_pmv_field(
    xname, yname, x, y, fixed, model="ISO", tr_match_tdb=True, limit_inputs=False
) -> PMVField:
    X, Y = np.meshgrid(x, y)
    nrows, ncols = X.shape

    
    params = {k: np.full((nrows, ncols), v) for k, v in fixed.items()}
    params[xname] = X
    params[yname] = Y

   
    if tr_match_tdb:
        params["tr"] = params["tdb"]

   
    if "vr" not in params:
        if "v" in params:
            params["vr"] = params["v"]
        else:
            
            params["vr"] = np.full_like(params["tdb"], 0.1)

   
    if "v" in params:
        del params["v"]

   
    if model.upper() == "ISO":
        res = pmv_ppd_iso(limit_inputs=limit_inputs, **params)
    else:
        res = pmv_ppd_ashrae(limit_inputs=limit_inputs, **params)

    return PMVField(
        X=X, Y=Y,
        PMV=res["pmv"], PPD=res["ppd"],
        xname=xname, yname=yname,
        meta={**fixed, "model": model}
    )


def plot_pmv_contours(
    field: PMVField,
    title="PMV (ISO 7730) — Thermal Comfort Zones",
    levels=DEFAULT_PMV_LEVELS,
    figsize=(10.0, 5.4),            
    show_comfort_band=True,
    comfort_band=(-0.5, 0.5),
    show_lines=True,
    line_levels=(-3, -2, -1, 0, 1, 2, 3),
    line_color="black",
    line_width=1.2,
    font_size=10,                  
    colors=None,
    show_colorbar=True,
    axes_label=True,
):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    X, Y, Z = field.X, field.Y, field.PMV

    xlab = "Air Temperature (°C)" if field.xname in ("tdb", "ta") else field.xname
    ylab = "Relative Humidity (%)" if field.yname == "rh" else field.yname

   
    if colors is None:
        colors = [
            (0.18, 0.31, 0.60), (0.28, 0.52, 0.85), (0.60, 0.78, 0.94),
            (0.88, 0.92, 0.97), (0.98, 0.98, 0.98),
            (0.99, 0.87, 0.78), (0.98, 0.59, 0.47), (0.80, 0.20, 0.20),
        ]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

   
    mpl.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })
    title_fs = font_size + 4
    subtitle_fs = font_size + 2

    
    fig, ax = plt.subplots(figsize=figsize)
    
    fig.subplots_adjust(left=0.12, bottom=0.12, top=0.86, right=0.86)

    
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, antialiased=True)

    
    if show_comfort_band and comfort_band:
        lo, hi = comfort_band
        ax.contour(X, Y, Z, levels=[lo, hi], colors="k", linewidths=1.6, linestyles="--")
        ax.contourf(X, Y, Z, levels=[lo, hi], colors=[(0.97, 0.97, 0.97, 0.45)])

    
    if show_lines:
        cs = ax.contour(X, Y, Z, levels=line_levels, colors=line_color, linewidths=line_width)
        ax.clabel(cs, cs.levels, inline=True, fmt=lambda v: f"{v:.0f}", fontsize=font_size)

    
    if axes_label:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    ax.grid(which="major", alpha=0.10)

    
    fig.suptitle(title, y=0.965, fontsize=title_fs, fontweight="bold")
    if getattr(field, "meta", None):
        m = field.meta
        sub = f"met={m.get('met','?')}, clo={m.get('clo','?')}, v={m.get('v','?')} m/s, tr={m.get('tr','?')} °C"
        ax.set_title(sub, fontsize=subtitle_fs, pad=4, color="#222")

    
    if show_colorbar:
        
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.92)
        mids = [(levels[i] + levels[i+1]) / 2 for i in range(len(levels)-1)]
        labels = [f"{levels[i]} to {levels[i+1]}" for i in range(len(levels)-1)]
        cbar.set_ticks(mids)
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=font_size)
        cbar.set_label("PMV bands", fontsize=font_size + 1)

    return fig
