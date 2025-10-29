# Fixed version of plots.py for latest pythermalcomfort compatibility

from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from pythermalcomfort.models import pmv_ppd_iso, utci, heat_index_rothfusz, two_nodes_gagge


class ComfortPlotter:
    def __init__(self, use_plotly: bool = True):
        self.use_plotly = use_plotly
        self.colorscales = {
            "pmv": {
                "plotly": "RdBu_r",
                "matplotlib": self._create_pmv_colormap(),
                "levels": [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3],
                "labels": {
                    -3: "Cold", -2: "Cool", -1: "Slightly cool", 0: "Neutral",
                    1: "Slightly warm", 2: "Warm", 3: "Hot"
                },
            },
            "utci": {
                "plotly": "Jet", "matplotlib": "jet",
                "levels": [9, 26, 32, 38, 46],
                "labels": {
                    9: "Strong cold stress", 26: "No thermal stress",
                    32: "Moderate heat stress", 38: "Strong heat stress",
                    46: "Extreme heat stress"
                },
            },
            "hi": {
                "plotly": "Jet", "matplotlib": "jet",
                "levels": [9, 26, 32, 38, 46],
                "labels": {
                    9: "Strong cold stress", 26: "No thermal stress",
                    32: "Moderate heat stress", 38: "Strong heat stress",
                    46: "Extreme heat stress"
                },
            },
        }

    def _create_pmv_colormap(self):
        colors = [(0, 0, 1), (0.5, 0.5, 1), (1, 1, 1), (1, 0.5, 0.5), (1, 0, 0)]
        return LinearSegmentedColormap.from_list("pmv_cmap", colors, N=100)

    def plot_hourly_data(self, data: dict[str, list[float]],
                          x_var="ta", y_var="rh", comfort_var="pmv",
                          plot_type="scatter", title=None, colorbar_label=None, **kwargs) -> Union[plt.Figure, go.Figure]:
        x_data = data[x_var]
        y_data = data[y_var]
        z_data = data[comfort_var]

        colorscale = self.colorscales.get(comfort_var.lower(), {"plotly": "Viridis", "matplotlib": "viridis"})

        if self.use_plotly:
            if plot_type == "scatter":
                fig = px.scatter(x=x_data, y=y_data, color=z_data,
                                 labels={"x": x_var, "y": y_var, "color": comfort_var},
                                 color_continuous_scale=colorscale["plotly"], title=title, **kwargs)
                return fig
            elif plot_type == "heatmap":
                fig = go.Figure(data=go.Heatmap(x=x_data, y=y_data, z=z_data,
                                                colorscale=colorscale["plotly"],
                                                colorbar=dict(title=colorbar_label or comfort_var)))
                fig.update_layout(title=title, xaxis_title=x_var, yaxis_title=y_var)
                return fig
        else:
            fig, ax = plt.subplots()
            sc = ax.scatter(x_data, y_data, c=z_data, cmap=colorscale["matplotlib"])
            plt.colorbar(sc, ax=ax, label=colorbar_label or comfort_var)
            ax.set_title(title)
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            return fig

    def plot_comfort_contours(self, comfort_model: Callable, x_var="ta", y_var="rh",
                              x_range=(18, 35), y_range=(0, 100), fixed_params=None,
                              comfort_var="pmv", **kwargs):
        if fixed_params is None:
            fixed_params = {}

        x_vals = np.linspace(*x_range, 100)
        y_vals = np.linspace(*y_range, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                params = {**fixed_params, x_var: X[i, j], y_var: Y[i, j]}
                try:
                    res = comfort_model(**params)
                    Z[i, j] = res[comfort_var] if isinstance(res, dict) else res
                except Exception:
                    Z[i, j] = np.nan

        if self.use_plotly:
            fig = go.Figure(data=go.Contour(x=x_vals, y=y_vals, z=Z,
                                            colorscale=self.colorscales.get(comfort_var, {}).get("plotly", "Viridis"),
                                            colorbar=dict(title=comfort_var.upper())))
            fig.update_layout(title=f"{comfort_var.upper()} Contours",
                              xaxis_title=x_var, yaxis_title=y_var)
            return fig
        else:
            fig, ax = plt.subplots()
            cs = ax.contourf(X, Y, Z, cmap=self.colorscales.get(comfort_var, {}).get("matplotlib", "viridis"))
            fig.colorbar(cs, ax=ax, label=comfort_var.upper())
            ax.set_title(f"{comfort_var.upper()} Contours")
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            return fig


# Example usage:
if __name__ == "__main__":
    hourly_data = {
        "ta": np.random.uniform(18, 30, 1000),
        "rh": np.random.uniform(30, 70, 1000),
    }

    hourly_data["pmv"] = [
        pmv_ppd_iso(tdb=ta, rh=rh, tr=ta, vr=0.1, met=1.2, clo=0.5)["pmv"]
        for ta, rh in zip(hourly_data["ta"], hourly_data["rh"])
    ]

    hourly_data["utci"] = [
        utci(tdb=ta, tr=ta, rh=rh, v=3).utci
        for ta, rh in zip(hourly_data["ta"], hourly_data["rh"])
    ]

    plotter = ComfortPlotter(use_plotly=True)

    # PMV Scatter Plot
    fig1 = plotter.plot_hourly_data(hourly_data, comfort_var="pmv", plot_type="scatter", title="PMV Scatter")
    fig1.show()

    # UTCI Scatter Plot
    fig2 = plotter.plot_hourly_data(hourly_data, comfort_var="utci", plot_type="scatter", title="UTCI Scatter")
    fig2.show()

    # PMV Contours
    pmv_contour = plotter.plot_comfort_contours(
        lambda tdb, rh: pmv_ppd_iso(tdb=tdb, rh=rh, tr=tdb, vr=0.1, met=1.2, clo=0.5),
        x_var="tdb", y_var="rh", comfort_var="pmv")
    pmv_contour.show()

    # UTCI Contours
    utci_contour = plotter.plot_comfort_contours(
        lambda tdb, rh: utci(tdb=tdb, tr=tdb, rh=rh, v=3).utci,
        x_var="tdb", y_var="rh", comfort_var="utci")
    utci_contour.show()

    # Heat Index Contours
    hi_contour = plotter.plot_comfort_contours(
        lambda tdb, rh: heat_index_rothfusz(tdb=tdb, rh=rh),
        x_var="tdb", y_var="rh", comfort_var="hi")
    hi_contour.show()