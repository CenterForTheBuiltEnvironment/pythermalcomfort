from math import ceil, floor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

from pythermalcomfort.models import pmv_ppd_ashrae
from pythermalcomfort.utilities import v_relative, clo_dynamic_iso, psy_ta_rh
from pythermalcomfort.charts.theme import (
    tight_margins,
    index_mapping_dictionary,
)


def pmv_chart(df, pmv_constants, show_summary=True, si_ip="si"):
    """Generates a psychrometric chart with the Predicted Mean Vote (PMV) comfort zone, ASHRAE 55.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the input data. Must have the following columns with matching column names:
        'tdb' (Dry bulb temperature [°C]), 'rh' (Relative humidity [%]).

    pmv_constants : dict
        A dictionary containing the constants required for the PMV calculation. It must have the following keys
        'tr' (Mean radiant temperature [°C]), 'v' (Air speed [m/s]), 'met' (Metabolic rate [met]), 'clo' (Clothing
        insulation [clo]).

    show_summary : bool, optional
        If True, a summary of the data is shown. Default is False.

    si_ip : str, optional
        The unit system to use. It can be 'si' (default) or 'ip'.

    Returns
    -------
    fig : graph_objects.Figure
        Plotly figure object.

    """

    var = "pmv"
    var_metadata = index_mapping_dictionary[var]
    var_name, var_unit, var_range = (
        var_metadata["name"],
        var_metadata[si_ip]["unit"],
        var_metadata[si_ip]["range"],
    )
    cat_color, cat_order = (
        var_metadata["colors_categories"],
        var_metadata["order_categories"],
    )

    t_dry = np.linspace(10, 36, 500)
    rh_levels = np.linspace(0, 100, 10)
    tr, v, met, clo = (
        pmv_constants["tr"],
        pmv_constants["v"],
        pmv_constants["met"],
        pmv_constants["clo"],
    )

    vr = v_relative(v=v, met=met)
    clo_d = clo_dynamic_iso(clo=clo, met=met, v=vr)

    observations = {
        "t_dry": df["tdb"].to_list(),
        "rh": df["rh"].to_list(),
    }

    # helper functions start

    def pmv_eq(T, RH):
        return pmv_ppd_ashrae(
            tdb=T, tr=tr, vr=vr, rh=RH, met=met, clo=clo_d, limit_inputs=False
        ).pmv

    def convert_rh_to_hr(T, RH):
        return psy_ta_rh(T, RH / 100, 101325)["hr"] * 100000

    def find_comfort_temp(RH, target_pmv):
        def pmv_root(T):
            return pmv_eq(T, RH) - target_pmv

        try:
            return brentq(pmv_root, 0, 120)
        except ValueError:
            return None

    np.seterr(divide="ignore", invalid="ignore")

    def extract_comfort_data(zone):
        RH, T = zip(*zone)
        return T, [convert_rh_to_hr(T, rh) for T, rh in zip(T, RH)]

    ### helper functions end

    humidity_ratios = np.array(
        [[convert_rh_to_hr(T, RH) for T in t_dry] for RH in rh_levels]
    )

    target_pmvs = {"Neutral": 0, "Upper": 0.5, "Lower": -0.5}
    comfort_zones = {
        name: [(RH, find_comfort_temp(RH, target)) for RH in rh_levels]
        for name, target in target_pmvs.items()
    }

    neutral_line = extract_comfort_data(comfort_zones["Neutral"])
    upper_line = extract_comfort_data(comfort_zones["Upper"])
    lower_line = extract_comfort_data(comfort_zones["Lower"])

    fig = (
        make_subplots(
            rows=1, cols=2, column_widths=[0.75, 0.25], horizontal_spacing=0.175
        )
        if show_summary
        else go.Figure()
    )

    for i, RH in enumerate(rh_levels):
        fig.add_trace(
            go.Scatter(
                x=t_dry,
                y=humidity_ratios[i],
                mode="lines",
                name=f"{RH}% RH",
                line=dict(width=1, color="lightgrey"),
                showlegend=False,
                hovertemplate="t_dry: %{x:.1f}°C<br>HR: %{y:.2f}<br><extra></extra>",
            ),
            row=1 if show_summary else None,
            col=1 if show_summary else None,
        )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([upper_line[0], lower_line[0][::-1]]),
            y=np.concatenate([upper_line[1], lower_line[1][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Comfort Zone",
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    fig.add_trace(
        go.Scatter(
            x=neutral_line[0],
            y=neutral_line[1],
            mode="lines",
            name="Neutral PMV",
            line=dict(color="black", width=1.5, dash="dash"),
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    fig.add_trace(
        go.Scatter(
            x=observations["t_dry"],
            y=[
                convert_rh_to_hr(T, RH)
                for T, RH in zip(observations["t_dry"], observations["rh"])
            ],
            mode="markers",
            name="Observations",
            marker=dict(color="#ea536e", size=5, symbol="circle"),
            hovertemplate="t_dry: %{x:.1f}°C<br>HR: %{y:.2f}<br><extra></extra>",
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    if show_summary:

        df["pmv"] = df.apply(lambda x: pmv_eq(x["tdb"], x["rh"]), axis=1)

        df.loc[df["pmv"] > 0.5, "categorical"] = "Too Hot"
        df.loc[df["pmv"] < -0.6, "categorical"] = "Too Cold"
        df.loc[df["pmv"].between(-0.5, 0.5), "categorical"] = "Comfortable"
        df.loc[pd.isna(df["pmv"]), "categorical"] = "Out of Range"

        category_counts = df["categorical"].value_counts()
        category_percentages = df["categorical"].value_counts(normalize=True) * 100

        categories = cat_order
        percentages = [0] * len(cat_order)
        sample_counts = [0] * len(cat_order)

        for i, cat in enumerate(cat_order):
            if cat in category_counts.index:
                sample_counts[i] = category_counts[cat]
                percentages[i] = category_percentages[cat]

        fig.add_trace(
            go.Bar(
                x=percentages,
                y=categories,
                orientation="h",
                width=0.6,
                marker=dict(color=cat_color),
                text=[f"{percentage:.1f}%" for percentage in percentages],
                textposition="outside",
                cliponaxis=False,
                customdata=sample_counts,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Percentage: %{x:.1f}%<br>"
                    "Samples: %{customdata} <extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(
            showgrid=False,
            side="left",
            tickfont=dict(weight="bold"),
            ticklabelstandoff=10,
            row=1,
            col=2,
        )

        fig.update_layout(
            title=dict(
                text="Psychrometric Chart with PMV Comfort Zone, ASHRAE 55",
                x=0,
                xref="paper",
                y=0.95,
                yref="container",
            ),
            legend=dict(
                font=dict(size=12), x=0, y=0.85, xanchor="left", traceorder="normal"
            ),
            annotations=[
                dict(
                    x=0,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    text=f"MRT: {tr}°C | v: {v} m/s | {met} met | {clo} clo",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
            width=1000,
            height=500,
            margin=tight_margins,
        )

        fig.update_xaxes(
            title="Dry Bulb Temperature [°C]",
            range=[10, 36],
            dtick=5,
            showgrid=False,
            showline=False,
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title="Humidity Ratio [g water / kg dry air]",
            range=[0, 35],
            dtick=5,
            side="right",
            showgrid=False,
            mirror=False,
            linewidth=1,
            linecolor="lightgrey",
            row=1,
            col=1,
        )

    else:

        fig.update_layout(
            title=dict(
                text="Psychrometric Chart with PMV Comfort Zone, ASHRAE 55",
                x=0,
                xref="paper",
                y=0.95,
                yref="container",
            ),
            xaxis=dict(
                title="Dry Bulb Temperature [°C]",
                showgrid=False,
                showline=False,
                linewidth=1,
                linecolor="lightgrey",
                mirror=False,
                range=[10, 36],
            ),
            yaxis=dict(
                title="Humidity Ratio [g water / kg dry air]",
                side="right",
                showgrid=False,
                linewidth=1,
                linecolor="lightgrey",
                mirror=False,
                range=[0, 35],
            ),
            legend=dict(
                font=dict(size=12), x=0, y=0.85, xanchor="left", traceorder="normal"
            ),
            annotations=[
                dict(
                    x=0,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    text=f"MRT: {tr}°C | v: {v} m/s | {met} met | {clo} clo",
                    showarrow=False,
                    font=dict(size=12),
                )
            ],
            height=600,
            width=800,
            margin=tight_margins,
        )

    return fig
