import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pythermalcomfort.charts.theme import (
    index_mapping_dictionary,
)


def adaptive_chart(df, show_summary=False, si_ip="si"):
    """Generates a scatter plot with the Adaptive Comfort Model, ASHRAE 55.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the input data. Must have the following columns:
        't_out' (Outdoor temperature [°C]), 'top' (Operative temperature [°C]),
        'adaptive_acceptability_80%' (Acceptability for 80% of the occupants),
        'adaptive_acceptability_90%' (Acceptability for 90% of the occupants).

    show_summary : bool, optional
        If True, a summary of the data is shown. Default is False.

    si_ip : str, optional
        The unit system to use. It can be 'si' (default) or 'ip'.

    Returns
    -------
    fig : graph_objects.Figure
        Plotly figure object.

    """

    var = "adaptive"
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

    total_number_of_samples = len(df)

    t_outdoor_range = var_range

    t_outdoor = np.linspace(t_outdoor_range[0], t_outdoor_range[1], 100)
    t_comfort = 0.31 * t_outdoor + 17.8
    t_80_upper, t_80_lower = t_comfort + 3.5, t_comfort - 3.5
    t_90_upper, t_90_lower = t_comfort + 2.5, t_comfort - 2.5

    sum_80 = df["adaptive_acceptability_80%"].sum()
    sum_90 = df["adaptive_acceptability_90%"].sum()

    if show_summary:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2])
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_outdoor,
            y=t_comfort,
            mode="lines",
            name="Comfort Temperature",
            line=dict(color="black", width=1.5, dash="dash"),
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_outdoor, t_outdoor[::-1]]),
            y=np.concatenate([t_90_upper, t_90_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Acceptability",
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_outdoor, t_outdoor[::-1]]),
            y=np.concatenate([t_80_upper, t_80_lower[::-1]]),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Acceptability",
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    fig.add_trace(
        go.Scatter(
            x=df["t_out"],
            y=df["top"],
            mode="markers",
            name="Observations",
            marker=dict(color="#ea536e", size=5, symbol="circle"),
            hovertemplate="t_out: %{x:.1f}<br>t_op: %{y:.1f}<br><extra></extra>",
        ),
        row=1 if show_summary else None,
        col=1 if show_summary else None,
    )

    if show_summary:

        conditions = [
            pd.isna(df["adaptive_acceptability_80%"]),
            df["adaptive_acceptability_90%"] == True,
            df["adaptive_acceptability_80%"] == True,
            (df["adaptive_acceptability_80%"] == False)
            & (df["adaptive_acceptability_90%"] == False),
        ]

        df["categorical"] = np.select(conditions, cat_order, default="Out of Range")

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
            ticklabelstandoff=10,
            tickfont=dict(
                weight="bold",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Adaptive Comfort Model, ASHRAE 55",
            xaxis_title="Prevailing Mean Outdoor Temperature [°C]",
            yaxis_title="Operative Temperature [°C]",
            legend=dict(
                font=dict(size=12),
                orientation="h",
                yanchor="bottom",
                y=1.025,
                xanchor="left",
                x=0,
            ),
            xaxis=dict(range=[9, 34.5], dtick=2),
            yaxis=dict(range=[13, 37], dtick=2),
            width=1000,
            height=500,
        )

    else:
        fig.update_layout(
            title="Adaptive Comfort Model, ASHRAE 55",
            xaxis_title="Prevailing Mean Outdoor Temperature [°C]",
            yaxis_title="Operative Temperature [°C]",
            legend=dict(font=dict(size=12)),
            xaxis=dict(range=[9, 34.5], dtick=2),
            yaxis=dict(range=[13, 37], dtick=2),
            height=600,
            width=800,
            annotations=[
                dict(
                    x=1.025,
                    y=0.15,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    text=f"Within 80% range: {sum_80/total_number_of_samples*100:.1f}%",
                    showarrow=False,
                    font=dict(size=12),
                ),
                dict(
                    x=1.025,
                    y=0.1,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    text=f"Within 90% range: {sum_90/total_number_of_samples*100:.1f}%",
                    showarrow=False,
                    font=dict(size=12),
                ),
            ],
        )

    return fig
