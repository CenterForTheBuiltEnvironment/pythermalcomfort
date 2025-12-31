import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charts.theme import (
    index_mapping_dictionary,
)

from charts.classes import AdaptivePlot, Scatter, AdaptiveSummary


def adaptive_chart(df, units="si", show_summary=True):
    """Generates a scatter plot with the Adaptive cmf Model, ASHRAE 55.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the input data. Must have the following columns with matching column names:
        't_out' (Outdoor temperature [°C]),
        'top' (Operative temperature [°C]),
        'adaptive_acceptability_80%' (Acceptability for 80% of the occupants),
        'adaptive_acceptability_90%' (Acceptability for 90% of the occupants).

    show_summary : bool, optional
    A horizontal Plotly bar chart with the percentage of samples within the 80% and 90% acceptability range is shown.
        If True, a summary of the data is shown. Default is False.

    si_ip : str, optional
        The unit system to use. It can be 'si' (default) or 'ip'.

    Returns
    -------
    fig : graph_objects.Figure
        Plotly figure object.

    """

    base = AdaptivePlot(show_summary=show_summary)

    total_number_of_samples = len(df)  # int to calculate summary percentages

    if base.show_summary:
        fig = make_subplots(
            rows=1, cols=2, column_widths=[1 - base.summary_width, base.summary_width]
        )
    else:
        fig = go.Figure()

    t_outdoor = np.linspace(min(base.range), max(base.range), 100)

    t_cmf = 0.31 * t_outdoor + 17.8
    t_80_upper, t_80_lower = t_cmf + 3.5, t_cmf - 3.5
    t_90_upper, t_90_lower = t_cmf + 2.5, t_cmf - 2.5

    sum_80 = df["adaptive_acceptability_80%"].sum()
    sum_90 = df["adaptive_acceptability_90%"].sum()

    # fig.add_trace(
    #     go.Scatter(
    #         x=t_outdoor,
    #         y=t_cmf,
    #         mode="lines",
    #         name="Comfort Temperature",
    #         line=dict(color="black", width=1.5, dash="dash"),
    #     ),
    #     row=1 if base.show_summary else None,
    #     col=1 if base.show_summary else None,
    # )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_outdoor, t_outdoor[::-1]]),
            y=np.concatenate([t_90_upper, t_90_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Acceptability",
        ),
        row=1 if base.show_summary else None,
        col=1 if base.show_summary else None,
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
        row=1 if base.show_summary else None,
        col=1 if base.show_summary else None,
    )

    # ------------------------------------------------------------------------

    # adding observations

    samples = Scatter(base)

    if base.show_summary:

        fig.add_trace(
            samples.plot(df=df),
            row=1,
            col=1,
        )

        summary = AdaptiveSummary()

        fig.add_trace(
            summary.plot(df=df),
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
            title=base.title,
            xaxis_title=base.x_axis_label,
            yaxis_title=base.y_axis_label,
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

    else:  # if no summary

        fig.add_trace(samples.plot(df=df))

        fig.update_layout(
            title=base.title,
            xaxis_title=base.x_axis_label,
            yaxis_title=base.y_axis_label,
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
                    text=f"Within 90% range: {sum_90+sum_80/total_number_of_samples*100:.1f}%",
                    showarrow=False,
                    font=dict(size=12),
                ),
            ],
        )

    return fig
