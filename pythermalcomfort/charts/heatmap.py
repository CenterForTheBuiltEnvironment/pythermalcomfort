from math import ceil, floor
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pythermalcomfort.charts.theme import (
    tight_margins,
    index_mapping_dictionary,
)


def heatmap(df, var, global_local, si_ip, show_summary=False):
    """Generates a heatmap with the desired variable.

    Parameters
    ----------
    df : DataFrame
        A DataFrame containing the input data. Must have the following columns:
        'DateTime' (Date and time), 'Hour' (Hour of the day), 'Month Name' (Month name), 'Day' (Day of the month),
        and the variable to be plotted.

    var : str
        The variable to be plotted.

    global_local : str
        The scope of the data. It can be 'global' or 'local'.

    si_ip : str
        The unit system to use. It can be 'si' or 'ip'.

    show_summary : bool, optional
        If True, a summary of the data is shown. Default is False.

    Returns
    -------
    fig : graph_objects.Figure
        Plotly figure object.

    """

    var_name = index_mapping_dictionary[var]["name"]
    var_unit = index_mapping_dictionary[var][si_ip]["unit"]
    var_range = index_mapping_dictionary[var][si_ip]["range"]
    var_color = index_mapping_dictionary[var]["colors"]

    if global_local == "global":
        range_z = var_range
    else:
        data_max = 5 * ceil(df[var].max() / 5)
        data_min = 5 * floor(df[var].min() / 5)
        range_z = [data_min, data_max]

    if not show_summary:
        fig = go.Figure(
            go.Heatmap(
                y=df["Hour"] + 0.5,  # Offset hour for better display
                x=df["DateTime"].dt.date,
                z=df[var],
                zmin=range_z[0],
                zmax=range_z[1],
                customdata=np.stack(
                    (df["Month Name"], df["Day"], df["Hour"], df["Hour"] + 1), axis=-1
                ),
                hovertemplate=(
                    "<b>"
                    + var
                    + ": %{z:.2f} "
                    + var_unit
                    + "</b><br>Month: %{customdata[0]}<br>Day: %{customdata[1]}<br>Hour:"
                    " %{customdata[2]}-%{customdata[3]}<br>"
                ),
                name="",
                colorscale=var_color,
                colorbar=dict(
                    title=dict(
                        text=f"{var_name}" + " in " + f"{var_unit} ", side="right"
                    ),
                    orientation="h",
                    thickness=20,
                    x=0,
                    xref="paper",
                    xanchor="left",
                    y=0.85,
                    yref="container",
                    yanchor="bottom",
                    ticks="inside",
                    ticklabelposition="outside top",
                ),
            )
        )

        fig.update_yaxes(
            title_text="Hour", mirror=True, range=[0, 24], zerolinewidth=1, dtick=2
        )
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b",
            ticklabelmode="period",
            title_text="Day",
            mirror=True,
            zeroline=False,
        )

        fig.update_layout(margin=tight_margins, width=800, height=500)

    else:
        cat_color = index_mapping_dictionary[var]["colors_categories"]
        cat_order = index_mapping_dictionary[var]["order_categories"]

        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.8, 0.2],
        )

        fig.add_trace(
            go.Heatmap(
                y=df["Hour"] + 0.5,
                x=df["DateTime"].dt.date,
                z=df[var],
                zmin=range_z[0],
                zmax=range_z[1],
                customdata=np.stack(
                    (df["Month Name"], df["Day"], df["Hour"], df["Hour"] + 1), axis=-1
                ),
                hovertemplate=(
                    "<b>"
                    + var
                    + ": %{z:.2f} "
                    + var_unit
                    + "</b><br>Month: %{customdata[0]}<br>Day: %{customdata[1]}<br>Hour:"
                    " %{customdata[2]}-%{customdata[3]}<br>"
                ),
                name="",
                colorscale=var_color,
                colorbar=dict(
                    title=dict(
                        text=f"{var_name}" + " in " + f"{var_unit} ", side="right"
                    ),
                    orientation="h",
                    thickness=20,
                    x=0,
                    xref="paper",
                    xanchor="left",
                    y=0.85,
                    yref="container",
                    yanchor="bottom",
                    ticks="inside",
                    ticklabelposition="outside top",
                ),
            ),
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text="Hour",
            mirror=True,
            range=[0, 24],
            zerolinewidth=1,
            dtick=2,
            row=1,
            col=1,
        )
        fig.update_xaxes(
            dtick="M1",
            tickformat="%b",
            ticklabelmode="period",
            title_text="Day",
            mirror=True,
            zeroline=False,
            row=1,
            col=1,
        )

        category_counts = df["categorical"].value_counts()
        category_percentages = df["categorical"].value_counts(normalize=True) * 100

        categories = cat_order
        percentages = [0] * len(cat_order)
        hour_counts = [0] * len(cat_order)

        for i, cat in enumerate(cat_order):
            if cat in category_counts.index:
                hour_counts[i] = category_counts[cat]
                percentages[i] = category_percentages[cat]

        fig.add_trace(
            go.Bar(
                x=percentages,
                y=categories,
                orientation="h",
                marker=dict(color=cat_color),
                text=[f"{percentage:.1f}%" for percentage in percentages],
                textposition="outside",
                cliponaxis=False,
                customdata=hour_counts,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Percentage: %{x:.1f}%<br>"
                    "Hours: %{customdata} h<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2)
        fig.update_yaxes(
            showgrid=False, side="left", tickfont=dict(weight="bold"), row=1, col=2
        )

        fig.update_layout(margin=dict(tight_margins, pad=10), width=1000, height=500)

    return fig
