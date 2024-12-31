from math import ceil, floor
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import v_relative, clo_dynamic, psy_ta_rh
from pythermalcomfort.theme import (
    tight_margins,
    index_mapping_dictionary,
)


def adaptive_chart(df):

    OUTDOOR_TEMP_RANGE = (10, 33.5)
    VELOCITY_RANGE = (0, 2)
    TEMPERATURE_RANGE = (10, 40)

    applicable_df = df.loc[
        (df["ta"].between(*TEMPERATURE_RANGE))
        & (df["tr"].between(*TEMPERATURE_RANGE))
        & (df["vel"].between(*VELOCITY_RANGE))
        & (df["t_out"].between(*OUTDOOR_TEMP_RANGE))
    ].dropna()

    len_total = len(df)
    len_applicable = len(applicable_df)

    sum_80 = applicable_df["adaptive_acceptability_80%"].sum()
    sum_90 = applicable_df["adaptive_acceptability_90%"].sum()

    T_outdoor = np.linspace(OUTDOOR_TEMP_RANGE[0], OUTDOOR_TEMP_RANGE[1], 100)
    T_comfort = 0.31 * T_outdoor + 17.8
    T_80_upper, T_80_lower = T_comfort + 2.5, T_comfort - 2.5
    T_90_upper, T_90_lower = T_comfort + 2.0, T_comfort - 2.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=T_outdoor,
            y=T_comfort,
            mode="lines",
            name="Comfort Temperature",
            line=dict(color="black", width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([T_outdoor, T_outdoor[::-1]]),
            y=np.concatenate([T_90_upper, T_90_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Acceptability",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([T_outdoor, T_outdoor[::-1]]),
            y=np.concatenate([T_80_upper, T_80_lower[::-1]]),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Acceptability",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=applicable_df["t_out"],
            y=applicable_df["top"],
            mode="markers",
            name="Observations",
            marker=dict(color="#ea536e", size=5, symbol="circle"),
            hovertemplate="t_out: %{x:.1f}<br>t_op: %{y:.1f}<br><extra></extra>",
        )
    )

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
                y=0.2,
                xref="paper",
                yref="paper",
                xanchor="left",
                text=f"Within bounds: {len_applicable/len_total*100:.1f}%",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=1.025,
                y=0.15,
                xref="paper",
                yref="paper",
                xanchor="left",
                text=f"Within 80% range: {sum_80/len_total*100:.1f}%",
                showarrow=False,
                font=dict(size=12),
            ),
            dict(
                x=1.025,
                y=0.1,
                xref="paper",
                yref="paper",
                xanchor="left",
                text=f"Within 90% range: {sum_90/len_total*100:.1f}%",
                showarrow=False,
                font=dict(size=12),
            ),
        ],
    )

    return fig


def adaptive_chart_with_summary(df):

    OUTDOOR_TEMP_RANGE = (10, 33.5)
    VELOCITY_RANGE = (0, 2)
    TEMPERATURE_RANGE = (10, 40)

    applicable_df = df.loc[
        (df["ta"].between(*TEMPERATURE_RANGE))
        & (df["tr"].between(*TEMPERATURE_RANGE))
        & (df["vel"].between(*VELOCITY_RANGE))
        & (df["t_out"].between(*OUTDOOR_TEMP_RANGE))
    ].dropna()

    len_total = len(df)
    len_applicable = len(applicable_df)

    sum_80 = applicable_df["adaptive_acceptability_80%"].sum()
    sum_90 = applicable_df["adaptive_acceptability_90%"].sum()

    T_outdoor = np.linspace(OUTDOOR_TEMP_RANGE[0], OUTDOOR_TEMP_RANGE[1], 100)
    T_comfort = 0.31 * T_outdoor + 17.8
    T_80_upper, T_80_lower = T_comfort + 2.5, T_comfort - 2.5
    T_90_upper, T_90_lower = T_comfort + 2.0, T_comfort - 2.0

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.85, 0.15],
    )

    # --- First Subplot: Adaptive Comfort Model ---

    fig.add_trace(
        go.Scatter(
            x=T_outdoor,
            y=T_comfort,
            mode="lines",
            name="Comfort Temperature",
            line=dict(color="black", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([T_outdoor, T_outdoor[::-1]]),
            y=np.concatenate([T_90_upper, T_90_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Acceptability",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([T_outdoor, T_outdoor[::-1]]),
            y=np.concatenate([T_80_upper, T_80_lower[::-1]]),
            fill="tonexty",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% Acceptability",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=applicable_df["t_out"],
            y=applicable_df["top"],
            mode="markers",
            name="Observations",
            marker=dict(color="#ea536e", size=5, symbol="circle"),
            hovertemplate="t_out: %{x:.1f}<br>t_op: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Second Subplot: Summary ---

    categories = ["Applicability", "80% Acceptability", "90% Acceptability"]
    percentages = [
        len_applicable / len_total * 100,
        sum_80 / len_total * 100,
        sum_90 / len_total * 100,
    ]

    colors = ["#ea536e", "rgba(0, 0, 255, 0.2)", "rgba(0, 128, 0, 0.2)"]

    for i, (category, percentage, color) in enumerate(
        zip(categories, percentages, colors)
    ):
        y_pos = len(categories) - i

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y_pos],
                mode="markers",
                marker=dict(
                    size=75,
                    color="rgba(200,200,200,0.2)",
                    line=dict(width=0.25, color="black"),
                ),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[y_pos],
                mode="markers",
                marker=dict(
                    size=percentage * 0.75,
                    color=color,
                    line=dict(width=1, color="black"),
                ),
                name=f"{category}: {percentage:.1f}%",
                text=[f"{int(percentage)}% of {len_total} samples"],
                hoverinfo="text",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_annotation(
            x=0,
            y=y_pos + 0.375,
            xref="x2",
            yref="y2",
            text=category,
            showarrow=False,
            font=dict(size=12),
            align="center",
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
        height=600,
        width=800,
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=2, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=2)

    return fig


def pmv_chart(df, pmv_constants):

    T_dry = np.linspace(10, 36, 1000)
    RH_levels = np.linspace(0, 100, 10)
    MRT, v, met, clo = (
        pmv_constants["tr"],
        pmv_constants["v"],
        pmv_constants["met"],
        pmv_constants["clo"],
    )

    vr = v_relative(v=v, met=met)
    clo_d = clo_dynamic(clo=clo, met=met)

    observations = {
        "T_dry": df["ta"].to_list(),
        "RH": df["rh"].to_list(),
    }

    def pmv_eq(Tair, RH):
        return pmv(
            tdb=Tair, tr=MRT, vr=vr, rh=RH, met=met, clo=clo_d, limit_inputs=False
        ).pmv

    def convert_rh_to_hr(T, RH):
        return psy_ta_rh(T, RH / 100, 101325)["hr"] * 100000

    np.seterr(divide="ignore", invalid="ignore")

    humidity_ratios = np.array(
        [[convert_rh_to_hr(T, RH) for T in T_dry] for RH in RH_levels]
    )

    def find_comfort_temp(RH, target_pmv):
        def pmv_root(T):
            return pmv_eq(T, RH) - target_pmv

        try:
            return brentq(pmv_root, 0, 120)
        except ValueError:
            return None

    def extract_comfort_data(zone):
        RH, Tair = zip(*zone)
        return Tair, [convert_rh_to_hr(T, rh) for T, rh in zip(Tair, RH)]

    target_pmvs = {"Neutral": 0, "Upper": 0.5, "Lower": -0.5}
    comfort_zones = {
        name: [(RH, find_comfort_temp(RH, target)) for RH in RH_levels]
        for name, target in target_pmvs.items()
    }

    neutral_line = extract_comfort_data(comfort_zones["Neutral"])
    upper_line = extract_comfort_data(comfort_zones["Upper"])
    lower_line = extract_comfort_data(comfort_zones["Lower"])

    # Plotly Chart

    fig = go.Figure()

    for i, RH in enumerate(RH_levels):
        fig.add_trace(
            go.Scatter(
                x=T_dry,
                y=humidity_ratios[i],
                mode="lines",
                name=f"{RH}% RH",
                line=dict(width=1, color="lightgrey"),
                showlegend=False,
                hovertemplate="T_dry: %{x:.1f}°C<br>HR: %{y:.2f}<br><extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([upper_line[0], lower_line[0][::-1]]),
            y=np.concatenate([upper_line[1], lower_line[1][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 128, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Comfort Zone",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=neutral_line[0],
            y=neutral_line[1],
            mode="lines",
            name="Neutral PMV",
            line=dict(color="black", width=1.5, dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=observations["T_dry"],
            y=[
                convert_rh_to_hr(T, RH)
                for T, RH in zip(observations["T_dry"], observations["RH"])
            ],
            mode="markers",
            name="Observations",
            marker=dict(color="#ea536e", size=5, symbol="circle"),
            hovertemplate="T_dry: %{x:.1f}°C<br>HR: %{y:.2f}<br><extra></extra>",
        )
    )

    fig.update_layout(
        title="Psychrometric Chart with PMV Comfort Zone, ASHRAE 55",
        title_x=0,
        title_xref="paper",
        title_y=0.95,
        title_yref="container",
        xaxis_title="Dry Bulb Temperature [°C]",
        yaxis_title="Humidity Ratio [g water / kg dry air]",
        yaxis=dict(side="right", showgrid=False),
        xaxis=dict(showgrid=False, showline=False),
        legend=dict(font=dict(size=12), x=0, xanchor="left"),
        height=600,
        width=800,
        margin=tight_margins,
        annotations=[
            dict(
                x=0,
                y=0.99,
                xref="paper",
                yref="paper",
                text=f"MRT: {MRT}°C | v: {v} m/s | {met} met | {clo} clo",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )

    fig.update_legends(dict(traceorder="normal"), x=0.0, y=0.85)

    fig.update_xaxes(linewidth=1, linecolor="lightgrey", mirror=False, range=[10, 36])
    fig.update_yaxes(linewidth=1, linecolor="lightgrey", mirror=False, range=[0, 35])

    return fig


def heatmap(df, var, global_local, si_ip):

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

    fig = go.Figure(
        go.Heatmap(
            y=df["Hour"] + 0.5,  # ? offset hour for better display
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
                title=dict(text=f"{var_name}" + " in " + f"{var_unit} ", side="right"),
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

    return fig


def heatmap_with_summary(df, var, global_local, si_ip):

    var_name = index_mapping_dictionary[var]["name"]
    var_unit = index_mapping_dictionary[var][si_ip]["unit"]
    var_range = index_mapping_dictionary[var][si_ip]["range"]
    var_color = index_mapping_dictionary[var]["colors"]
    cat_color = index_mapping_dictionary[var]["colors_categories"]
    cat_order = index_mapping_dictionary[var]["order_categories"]

    if global_local == "global":
        range_z = var_range
    else:
        data_max = 5 * ceil(df[var].max() / 5)
        data_min = 5 * floor(df[var].min() / 5)
        range_z = [data_min, data_max]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.8, 0.2],
    )

    fig.add_trace(
        go.Heatmap(
            y=df["Hour"] + 0.5,  # ? offset hour for better display
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
                title=dict(text=f"{var_name}" + " in " + f"{var_unit} ", side="right"),
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

    categories = [cat for cat in cat_order if cat in category_counts.index]
    percentages = [category_percentages[cat] for cat in categories]
    hour_counts = [category_counts[cat] for cat in categories]

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
        showgrid=False, row=1, col=2, side="left", tickfont=dict(weight="bold")
    )

    fig.update_layout(margin=dict(tight_margins, pad=10), width=1000, height=500)

    return fig
