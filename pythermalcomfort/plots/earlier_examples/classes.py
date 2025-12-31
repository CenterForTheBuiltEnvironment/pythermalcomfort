from dataclasses import dataclass, field
from typing import List, Optional, Union
import plotly.graph_objects as go
import numpy as np
import pandas as pd


@dataclass
class AdaptivePlot:
    title: str = "Adaptive Comfort Model"
    x_axis_label: str = "Outdoor Temperature (°C)"
    y_axis_label: str = "Indoor Temperature (°C)"
    show_summary: bool = True
    width: int = 800
    height: int = 800
    summary_width: float = 0.2  # between 0 and 1
    marker_color: str = "#ea536e"
    marker_size: int = 5
    marker_symbol: str = "circle"
    x_point_label: str = "t_out"
    y_point_label: str = "top"
    range: List[int] = field(default_factory=lambda: [10, 33.5])


@dataclass
class Scatter:
    ref_class: Union[AdaptivePlot]

    name: str = "Observations"
    type: str = "scatter"
    mode: str = "markers"
    marker_color: str = ""
    marker_size: int = 5
    marker_symbol: str = ""
    x_label: str = ""
    y_label: str = ""
    hovertemplate: str = (
        f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br><extra></extra>"
    )

    def __post_init__(self):
        """Sets values dynamically from the reference class."""
        self.marker_color = self.ref_class.marker_color
        self.marker_size = self.ref_class.marker_size
        self.marker_symbol = self.ref_class.marker_symbol
        self.x_label = self.ref_class.x_point_label
        self.y_label = self.ref_class.y_point_label

    def plot(self, df):
        return go.Scatter(
            x=df[self.x_label],
            y=df[self.y_label],
            mode=self.mode,
            name=self.name,
            marker=dict(
                color=self.marker_color,
                size=self.marker_size,
                symbol=self.marker_symbol,
            ),
            hovertemplate=self.hovertemplate,
        )


@dataclass
class AdaptiveSummary:
    name: str = "Summary"
    bar_width: int = 0.6
    showlegend: bool = False
    textposition: str = "outside"
    orientation: str = "h"
    cat_default: str = "out of range"
    cat_order: List[str] = field(
        default_factory=lambda: [
            "out of<br>applicable range",
            "acceptable<br>for 90%",
            "acceptable<br>for 80%",
            "discomfort",
        ]
    )
    cat_color: List[str] = field(
        default_factory=lambda: [
            "grey",
            "rgba(0, 128, 0, 0.2)",
            "rgba(0, 0, 255, 0.2)",
            "rgba(255, 0, 0, 0.2)",
        ]
    )
    hovertemplate: str = (
        "<b>%{y}</b><br>"
        "Percentage: %{x:.1f}%<br>"
        "Samples: %{customdata} <extra></extra>"
    )

    def plot(self, df):

        conditions = [
            pd.isna(df["adaptive_acceptability_80%"]),
            df["adaptive_acceptability_90%"] == True,
            df["adaptive_acceptability_80%"] == True,
            (df["adaptive_acceptability_80%"] == False)
            & (df["adaptive_acceptability_90%"] == False),
        ]

        df["categorical"] = np.select(
            conditions, self.cat_order, default=self.cat_default
        )

        category_counts = df["categorical"].value_counts()
        category_percentages = df["categorical"].value_counts(normalize=True) * 100

        percentages = [0] * len(self.cat_order)
        sample_counts = [0] * len(self.cat_order)

        for i, cat in enumerate(self.cat_order):
            if cat in category_counts.index:
                sample_counts[i] = category_counts[cat]
                percentages[i] = category_percentages[cat]

        return go.Bar(
            x=percentages,
            y=[category.title() for category in self.cat_order],
            orientation=self.orientation,
            width=self.bar_width,
            marker=dict(color=self.cat_color),
            text=[f"{percentage:.1f}%" for percentage in percentages],
            textposition=self.textposition,
            cliponaxis=False,
            customdata=sample_counts,
            hovertemplate=self.hovertemplate,
            showlegend=self.showlegend,
        )
