from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import plotly.graph_objects as go
import numpy as np
import pandas as pd


@dataclass
class Base:
    title: str
    x_axis_label: str
    y_axis_label: str
    show_summary: bool
    width: int
    height: int
    summary_width: float  # between 0 and 1


@dataclass
class AdaptiveBase(Base):
    title: str = "Adaptive Comfort Model"
    x_axis_label: str = "Outdoor Temperature (°C)"
    y_axis_label: str = "Indoor Temperature (°C)"
    show_summary: bool = True
    width: int = 800
    height: int = 800
    summary_width: float = 0.2


@dataclass
class Trace(ABC):
    name: str  # Name of the element (e.g., "Observations" or "Comfort Range")
    type: str
    mode: Optional[str] = None
    marker_color: Union[str, List[str]] = None
    marker_size: Optional[int] = None
    marker_symbol: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    hovertemplate: Optional[str] = None
    bar_width: Optional[float] = None
    cat_order: Optional[List[str]] = None
    cat_default: Optional[str] = None  # default category for missing values
    showlegend: Optional[bool] = None
    textposition: Optional[str] = None
    orientation: Optional[str] = None
    fill: Optional[str] = None
    fillcolor: Optional[str] = None
    line_color: Optional[str] = None
    line_width: Optional[int] = None
    line_dash: Optional[str] = None

    @abstractmethod
    def generate_trace(self):
        """Abstract method to generate the Plotly trace."""
        pass


@dataclass
class Observations(Trace):  # use same styling for all plots
    name: str = "Observations"
    type: str = "scatter"
    mode: str = "markers"
    marker_color: str = "#ea536e"
    marker_size: int = 5
    marker_symbol: str = "circle"
    x_label: str = "t_out"
    y_label: str = "top"
    hovertemplate: str = (
        f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br><extra></extra>"
    )

    def generate_trace(self, df):
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
class AdaptiveSummary(Trace):
    name: str = "Summary"
    type: str = "bar"
    cat_order: List[str] = field(
        default_factory=lambda: [
            "out of range",
            "acceptable<br>for 90%",
            "acceptable<br>for 80%",
            "discomfort",
        ]
    )
    cat_default: str = "out of range"
    marker_color: List[str] = field(
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
    bar_width: int = 0.6
    showlegend: bool = False
    textposition: str = "outside"
    orientation: str = "h"

    def generate_trace(self, df):

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
            marker=dict(color=self.marker_color),
            text=[f"{percentage:.1f}%" for percentage in percentages],
            textposition=self.textposition,
            cliponaxis=False,
            customdata=sample_counts,
            hovertemplate=self.hovertemplate,
            showlegend=self.showlegend,
        )
