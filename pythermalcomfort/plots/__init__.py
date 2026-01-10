"""Plotting functionality for pythermalcomfort.

This module provides the Plot class and related components for creating
thermal comfort visualizations.

Primary API
-----------
Plot : class
    Main facade for creating plots. Use Plot.range() for threshold-based
    region plots, or Plot.adaptive() for adaptive comfort model plots.
Style : class
    Mutable styling configuration. The only component that can be
    modified after plot creation.

Scenes
------
BaseScene : class
    Abstract base class for scene types.
RangeScene : class
    Generic threshold-based region visualization.
AdaptiveScene : class
    ASHRAE 55 Adaptive Comfort Model visualization.

Data
----
DataSeries : class
    Overlay data points for scenes.

Presets
-------
PMV_PRESET : Preset
    Predefined configuration for PMV (ISO 7730) plots.
UTCI_PRESET : Preset
    Predefined configuration for UTCI thermal stress plots.
HEAT_INDEX_PRESET : Preset
    Predefined configuration for Heat Index plots.
SET_PRESET : Preset
    Predefined configuration for SET plots.

Examples
--------
Simple range plot with automatic preset detection:

>>> from pythermalcomfort.models import utci
>>> from pythermalcomfort.plots import Plot
>>> plot = Plot.range(utci, fixed_params={"v": 1.0, "tr": 25})
>>> plot.style.title = "UTCI Thermal Stress"
>>> fig, ax = plot.render()

Adaptive comfort with data overlay and summary:

>>> from pythermalcomfort.plots import Plot
>>> plot = Plot.adaptive()
>>> plot = plot.add_data(x=t_outdoor, y=t_operative)
>>> plot.style.show_summary = True
>>> fig, axes = plot.render()

Custom styling:

>>> from pythermalcomfort.plots import Plot, Style
>>> style = Style(cmap="viridis", band_alpha=0.6)
>>> plot = Plot.range(model_func, fixed_params={...}, style=style)
"""

from pythermalcomfort.plots.data_series import DataSeries
from pythermalcomfort.plots.plot import Plot
from pythermalcomfort.plots.presets import (
    HEAT_INDEX_PRESET,
    PMV_EXTENDED_PRESET,
    PMV_PRESET,
    ADAPTIVE_PRESET,
    PSYCHROMETRIC_PRESET,
    PSYCHROMETRIC_EXTENDED_PRESET,
    Preset,
    get_preset,
)
from pythermalcomfort.plots.scenes import AdaptiveScene, BaseScene, RangeScene, PsychrometricScene
from pythermalcomfort.plots.style import Style
from pythermalcomfort.plots.summary import SummaryRenderer

__all__ = [
    # Primary API
    "Plot",
    "Style",
    # Scenes
    "BaseScene",
    "RangeScene",
    "AdaptiveScene",
    "PsychrometricScene",
    # Data
    "DataSeries",
    "SummaryRenderer",
    # Presets
    "Preset",
    "get_preset",
    "PMV_PRESET",
    "PMV_EXTENDED_PRESET",
    "HEAT_INDEX_PRESET",
    "ADAPTIVE_PRESET",
    "PSYCHROMETRIC_PRESET",
    "PSYCHROMETRIC_EXTENDED_PRESET",
]
