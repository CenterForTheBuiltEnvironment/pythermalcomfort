"""Plotting functionality for pythermalcomfort.

This module provides the Plot class and related components for creating
thermal comfort visualizations.

Primary API
-----------
Plot : class
    Main facade for creating plots. Use Plot.ranges() to create range plots.
Style : class
    Mutable styling configuration. The only component that can be
    modified after plot creation.

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
Simple usage with automatic preset detection:

>>> from pythermalcomfort.models import pmv_ppd_iso
>>> from pythermalcomfort.plots import Plot
>>> plot = Plot.ranges(
...     pmv_ppd_iso,
...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
... )
>>> plot.style.title = "PMV Comfort Regions"
>>> fig, ax = plot.render()

Custom styling:

>>> from pythermalcomfort.plots import Plot, Style
>>> style = Style(cmap="viridis", band_alpha=0.6)
>>> plot = Plot.ranges(model_func, fixed_params={...}, style=style)
"""

from pythermalcomfort.plots.plot import Plot
from pythermalcomfort.plots.presets import (
    HEAT_INDEX_PRESET,
    PMV_EXTENDED_PRESET,
    PMV_PRESET,
    SET_PRESET,
    UTCI_PRESET,
    UTCI_SIMPLE_PRESET,
    Preset,
    get_preset,
)
from pythermalcomfort.plots.ranges import Ranges
from pythermalcomfort.plots.regions import Regions
from pythermalcomfort.plots.style import Style

__all__ = [
    # Primary API
    "Plot",
    "Style",
    # Data and context (for advanced use)
    "Ranges",
    "Regions",
    # Presets
    "Preset",
    "get_preset",
    "PMV_PRESET",
    "PMV_EXTENDED_PRESET",
    "UTCI_PRESET",
    "UTCI_SIMPLE_PRESET",
    "HEAT_INDEX_PRESET",
    "SET_PRESET",
]
