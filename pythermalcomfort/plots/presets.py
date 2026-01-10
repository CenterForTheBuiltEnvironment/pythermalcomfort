"""Predefined presets for common thermal comfort models.

This module provides Preset configurations that reduce boilerplate
when creating plots for common models like PMV, UTCI, Heat Index, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class Preset:
    """Predefined configuration for plotting a specific model.

    Contains default thresholds, labels, and axis settings that work
    well for a particular model or index.

    Parameters
    ----------
    name : str
        Human-readable name for the preset.
    thresholds : list[float]
        Default threshold values for region boundaries.
    labels : list[str]
        Labels for each region (len = len(thresholds) + 1).
    metric_attr : str or None
        Attribute name to extract from model result.
    xlabel : str
        Default x-axis label.
    ylabel : str
        Default y-axis label.
    x_range : tuple[float, float] or None
        Default (min, max) for x-axis.
    y_range : tuple[float, float] or None
        Default (min, max) for y-axis.
    xy_mapper_name : str
        Name of the mapper function to use (e.g., "mapper_tdb_rh").
    """

    name: str
    thresholds: list[float]
    labels: list[str]
    metric_attr: str | None
    xlabel: str
    ylabel: str
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    xy_mapper_name: str = "mapper_tdb_rh"
    cmap: str = "coolwarm"  # Default colormap
    band_colors: tuple | None = None # explicit colors override cmap

    def get_xy_mapper(self) -> Callable[[float, float, dict[str, Any]], dict[str, Any]]:
        """Get the xy_to_kwargs mapper function for this preset."""
        from pythermalcomfort.plots import utils

        return getattr(utils, self.xy_mapper_name)


# -----------------------------------------------------------------------------
# Adaptive Preset
# -----------------------------------------------------------------------------

ADAPTIVE_PRESET = Preset(
    name="Adaptive Comfort (ASHRAE 55)",
    thresholds=[2.5, 3.5],
    labels=["90% Acceptability", "80% Acceptability", "Discomfort"],
    metric_attr=None,
    xlabel="Outdoor Temperature [°C]",
    ylabel="Operative Temperature [°C]",
    x_range=(10.0, 33.5),
    y_range=(14.0, 36.0),
    band_colors=(
        (1.0, 0.0, 0.0, 0.3),      # Discomfort (red)
        (0.0, 0.0, 1.0, 0.3),      # 80% (blue)
        (0.0, 0.5, 0.0, 0.3),      # 90% (green)
    ),
)


# -----------------------------------------------------------------------------
# PMV Presets
# -----------------------------------------------------------------------------

PMV_PRESET = Preset(
    name="PMV (ISO 7730)",
    thresholds=[-0.5, 0.5],
    labels=["Cool", "Neutral", "Warm"],
    metric_attr="pmv",
    xlabel="Air temperature [°C]",
    ylabel="Relative humidity [%]",
    x_range=(10.0, 36.0),
    y_range=(0.0, 100.0),
    cmap="coolwarm",
)

PMV_EXTENDED_PRESET = Preset(
    name="PMV Extended Categories",
    thresholds=[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
    labels=[
        "Cold",
        "Cool",
        "Slightly cool",
        "Neutral",
        "Slightly warm",
        "Warm",
        "Hot",
    ],
    metric_attr="pmv",
    xlabel="Air temperature [°C]",
    ylabel="Relative humidity [%]",
    x_range=(10.0, 40.0),
    y_range=(0.0, 100.0),
    cmap="coolwarm",
)


# -----------------------------------------------------------------------------
# Psychrometric Presets
# -----------------------------------------------------------------------------

PSYCHROMETRIC_PRESET = Preset(
    name="Psychrometric (PMV Simple)",
    thresholds=[-0.5, 0.5],
    labels=["Cooler than Comfortable", "Comfortable", "Warmer than Comfortable"],
    metric_attr="pmv",
    xlabel="Dry Bulb Temperature [°C]",
    ylabel="Humidity Ratio [g/kg]",
    x_range=(10.0, 36.0),
    y_range=(0.0, 30.0),
    band_colors=(
        (0.0, 0.0, 1.0, 0.5),      # Cooler (blue)
        (0.0, 0.5, 0.0, 0.5),      # Comfortable (green)
        (1.0, 0.0, 0.0, 0.5),      # Warmer (red)
    ),
)

PSYCHROMETRIC_EXTENDED_PRESET = Preset(
    name="Psychrometric (PMV Extended)",
    thresholds=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
    labels=["Cold", "Cool", "Slightly Cool", "Neutral", "Slightly Warm", "Warm", "Hot"],
    metric_attr="pmv",
    xlabel="Dry Bulb Temperature [°C]",
    ylabel="Humidity Ratio [g/kg]",
    x_range=(10.0, 36.0),
    y_range=(0.0, 30.0),
    cmap="coolwarm",  # Will use colormap if band_colors not set
)

# -----------------------------------------------------------------------------
# Heat Index Preset
# -----------------------------------------------------------------------------

HEAT_INDEX_PRESET = Preset(
    name="Heat Index (Rothfusz)",
    thresholds=[27.0, 32.0, 41.0, 54.0],
    labels=[
        "No risk",
        "Caution",
        "Extreme caution",
        "Danger",
        "Extreme danger",
    ],
    metric_attr="hi",
    xlabel="Air temperature [°C]",
    ylabel="Relative humidity [%]",
    x_range=(20.0, 50.0),
    y_range=(40.0, 100.0),
    cmap="YlOrRd",
)

# -----------------------------------------------------------------------------
# Registry for lookup by model function name
# -----------------------------------------------------------------------------

_PRESET_REGISTRY: dict[str, Preset] = {
    "pmv_ppd_iso": PMV_PRESET,
    "pmv_ppd_ashrae": PMV_PRESET,
    "heat_index_rothfusz": HEAT_INDEX_PRESET,
}


def get_preset(model_func) -> Preset | None:
    """Get the predefined preset for a model function.

    Parameters
    ----------
    model_func : Callable
        A pythermalcomfort model function.

    Returns
    -------
    Preset or None
        The preset for this model, or None if no preset exists.

    Examples
    --------
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> preset = get_preset(pmv_ppd_iso)
    >>> preset.name
    'PMV (ISO 7730)'
    """
    name = getattr(model_func, "__name__", "")
    return _PRESET_REGISTRY.get(name)
