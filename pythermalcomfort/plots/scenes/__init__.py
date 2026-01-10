"""Scene classes for thermal comfort plots.

Scenes represent the context/background of a plot, typically showing
threshold-based regions computed from thermal comfort models.
"""

from pythermalcomfort.plots.scenes.adaptive_scene import AdaptiveScene
from pythermalcomfort.plots.scenes.base import BaseScene
from pythermalcomfort.plots.scenes.range_scene import RangeScene
from pythermalcomfort.plots.scenes.psy_scene import PsychrometricScene

__all__ = [
    "BaseScene",
    "RangeScene",
    "AdaptiveScene",
    "PsychrometricScene",
]
