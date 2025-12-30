"""Ranges data representation for threshold-based plots.

This module provides the Ranges class, a frozen dataclass that holds
computed threshold curves from model functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt


@dataclass(frozen=True)
class Ranges:
    """Computed threshold curves for region-based plotting.

    This is a frozen (immutable) data representation holding the results
    of threshold curve computation. Use the `from_model` factory method
    to compute ranges from a pythermalcomfort model function.

    Parameters
    ----------
    curves : list[np.ndarray]
        List of x(y) curves, one per threshold. Each array has the same
        shape as y_values. NaN indicates no solution at that y value.
    y_values : np.ndarray
        Y-axis values at which curves were computed.
    thresholds : list[float]
        Threshold values that define region boundaries.
    x_bounds : tuple[float, float]
        (min, max) bounds for the x-axis.

    Notes
    -----
    This class is frozen (immutable) after creation. To modify ranges,
    create a new Ranges instance.

    Examples
    --------
    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> from pythermalcomfort.plots.utils import mapper_tdb_rh
    >>> ranges = Ranges.from_model(
    ...     model_func=pmv_ppd_iso,
    ...     xy_to_kwargs=mapper_tdb_rh,
    ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
    ...     thresholds=[-0.5, 0.5],
    ...     x_bounds=(18, 30),
    ...     y_values=np.arange(20, 80, 2),
    ... )
    """

    curves: list[npt.NDArray[np.floating[Any]]]
    y_values: npt.NDArray[np.floating[Any]]
    thresholds: list[float]
    x_bounds: tuple[float, float]

    @classmethod
    def from_model(
        cls,
        model_func: Callable[..., Any],
        xy_to_kwargs: Callable[[float, float, dict[str, Any]], dict[str, Any]],
        fixed_params: dict[str, Any],
        thresholds: Sequence[float],
        x_bounds: tuple[float, float],
        y_values: Sequence[float] | npt.NDArray[np.floating[Any]],
        *,
        metric_attr: str | None = None,
        x_scan_step: float = 1.0,
        smooth_sigma: float = 0.8,
    ) -> Ranges:
        """Compute ranges from a model function.

        This factory method uses the existing solve_threshold_curves
        infrastructure to compute where metric(x, y) == threshold.

        Parameters
        ----------
        model_func : Callable
            A pythermalcomfort model function (e.g., pmv_ppd_iso, utci).
        xy_to_kwargs : Callable
            Function mapping (x, y, fixed_params) to model kwargs.
            See pythermalcomfort.plots.utils for predefined mappers.
        fixed_params : dict
            Model parameters held constant for all evaluations.
        thresholds : Sequence[float]
            Threshold values to solve for.
        x_bounds : tuple[float, float]
            (min, max) bounds for the x-axis search.
        y_values : Sequence[float] or np.ndarray
            Y-axis values at which to compute curves.
        metric_attr : str or None, optional
            Attribute name to extract from model result (e.g., "pmv").
            If None, attempts to infer or coerce to float.
        x_scan_step : float, default 1.0
            Step size for scanning x-axis when finding thresholds.
        smooth_sigma : float, default 0.8
            Gaussian smoothing sigma for curves. Set 0 to disable.

        Returns
        -------
        Ranges
            Computed threshold curves ready for rendering.

        Examples
        --------
        >>> from pythermalcomfort.models import pmv_ppd_iso
        >>> from pythermalcomfort.plots.utils import mapper_tdb_rh
        >>> ranges = Ranges.from_model(
        ...     model_func=pmv_ppd_iso,
        ...     xy_to_kwargs=mapper_tdb_rh,
        ...     fixed_params={"tr": 25, "met": 1.2, "clo": 0.5, "vr": 0.1, "wme": 0},
        ...     thresholds=[-0.5, 0.5],
        ...     x_bounds=(18, 30),
        ...     y_values=np.arange(20, 80, 2),
        ...     metric_attr="pmv",
        ... )
        """
        from pythermalcomfort.plots.utils import make_metric_eval, solve_threshold_curves

        # Build metric evaluator
        metric_xy = make_metric_eval(
            model_func=model_func,
            xy_to_kwargs=xy_to_kwargs,
            fixed_params=fixed_params,
            metric_attr=metric_attr,
        )

        # Solve for threshold curves
        result = solve_threshold_curves(
            metric_xy=metric_xy,
            thresholds=list(thresholds),
            y_values=y_values,
            x_bounds=x_bounds,
            x_scan_step=x_scan_step,
            smooth_sigma=smooth_sigma,
        )

        return cls(
            curves=result["curves"],
            y_values=result["y_values"],
            thresholds=result["thresholds"],
            x_bounds=x_bounds,
        )

    @property
    def n_regions(self) -> int:
        """Number of regions (len(thresholds) + 1)."""
        return len(self.thresholds) + 1
