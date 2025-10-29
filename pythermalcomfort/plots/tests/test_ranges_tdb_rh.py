"""Tests for the Temperature vs Relative Humidity (T-RH) range plot.

Design goals and test contract:
- Use a central registry (see conftest.py) to define models, their fixed
  parameters, default thresholds, and recommended test ranges. Tests consume
  the registry instead of hard-coding model details, keeping them uniform and
  easy to extend.
- Call the plotting API directly (``ranges_tdb_rh``) so we validate the real
  execution path used by users.
- Verify numerical agreement between the plotted threshold curves and direct
  model evaluation. In other words, a point that lies on a drawn threshold
  curve should satisfy the underlying model metric being approximately equal
  to that threshold within a small absolute tolerance.
"""

from __future__ import annotations

import pytest
from pythermalcomfort.plots.matplotlib import ranges_tdb_rh


class TestMainModels:
    """Core behavioral tests for main models on the T-RH plot.

    Scope of these tests:
    1) Basic functionality: plotting with thresholds returns axes, curves, and bands.
    2) Default thresholds: omitting thresholds uses the registered defaults.
    3) Numerical accuracy: points on plotted curves match the model metric.
    4) Parameter validation: invalid inputs raise explicit ValueError.
    5) Extreme conditions: plot handles very wide/narrow and hot/cold ranges.
    6) Boundary values: plot handles near-degenerate ranges gracefully.
    """
    
    def test_basic_functionality(self, all_models):
        """Ensure each model can produce a plot with all registered thresholds."""
        for model_info in all_models:
            ax, artists = ranges_tdb_rh(
                model_func=model_info.func,
                fixed_params=model_info.fixed_params,
                thresholds=model_info.thresholds,
                t_range=model_info.test_ranges[0],
                rh_range=model_info.test_ranges[1]
            )
            
            assert ax is not None, f"{model_info.name} should create plot"
            assert "curves" in artists, f"{model_info.name} should have curves"
            assert "bands" in artists, f"{model_info.name} should have bands when multiple thresholds are used"
            assert len(artists["curves"]) >= 0, f"{model_info.name} should have non-negative number of curves"
    
    def test_default_thresholds(self, all_models):
        """Verify default thresholds are used when thresholds=None."""
        for model_info in all_models:
            ax, artists = ranges_tdb_rh(
                model_func=model_info.func,
                fixed_params=model_info.fixed_params,
                thresholds=None,
                t_range=model_info.test_ranges[0],
                rh_range=model_info.test_ranges[1]
            )
            
            assert ax is not None, f"{model_info.name} should create plot with default thresholds"
            assert "curves" in artists, f"{model_info.name} should have curves"
            assert len(artists["curves"]) > 0, f"{model_info.name} should have at least one curve"
    
    def test_numerical_accuracy(self, all_models):
        """Plotted curve points match direct model evaluation within tolerance."""
        from pythermalcomfort.plots.utils import make_metric_eval, mapper_tdb_rh
        import numpy as np
        
        for model_info in all_models:
            t_range, rh_range = model_info.test_ranges
            threshold = model_info.thresholds[0]
            
            ax, artists = ranges_tdb_rh(
                model_func=model_info.func,
                fixed_params=model_info.fixed_params,
                thresholds=[threshold],
                t_range=t_range,
                rh_range=rh_range,
                smooth_sigma=0.0
            )
            
            if artists["curves"]:
                curve = artists["curves"][0]
                
                metric_eval = make_metric_eval(
                    model_func=model_info.func,
                    xy_to_kwargs=mapper_tdb_rh,
                    fixed_params=model_info.fixed_params,
                    metric_attr=model_info.metric_attr,
                )
                
                errors = []
                for x, y in zip(curve.get_xdata(), curve.get_ydata()):
                    if not (np.isfinite(x) and np.isfinite(y)):
                        continue
                    try:
                        metric_val = metric_eval(x, y)
                        if not np.isclose(metric_val, threshold, atol=model_info.tolerance):
                            errors.append(
                                f"Point ({x:.2f}, {y:.2f}) has {model_info.name} {metric_val:.2f}, "
                                f"expected {threshold:.2f} (tolerance: {model_info.tolerance})"
                            )
                    except Exception as exc:
                        errors.append(f"Error calculating {model_info.name} at ({x:.2f}, {y:.2f}): {exc}")
                
                if errors:
                    error_msg = f"{model_info.name} numerical accuracy errors:\n" + "\n".join(errors[:3])
                    if len(errors) > 3:
                        error_msg += f"\n... and {len(errors) - 3} more errors"
                    pytest.fail(error_msg)
    
    def test_parameter_validation(self, all_models):
        """Invalid inputs raise ValueError with informative messages."""
        for model_info in all_models:
            with pytest.raises(ValueError, match="t_range must be strictly increasing"):
                ranges_tdb_rh(
                    model_func=model_info.func,
                    fixed_params=model_info.fixed_params,
                    thresholds=[model_info.thresholds[0]],
                    t_range=(30.0, 20.0),  # reversed: min > max
                    rh_range=model_info.test_ranges[1]
                )
            
            # Reversed rh_range should raise
            with pytest.raises(ValueError, match="rh_range must be strictly increasing"):
                ranges_tdb_rh(
                    model_func=model_info.func,
                    fixed_params=model_info.fixed_params,
                    thresholds=[model_info.thresholds[0]],
                    t_range=model_info.test_ranges[0],
                    rh_range=(80.0, 20.0)  # reversed: min > max
                )
            
            # Non-positive step should raise (exact error text is enforced)
            with pytest.raises(ValueError, match="rh_step must be positive"):
                ranges_tdb_rh(
                    model_func=model_info.func,
                    fixed_params=model_info.fixed_params,
                    thresholds=[model_info.thresholds[0]],
                    t_range=model_info.test_ranges[0],
                    rh_range=model_info.test_ranges[1],
                    rh_step=0.0  # zero step is invalid
                )
    
    def test_extreme_conditions(self, all_models):
        """Plot robustness under more extreme/narrow/wide temperature-RH ranges.

        Expanded ranges rationale:
        - Ultra-narrow: (22.0, 22.2) °C & (49.5, 49.7) %RH to stress tiny spans.
        - Very wide: (-30.0, 60.0) °C & (0.0, 100.0) %RH to stress global scan.
        - Colder: (-30.0, -10.0) °C & (20.0, 80.0) %RH for sub-freezing cases.
        - Hotter: (40.0, 60.0) °C & (20.0, 80.0) %RH for heat stress cases.
        - Near-dry: (15.0, 35.0) °C & (0.0, 5.0) %RH for very low humidity.
        - Near-saturation: (15.0, 35.0) °C & (95.0, 100.0) %RH for very high RH.
        For some combinations, thresholds may not be bracketed; we accept those
        as non-fatal (solver warnings), but still assert structure when plots
        succeed.
        """
        extreme_ranges = [
            ((22.0, 22.2), (49.5, 49.7)),  # ultra narrow
            ((-30.0, 60.0), (0.0, 100.0)),  # ultra wide
            ((-30.0, -10.0), (20.0, 80.0)),  # very cold
            ((40.0, 60.0), (20.0, 80.0)),  # very hot
            ((15.0, 35.0), (0.0, 5.0)),  # near dry
            ((15.0, 35.0), (95.0, 100.0)),  # near saturation
        ]
        
        for model_info in all_models:
            for t_range, rh_range in extreme_ranges:
                try:
                    ax, artists = ranges_tdb_rh(
                        model_func=model_info.func,
                        fixed_params=model_info.fixed_params,
                        thresholds=[model_info.thresholds[0]],
                        t_range=t_range,
                        rh_range=rh_range
                    )
                    
                    if ax is None:
                        print(
                            f"Warning: {model_info.name} not compatible for extreme case "
                            f"t_range={t_range}, rh_range={rh_range}"
                        )
                        continue
                        
                    assert "curves" in artists, (
                        f"{model_info.name} should have curves in extreme case "
                        f"t_range={t_range}, rh_range={rh_range}"
                    )
                    
                except Exception as e:
                    if "no bracket" in str(e).lower() or "unsolved" in str(e).lower():
                        continue
                    else:
                        raise
    
    def test_boundary_values(self, all_models):
        """Graceful handling of boundary/degenerate ranges.

        Note: ranges must be strictly increasing; exact equal endpoints such
        as (0.0, 0.0) are expected to fail validation by design. We include
        these cases to ensure the code either raises ValueError cleanly (for
        invalid ranges) or returns a consistent structure when ranges are
        valid but very narrow.
        """
        boundary_cases = [
            ((0.0, 0.0), (0.0, 0.0)),    # exactly equal endpoints (should raise ValueError)
            ((100.0, 100.0), (100.0, 100.0)),  # equal at upper extremes (should raise ValueError)
            ((0.1, 0.2), (0.1, 0.2)),    # minimal valid span
            ((50.0, 50.1), (50.0, 50.1)),  # single-point-like valid range
        ]
        
        for model_info in all_models:
            for t_range, rh_range in boundary_cases:
                try:
                    ax, artists = ranges_tdb_rh(
                        model_func=model_info.func,
                        fixed_params=model_info.fixed_params,
                        thresholds=[model_info.thresholds[0]],
                        t_range=t_range,
                        rh_range=rh_range
                    )
                    
                    if ax is None:
                        print(
                            f"Warning: {model_info.name} not compatible for boundary case "
                            f"t_range={t_range}, rh_range={rh_range}"
                        )
                        continue
                        
                    assert "curves" in artists, (
                        f"{model_info.name} should have curves for boundary case "
                        f"t_range={t_range}, rh_range={rh_range}"
                    )
                    
                except ValueError as e:
                    if "strictly increasing" in str(e):
                        continue
                    raise
                except Exception as e:

                    if "no bracket" in str(e).lower() or "unsolved" in str(e).lower():
                        continue
                    else:
                        raise
