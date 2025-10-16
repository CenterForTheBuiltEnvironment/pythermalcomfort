"""Tests for pythermalcomfort.plots.utils module.

Following seaborn testing approach:
- Property-level assertions (not pixel comparisons)
- Smoke tests for various parameter combinations
- Validation of data transformations and calculations
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.utils import (
    DEFAULT_THRESHOLDS,
    extract_metric,
    get_default_thresholds,
    humidity_ratio_from_t_rh,
    make_metric_eval,
    mapper_tdb_rh,
    mapper_tdb_vr,
    mapper_tdb_w,
    mapper_top_rh,
    rh_from_t_w,
    solve_threshold_curves,
)


class TestDefaultThresholds:
    """Test default thresholds registry functionality.
    
    Tests the get_default_thresholds function which provides
    default threshold values for thermal comfort models.
    """

    def test_get_default_thresholds_known_model(self):
        """Test retrieving default thresholds for known models."""
        thresholds = get_default_thresholds(pmv_ppd_iso)
        assert thresholds is not None
        assert isinstance(thresholds, list)
        assert len(thresholds) > 0
        assert all(isinstance(t, float | int) for t in thresholds)

    def test_get_default_thresholds_unknown_model(self):
        """Test retrieving thresholds for unregistered model returns None."""

        def unknown_model():
            pass

        assert get_default_thresholds(unknown_model) is None


class TestMetricExtraction:
    """Test metric extraction from model results.
    
    Tests the extract_metric function which extracts specific
    attributes from thermal comfort model result objects.
    """

    def test_extract_metric_with_attribute(self, sample_model_result):
        """Test extracting a specific attribute from result object."""
        result = sample_model_result(pmv=0.5, set=26.0)
        assert extract_metric(result, "pmv") == 0.5
        assert extract_metric(result, "set") == 26.0

    def test_extract_metric_missing_attribute_raises(self, sample_model_result):
        """Test that requesting missing attribute raises ValueError."""
        result = sample_model_result()
        with pytest.raises(ValueError, match="no readable attribute"):
            extract_metric(result, "nonexistent_attr")


class TestPsychrometricConversions:
    """Test humidity ratio and RH conversions.
    
    Tests psychrometric conversion functions including humidity ratio
    calculations and relative humidity conversions with proper bounds checking.
    """

    def test_humidity_ratio_from_t_rh_normal(self):
        """Test humidity ratio calculation for normal conditions.
        
        Verifies that humidity ratio calculation produces reasonable
        values for typical indoor conditions (25°C, 50% RH).
        """
        w = humidity_ratio_from_t_rh(25.0, 50.0)
        assert 0.005 < w < 0.015

    def test_humidity_ratio_from_t_rh_extreme_temps(self):
        """Test humidity ratio at extreme temperatures.
        
        Users may input extreme values - system should handle gracefully.
        Tests both very cold (-10°C) and very hot (50°C) conditions.
        """
        w_cold = humidity_ratio_from_t_rh(-10.0, 50.0)
        assert w_cold >= 0.0
        assert w_cold < 0.01

        w_hot = humidity_ratio_from_t_rh(50.0, 50.0)
        assert w_hot > 0.0
        assert w_hot < 0.1

    def test_humidity_ratio_clamps_rh(self):
        """Test that RH values are properly clamped to [0, 100].
        
        Verifies that negative RH values are clamped to 0 and
        values over 100% are clamped to 100%.
        """
        w_neg = humidity_ratio_from_t_rh(25.0, -10.0)
        w_zero = humidity_ratio_from_t_rh(25.0, 0.0)
        assert np.isclose(w_neg, w_zero)

        w_over = humidity_ratio_from_t_rh(25.0, 150.0)
        w_sat = humidity_ratio_from_t_rh(25.0, 100.0)
        assert np.isclose(w_over, w_sat)

    def test_rh_from_t_w_roundtrip(self):
        """Test RH recovery from (T, W) roundtrip.
        
        Verifies that converting from RH to W and back to RH
        produces the original RH value within acceptable tolerance.
        """
        t, rh_orig = 25.0, 60.0
        w = humidity_ratio_from_t_rh(t, rh_orig)
        rh_recovered = rh_from_t_w(t, w)
        assert np.isclose(rh_recovered, rh_orig, rtol=1e-3)

    def test_rh_from_t_w_negative_w_clamped(self):
        """Test that negative W is clamped to 0.
        
        Ensures that negative humidity ratio values are properly
        handled by clamping to 0, which corresponds to 0% RH.
        """
        rh = rh_from_t_w(25.0, -0.001)
        assert rh == 0.0


class TestMappers:
    """Test mapper functions that convert (x, y, fixed) to model kwargs.
    
    Tests various mapper functions that transform plotting coordinates
    and fixed parameters into thermal comfort model keyword arguments.
    """

    def test_mapper_tdb_rh_basic(self):
        """Test basic T-RH mapper."""
        fixed = {"met": 1.2, "clo": 0.5}
        result = mapper_tdb_rh(25.0, 50.0, fixed)
        assert result["tdb"] == 25.0
        assert result["rh"] == 50.0
        assert result["met"] == 1.2
        assert result["clo"] == 0.5

    def test_mapper_tdb_vr_basic(self):
        """Test basic T-v (air speed) mapper."""
        fixed = {"rh": 50.0, "met": 1.2}
        result = mapper_tdb_vr(25.0, 0.5, fixed)
        assert result["tdb"] == 25.0
        assert result["vr"] == 0.5
        assert result["rh"] == 50.0

    def test_mapper_top_rh_sets_both_tdb_tr(self):
        """Test operative temperature mapper sets both tdb and tr."""
        fixed = {"met": 1.2}
        result = mapper_top_rh(26.0, 60.0, fixed)
        assert result["tdb"] == 26.0
        assert result["tr"] == 26.0
        assert result["rh"] == 60.0

    def test_mapper_tdb_w_converts_to_rh(self):
        """Test psychrometric mapper converts W to RH."""
        fixed = {"p_atm": 101325.0, "met": 1.2}
        w = 0.01  # 10 g/kg
        result = mapper_tdb_w(25.0, w, fixed)
        assert result["tdb"] == 25.0
        assert "rh" in result
        assert 0.0 <= result["rh"] <= 100.0


class TestMetricEvaluatorBuilder:
    """Test make_metric_eval function.
    
    Tests the make_metric_eval function which creates metric evaluators
    for plotting thermal comfort metrics across parameter ranges.
    """

    def test_make_metric_eval_basic(self, simple_linear_model):
        """Test building a metric evaluator from a simple model."""
        metric = make_metric_eval(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={"a": 2.0, "b": 0.5},
            metric_attr="pmv",
        )
        result = metric(10.0, 20.0)
        expected = 2.0 * 10.0 + 0.5 * 20.0
        assert result == expected


class TestThresholdSolver:
    """Test solve_threshold_curves function.
    
    Tests the threshold curve solver which finds parameter combinations
    that satisfy specific thermal comfort metric thresholds for plotting.
    """

    def test_solve_threshold_curves_simple_linear(self):
        """Test solver on simple linear metric.
        
        Similar to seaborn's data validation - verify computed values.
        """

        def metric(x, y):
            return x + 0.1 * y  # linear

        result = solve_threshold_curves(
            metric_xy=metric,
            thresholds=[15.0, 20.0],
            y_values=[0.0, 50.0, 100.0],
            x_bounds=(0.0, 30.0),
            x_scan_step=0.5,
            smooth_sigma=0.0,  # no smoothing for exact test
            warn_on_unsolved=False,
        )

        curves = result["curves"]
        assert len(curves) == 2
        assert all(c.shape == (3,) for c in curves)

        assert np.isclose(curves[0][0], 15.0, atol=0.1)
        
        # For linear metric x + 0.1*y = threshold:
        # threshold=15: at y=0->x=15, y=50->x=10, y=100->x=5
        expected_curve_0 = np.array([15.0, 10.0, 5.0])
        assert_array_almost_equal(curves[0], expected_curve_0, decimal=1)

    def test_solve_threshold_curves_no_solution_returns_nan(self):
        """Test that unsolved points return NaN."""

        def metric(x, y):
            return 100.0  # constant, threshold never reached

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = solve_threshold_curves(
                metric_xy=metric,
                thresholds=[50.0],
                y_values=[10.0],
                x_bounds=(0.0, 10.0),
                x_scan_step=1.0,
                warn_on_unsolved=False,
            )

        curves = result["curves"]
        assert np.isnan(curves[0][0])  # no solution found

    def test_solve_threshold_curves_warns_on_unsolved(self):
        """Test that solver warns when points are unsolved."""

        def metric(x, y):
            return 100.0

        with pytest.warns(RuntimeWarning, match="some y-values had no bracket"):
            solve_threshold_curves(
                metric_xy=metric,
                thresholds=[50.0],
                y_values=[10.0],
                x_bounds=(0.0, 10.0),
                x_scan_step=1.0,
                warn_on_unsolved=True,
            )

    def test_solve_threshold_curves_smoothing(self):
        """Test that smoothing reduces noise in curves."""

        def noisy_metric(x, y):
            return x + 0.05 * np.sin(10 * y)  # oscillatory in y

        # Without smoothing
        result_raw = solve_threshold_curves(
            metric_xy=noisy_metric,
            thresholds=[15.0],
            y_values=np.linspace(0, 10, 50),
            x_bounds=(0.0, 30.0),
            x_scan_step=0.1,
            smooth_sigma=0.0,
            warn_on_unsolved=False,
        )
        # With smoothing
        result_smooth = solve_threshold_curves(
            metric_xy=noisy_metric,
            thresholds=[15.0],
            y_values=np.linspace(0, 10, 50),
            x_bounds=(0.0, 30.0),
            x_scan_step=0.1,
            smooth_sigma=2.0,
            warn_on_unsolved=False,
        )

        raw_curve = result_raw["curves"][0]
        smooth_curve = result_smooth["curves"][0]

        # Smoothed curve should have less variance
        if np.isfinite(raw_curve).any() and np.isfinite(smooth_curve).any():
            var_raw = np.nanvar(raw_curve)
            var_smooth = np.nanvar(smooth_curve)
            # Smoothing should reduce variance (with some tolerance)
            assert var_smooth <= var_raw * 1.1  # allow small numerical differences

    def test_solve_threshold_curves_invalid_x_bounds_raises(self):
        """Test that invalid x_bounds raises ValueError."""

        def metric(x, y):
            return x

        with pytest.raises(ValueError, match="strictly increasing"):
            solve_threshold_curves(
                metric_xy=metric,
                thresholds=[10.0],
                y_values=[0.0],
                x_bounds=(30.0, 10.0),  # reversed
            )

    def test_solve_threshold_curves_invalid_step_raises(self):
        """Test that non-positive step raises ValueError."""

        def metric(x, y):
            return x

        with pytest.raises(ValueError, match="must be positive"):
            solve_threshold_curves(
                metric_xy=metric,
                thresholds=[10.0],
                y_values=[0.0],
                x_bounds=(0.0, 30.0),
                x_scan_step=0.0,  # invalid
            )

    def test_solve_threshold_curves_extreme_thresholds(self):
        """Test solver with extreme threshold values.
        
        Users may input extreme values - system should handle gracefully.
        """

        def metric(x, y):
            return x * y

        # Very large thresholds may not be solvable
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = solve_threshold_curves(
                metric_xy=metric,
                thresholds=[1e6],
                y_values=[1.0],
                x_bounds=(0.0, 100.0),
                warn_on_unsolved=False,
            )

        # Should handle gracefully with NaN
        assert "curves" in result
        assert result["unsolved_counts"][0] > 0

    def test_solve_threshold_curves_multiple_thresholds_ordered(self):
        """Test solver with multiple thresholds.
        
        Using numpy.testing assertions like seaborn.
        """

        def metric(x, y):
            return x + y

        result = solve_threshold_curves(
            metric_xy=metric,
            thresholds=[10.0, 20.0, 30.0],
            y_values=[5.0],
            x_bounds=(0.0, 50.0),
            warn_on_unsolved=False,
        )

        curves = result["curves"]
        assert len(curves) == 3
        
        # At y=5: x should be 5, 15, 25 for thresholds 10, 20, 30
        expected_x_values = np.array([5.0, 15.0, 25.0])
        actual_x_values = np.array([curves[0][0], curves[1][0], curves[2][0]])
        assert_array_almost_equal(actual_x_values, expected_x_values, decimal=1)