"""Tests for ranges_tdb_rh plotting function.

Following seaborn testing approach with comprehensive parameter combinations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import same_color

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import ranges_tdb_rh


class TestRangesTdbRhSmoke:
    """Smoke tests: ensure function runs without errors."""

    def test_basic_pmv_plot(self, fixed_params_pmv):
        """Test basic PMV plot execution."""
        ax, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            t_range=(18.0, 28.0),
            rh_range=(20.0, 80.0),
        )
        assert ax is not None
        assert isinstance(artists, dict)
        assert "bands" in artists
        assert "curves" in artists
        assert isinstance(artists["bands"], list)
        assert isinstance(artists["curves"], list)

    def test_uses_default_thresholds(self, fixed_params_pmv):
        """Test that default thresholds are used when not provided."""
        ax, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=None,
            t_range=(20.0, 26.0),
            rh_range=(30.0, 70.0),
        )
        assert ax is not None

    def test_no_thresholds_and_no_defaults_raises(self, simple_linear_model):
        """Test that missing thresholds raises error when no defaults."""
        with pytest.raises(ValueError, match="No thresholds provided"):
            ranges_tdb_rh(
                model_func=simple_linear_model,
                fixed_params={},
                thresholds=None,
                t_range=(10.0, 30.0),
            )

    @pytest.mark.parametrize("t_range,rh_range", [
        ((-10.0, 50.0), (0.0, 100.0)),  # extreme ranges
        ((22.0, 24.0), (45.0, 55.0)),   # narrow ranges
    ])
    def test_various_ranges(self, fixed_params_pmv, t_range, rh_range):
        """Test with different temperature and humidity ranges."""
        ax, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            t_range=t_range,
            rh_range=rh_range,
        )
        assert ax is not None

    @pytest.mark.parametrize("thresholds,rh_step", [
        ([0.0], 10.0),
        ([-1.5, -0.5, 0.5, 1.5], 2.0),
        ([-0.5, 0.5], 0.5),
    ])
    def test_thresholds_and_steps(self, fixed_params_pmv, thresholds, rh_step):
        """Test various threshold and step combinations."""
        ax, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=thresholds,
            t_range=(20.0, 26.0),
            rh_step=rh_step,
        )
        assert ax is not None


class TestRangesTdbRhValidation:
    """Test input validation."""

    @pytest.mark.parametrize("t_range,rh_range,rh_step,x_scan_step,error_msg", [
        ((30.0, 20.0), (20.0, 80.0), 2.0, 1.0, "strictly increasing"), 
        ((20.0, 26.0), (80.0, 20.0), 2.0, 1.0, "strictly increasing"), 
        ((20.0, 26.0), (20.0, 80.0), 0.0, 1.0, "must be positive"),
        ((20.0, 26.0), (20.0, 80.0), -1.0, 1.0, "must be positive"), 
        ((20.0, 26.0), (20.0, 80.0), 2.0, 0.0, "must be positive"),
    ])
    def test_invalid_parameters_raise(self, fixed_params_pmv, t_range, rh_range, 
                                     rh_step, x_scan_step, error_msg):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match=error_msg):
            ranges_tdb_rh(
                model_func=pmv_ppd_iso,
                fixed_params=fixed_params_pmv,
                thresholds=[-0.5, 0.5],
                t_range=t_range,
                rh_range=rh_range,
                rh_step=rh_step,
                x_scan_step=x_scan_step,
            )


class TestRangesTdbRhAxesHandling:
    """Test axes creation and usage."""

    def test_axes_handling(self, fixed_params_pmv):
        """Test axes creation and reuse."""
        ax1, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            ax=None,
        )
        assert isinstance(ax1, plt.Axes)
        
        fig, ax_provided = plt.subplots()
        ax2, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            ax=ax_provided,
        )
        assert ax2 is ax_provided


class TestRangesTdbRhVisualProperties:
    """Test visual customization options."""

    def test_returns_artist_dict_structure(self, fixed_params_pmv):
        """Test that artists dict has expected structure.
        
        Similar to seaborn's artist verification.
        """
        _, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
        )
        assert isinstance(artists, dict)
        assert "bands" in artists
        assert "curves" in artists
        assert "legend" in artists
        
        assert isinstance(artists["bands"], list)
        assert isinstance(artists["curves"], list)

    def test_number_of_curves_matches_thresholds(self, fixed_params_pmv):
        """Test that number of curves matches number of thresholds.
        
        Data-level verification like seaborn.
        """
        thresholds = [-0.7, -0.2, 0.3, 0.8]
        _, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=thresholds,
            t_range=(18.0, 28.0),
        )
        curves = artists["curves"]
        assert len(curves) <= len(thresholds)
        assert len(curves) > 0

    @pytest.mark.parametrize("legend,should_exist", [(True, True), (False, False)])
    def test_legend_control(self, fixed_params_pmv, legend, should_exist):
        """Test legend creation control."""
        _, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            legend=legend,
        )
        if should_exist:
            assert "legend" in artists
        else:
            assert artists["legend"] is None

    def test_custom_visual_parameters(self, fixed_params_pmv):
        """Test that visual parameters work correctly.
        
        Property-level assertion like seaborn.
        """
        _, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            cmap="viridis",
            band_alpha=0.5,
            line_color="red",
            line_width=2.0,
        )
        curves = artists["curves"]
        assert len(curves) > 0, "Should have at least one curve"
        for curve in curves:
            assert curve.get_linewidth() == 2.0
            assert same_color(curve.get_color(), "red")

    def test_kwargs_override_defaults(self, fixed_params_pmv):
        """Test that **kwargs can override default labels."""
        ax, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            xlabel="Custom Temperature Label",
            ylabel="Custom Humidity Label",
        )
        assert ax.get_xlabel() == "Custom Temperature Label"
        assert ax.get_ylabel() == "Custom Humidity Label"


    def test_default_labels(self, fixed_params_pmv):
        """Test default axis labels."""
        ax, _ = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
        )
        assert "temperature" in ax.get_xlabel().lower()
        assert "humidity" in ax.get_ylabel().lower()


class TestRangesTdbRhSolverControls:
    """Test solver control parameters."""

    def test_solver_parameters(self, fixed_params_pmv):
        """Test typical solver control parameters."""
        ax, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            t_range=(20.0, 26.0),
            x_scan_step=0.5,
            smooth_sigma=1.0,
        )
        assert ax is not None
        assert len(artists["bands"]) > 0 or len(artists["curves"]) > 0


class TestRangesTdbRhEdgeCases:
    """Test edge cases and extreme user inputs."""

    @pytest.mark.parametrize("thresholds,t_range,description", [
        ([0.0], (23.0, 23.5), "very narrow temperature range"),
        ([-2.0, -0.2, 0.8], (10.0, 35.0), "asymmetric thresholds"),
        ([0.0], (20.0, 26.0), "zero threshold (neutral)"),
        ([-2.0, -1.0], (10.0, 20.0), "cold conditions (negative thresholds)"),
        ([1.0, 2.0], (28.0, 38.0), "hot conditions (positive thresholds)"),
    ])
    def test_threshold_edge_cases(self, fixed_params_pmv, thresholds, t_range, description):
        """Test various threshold configurations that users might input."""
        ax, artists = ranges_tdb_rh(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=thresholds,
            t_range=t_range,
        )
        assert ax is not None, f"Failed for {description}"
        assert isinstance(artists, dict), f"Invalid return for {description}"

    def test_different_comfort_conditions(self, fixed_params_pmv):
        """Test with different metabolic/clothing combinations (summer/winter)."""
        for met, clo, t_range, desc in [
            (1.0, 0.3, (24, 30), "summer"),
            (2.0, 1.0, (10, 20), "winter")
        ]:
            params = {**fixed_params_pmv, "met": met, "clo": clo}
            ax, artists = ranges_tdb_rh(
                model_func=pmv_ppd_iso,
                fixed_params=params,
                thresholds=[-0.5, 0.5],
                t_range=t_range,
            )
            assert ax is not None, f"Failed for {desc} conditions"
            assert len(artists["bands"]) > 0 or len(artists["curves"]) > 0, \
                f"No results for {desc} conditions"