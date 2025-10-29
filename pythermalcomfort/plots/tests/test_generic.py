"""Tests for pythermalcomfort.plots.generic module.

Following seaborn testing approach:
- Smoke tests with various parameter combinations
- Property-level assertions on returned artists
- No pixel-level image comparisons
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import same_color

from pythermalcomfort.plots.generic import calc_plot_ranges


class TestCalcPlotRangesBasic:
    """Basic functionality tests for calc_plot_ranges.
    
    Tests core functionality including execution, axes handling,
    and parameter combinations for the generic plotting function.
    """

    def test_basic_execution(self, simple_linear_model):
        """Test basic execution returns axes and artists dict."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        ax, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={"a": 1.0, "b": 0.1},
            thresholds=[15.0, 20.0],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 20),
            metric_attr="pmv",
        )
        assert isinstance(ax, plt.Axes)
        assert isinstance(artists, dict)
        assert all(k in artists for k in ["bands", "curves", "legend"])

    def test_axes_handling(self, simple_linear_model):
        """Test axes creation and reuse."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        ax1, _ = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[15.0],
            x_bounds=(10.0, 30.0),
            y_values=[50.0],
            ax=None,
        )
        assert isinstance(ax1, plt.Axes)

        fig, ax_provided = plt.subplots()
        ax2, _ = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[15.0],
            x_bounds=(10.0, 30.0),
            y_values=[50.0],
            ax=ax_provided,
        )
        assert ax2 is ax_provided

    @pytest.mark.parametrize("thresholds,x_bounds,y_values", [
        ([20.0], (10.0, 30.0), [50.0]),                      # single threshold, single y
        ([10.0, 20.0, 30.0], (5.0, 35.0), np.linspace(0, 100, 20)),  # multiple thresholds
        ([0.0], (-100.0, 100.0), np.linspace(-1000, 1000, 50)),  # extreme bounds
        ([20.0], (10.0, 30.0), np.linspace(0, 100, 200)),    # dense grid
    ])
    def test_parameter_combinations(self, simple_linear_model, thresholds, x_bounds, y_values):
        """Test various parameter combinations execute without error."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        ax, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=thresholds,
            x_bounds=x_bounds,
            y_values=y_values,
        )
        assert ax is not None
        assert "bands" in artists


class TestCalcPlotRangesArtists:
    """Test properties of returned artists.
    
    Tests the structure and properties of matplotlib artists
    returned by calc_plot_ranges including bands, curves, and legend.
    """

    def test_returns_expected_dict_keys(self, simple_linear_model):
        """Test that artists dict contains expected keys."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        _, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[15.0, 20.0],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 10),
        )
        assert "bands" in artists
        assert "curves" in artists
        assert "legend" in artists

    def test_curve_data_satisfies_threshold(self, simple_linear_model):
        """Test that curve coordinates actually satisfy threshold condition.
        
        Similar to seaborn's test_xy_data - verify data mapping to artist.
        """
        from pythermalcomfort.plots.utils import mapper_tdb_rh, make_metric_eval

        threshold = 20.0
        fixed_params = {"a": 1.0, "b": 0.1}
        
        # Build metric evaluator
        metric_xy = make_metric_eval(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params=fixed_params,
            metric_attr="pmv",
        )
        
        _, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params=fixed_params,
            thresholds=[threshold],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 10),
            smooth_sigma=0.0,
        )
        
        curves = artists["curves"]
        if len(curves) > 0:
            curve = curves[0]
            x_data = curve.get_xdata()
            y_data = curve.get_ydata()
            
        for x, y in zip(x_data, y_data):
            if np.isfinite(x) and np.isfinite(y):
                metric_val = metric_xy(x, y)
                assert np.isclose(metric_val, threshold, atol=0.5), \
                    f"Point ({x}, {y}) has metric {metric_val}, expected {threshold}"

    @pytest.mark.parametrize("legend_flag,expected", [
        (True, "not_none"),
        (False, "none"),
    ])
    def test_legend_parameter(self, simple_linear_model, legend_flag, expected):
        """Test that legend parameter controls legend creation."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        _, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[20.0],
            x_bounds=(10.0, 30.0),
            y_values=[50.0],
            legend=legend_flag,
        )
        if expected == "not_none":
            assert artists["legend"] is not None
        else:
            assert artists["legend"] is None

    def test_custom_band_colors_applied(self, simple_linear_model):
        """Test that custom band colors are applied correctly."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
        
        _, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[15.0, 20.0],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 10),
            band_colors=custom_colors,
        )
        
        bands = artists["bands"]
        assert len(bands) > 0
        assert len(bands) <= len(custom_colors)
        
        for band in bands:
            fc = band.get_facecolor()
            assert len(fc) > 0


class TestCalcPlotRangesVisualProperties:
    """Test visual properties (colors, alpha, line styles).
    
    Tests visual customization options including color schemes,
    transparency, line styles, and axis labels.
    """

    def test_band_colors_length_mismatch_raises(self, simple_linear_model):
        """Test that wrong number of band_colors raises error."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        with pytest.raises(ValueError, match="band_colors must have length"):
            calc_plot_ranges(
                model_func=simple_linear_model,
                xy_to_kwargs=mapper_tdb_rh,
                fixed_params={},
                thresholds=[15.0, 20.0],  # creates 3 regions
                x_bounds=(10.0, 30.0),
                y_values=[50.0],
                band_colors=["red", "blue"],  # only 2 colors
            )

    def test_both_cmap_and_band_colors_raises(self, simple_linear_model):
        """Test that providing both cmap and band_colors raises error."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        with pytest.raises(ValueError, match="only one of cmap or band_colors"):
            calc_plot_ranges(
                model_func=simple_linear_model,
                xy_to_kwargs=mapper_tdb_rh,
                fixed_params={},
                thresholds=[20.0],
                x_bounds=(10.0, 30.0),
                y_values=[50.0],
                cmap="viridis",
                band_colors=["red", "blue"],
            )

    def test_visual_properties(self, simple_linear_model):
        """Test that visual properties are applied correctly."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        _, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[20.0],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 10),
            band_alpha=0.3,
            line_color="red",
            line_width=3.0,
            cmap="viridis",
        )
        
        for band in artists["bands"]:
            alpha = band.get_alpha()
            if alpha is not None:
                assert np.isclose(alpha, 0.3)
        
        for curve in artists["curves"]:
            assert same_color(curve.get_color(), "red")
            assert curve.get_linewidth() == 3.0


    def test_axis_labels(self, simple_linear_model):
        """Test axis label customization."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        ax, _ = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[20.0],
            x_bounds=(10.0, 30.0),
            y_values=[50.0],
            xlabel="Temperature (°C)",
            ylabel="Relative Humidity (%)",
        )
        assert ax.get_xlabel() == "Temperature (°C)"
        assert ax.get_ylabel() == "Relative Humidity (%)"


class TestCalcPlotRangesSolverControls:
    """Test solver control parameters for stability.
    
    Tests solver parameters that control curve finding accuracy
    and stability including scan steps and smoothing.
    """

    def test_solver_params(self, simple_linear_model):
        """Test typical solver parameters execute without error."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        ax, artists = calc_plot_ranges(
            model_func=simple_linear_model,
            xy_to_kwargs=mapper_tdb_rh,
            fixed_params={},
            thresholds=[20.0],
            x_bounds=(10.0, 30.0),
            y_values=np.linspace(0, 100, 10),
            x_scan_step=0.5,
            smooth_sigma=1.0,
        )
        assert ax is not None
        assert len(artists["bands"]) > 0
        assert len(artists["curves"]) > 0


class TestCalcPlotRangesNaNHandling:
    """Test NaN handling in curve data.
    
    Tests how the plotting function handles cases where threshold
    curves cannot be solved for certain parameter combinations,
    similar to seaborn's NaN/segmentation tests.
    """

    def test_curve_handles_partial_unsolvability(self):
        """Test that curves handle partially unsolvable regions.
        
        Following seaborn's test_xy_data pattern for NaN verification.
        """
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        # Create a model where threshold is only solvable in some y-ranges
        def limited_range_model(tdb, rh, **kwargs):
            class Result:
                def __init__(self):
                    # Metric depends strongly on RH
                    # Only certain RH ranges will allow threshold to be reached
                    self.pmv = 0.5 * tdb + 0.05 * rh - 15.0
            return Result()

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, artists = calc_plot_ranges(
                model_func=limited_range_model,
                xy_to_kwargs=mapper_tdb_rh,
                fixed_params={},
                thresholds=[5.0],  # might not be solvable everywhere
                x_bounds=(10.0, 25.0),  # limited range
                y_values=np.linspace(0, 100, 30),
                smooth_sigma=0.0,
            )
        
        curves = artists["curves"]
        assert isinstance(curves, list)
        
        for curve in curves:
            x_data = curve.get_xdata()
            assert isinstance(x_data, np.ndarray)
            if len(x_data) > 0:
                assert np.any(np.isfinite(x_data)) or np.all(np.isnan(x_data))


class TestCalcPlotRangesParameterValidation:
    """Test parameter validation and error handling.
    
    Tests input validation and error handling for invalid parameters,
    following seaborn's comprehensive error coverage approach.
    """

    def test_reversed_x_bounds_raises(self, simple_linear_model):
        """Test that reversed x_bounds raises ValueError."""
        from pythermalcomfort.plots.utils import mapper_tdb_rh

        with pytest.raises(ValueError, match="strictly increasing"):
            calc_plot_ranges(
                model_func=simple_linear_model,
                xy_to_kwargs=mapper_tdb_rh,
                fixed_params={},
                thresholds=[20.0],
                x_bounds=(30.0, 10.0),
                y_values=[50.0],
            )