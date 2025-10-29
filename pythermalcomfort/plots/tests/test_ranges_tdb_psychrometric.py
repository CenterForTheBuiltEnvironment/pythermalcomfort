"""Tests for ranges_tdb_psychrometric plotting function (Psychrometric Chart)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import same_color

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import ranges_tdb_psychrometric


class TestRangesTdbPsychrometricSmoke:
    """Smoke tests for psychrometric chart plots."""

    def test_basic_execution(self, fixed_params_pmv):
        """Test basic plot execution."""
        ax, artists = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            t_range=(18.0, 28.0),
            w_range=(0.005, 0.015),
        )
        assert ax is not None
        assert isinstance(artists, dict)
        assert "bands" in artists
        assert "curves" in artists
        assert isinstance(artists["bands"], list)
        assert isinstance(artists["curves"], list)

    def test_uses_default_thresholds(self, fixed_params_pmv):
        """Test with default thresholds."""
        ax, _ = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=None,
        )
        assert ax is not None

    @pytest.mark.parametrize("w_range,w_step", [
        ((0.0, 0.03), 0.002),   
        ((0.009, 0.011), 0.0001), 
        ((0.015, 0.025), 0.001), 
    ])
    def test_various_humidity_ranges(self, fixed_params_pmv, w_range, w_step):
        """Test different humidity ratio ranges and steps."""
        ax, artists = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            w_range=w_range,
            w_step=w_step,
        )
        assert ax is not None
        assert len(artists["bands"]) > 0 or len(artists["curves"]) > 0


class TestRangesTdbPsychrometricBackground:
    """Test psychrometric chart background features."""

    @pytest.mark.parametrize("draw_bg,rh_isolines", [
        (True, [20, 50, 80, 100]),        
        (True, [100]),                     
        (True, list(range(10, 101, 10))),  
        (False, []),                       
    ])
    def test_background_options(self, fixed_params_pmv, draw_bg, rh_isolines):
        """Test background RH isoline options."""
        ax, _ = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            draw_background=draw_bg,
            rh_isolines=rh_isolines,
        )
        assert ax is not None


class TestRangesTdbPsychrometricValidation:
    """Test input validation."""

    @pytest.mark.parametrize("t_range,w_range,w_step,x_scan_step,error_msg", [
        ((30.0, 20.0), (0.005, 0.015), 0.001, 1.0, "strictly increasing"), 
        ((20.0, 30.0), (0.02, 0.01), 0.001, 1.0, "strictly increasing"),   
        ((20.0, 30.0), (0.005, 0.015), 0.0, 1.0, "must be positive"),      
        ((20.0, 30.0), (0.005, 0.015), -0.001, 1.0, "must be positive"),   
        ((20.0, 30.0), (0.005, 0.015), 0.001, 0.0, "must be positive"),    
    ])
    def test_invalid_parameters_raise(self, fixed_params_pmv, t_range, w_range,
                                     w_step, x_scan_step, error_msg):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match=error_msg):
            ranges_tdb_psychrometric(
                model_func=pmv_ppd_iso,
                fixed_params=fixed_params_pmv,
                thresholds=[-0.5, 0.5],
                t_range=t_range,
                w_range=w_range,
                w_step=w_step,
                x_scan_step=x_scan_step,
            )


class TestRangesTdbPsychrometricVisualProperties:
    """Test visual customization."""

    def test_visual_customization(self, fixed_params_pmv):
        """Test **kwargs and labels."""
        ax, artists = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            line_width=2.5,
            xlabel="Custom T",
            ylabel="Custom W",
        )
        assert ax.get_xlabel() == "Custom T"
        assert ax.get_ylabel() == "Custom W"
        if artists["curves"]:
            assert artists["curves"][0].get_linewidth() == 2.5


class TestRangesTdbPsychrometricEdgeCases:
    """Test edge cases and extreme user inputs."""

    def test_saturation_clipping(self, fixed_params_pmv):
        """Test that regions are clipped at saturation boundary.
        
        Wide humidity range that crosses saturation line.
        """
        ax, artists = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params=fixed_params_pmv,
            thresholds=[-0.5, 0.5],
            t_range=(10.0, 35.0),
            w_range=(0.0, 0.03),
        )
        assert ax is not None
        assert "bands" in artists
        assert len(artists["bands"]) > 0

    @pytest.mark.parametrize("thresholds,t_range,w_range,clo,description", [
        ([0.0], (23.0, 23.5), (0.0095, 0.0105), 0.5, "very narrow range"),
        ([-1.5, 0.0, 1.5], (10.0, 35.0), (0.002, 0.025), 0.5, "many thresholds"),
        ([-0.5, 0.5], (-5.0, 45.0), (0.0, 0.035), 0.5, "extreme temperatures"),
        ([-0.5, 0.5], (24.0, 32.0), (0.010, 0.025), 0.3, "summer conditions"),
        ([-0.5, 0.5], (16.0, 24.0), (0.002, 0.012), 1.0, "winter conditions"),
    ])
    def test_edge_case_combinations(self, thresholds, t_range, w_range, clo, description):
        """Test various edge case combinations that users might input."""
        ax, artists = ranges_tdb_psychrometric(
            model_func=pmv_ppd_iso,
            fixed_params={"tr": 25.0, "met": 1.2, "clo": clo, "vr": 0.1, "wme": 0.0},
            thresholds=thresholds,
            t_range=t_range,
            w_range=w_range,
        )
        assert ax is not None, f"Failed for {description}"
        assert isinstance(artists, dict), f"Invalid return type for {description}"