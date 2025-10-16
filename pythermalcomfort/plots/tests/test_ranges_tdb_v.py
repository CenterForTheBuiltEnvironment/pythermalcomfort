"""Tests for ranges_tdb_v plotting function (Temperature vs Air Speed)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import same_color

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import ranges_tdb_v


@pytest.fixture
def params_for_v(fixed_params_pmv):
    """Prepare params for tdb-v plot (remove vr, ensure rh exists)."""
    params = {**fixed_params_pmv}
    params.pop("vr", None)
    if "rh" not in params:
        params["rh"] = 50.0
    return params


class TestRangesTdbVSmoke:
    """Smoke tests for temperature vs air speed plots."""

    def test_basic_execution(self, params_for_v):
        """Test basic plot execution."""
        ax, artists = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_v,
            thresholds=[-0.5, 0.5],
            t_range=(18.0, 28.0),
            v_range=(0.0, 1.0),
        )
        assert ax is not None
        assert isinstance(artists, dict)
        assert "bands" in artists
        assert "curves" in artists
        assert isinstance(artists["bands"], list)
        assert isinstance(artists["curves"], list)

    def test_uses_default_thresholds(self, params_for_v):
        """Test with default thresholds."""
        ax, _ = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_v,
            thresholds=None,
        )
        assert ax is not None

    @pytest.mark.parametrize("v_range,v_step,description", [
        ((0.0, 3.0), 0.05, "extreme high speed range"),
        ((0.1, 0.3), 0.05, "narrow range (typical indoor)"),
        ((0.0, 1.0), 0.2, "coarse step"),
    ])
    def test_various_speed_ranges(self, params_for_v, v_range, v_step, description):
        """Test different air speed ranges and steps."""
        ax, artists = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_v,
            thresholds=[-0.5, 0.5],
            v_range=v_range,
            v_step=v_step,
        )
        assert ax is not None, f"Failed for {description}"
        assert len(artists["bands"]) > 0 or len(artists["curves"]) > 0, \
            f"No results for {description}"


class TestRangesTdbVValidation:
    """Test input validation."""

    @pytest.mark.parametrize("v_range,v_step,error_msg", [
        ((1.0, 0.0), 0.05, "strictly increasing"), 
        ((0.0, 1.0), 0.0, "must be positive"),     
        ((0.0, 1.0), -0.1, "must be positive"),    
    ])
    def test_invalid_parameters_raise(self, params_for_v, v_range, v_step, error_msg):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match=error_msg):
            ranges_tdb_v(
                model_func=pmv_ppd_iso,
                fixed_params=params_for_v,
                thresholds=[-0.5, 0.5],
                v_range=v_range,
                v_step=v_step,
            )


class TestRangesTdbVVisualProperties:
    """Test visual customization."""

    def test_custom_visual_parameters(self, params_for_v):
        """Test plot customization."""
        _, artists = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_v,
            thresholds=[-0.5, 0.5],
            line_width=3.0,
            line_color="blue",
        )
        assert len(artists["curves"]) > 0, "Should have at least one curve"
        for curve in artists["curves"]:
            assert curve.get_linewidth() == 3.0
            assert same_color(curve.get_color(), "blue")

    def test_default_labels(self, params_for_v):
        """Test default axis labels."""
        ax, _ = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_v,
            thresholds=[-0.5, 0.5],
        )
        assert "temperature" in ax.get_xlabel().lower()
        assert "speed" in ax.get_ylabel().lower()


class TestRangesTdbVEdgeCases:
    """Test edge cases and extreme user inputs."""

    @pytest.mark.parametrize("v_range,t_range,met,description", [
        ((0.0, 0.5), (20.0, 26.0), 1.2, "low speeds (still air)"),
        ((1.0, 5.0), (20.0, 30.0), 1.2, "extreme high speeds"),
        ((0.0, 2.0), (10.0, 25.0), 3.0, "high metabolic activity"),
    ])
    def test_edge_case_combinations(self, params_for_v, v_range, t_range, met, description):
        """Test various edge case combinations that users might input."""
        params = {**params_for_v, "met": met}
        ax, artists = ranges_tdb_v(
            model_func=pmv_ppd_iso,
            fixed_params=params,
            thresholds=[-0.5, 0.5],
            t_range=t_range,
            v_range=v_range,
        )
        assert ax is not None, f"Failed for {description}"
        assert isinstance(artists, dict), f"Invalid return for {description}"