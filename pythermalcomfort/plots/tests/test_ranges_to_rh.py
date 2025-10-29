"""Tests for ranges_to_rh plotting function (Operative Temperature vs RH)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import same_color

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import ranges_to_rh


@pytest.fixture
def params_for_to(fixed_params_pmv):
    """Prepare params for operative temp plot (remove tr)."""
    params = {**fixed_params_pmv}
    params.pop("tr", None)
    return params


class TestRangesToRhSmoke:
    """Smoke tests for operative temperature vs RH plots."""

    def test_basic_execution(self, params_for_to):
        """Test basic plot execution."""
        ax, artists = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
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

    def test_uses_default_thresholds(self, params_for_to):
        """Test with default thresholds."""
        ax, _ = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=None,
        )
        assert ax is not None

    @pytest.mark.parametrize("t_range,rh_range,rh_step,description", [
        ((0.0, 40.0), (10.0, 90.0), 2.0, "extreme temperature range"),
        ((22.0, 24.0), (45.0, 55.0), 0.5, "narrow range (fine control)"),
        ((20.0, 26.0), (20.0, 80.0), 10.0, "coarse step"),
    ])
    def test_various_ranges(self, params_for_to, t_range, rh_range, rh_step, description):
        """Test different temperature and RH ranges."""
        ax, artists = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=[-0.5, 0.5],
            t_range=t_range,
            rh_range=rh_range,
            rh_step=rh_step,
        )
        assert ax is not None, f"Failed for {description}"
        assert len(artists["bands"]) > 0 or len(artists["curves"]) > 0, \
            f"No results for {description}"


class TestRangesToRhValidation:
    """Test input validation."""

    @pytest.mark.parametrize("t_range,rh_range,rh_step,error_msg", [
        ((30.0, 20.0), (20.0, 80.0), 2.0, "strictly increasing"),
        ((20.0, 30.0), (80.0, 20.0), 2.0, "strictly increasing"),
        ((20.0, 30.0), (20.0, 80.0), 0.0, "must be positive"),
    ])
    def test_invalid_parameters_raise(self, params_for_to, t_range, rh_range, rh_step, error_msg):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match=error_msg):
            ranges_to_rh(
                model_func=pmv_ppd_iso,
                fixed_params=params_for_to,
                thresholds=[-0.5, 0.5],
                t_range=t_range,
                rh_range=rh_range,
                rh_step=rh_step,
            )


class TestRangesToRhVisualProperties:
    """Test visual customization."""

    def test_custom_labels(self, params_for_to):
        """Test custom axis labels."""
        ax, _ = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=[-0.5, 0.5],
            xlabel="Custom T",
            ylabel="Custom RH",
        )
        assert ax.get_xlabel() == "Custom T"
        assert ax.get_ylabel() == "Custom RH"

    def test_default_labels(self, params_for_to):
        """Test default axis labels."""
        ax, _ = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=[-0.5, 0.5],
        )
        assert "operative" in ax.get_xlabel().lower()
        assert "humidity" in ax.get_ylabel().lower()

    @pytest.mark.parametrize("legend,should_exist", [(True, True), (False, False)])
    def test_legend_control(self, params_for_to, legend, should_exist):
        """Test legend control."""
        _, artists = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=[-0.5, 0.5],
            legend=legend,
        )
        if should_exist:
            assert "legend" in artists
        else:
            assert artists["legend"] is None


class TestRangesToRhEdgeCases:
    """Test edge cases and extreme user inputs."""

    @pytest.mark.parametrize("thresholds,t_range,rh_range,description", [
        ([0.0], (20.0, 26.0), (30.0, 70.0), "single neutral threshold"),
        ([-1.5, -0.5, 0.5, 1.5], (10.0, 35.0), (10.0, 90.0), "many thresholds (wide coverage)"),
        ([-0.5, 0.5], (20.0, 26.0), (0.0, 10.0), "very low humidity (arid)"),
        ([-0.5, 0.5], (20.0, 26.0), (90.0, 100.0), "very high humidity (tropical)"),
    ])
    def test_edge_case_combinations(self, params_for_to, thresholds, t_range, rh_range, description):
        """Test various edge case combinations that users might input."""
        ax, artists = ranges_to_rh(
            model_func=pmv_ppd_iso,
            fixed_params=params_for_to,
            thresholds=thresholds,
            t_range=t_range,
            rh_range=rh_range,
        )
        assert ax is not None, f"Failed for {description}"
        assert isinstance(artists, dict), f"Invalid return for {description}"