import numpy as np

from pythermalcomfort.models import adaptive_en


def test_adaptive_en():
    np.testing.assert_equal(
        adaptive_en(
            tdb=[25, 25, 23.5], tr=[25, 25, 23.5], t_running_mean=[9, 20, 28], v=0.1
        ),
        {
            "tmp_cmf": [np.nan, 25.4, 28.0],
            "acceptability_cat_i": [False, True, False],
            "acceptability_cat_ii": [False, True, False],
            "acceptability_cat_iii": [False, True, True],
            "tmp_cmf_cat_i_up": [np.nan, 27.4, 30.0],
            "tmp_cmf_cat_ii_up": [np.nan, 28.4, 31.0],
            "tmp_cmf_cat_iii_up": [np.nan, 29.4, 32.0],
            "tmp_cmf_cat_i_low": [np.nan, 22.4, 25.0],
            "tmp_cmf_cat_ii_low": [np.nan, 21.4, 24.0],
            "tmp_cmf_cat_iii_low": [np.nan, 20.4, 23.0],
        },
    )
