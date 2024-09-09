import numpy as np
from pythermalcomfort.models import adaptive_en


def test_adaptive_en(get_adaptive_en_test_data, is_equal):
    for entry in get_adaptive_en_test_data["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"], inputs["tr"], inputs["t_running_mean"], inputs["v"]
        )

        assert is_equal(result["tmp_cmf"], outputs["tmp_cmf"])
        assert is_equal(result["acceptability_cat_i"], outputs["acceptability_cat_i"])
        assert is_equal(result["acceptability_cat_ii"], outputs["acceptability_cat_ii"])
        assert is_equal(
            result["acceptability_cat_iii"], outputs["acceptability_cat_iii"]
        )
        assert is_equal(result["tmp_cmf_cat_i_up"], outputs["tmp_cmf_cat_i_up"])
        assert is_equal(result["tmp_cmf_cat_ii_up"], outputs["tmp_cmf_cat_ii_up"])
        assert is_equal(result["tmp_cmf_cat_iii_up"], outputs["tmp_cmf_cat_iii_up"])
        assert is_equal(result["tmp_cmf_cat_i_low"], outputs["tmp_cmf_cat_i_low"])
        assert is_equal(result["tmp_cmf_cat_ii_low"], outputs["tmp_cmf_cat_ii_low"])
        assert is_equal(result["tmp_cmf_cat_iii_low"], outputs["tmp_cmf_cat_iii_low"])
