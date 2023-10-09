from pythermalcomfort.models import two_nodes


def test_two_nodes():
    assert two_nodes(25, 25, 1.1, 50, 2, 0.5)["disc"] == 0.4
    assert two_nodes(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 0.3
    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 1.0
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["disc"] == 1.6
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["disc"] == 0.8

    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 0.9
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 1.5
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["pmv_gagge"] == 0.8

    assert two_nodes(tdb=30, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_set"] == 1.0
    assert two_nodes(tdb=30, tr=30, v=0.1, rh=50, met=1.2, clo=0.5)["pmv_set"] == 1.4
    assert two_nodes(tdb=28, tr=28, v=0.4, rh=50, met=1.2, clo=0.5)["pmv_set"] == 0.5

    # testing limiting w_max
    assert two_nodes(40, 40, 1.1, 50, 2, 0.5, w_max=False)["t_core"] == 37.9
    assert two_nodes(40, 40, 1.1, 50, 2, 0.5, w_max=0.2)["t_core"] == 39.0

    # testing limiting max_sweating
    assert two_nodes(45, 45, 1.1, 20, 3, 0.2)["e_rsw"] == 219.3
    assert two_nodes(45, 45, 1.1, 20, 3, 0.2, max_sweating=300)["e_rsw"] == 204.0

    # testing limiting max skin blood flow
    assert two_nodes(45, 45, 1.1, 20, 3, 0.2)["t_core"] == 38.0
    assert two_nodes(45, 45, 1.1, 20, 3, 0.2, max_skin_blood_flow=60)["t_core"] == 38.2
