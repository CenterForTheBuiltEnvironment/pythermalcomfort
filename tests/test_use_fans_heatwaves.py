import numpy as np

from pythermalcomfort.models import (
    use_fans_heatwaves,
)


def test_use_fans_heatwaves():
    # checking that returns np.nan when outside standard applicability limits
    np.testing.assert_equal(
        use_fans_heatwaves(
            tdb=[41, 60],
            tr=40,
            v=0.1,
            rh=50,
            met=1.1,
            clo=0.5,
        )["e_skin"],
        [65.2, np.nan],
    )

    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.5, body_position="sitting"
        )["q_skin"]
        == 37.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=0.7, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 68.6
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.3, body_position="sitting"
        )["m_rsw"]
        == 118.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.5, body_position="sitting"
        )["m_rsw"]
        == 117.3
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=1.3, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 116.4
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.3, body_position="sitting"
        )["heat_strain_w"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.5, body_position="sitting"
        )["w"]
        == 0.5
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=20, met=2, clo=0.7, body_position="sitting"
        )["t_skin"]
        == 36.2
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.3, body_position="sitting"
        )["heat_strain_blood_flow"]
        == False
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.5, body_position="sitting"
        )["t_core"]
        == 36.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=0.7, clo=0.7, body_position="sitting"
        )["m_rsw"]
        == 73.9
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=1.3, clo=0.3, body_position="sitting"
        )["m_rsw"]
        == 126.8
    )
    assert (
        use_fans_heatwaves(
            tdb=39, tr=39, v=0.2, rh=40, met=1.3, clo=0.5, body_position="sitting"
        )["e_rsw"]
        == 84.9
    )
