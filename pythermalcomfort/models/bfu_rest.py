from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import BFU_rest_Groups, BFU_rest_Inputs
from pythermalcomfort.classes_return import BFU_rest as BFURestResult


def morris_lph_fans(
    tdb: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    group: str = BFU_rest_Groups.young.value,
    round_output: bool = True,
) -> BFURestResult:
    """Calculate the net heat storage between fans vs still air using a biophysical
    model for resting individuals.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, :math:`°C`.
    rh : float or list of floats
        Relative humidity, :math:`%`.
    group : str, optional
        Population group considered in the model. Choices are ``'YNG'``, ``'OLD'``, and ``'MEDS'``. Defaults to ``'YNG'``.
    round_output : bool, optional
        If ``True`` the returned values are rounded to one decimal place. Defaults to ``True``.

    Returns
    -------
    BFU_rest
        A dataclass containing the net difference in heat storage and evaporative requirements between fans and still air. See :py:class:`~pythermalcomfort.classes_return.MorrisLPHFans` for details.
    """

    def _psat_kpa_air(t_c: np.ndarray) -> np.ndarray:
        """Saturation vapour pressure over water, [kPa]."""
        return np.exp(18.956 - (4030.18 / (t_c + 235))) / 10

    def _icl_from_rcl(rcl_m2k_w: float) -> float:
        """Convert dry resistance to clothing insulation [clo]."""
        return rcl_m2k_w / 0.155

    def _fcl_from_icl(icl_clo: float) -> float:
        """Clothing area factor for light clothing (ISO 9920)."""
        return 1.0 + 0.31 * icl_clo

    def _hc_from_v(v_ms: float) -> float:
        """Convective heat transfer coefficient (Eq.

        6).
        """
        return 8.3 * (v_ms**0.6)

    def _he_from_hc(hc: float) -> float:
        """Lewis relation (Eq.

        8).
        """
        return 16.5 * hc

    def _ambient_pa_kpa(tdb_arr: np.ndarray, rh_arr: np.ndarray) -> np.ndarray:
        """Ambient water vapour partial pressure, [kPa]."""
        return (rh_arr / 100.0) * _psat_kpa_air(tdb_arr)

    def _dry_heat_flux(
        tdb_arr: np.ndarray,
        rcl: float,
        v_ms: float,
        t_skin: float,
        hr: float,
    ) -> np.ndarray:
        """Compute combined convective and radiative heat loss [W/m²]."""
        icl = _icl_from_rcl(rcl)
        fcl = _fcl_from_icl(icl)
        hc = _hc_from_v(v_ms)
        h = hc + hr
        r_total = rcl + (1.0 / (fcl * h))
        return (t_skin - tdb_arr) / r_total

    def _respiratory(
        tdb_arr: np.ndarray,
        rh_arr: np.ndarray,
        metabolic_rate: float,
    ) -> np.ndarray:
        """Respiratory sensible and latent heat losses [W/m²]."""
        pa = _ambient_pa_kpa(tdb_arr, rh_arr)
        return (0.0014 * metabolic_rate * (34.0 - tdb_arr)) + (
            0.0173 * metabolic_rate * (5.87 - pa)
        )

    def _required_evaporation(
        tdb_arr: np.ndarray,
        rh_arr: np.ndarray,
        rcl: float,
        v_ms: float,
        metabolic_rate: float,
        external_work: float,
        t_skin: float,
        hr: float,
    ) -> np.ndarray:
        c_res_e_res = _respiratory(tdb_arr, rh_arr, metabolic_rate)
        c_plus_r = _dry_heat_flux(tdb_arr, rcl, v_ms, t_skin, hr)
        return metabolic_rate - external_work - c_plus_r - c_res_e_res

    def _max_evaporation(
        tdb_arr: np.ndarray,
        rh_arr: np.ndarray,
        rcl: float,
        recl: float,
        v_ms: float,
        wcrit: float,
        t_skin_sat_pa: float,
        hr: float,
    ) -> np.ndarray:
        """Compute the maximum physical evaporation rate [W/m²]."""
        icl = _icl_from_rcl(rcl)
        fcl = _fcl_from_icl(icl)
        hc = _hc_from_v(v_ms)
        he = _he_from_hc(hc)
        pa = _ambient_pa_kpa(tdb_arr, rh_arr)
        denominator = recl + (1.0 / (he * fcl))
        emax = wcrit * (t_skin_sat_pa - pa) / denominator
        return emax

    def _sweat_limited_max_evaporation(
        ereq: np.ndarray,
        emax_physical: np.ndarray,
        wcrit: float,
        sweating_efficiency_min: float,
        sweat_rate_ml_h: float,
        latent_heat_j_g: float,
        body_surface_area: float,
    ) -> np.ndarray:
        w_req = np.divide(
            ereq,
            emax_physical,
            out=np.full_like(ereq, np.inf, dtype=float),
            where=~np.isnan(emax_physical),
        )
        sweff = 1.0 - ((w_req * w_req) / 2)
        sweff = np.maximum(sweff, sweating_efficiency_min)

        sweat_rate_w = ((sweat_rate_ml_h * latent_heat_j_g) / 3600) * sweff
        sweat_rate_w_m2 = sweat_rate_w / body_surface_area
        return sweat_rate_w_m2

    inputs = BFU_rest_Inputs(tdb=tdb, rh=rh, group=group, round_output=round_output)

    tdb_arr = np.asarray(inputs.tdb, dtype=float)
    rh_arr = np.asarray(inputs.rh, dtype=float)

    try:
        tdb_arr, rh_arr = np.broadcast_arrays(tdb_arr, rh_arr)
    except ValueError as err:
        raise ValueError(
            "Input arrays are not broadcastable to a common shape.",
        ) from err

    params = inputs.params
    group_key = inputs.group

    def _fan_state(fan_on: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wcrit = params.wcrit_on[group_key] if fan_on else params.wcrit_off[group_key]
        sweat_rate = params.sweat_rate[group_key]
        rcl = params.rcl_on if fan_on else params.rcl_off
        recl = (
            0.5 * (params.recl_on_front + params.recl_on_rear)
            if fan_on
            else params.recl_off
        )
        v = params.v_on if fan_on else params.v_off

        ereq = _required_evaporation(
            tdb_arr,
            rh_arr,
            rcl,
            v,
            params.metabolic_rate,
            params.external_work,
            params.t_skin,
            params.hr,
        )
        emax_physical = _max_evaporation(
            tdb_arr,
            rh_arr,
            rcl,
            recl,
            v,
            wcrit,
            params.p_skin_sat_kpa,
            params.hr,
        )
        emax_sweat = _sweat_limited_max_evaporation(
            ereq,
            emax_physical,
            wcrit,
            params.sweating_efficiency_min,
            sweat_rate,
            params.latent_heat_j_g,
            params.body_surface_area,
        )
        emax = np.minimum(emax_physical, emax_sweat)
        return ereq, emax

    ereq_on, emax_on = _fan_state(fan_on=True)
    ereq_off, emax_off = _fan_state(fan_on=False)

    heat_storage = (ereq_off - ereq_on) - (emax_off - emax_on)

    if round_output:
        ereq_on = np.around(ereq_on, 1)
        emax_on = np.around(emax_on, 1)
        ereq_off = np.around(ereq_off, 1)
        emax_off = np.around(emax_off, 1)
        heat_storage = np.around(heat_storage, 1)

    return BFURestResult(
        heat_storage=heat_storage,
        e_req_fan=ereq_on,
        e_max_fan=emax_on,
        e_req_no_fan=ereq_off,
        e_max_no_fan=emax_off,
        group=group_key,
    )


def BFU_rest(
    tdb: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    group: str = BFU_rest_Groups.young.value,
    round_output: bool = True,
) -> BFURestResult:
    """Backward-compatible wrapper for ``morris_lph_fans``."""

    return morris_lph_fans(
        tdb=tdb,
        rh=rh,
        group=group,
        round_output=round_output,
    )
