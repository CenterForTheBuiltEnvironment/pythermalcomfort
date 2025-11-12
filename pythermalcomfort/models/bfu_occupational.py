from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import BFU_occupational_Inputs
from pythermalcomfort.classes_return import BFU_occupational as BFUOccupationalResult


def bfu_occupational(
    tdb: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    tr: float | list[float] | np.ndarray | None = None,
    clo: float | list[float] | np.ndarray = 0.28,
    met: float | list[float] | np.ndarray = 125.0,
    fan_velocity: float | list[float] | np.ndarray = 3.5,
    fan_off_velocity: float | list[float] | np.ndarray = 0.2,
    position: str = "standing",
    activity: str = "walk",
    speed: float | list[float] | np.ndarray = 1.66,
    fixed_sweat_rate_lph: float | list[float] | np.ndarray | None = 1.0,
    body_surface_area: float | list[float] | np.ndarray = 1.81,
    round_output: bool = True,
) -> BFUOccupationalResult:
    """Compute the potential benefit of fans in an occupational context using the
    validated model.

    Parameters
    ----------
    tdb : float or list of floats
        Dry-bulb air temperature, :math:`°C`.
    rh : float or list of floats
        Relative humidity, :math:`%`.
    tr : float or list of floats, optional
        Mean radiant temperature, :math:`°C`. If omitted or equal to ``999`` it is assumed equal to ``tdb``.
    clo : float or list of floats, optional
        Clothing insulation, [clo]. Defaults to 0.28.
    met : float or list of floats, optional
        Metabolic rate, [W/m²]. Defaults to 125.0.
    fan_velocity : float or list of floats, optional
        Air speed supplied by the fan, [m/s]. Defaults to 3.5.
    fan_off_velocity : float or list of floats, optional
        Background air speed without the fan, [m/s]. Defaults to 0.2.
    position : str, optional
        Worker posture. Use ``"standing"`` or ``"seated"``. Defaults to ``"standing"``.
    activity : str, optional
        Activity classification. Use ``"walk"`` to apply the walking coefficients, ``"rest"`` to remove additional airflow from movement, and any other value will default to the cycling activity style, interpretting speed as RPM. Defaults to ``"walk"``.
    speed : float or list of floats, optional
        Worker speed, [m/s]. Defaults to 1.66.
    fixed_sweat_rate_lph : float or list of floats, optional
        Fixed sweat production rate, [L/h]. If ``None`` the empirical humidity function defined by Foster is used. Defaults to ``1.0``.
    body_surface_area : float or list of floats, optional
        Body surface area, [m²]. Defaults to 1.81.
    round_output : bool, optional
        If ``True`` numeric outputs are rounded (storage components to one decimal, efficiencies to two decimals). Defaults to ``True``.

    Returns
    -------
    BFU_occupational
        A dataclass containing fan-on / fan-off storage rates, their components, and
        an interpretation code where ``1`` indicates beneficial fan use, ``0`` neutral,
        and ``-1`` harmful.
    """
    inputs = BFU_occupational_Inputs(
        tdb=tdb,
        rh=rh,
        tr=tr,
        clo=clo,
        met=met,
        fan_velocity=fan_velocity,
        fan_off_velocity=fan_off_velocity,
        position=position,
        activity=activity,
        speed=speed,
        fixed_sweat_rate_lph=fixed_sweat_rate_lph,
        body_surface_area=body_surface_area,
        round_output=round_output,
    )

    tdb_arr = np.asarray(inputs.tdb, dtype=float)
    tr_arr = np.asarray(inputs.tr, dtype=float)
    rh_arr = np.asarray(inputs.rh, dtype=float)
    clo_arr = np.asarray(inputs.clo, dtype=float)
    met_arr = np.asarray(inputs.met, dtype=float)
    bsa_arr = np.asarray(inputs.body_surface_area, dtype=float)
    speed_arr = np.asarray(inputs.speed, dtype=float)
    fan_on_arr = np.asarray(inputs.fan_velocity, dtype=float)
    fan_off_arr = np.asarray(inputs.fan_off_velocity, dtype=float)

    (
        tdb_arr,
        tr_arr,
        rh_arr,
        clo_arr,
        met_arr,
        bsa_arr,
        speed_arr,
        fan_on_arr,
        fan_off_arr,
    ) = np.broadcast_arrays(
        tdb_arr,
        tr_arr,
        rh_arr,
        clo_arr,
        met_arr,
        bsa_arr,
        speed_arr,
        fan_on_arr,
        fan_off_arr,
    )

    shape = tdb_arr.shape

    fixed_sweat = inputs.fixed_sweat_rate_lph
    if fixed_sweat is not None:
        fixed_sweat_arr = np.broadcast_to(
            np.asarray(fixed_sweat, dtype=float),
            shape,
        )
    else:
        fixed_sweat_arr = None

    position_arr = np.broadcast_to(
        np.asarray(inputs.position, dtype=object),
        shape,
    )
    activity_arr = np.broadcast_to(
        np.asarray(inputs.activity, dtype=object),
        shape,
    )

    is_seated = position_arr == "seated"
    is_walk = activity_arr == "walk"
    is_rest = activity_arr == "rest"
    effective_speed = np.where(is_rest, 0.0, speed_arr)

    # Base physiological and physical properties
    vo = np.where(is_seated, 0.07, 0.11)
    vact = np.where(
        is_walk,
        0.67 * effective_speed,
        0.0043 * effective_speed,
    )
    emissivity = 0.95
    sigma = 5.67e-8
    ar_ad = np.where(is_seated, 0.70, 0.77)

    tsk = 25.883 + (0.23 * tdb_arr) + (0.024 * rh_arr) - (0.304 * clo_arr)
    hr_kelvin = 273.2 + (tsk + tr_arr) / 2.0
    hr = 4.0 * emissivity * sigma * ar_ad * (hr_kelvin**3)

    def _hc_from(vwind: np.ndarray) -> np.ndarray:
        veff = vo + vwind + vact
        return 8.3 * np.sqrt(np.maximum(veff, 0.0))

    hc_off = _hc_from(fan_off_arr)
    t_operative = (hr * tr_arr + hc_off * tdb_arr) / (hr + hc_off)

    icl = 0.155 * clo_arr
    fcl = 1.0 + 1.81 * icl
    ia_off = 1.0 / np.maximum(hc_off + hr, 1e-9)
    it_off = icl + ia_off / fcl

    psat_ta = np.exp(18.956 - 4030.18 / (tdb_arr + 235.0)) / 10.0
    pa = psat_ta * (rh_arr / 100.0)
    psat_tsk = np.exp(18.956 - 4030.18 / (tsk + 235.0)) / 10.0

    if fixed_sweat_arr is None:
        gph = (459.5 - 1000.0) * np.exp(-0.519 * pa) + 1000.0
        lph = np.maximum(gph, 0.0) / 1000.0
    else:
        lph = np.asarray(fixed_sweat_arr, dtype=float)

    swmax = (lph * 1000.0) / np.maximum(bsa_arr, 1e-9)
    swmax = (swmax * 2430.0) / 3600.0

    recl_off = icl * 0.18
    rea_off = 1.0 / np.maximum(16.5 * hc_off, 1e-9)
    ret_off = recl_off + rea_off / fcl

    resp = (0.0014 * met_arr * (34.0 - tdb_arr)) + (0.0173 * met_arr * (5.87 - pa))

    def _state(vwind: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return storage, dry, evaporative, efficiency, wettedness for a fan state."""
        hc = _hc_from(vwind)

        corr_it = np.exp(
            -0.281 * (vwind - 0.15)
            + 0.044 * (vwind - 0.15) ** 2
            - 0.492 * effective_speed
            + 0.176 * effective_speed**2,
        )
        corr_ia = np.exp(
            -0.533 * (vwind - 0.15)
            + 0.069 * (vwind - 0.15) ** 2
            - 0.462 * effective_speed
            + 0.201 * effective_speed**2,
        )

        clothed_it = it_off * corr_it
        nude_it = ia_off * corr_ia
        itr = np.where(
            (icl > 0) & (icl < 0.093),
            ((0.093 - icl) * nude_it + icl * clothed_it) / 0.093,
            clothed_it,
        )
        dry = (tsk - t_operative) / np.maximum(itr, 1e-9)

        hedyn = 16.5 * hc
        corr_ret = np.exp(
            -0.468 * (vwind - 0.15)
            + 0.080 * (vwind - 0.15) ** 2
            - 0.87 * effective_speed
            + 0.358 * effective_speed**2,
        )
        clothed_ret = ret_off * corr_ret
        nude_dyn = 1.0 / np.maximum(hedyn, 1e-9)
        ret_r = np.where(
            icl <= 1e-9,
            nude_dyn,
            np.where(
                (icl > 0) & (icl < 0.093),
                ((0.093 - icl) * nude_dyn + icl * clothed_ret) / 0.093,
                clothed_ret,
            ),
        )

        emax = np.maximum(psat_tsk - pa, 0.0) / np.maximum(ret_r, 1e-9)

        wettedness = np.divide(
            swmax,
            emax,
            out=np.full_like(emax, np.inf),
            where=emax > 1e-12,
        )

        sweff = np.where(
            wettedness <= 1.0,
            1.0 - ((wettedness**2) / 2),
            np.where(
                wettedness <= 1.7,
                ((2.0 - wettedness) ** 2) / 2,
                0.05,
            ),
        )
        sweff = np.clip(sweff, 0.0, 1.0)
        esweat = swmax * sweff
        storage = met_arr - (dry + esweat + resp)

        return storage, dry, esweat, sweff, wettedness

    storage_fan, dry_fan, evap_fan, eff_fan, _ = _state(fan_on_arr)
    storage_off, dry_off, evap_off, eff_off, _ = _state(fan_off_arr)

    delta_storage_raw = storage_fan - storage_off
    interpretation = np.select(
        [delta_storage_raw < -50, delta_storage_raw <= 50],
        [1, 0],
        default=-1,
    )
    if isinstance(interpretation, np.ndarray) and interpretation.shape == ():
        interpretation = interpretation.item()

    if round_output:
        storage_fan = np.around(storage_fan, 1)
        storage_off = np.around(storage_off, 1)
        delta_storage = np.around(delta_storage_raw, 1)
        dry_fan = np.around(dry_fan, 1)
        dry_off = np.around(dry_off, 1)
        evap_fan = np.around(evap_fan, 1)
        evap_off = np.around(evap_off, 1)
        resp = np.around(resp, 1)
        eff_fan = np.around(eff_fan, 2)
        eff_off = np.around(eff_off, 2)
    else:
        delta_storage = delta_storage_raw

    return BFUOccupationalResult(
        storage_fan=storage_fan,
        storage_no_fan=storage_off,
        delta_storage=delta_storage,
        dry_heat_fan=dry_fan,
        dry_heat_no_fan=dry_off,
        evaporative_heat_fan=evap_fan,
        evaporative_heat_no_fan=evap_off,
        respiratory_heat=resp,
        sweat_efficiency_fan=eff_fan,
        sweat_efficiency_no_fan=eff_off,
        interpretation=interpretation,
    )


def BFU_occupational(
    tdb: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    tr: float | list[float] | np.ndarray | None = None,
    clo: float | list[float] | np.ndarray = 0.28,
    met: float | list[float] | np.ndarray = 125.0,
    fan_velocity: float | list[float] | np.ndarray = 3.5,
    fan_off_velocity: float | list[float] | np.ndarray = 0.2,
    position: str = "standing",
    activity: str = "walk",
    speed: float | list[float] | np.ndarray = 1.66,
    fixed_sweat_rate_lph: float | list[float] | np.ndarray | None = 1.0,
    body_surface_area: float | list[float] | np.ndarray = 1.81,
    round_output: bool = True,
) -> BFUOccupationalResult:
    """Backward-compatible wrapper for ``bfu_occupational``."""

    return bfu_occupational(
        tdb=tdb,
        rh=rh,
        tr=tr,
        clo=clo,
        met=met,
        fan_velocity=fan_velocity,
        fan_off_velocity=fan_off_velocity,
        position=position,
        activity=activity,
        speed=speed,
        fixed_sweat_rate_lph=fixed_sweat_rate_lph,
        body_surface_area=body_surface_area,
        round_output=round_output,
    )
