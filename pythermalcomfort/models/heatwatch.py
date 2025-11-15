from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from pythermalcomfort.classes_return import HeatWatch
from pythermalcomfort.models import two_nodes_gagge


def calculate_w_max(persona: Persona) -> float:
    w_max = 1.0
    if persona.age >= 80:
        w_max *= 0.65
    elif persona.age >= 70:
        w_max *= 0.7
    elif persona.age >= 60:
        w_max *= 0.75
    elif persona.age >= 40:
        w_max *= 0.8
    elif persona.age >= 18:
        w_max *= 0.85
    else:
        w_max *= 0.9

    if persona.obesity:
        w_max *= 0.95
    if persona.medication:
        w_max *= 0.95
    if persona.pregnancy:
        w_max *= 0.95
    if persona.mobility_impairment:
        w_max *= 0.95
    if persona.homelessness:
        pass  # no effect on w_max
    if persona.illness:
        w_max *= 0.95

    return w_max


def calculate_coefficient_disc_m_bl(persona: Persona) -> float:
    coefficient = 1.0
    if persona.age >= 80:
        min(coefficient, 0.3)
    elif persona.age >= 70:
        min(coefficient, 0.4)
    elif persona.age >= 60:
        min(coefficient, 0.5)
    elif persona.age >= 40:
        min(coefficient, 0.55)
    elif persona.age >= 18:
        min(coefficient, 0.7)
    else:
        min(coefficient, 0.6)

    if persona.obesity:
        min(coefficient, 0.6)
    if persona.homelessness:
        min(coefficient, 0.5)
    if persona.illness:
        min(coefficient, 0.6)

    return coefficient


@dataclass()
class Persona:
    age: int
    obesity: bool
    medication: bool
    pregnancy: bool = False
    mobility_impairment: bool = False
    homelessness: bool = False
    illness: bool = False


def heatwatch(
    tdb: float | list[float] | np.ndarray,
    tr: float | list[float] | np.ndarray,
    v: float | list[float] | np.ndarray,
    rh: float | list[float] | np.ndarray,
    met: float | list[float] | np.ndarray,
    clo: float | list[float] | np.ndarray,
    persona: Persona,
):
    """
    Calculate the HeatWatch thermal comfort index.

    The HeatWatch model assesses thermal comfort based on environmental conditions
    and individual characteristics defined in the `persona` dataclass. It combines
    the discomfort index and skin blood flow to provide a personalized heat stress score.
    The model is based on the two-node Gagge thermal regulation model. Adjustments
    are made based on the personal attributes that may affect the body's response to heat.

    Parameters
    ----------
    tdb : float
        Dry-bulb air temperature in °C.
    tr : float
        Mean radiant temperature in °C.
    v : float
        Relative air velocity in m/s.
    rh : float
        Relative humidity in %.
    met : float
        Metabolic rate in met.
    clo : float
        Clothing insulation in clo.
    persona : Persona
        A dataclass defining the persona attributes.

    Returns
    -------
    heatwatch_index : float
        The HeatWatch thermal comfort index.
    """
    # # Validate inputs using the PMVPPDInputs class
    # PMVPPDInputs(
    #     tdb=tdb,
    #     tr=tr,
    #     v=v,
    #     rh=rh,
    #     met=met,
    #     clo=clo,
    #     wme=wme,
    #     units=units,
    #     limit_inputs=limit_inputs,
    # )

    tdb = np.asarray(tdb, dtype=np.float64)
    tr = np.asarray(tr, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    met = np.asarray(met, dtype=np.float64)
    clo = np.asarray(clo, dtype=np.float64)

    w_max = calculate_w_max(persona)
    coefficient_disc_m_bl = calculate_coefficient_disc_m_bl(persona)

    result = two_nodes_gagge(tdb=tdb, tr=tr, v=v, rh=rh, met=met, clo=clo, w_max=w_max, round_output=False)

    m_bl = result["m_bl"]
    disc = result["disc"]

    m_bl_min = 1.2  # min skin blood flow
    m_bl_max = 90  # max skin blood flow
    w_min = 0.06  # min skin wettedness
    tdb_reduce_risk_ratio = 0.7
    tdb_reduce_risk = 28.0  # °C
    disc_max = 5.0  # max discomfort index

    # Thermal discomfort
    ratio_disc = disc / disc_max

    # Skin blood flow
    ratio_m_bl = (m_bl - m_bl_min) / (m_bl_max - m_bl_min)

    # Combine discomfort index with the normalised skin blood flow
    disc_m_bl = ratio_disc * coefficient_disc_m_bl + ratio_m_bl * (
        1 - coefficient_disc_m_bl
    )

    # disc_m_bl should not be lower than ratio_disc
    disc_m_bl = np.where(disc_m_bl >= ratio_disc, disc_m_bl, ratio_disc)

    # Reduce the heat stress risk if the temperature is lower than assumption_tdb_reduce_risk
    disc_m_bl = np.where(
        tdb < tdb_reduce_risk,
        disc_m_bl * tdb_reduce_risk_ratio,
        disc_m_bl,
    )

    # limit the variable to map to 1
    disc_m_bl = np.where(disc_m_bl > 0.99, 0.99, disc_m_bl)

    hss = disc_m_bl * 6.0

    # # do we want to implement lower T limit?
    # if persona.lower_t_limit:
    #     hss = np.where(tdb.values < persona.lower_t_limit, 0.0, hss)

    # Are negative values of hss sensible? Set to zero?
    hss = np.where(hss < 0.0, 0.0, hss)

    risk_categories = {
        1: "Minimal",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Severe",
        6: "Extreme",
    }

    return HeatWatch(
        hss=hss, risk_category=np.vectorize(risk_categories.get)(np.ceil(hss))
    )


if __name__ == "__main__":
    # Example usage
    # from pythermalcomfort.heatwatch import Persona, heatwatch

    tdb = [30, 35, 40]
    tr = [32, 37, 42]
    v = [0.3, 0.5, 0.7]
    rh = [50, 60, 70]
    met = 1.2
    clo = 0.5

    p = Persona(
        age=65,
        obesity=True,
        medication=False,
        pregnancy=False,
        mobility_impairment=False,
        homelessness=False,
        illness=True,
    )

    results = heatwatch(tdb, tr, v, rh, met, clo, p)
    print(results)
