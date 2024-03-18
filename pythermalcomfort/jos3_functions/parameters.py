# -*- coding: utf-8 -*-
"""This code defines a dataclass for default input and a dictionary called ALL_OUT_PARAMS that contains
information about various output parameters related to human body properties,
heat exchange, and environmental conditions.

It also includes a function called show_outparam_docs() that generates a formatted string with the documentation
of the output parameters.

The show_outparam_docs() function uses text wrapping to create a readable documentation string
for both regular output parameters and extra output parameters.

It sorts the parameters alphabetically by key and formats each line with the parameter's name, meaning, and unit.
The resulting documentation string can be displayed or printed for user reference.
"""
from typing import List

import textwrap
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class Default:
    # Body information
    height: float = 1.72  # [m]
    weight: float = 74.43  # [kg]
    age: int = 20  # [-]
    body_fat: float = 15  # [%]
    cardiac_index: float = 2.59  # [L/min/m2]
    blood_flow_rate: int = 290  # [L/h]
    physical_activity_ratio: float = 1.25  # [-]
    metabolic_rate: float = 1.0  # [met]
    sex: str = "male"
    posture: str = "standing"
    bmr_equation: str = "harris-benedict"
    bsa_equation: str = "dubois"
    local_bsa: ClassVar[List[float]] = np.array(  # body surface area [m2]
        [
            0.110,
            0.029,
            0.175,
            0.161,
            0.221,
            0.096,
            0.063,
            0.050,
            0.096,
            0.063,
            0.050,
            0.209,
            0.112,
            0.056,
            0.209,
            0.112,
            0.056,
        ]
    )

    # Environment information
    core_temperature: float = 37  # [°C]
    skin_temperature: float = 34  # [°C]
    other_body_temperature: float = 36  # [°C]
    dry_bulb_air_temperature: float = 28.8  # [°C]
    mean_radiant_temperature: float = 28.8  # [°C]
    relative_humidity: float = 50  # [%]
    air_speed: float = 0.1  # [m/s]
    # Clothing information
    clothing_insulation: float = 0  # [clo]
    clothing_vapor_permeation_efficiency: float = 0.45  # [-]
    lewis_rate = 16.5  # [K/kPa]


ALL_OUT_PARAMS = {
    "age": {"ex_output": True, "meaning": "age", "suffix": None, "unit": "years"},
    "bf_ava_foot": {
        "ex_output": True,
        "meaning": "AVA blood flow rate of one foot",
        "suffix": None,
        "unit": "L/h",
    },
    "bf_ava_hand": {
        "ex_output": True,
        "meaning": "AVA blood flow rate of one hand",
        "suffix": None,
        "unit": "L/h",
    },
    "bf_core": {
        "ex_output": True,
        "meaning": "core blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_fat": {
        "ex_output": True,
        "meaning": "fat blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_muscle": {
        "ex_output": True,
        "meaning": "muscle blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_skin": {
        "ex_output": True,
        "meaning": "skin blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bsa": {
        "ex_output": True,
        "meaning": "body surface area (each body part)",
        "suffix": "Body name",
        "unit": "m2",
    },
    "cardiac_output": {
        "ex_output": False,
        "meaning": "cardiac output (the sum of the whole blood flow)",
        "suffix": None,
        "unit": "L/h",
    },
    "cycle_time": {
        "ex_output": False,
        "meaning": "the counts of executing one cycle calculation",
        "suffix": None,
        "unit": "-",
    },
    "e_max": {
        "ex_output": True,
        "meaning": "maximum evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "e_skin": {
        "ex_output": True,
        "meaning": "evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "e_sweat": {
        "ex_output": True,
        "meaning": (
            "evaporative heat loss from the skin by only sweating (each body part)"
        ),
        "suffix": "Body name",
        "unit": "W",
    },
    "fat": {"ex_output": True, "meaning": "body fat rate", "suffix": None, "unit": "%"},
    "height": {
        "ex_output": True,
        "meaning": "body height",
        "suffix": None,
        "unit": "m",
    },
    "clo": {
        "ex_output": True,
        "meaning": "clothing insulation (each body part)",
        "suffix": "Body name",
        "unit": "clo",
    },
    "q_skin2env_latent": {
        "ex_output": True,
        "meaning": "latent heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_core": {
        "ex_output": True,
        "meaning": "core thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_fat": {
        "ex_output": True,
        "meaning": "fat thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_muscle": {
        "ex_output": True,
        "meaning": "muscle thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_skin": {
        "ex_output": True,
        "meaning": "skin thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_total": {
        "ex_output": False,
        "meaning": "total thermogenesis of the whole body",
        "suffix": None,
        "unit": "W",
    },
    "q_nst": {
        "ex_output": True,
        "meaning": "core thermogenesis by non-shivering (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "simulation_time": {
        "ex_output": False,
        "meaning": "simulation times",
        "suffix": None,
        "unit": "sec",
    },
    "q_shiv": {
        "ex_output": True,
        "meaning": "core or muscle thermogenesis by shivering (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_work": {
        "ex_output": True,
        "meaning": "core or muscle thermogenesis by work (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "name": {
        "ex_output": True,
        "meaning": "name of the model",
        "suffix": None,
        "unit": "-",
    },
    "par": {
        "ex_output": True,
        "meaning": "physical activity ratio",
        "suffix": None,
        "unit": "-",
    },
    "q_thermogenesis_core": {
        "ex_output": True,
        "meaning": "core total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_fat": {
        "ex_output": True,
        "meaning": "fat total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_muscle": {
        "ex_output": True,
        "meaning": "muscle total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_skin": {
        "ex_output": True,
        "meaning": "skin total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_res": {
        "ex_output": False,
        "meaning": "heat loss by respiration",
        "suffix": None,
        "unit": "W",
    },
    "q_res_latent": {
        "ex_output": True,
        "meaning": "latent heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_res_sensible": {
        "ex_output": True,
        "meaning": "sensible heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "rh": {
        "ex_output": True,
        "meaning": "relative humidity (each body part)",
        "suffix": "Body name",
        "unit": "%",
    },
    "r_et": {
        "ex_output": True,
        "meaning": "total clothing evaporative heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "(m2*kPa)/W",
    },
    "r_t": {
        "ex_output": True,
        "meaning": "total clothing heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "(m2*K)/W",
    },
    "q_skin2env_sensible": {
        "ex_output": True,
        "meaning": "sensible heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "t_skin_set": {
        "ex_output": True,
        "meaning": "skin set point temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_core_set": {
        "ex_output": True,
        "meaning": "core set point temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "sex": {
        "ex_output": True,
        "meaning": "sex",
        "suffix": None,
        "unit": "-",
    },
    "q_skin2env": {
        "ex_output": False,
        "meaning": "total heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "tdb": {
        "ex_output": True,
        "meaning": "dry bulb air temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_artery": {
        "ex_output": True,
        "meaning": "arterial temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_cb": {
        "ex_output": True,
        "meaning": "central blood temperature",
        "suffix": None,
        "unit": "°C",
    },
    "t_core": {
        "ex_output": False,
        "meaning": "core temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_fat": {
        "ex_output": True,
        "meaning": "fat temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_muscle": {
        "ex_output": True,
        "meaning": "muscle temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "to": {
        "ex_output": True,
        "meaning": "operative temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "tr": {
        "ex_output": True,
        "meaning": "mean radiant temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_skin": {
        "ex_output": False,
        "meaning": "skin temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_skin_mean": {
        "ex_output": False,
        "meaning": "mean skin temperature",
        "suffix": None,
        "unit": "°C",
    },
    "t_superficial_vein": {
        "ex_output": True,
        "meaning": "superficial vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_vein": {
        "ex_output": True,
        "meaning": "vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "v": {
        "ex_output": True,
        "meaning": "air velocity (each body part)",
        "suffix": "Body name",
        "unit": "m/s",
    },
    "weight": {
        "ex_output": True,
        "meaning": "body weight",
        "suffix": None,
        "unit": "kg",
    },
    "w": {
        "ex_output": False,
        "meaning": "skin wettedness (each body part)",
        "suffix": "Body name",
        "unit": "-",
    },
    "w_mean": {
        "ex_output": False,
        "meaning": "mean skin wettedness",
        "suffix": None,
        "unit": "-",
    },
    "weight_loss_by_evap_and_res": {
        "ex_output": False,
        "meaning": "weight loss by the evaporation and respiration of the whole body",
        "suffix": None,
        "unit": "g/sec",
    },
    "dt": {
        "ex_output": False,
        "meaning": "time step",
        "suffix": None,
        "unit": "sec",
    },
    "pythermalcomfort_version": {
        "ex_output": False,
        "meaning": "version of pythermalcomfort",
        "suffix": None,
        "unit": "-",
    },
}


def show_out_param_docs():
    """Show the documentation of the output parameters.

    Returns
    -------
    docstring : str
        Text of the documentation of the output parameters
    """

    outparams = textwrap.dedent(
        """
        Output parameters
        -------
        """
    )

    exoutparams = textwrap.dedent(
        """
        Extra output parameters
        -------
        """
    )

    sortkeys = list(ALL_OUT_PARAMS.keys())
    sortkeys.sort()
    for key in sortkeys:
        value = ALL_OUT_PARAMS[key]

        line = "{}: {} [{}]".format(key.ljust(8), value["meaning"], value["unit"])

        if value["ex_output"]:
            exoutparams += line + "\n"
        else:
            outparams += line + "\n"

    docs = outparams + "\n" + exoutparams
    docs = textwrap.indent(docs.strip(), "    ")

    return docs


if __name__ == "__main__":
    print(show_out_param_docs())
