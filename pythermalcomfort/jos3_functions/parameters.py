"""This code defines a dataclass for default input and a dictionary called
ALL_OUT_PARAMS that contains information about various output parameters
related to human body properties, heat exchange, and environmental conditions.

It also includes a function called show_outparam_docs() that generates a
formatted string with the documentation of the output parameters.

The show_outparam_docs() function uses text wrapping to create a
readable documentation string for both regular output parameters and
extra output parameters.

It sorts the parameters alphabetically by key and formats each line with
the parameter's name, meaning, and unit. The resulting documentation
string can be displayed or printed for user reference.
"""

import re
import textwrap
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# import pandas as pd
from pythermalcomfort.classes_return import JOS3BodyParts
from pythermalcomfort.utilities import BodySurfaceAreaEquations, Postures, Sex


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
    sex: str = Sex.male.value
    posture: str = Postures.standing.value
    bmr_equation: str = "harris-benedict"
    bsa_equation: str = BodySurfaceAreaEquations.dubois.value
    local_bsa: ClassVar[list[float]] = np.array(  # body surface area [m2]
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
    lewis_rate: float = 16.5  # [K/kPa]
    num_body_parts: int = 17  # [-]


ALL_OUT_PARAMS = {
    "age": {"meaning": "age", "suffix": None, "unit": "years"},
    "bf_ava_foot": {
        "meaning": "AVA blood flow rate of one foot",
        "suffix": None,
        "unit": "L/h",
    },
    "bf_ava_hand": {
        "meaning": "AVA blood flow rate of one hand",
        "suffix": None,
        "unit": "L/h",
    },
    "bf_core": {
        "meaning": "core blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_fat": {
        "meaning": "fat blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_muscle": {
        "meaning": "muscle blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bf_skin": {
        "meaning": "skin blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "bsa": {
        "meaning": "body surface area (each body part)",
        "suffix": "Body name",
        "unit": "m2",
    },
    "cardiac_output": {
        "meaning": "cardiac output (the sum of the whole blood flow)",
        "suffix": None,
        "unit": "L/h",
    },
    "e_max": {
        "meaning": "maximum evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "e_skin": {
        "meaning": "evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "e_sweat": {
        "meaning": (
            "evaporative heat loss from the skin by only sweating (each body part)"
        ),
        "suffix": "Body name",
        "unit": "W",
    },
    "fat": {"meaning": "body fat rate", "suffix": None, "unit": "%"},
    "height": {
        "meaning": "body height",
        "suffix": None,
        "unit": "m",
    },
    "clo": {
        "meaning": "clothing insulation (each body part)",
        "suffix": "Body name",
        "unit": "clo",
    },
    "q_skin2env_latent": {
        "meaning": "latent heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_core": {
        "meaning": "core thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_fat": {
        "meaning": "fat thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_muscle": {
        "meaning": "muscle thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_bmr_skin": {
        "meaning": "skin thermogenesis by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_total": {
        "meaning": "total thermogenesis of the whole body",
        "suffix": None,
        "unit": "W",
    },
    "q_nst": {
        "meaning": "core thermogenesis by non-shivering (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "simulation_time": {
        "meaning": "simulation times",
        "suffix": None,
        "unit": "sec",
    },
    "q_shiv": {
        "meaning": "core or muscle thermogenesis by shivering (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_work": {
        "meaning": "core or muscle thermogenesis by work (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "name": {
        "meaning": "name of the model",
        "suffix": None,
        "unit": "-",
    },
    "par": {
        "meaning": "physical activity ratio",
        "suffix": None,
        "unit": "-",
    },
    "q_thermogenesis_core": {
        "meaning": "core total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_fat": {
        "meaning": "fat total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_muscle": {
        "meaning": "muscle total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_thermogenesis_skin": {
        "meaning": "skin total thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_res": {
        "meaning": "heat loss by respiration",
        "suffix": None,
        "unit": "W",
    },
    "q_res_latent": {
        "meaning": "latent heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "q_res_sensible": {
        "meaning": "sensible heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "rh": {
        "meaning": "relative humidity (each body part)",
        "suffix": "Body name",
        "unit": "%",
    },
    "r_et": {
        "meaning": "total clothing evaporative heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "(m2*kPa)/W",
    },
    "r_t": {
        "meaning": "total clothing heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "(m2*K)/W",
    },
    "q_skin2env_sensible": {
        "meaning": "sensible heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "t_skin_set": {
        "meaning": "skin set point temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_core_set": {
        "meaning": "core set point temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "sex": {
        "meaning": "sex",
        "suffix": None,
        "unit": "-",
    },
    "q_skin2env": {
        "meaning": "total heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "tdb": {
        "meaning": "dry bulb air temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_artery": {
        "meaning": "arterial temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_cb": {
        "meaning": "central blood temperature",
        "suffix": None,
        "unit": "°C",
    },
    "t_core": {
        "meaning": "core temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_fat": {
        "meaning": "fat temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_muscle": {
        "meaning": "muscle temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "to": {
        "meaning": "operative temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "tr": {
        "meaning": "mean radiant temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_skin": {
        "meaning": "skin temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_skin_mean": {
        "meaning": "mean skin temperature",
        "suffix": None,
        "unit": "°C",
    },
    "t_superficial_vein": {
        "meaning": "superficial vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "t_vein": {
        "meaning": "vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "°C",
    },
    "v": {
        "meaning": "air velocity (each body part)",
        "suffix": "Body name",
        "unit": "m/s",
    },
    "weight": {
        "meaning": "body weight",
        "suffix": None,
        "unit": "kg",
    },
    "w": {
        "meaning": "skin wettedness (each body part)",
        "suffix": "Body name",
        "unit": "-",
    },
    "w_mean": {
        "meaning": "mean skin wettedness",
        "suffix": None,
        "unit": "-",
    },
    "weight_loss_by_evap_and_res": {
        "meaning": "weight loss by the evaporation and respiration of the whole body",
        "suffix": None,
        "unit": "g/sec",
    },
    "dt": {
        "meaning": "time step",
        "suffix": None,
        "unit": "sec",
    },
    "pythermalcomfort_version": {
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

        outparams += line + "\n"

    docs = outparams + "\n" + exoutparams
    docs = textwrap.indent(docs.strip(), "    ")

    return docs


# commented out this function since needs pandas
# def convert_and_print_local_clo_values_from_csv_to_dict(csv_name):
#     """Parameters
#     ----------
#     csv_name : file path that you want to convert (it should be in the same folder as this function)
#
#     Returns
#     -------
#     local_clo_dict : a dictionary including local clothing insulation values as well as that for the whole body
#
#     Notes
#     -----
#     The 17 body segments correspond to the JOS-3 model.
#
#     """
#     # Read the Excel file
#     df = pd.read_csv(csv_name)
#
#     # Create an empty dictionary
#     local_clo_dict = {}
#
#     # Add data from each row to the dictionary
#     for _index, row in df.iterrows():
#         # Get the name of the clothing combination as the key
#         key = row["clothing_ensemble"]
#         # Create a dictionary as the value
#         value = {"whole_body": row["whole_body"], "local_body_part": {}}
#         # Add data from each column to the Local body dictionary
#         for col in df.columns[2:]:
#             # Get the name of the body part
#             body_part = col
#             # Get the clo value for the local body part and add it to the Local body dictionary
#             clo = row[col]
#             value["local_body_part"][body_part] = clo
#         # Add the data to the dictionary
#         local_clo_dict[key] = value
#
#     return local_clo_dict


def add_prompt_to_code(code: str, prompt: str = ">>> ") -> str:
    lines = code.strip().split("\n")
    result = []
    for line in lines:
        if re.match(r"^\s*#", line):  # If it's a comment line
            result.append(line)
        else:
            result.append(prompt + line)
    return "\n".join(result)


# This dictionary contains the local and the whole body clothing insulation of typical clothing ensemble.
# It is based on the study by Juyoun et al. (https://escholarship.org/uc/item/18f0r375)
# and by Nomoto et al. (https://doi.org/10.1002/2475-8876.12124)
# Please note that value for the neck is the same as the measured value for the head
# and it does not take into account the insulation effect of the hair.
# Typically, the clothing insulation for the hair are quantified by assuming a head covering of approximately 0.6 to 1.0 clo.

local_clo_typical_ensembles = {
    "nude (mesh chair)": {
        "whole_body": 0.01,
        "local_body_part": JOS3BodyParts(
            head=0.13,
            neck=0.13,
            chest=0.01,
            back=0.01,
            pelvis=0.04,
            left_shoulder=0.02,
            left_arm=0.0,
            left_hand=0.01,
            right_shoulder=0.02,
            right_arm=0.0,
            right_hand=0.01,
            left_thigh=0.01,
            left_leg=0.03,
            left_foot=0.05,
            right_thigh=0.01,
            right_leg=0.03,
            right_foot=0.05,
        ),
    },
    "nude (nude chair)": {
        "whole_body": -0.02,
        "local_body_part": JOS3BodyParts(
            head=0.13,
            neck=0.13,
            chest=0.05,
            back=-0.14,
            pelvis=-0.01,
            left_shoulder=-0.01,
            left_arm=-0.01,
            left_hand=-0.02,
            right_shoulder=-0.01,
            right_arm=-0.01,
            right_hand=-0.02,
            left_thigh=-0.1,
            left_leg=0.0,
            left_foot=0.0,
            right_thigh=-0.1,
            right_leg=0.0,
            right_foot=0.0,
        ),
    },
    "panty": {
        "whole_body": 0.03,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.0,
            back=0.0,
            pelvis=0.24,
            left_shoulder=0.0,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.0,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.05,
            left_leg=0.0,
            left_foot=0.05,
            right_thigh=0.05,
            right_leg=0.0,
            right_foot=0.05,
        ),
    },
    "bra+panty": {
        "whole_body": 0.05,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.22,
            back=0.0,
            pelvis=0.18,
            left_shoulder=0.0,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.0,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.03,
            left_leg=0.03,
            left_foot=0.08,
            right_thigh=0.03,
            right_leg=0.03,
            right_foot=0.08,
        ),
    },
    "bra+panty, tanktop, shorts, sandals": {
        "whole_body": 0.22,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.57,
            back=0.27,
            pelvis=0.92,
            left_shoulder=0.04,
            left_arm=0.02,
            left_hand=0.02,
            right_shoulder=0.04,
            right_arm=0.02,
            right_hand=0.02,
            left_thigh=0.51,
            left_leg=0.01,
            left_foot=0.38,
            right_thigh=0.51,
            right_leg=0.01,
            right_foot=0.38,
        ),
    },
    "bra+panty, long-sleeve shirt, shorts, sandals": {
        "whole_body": 0.43,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.43,
            back=1.02,
            pelvis=1.45,
            left_shoulder=0.29,
            left_arm=0.22,
            left_hand=0.01,
            right_shoulder=0.29,
            right_arm=0.22,
            right_hand=0.01,
            left_thigh=0.57,
            left_leg=0.01,
            left_foot=0.4,
            right_thigh=0.57,
            right_leg=0.01,
            right_foot=0.4,
        ),
    },
    "bra+panty, sleeveless dress, sandals": {
        "whole_body": 0.29,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.85,
            back=0.48,
            pelvis=0.94,
            left_shoulder=0.0,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.0,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.72,
            left_leg=0.0,
            left_foot=0.41,
            right_thigh=0.72,
            right_leg=0.0,
            right_foot=0.41,
        ),
    },
    "bra+panty, T-shirt, long pants, socks, sneakers": {
        "whole_body": 0.52,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.14,
            back=0.84,
            pelvis=1.04,
            left_shoulder=0.42,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.42,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.58,
            left_leg=0.62,
            left_foot=0.82,
            right_thigh=0.58,
            right_leg=0.62,
            right_foot=0.82,
        ),
    },
    "bra+panty, sleeveless dress, cardigan, sandals": {
        "whole_body": 0.53,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.78,
            back=1.42,
            pelvis=1.19,
            left_shoulder=0.65,
            left_arm=0.41,
            left_hand=0.05,
            right_shoulder=0.65,
            right_arm=0.41,
            right_hand=0.05,
            left_thigh=0.77,
            left_leg=0.0,
            left_foot=0.39,
            right_thigh=0.77,
            right_leg=0.0,
            right_foot=0.39,
        ),
    },
    "bra+panty, song-sleeve dress, socks, sneakers": {
        "whole_body": 0.54,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.49,
            back=1.1,
            pelvis=0.91,
            left_shoulder=0.72,
            left_arm=0.58,
            left_hand=0.03,
            right_shoulder=0.72,
            right_arm=0.58,
            right_hand=0.03,
            left_thigh=0.73,
            left_leg=0.07,
            left_foot=0.77,
            right_thigh=0.73,
            right_leg=0.07,
            right_foot=0.77,
        ),
    },
    "bra+panty, long-sleeve dress, cardigan, socks, sneakers": {
        "whole_body": 0.67,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=2.05,
            back=1.32,
            pelvis=1.39,
            left_shoulder=1.14,
            left_arm=0.63,
            left_hand=0.04,
            right_shoulder=1.14,
            right_arm=0.63,
            right_hand=0.04,
            left_thigh=0.84,
            left_leg=0.05,
            left_foot=0.78,
            right_thigh=0.84,
            right_leg=0.05,
            right_foot=0.78,
        ),
    },
    "bra+panty, tank top, skirt, sandals": {
        "whole_body": 0.31,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.83,
            back=0.22,
            pelvis=0.99,
            left_shoulder=0.0,
            left_arm=0.0,
            left_hand=0.03,
            right_shoulder=0.0,
            right_arm=0.0,
            right_hand=0.03,
            left_thigh=0.88,
            left_leg=0.05,
            left_foot=0.44,
            right_thigh=0.88,
            right_leg=0.05,
            right_foot=0.44,
        ),
    },
    "bra+panty, long sleeve shirts, skirt, sandals": {
        "whole_body": 0.52,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.62,
            back=0.99,
            pelvis=1.41,
            left_shoulder=0.31,
            left_arm=0.28,
            left_hand=0.03,
            right_shoulder=0.31,
            right_arm=0.28,
            right_hand=0.03,
            left_thigh=0.82,
            left_leg=0.04,
            left_foot=0.41,
            right_thigh=0.82,
            right_leg=0.04,
            right_foot=0.41,
        ),
    },
    "bra+panty, dress shirts, skirt, stocking, formal shoes": {
        "whole_body": 0.62,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.58,
            back=0.99,
            pelvis=1.31,
            left_shoulder=0.91,
            left_arm=0.64,
            left_hand=0.04,
            right_shoulder=0.91,
            right_arm=0.64,
            right_hand=0.04,
            left_thigh=0.87,
            left_leg=0.05,
            left_foot=0.81,
            right_thigh=0.87,
            right_leg=0.05,
            right_foot=0.81,
        ),
    },
    "bra+panty, dress shirts, skirt, leggings, sandals": {
        "whole_body": 0.65,
        "local_body_part": JOS3BodyParts(
            head=0.13,
            neck=0.13,
            chest=1.59,
            back=1.04,
            pelvis=1.36,
            left_shoulder=0.91,
            left_arm=0.67,
            left_hand=0.07,
            right_shoulder=0.91,
            right_arm=0.67,
            right_hand=0.07,
            left_thigh=1.26,
            left_leg=0.12,
            left_foot=0.43,
            right_thigh=1.26,
            right_leg=0.12,
            right_foot=0.43,
        ),
    },
    "bra+panty, thin dress shirts, long pants, socks, sneakers": {
        "whole_body": 0.82,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=3.35,
            back=1.73,
            pelvis=1.63,
            left_shoulder=1.99,
            left_arm=1.49,
            left_hand=0.11,
            right_shoulder=1.99,
            right_arm=1.49,
            right_hand=0.11,
            left_thigh=0.6,
            left_leg=0.43,
            left_foot=0.68,
            right_thigh=0.6,
            right_leg=0.43,
            right_foot=0.68,
        ),
    },
    "bra+panty, long sleeve shirts, long pants, socks, sneakers": {
        "whole_body": 0.8,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=2.47,
            back=1.48,
            pelvis=1.58,
            left_shoulder=0.98,
            left_arm=0.58,
            left_hand=0.04,
            right_shoulder=0.98,
            right_arm=0.58,
            right_hand=0.04,
            left_thigh=0.69,
            left_leg=0.65,
            left_foot=0.89,
            right_thigh=0.69,
            right_leg=0.65,
            right_foot=0.89,
        ),
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, socks, sneakers": {
        "whole_body": 0.83,
        "local_body_part": JOS3BodyParts(
            head=0.25,
            neck=0.25,
            chest=3.88,
            back=2.28,
            pelvis=2.07,
            left_shoulder=1.89,
            left_arm=1.41,
            left_hand=0.16,
            right_shoulder=1.89,
            right_arm=1.41,
            right_hand=0.16,
            left_thigh=0.83,
            left_leg=0.66,
            left_foot=0.86,
            right_thigh=0.83,
            right_leg=0.66,
            right_foot=0.86,
        ),
    },
    "bra+panty, T-shirt, jeans, socks, sneakers": {
        "whole_body": 0.57,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.29,
            back=0.93,
            pelvis=1.3,
            left_shoulder=0.68,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.68,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.65,
            left_leg=0.47,
            left_foot=0.73,
            right_thigh=0.65,
            right_leg=0.47,
            right_foot=0.73,
        ),
    },
    "bra+panty, long sleeve shirts, jeans, socks, sneakers": {
        "whole_body": 0.74,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.58,
            back=0.98,
            pelvis=1.35,
            left_shoulder=0.86,
            left_arm=0.71,
            left_hand=0.07,
            right_shoulder=0.86,
            right_arm=0.71,
            right_hand=0.07,
            left_thigh=0.74,
            left_leg=0.48,
            left_foot=0.74,
            right_thigh=0.74,
            right_leg=0.48,
            right_foot=0.74,
        ),
    },
    "bra+panty, oxford shirts, long thin pants, socks, sneakers": {
        "whole_body": 0.83,
        "local_body_part": JOS3BodyParts(
            head=0.16,
            neck=0.16,
            chest=1.39,
            back=1.02,
            pelvis=1.34,
            left_shoulder=0.83,
            left_arm=0.69,
            left_hand=0.22,
            right_shoulder=0.83,
            right_arm=0.69,
            right_hand=0.22,
            left_thigh=1.02,
            left_leg=0.68,
            left_foot=0.8,
            right_thigh=1.02,
            right_leg=0.68,
            right_foot=0.8,
        ),
    },
    "bra+panty, thin dress shirts (roll-up), long pants, socks, sneakers": {
        "whole_body": 0.81,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=3.6,
            back=1.83,
            pelvis=1.71,
            left_shoulder=2.16,
            left_arm=1.49,
            left_hand=0.13,
            right_shoulder=2.16,
            right_arm=1.49,
            right_hand=0.13,
            left_thigh=0.64,
            left_leg=0.43,
            left_foot=0.69,
            right_thigh=0.64,
            right_leg=0.43,
            right_foot=0.69,
        ),
    },
    "bra+panty, T-shirt, short sleeve shirt, long pants, socks, sneakers": {
        "whole_body": 0.71,
        "local_body_part": JOS3BodyParts(
            head=0.12,
            neck=0.12,
            chest=2.15,
            back=1.4,
            pelvis=1.71,
            left_shoulder=1.22,
            left_arm=0.02,
            left_hand=0.05,
            right_shoulder=1.22,
            right_arm=0.02,
            right_hand=0.05,
            left_thigh=0.79,
            left_leg=0.48,
            left_foot=0.67,
            right_thigh=0.79,
            right_leg=0.48,
            right_foot=0.67,
        ),
    },
    "bra+panty, sports shirts, long pants, socks, sneakers": {
        "whole_body": 0.8,
        "local_body_part": JOS3BodyParts(
            head=0.05,
            neck=0.05,
            chest=1.92,
            back=1.31,
            pelvis=1.41,
            left_shoulder=1.14,
            left_arm=0.86,
            left_hand=0.18,
            right_shoulder=1.14,
            right_arm=0.86,
            right_hand=0.18,
            left_thigh=0.59,
            left_leg=0.49,
            left_foot=0.75,
            right_thigh=0.59,
            right_leg=0.49,
            right_foot=0.75,
        ),
    },
    "bra+panty, sports shirts, sports pants, sports socks, sports shoes": {
        "whole_body": 0.87,
        "local_body_part": JOS3BodyParts(
            head=0.07,
            neck=0.07,
            chest=1.87,
            back=1.17,
            pelvis=1.26,
            left_shoulder=1.2,
            left_arm=1.07,
            left_hand=0.09,
            right_shoulder=1.2,
            right_arm=1.07,
            right_hand=0.09,
            left_thigh=0.62,
            left_leg=0.77,
            left_foot=1.58,
            right_thigh=0.62,
            right_leg=0.77,
            right_foot=1.58,
        ),
    },
    "bra+panty, thin dress shirts, long pants, wool sweater, socks, sneakers": {
        "whole_body": 0.92,
        "local_body_part": JOS3BodyParts(
            head=0.09,
            neck=0.09,
            chest=2.39,
            back=1.64,
            pelvis=1.71,
            left_shoulder=1.36,
            left_arm=1.29,
            left_hand=0.21,
            right_shoulder=1.36,
            right_arm=1.29,
            right_hand=0.21,
            left_thigh=0.7,
            left_leg=0.52,
            left_foot=0.77,
            right_thigh=0.7,
            right_leg=0.52,
            right_foot=0.77,
        ),
    },
    "bra+panty, thin dress shirts, long pants, cashmere sweater, socks, sneakers": {
        "whole_body": 0.87,
        "local_body_part": JOS3BodyParts(
            head=0.1,
            neck=0.1,
            chest=2.4,
            back=1.72,
            pelvis=1.67,
            left_shoulder=1.33,
            left_arm=1.23,
            left_hand=0.08,
            right_shoulder=1.33,
            right_arm=1.23,
            right_hand=0.08,
            left_thigh=0.61,
            left_leg=0.47,
            left_foot=0.77,
            right_thigh=0.61,
            right_leg=0.47,
            right_foot=0.77,
        ),
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket, socks, sneakers": {
        "whole_body": 1.18,
        "local_body_part": JOS3BodyParts(
            head=0.65,
            neck=0.65,
            chest=5.26,
            back=3.07,
            pelvis=2.2,
            left_shoulder=3.14,
            left_arm=2.07,
            left_hand=0.08,
            right_shoulder=3.14,
            right_arm=2.07,
            right_hand=0.08,
            left_thigh=0.67,
            left_leg=0.54,
            left_foot=0.77,
            right_thigh=0.67,
            right_leg=0.54,
            right_foot=0.77,
        ),
    },
    "bra+panty, T-shirt, long sleeve shirts, jeans, sports jumper, socks, sneakers": {
        "whole_body": 1.07,
        "local_body_part": JOS3BodyParts(
            head=0.28,
            neck=0.28,
            chest=3.99,
            back=2.12,
            pelvis=2.0,
            left_shoulder=1.7,
            left_arm=1.36,
            left_hand=0.1,
            right_shoulder=1.7,
            right_arm=1.36,
            right_hand=0.1,
            left_thigh=0.92,
            left_leg=0.48,
            left_foot=1.07,
            right_thigh=0.92,
            right_leg=0.48,
            right_foot=1.07,
        ),
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, ventura jacket, socks, sneakers": {
        "whole_body": 0.9,
        "local_body_part": JOS3BodyParts(
            head=0.09,
            neck=0.09,
            chest=2.66,
            back=1.42,
            pelvis=1.57,
            left_shoulder=1.32,
            left_arm=0.99,
            left_hand=0.14,
            right_shoulder=1.32,
            right_arm=0.99,
            right_hand=0.14,
            left_thigh=0.73,
            left_leg=0.66,
            left_foot=0.85,
            right_thigh=0.73,
            right_leg=0.66,
            right_foot=0.85,
        ),
    },
    "bra+panty, turtle neck, long pants, short trench coat, socks, sneakers": {
        "whole_body": 1.24,
        "local_body_part": JOS3BodyParts(
            head=0.06,
            neck=0.06,
            chest=3.22,
            back=1.99,
            pelvis=2.03,
            left_shoulder=1.62,
            left_arm=1.5,
            left_hand=0.37,
            right_shoulder=1.62,
            right_arm=1.5,
            right_hand=0.37,
            left_thigh=1.51,
            left_leg=0.65,
            left_foot=0.8,
            right_thigh=1.51,
            right_leg=0.65,
            right_foot=0.8,
        ),
    },
    "bra+panty, tank top, long sleeve shirts, blazer, skirt, sandals": {
        "whole_body": 0.86,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=3.24,
            back=1.81,
            pelvis=2.06,
            left_shoulder=1.98,
            left_arm=1.13,
            left_hand=0.07,
            right_shoulder=1.98,
            right_arm=1.13,
            right_hand=0.07,
            left_thigh=1.19,
            left_leg=0.04,
            left_foot=0.44,
            right_thigh=1.19,
            right_leg=0.04,
            right_foot=0.44,
        ),
    },
    "bra+panty, long sleeve shirts, wool skirt, socks, formal shoes": {
        "whole_body": 0.59,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.21,
            back=0.74,
            pelvis=1.56,
            left_shoulder=0.44,
            left_arm=0.24,
            left_hand=0.17,
            right_shoulder=0.44,
            right_arm=0.24,
            right_hand=0.17,
            left_thigh=1.52,
            left_leg=0.09,
            left_foot=0.74,
            right_thigh=1.52,
            right_leg=0.09,
            right_foot=0.74,
        ),
    },
    "bra+panty, turtleneck, wool skirt, socks, formal shoes": {
        "whole_body": 0.7,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.11,
            back=0.94,
            pelvis=1.52,
            left_shoulder=0.73,
            left_arm=0.62,
            left_hand=0.14,
            right_shoulder=0.73,
            right_arm=0.62,
            right_hand=0.14,
            left_thigh=1.53,
            left_leg=0.09,
            left_foot=0.85,
            right_thigh=1.53,
            right_leg=0.09,
            right_foot=0.85,
        ),
    },
    "bra+panty, long sleeve shirt, wool skirt, sweater, socks, formal shoes": {
        "whole_body": 0.91,
        "local_body_part": JOS3BodyParts(
            head=0.14,
            neck=0.14,
            chest=2.82,
            back=1.53,
            pelvis=1.79,
            left_shoulder=1.22,
            left_arm=0.97,
            left_hand=0.08,
            right_shoulder=1.22,
            right_arm=0.97,
            right_hand=0.08,
            left_thigh=1.53,
            left_leg=0.11,
            left_foot=0.83,
            right_thigh=1.53,
            right_leg=0.11,
            right_foot=0.83,
        ),
    },
    "bra+panty, thin dress shirts, slacks, tie, socks, sneakers": {
        "whole_body": 0.57,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.69,
            back=0.8,
            pelvis=1.08,
            left_shoulder=0.67,
            left_arm=0.58,
            left_hand=0.07,
            right_shoulder=0.67,
            right_arm=0.58,
            right_hand=0.07,
            left_thigh=0.36,
            left_leg=0.39,
            left_foot=0.74,
            right_thigh=0.36,
            right_leg=0.39,
            right_foot=0.74,
        ),
    },
    "bra+panty, thin dress shirts, slacks, blazer, tie, belt, socks, formal shoes": {
        "whole_body": 0.93,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=3.6,
            back=1.83,
            pelvis=1.71,
            left_shoulder=2.16,
            left_arm=1.49,
            left_hand=0.13,
            right_shoulder=2.16,
            right_arm=1.49,
            right_hand=0.13,
            left_thigh=0.64,
            left_leg=0.43,
            left_foot=0.69,
            right_thigh=0.64,
            right_leg=0.43,
            right_foot=0.69,
        ),
    },
    "bra+panty, long sleeve shirts, long pants, blazer, socks, sneakers": {
        "whole_body": 0.96,
        "local_body_part": JOS3BodyParts(
            head=0.04,
            neck=0.04,
            chest=3.3,
            back=1.67,
            pelvis=2.2,
            left_shoulder=2.1,
            left_arm=1.43,
            left_hand=0.09,
            right_shoulder=2.1,
            right_arm=1.43,
            right_hand=0.09,
            left_thigh=0.72,
            left_leg=0.42,
            left_foot=0.67,
            right_thigh=0.72,
            right_leg=0.42,
            right_foot=0.67,
        ),
    },
    "bra+panty, T-shirt, long sleeve shirts, long pants, winter jacket (Notica)": {
        "whole_body": 1.05,
        "local_body_part": JOS3BodyParts(
            head=0.04,
            neck=0.04,
            chest=3.88,
            back=2.26,
            pelvis=1.97,
            left_shoulder=1.82,
            left_arm=1.46,
            left_hand=0.17,
            right_shoulder=1.82,
            right_arm=1.46,
            right_hand=0.17,
            left_thigh=0.81,
            left_leg=0.57,
            left_foot=0.78,
            right_thigh=0.81,
            right_leg=0.57,
            right_foot=0.78,
        ),
    },
    "bra+panty, turtle neck, ski-jumper, skin pants, sports socks, sports shoes": {
        "whole_body": 1.84,
        "local_body_part": JOS3BodyParts(
            head=0.89,
            neck=0.89,
            chest=5.24,
            back=2.87,
            pelvis=2.64,
            left_shoulder=2.55,
            left_arm=2.16,
            left_hand=0.46,
            right_shoulder=2.55,
            right_arm=2.16,
            right_hand=0.46,
            left_thigh=1.49,
            left_leg=1.82,
            left_foot=1.56,
            right_thigh=1.49,
            right_leg=1.82,
            right_foot=1.56,
        ),
    },
    "bra+panty, turtle neck, ski-jumper and hood, skin pants, sports socks, sports shoes": {
        "whole_body": 1.87,
        "local_body_part": JOS3BodyParts(
            head=1.63,
            neck=1.63,
            chest=5.12,
            back=2.7,
            pelvis=2.57,
            left_shoulder=2.58,
            left_arm=2.16,
            left_hand=0.49,
            right_shoulder=2.58,
            right_arm=2.16,
            right_hand=0.49,
            left_thigh=1.44,
            left_leg=1.76,
            left_foot=1.54,
            right_thigh=1.44,
            right_leg=1.76,
            right_foot=1.54,
        ),
    },
    "bra+panty, turtle neck, goose down, ski pants, sports socks, sports shoes": {
        "whole_body": 2.53,
        "local_body_part": JOS3BodyParts(
            head=1.17,
            neck=1.17,
            chest=15.44,
            back=5.5,
            pelvis=5.2,
            left_shoulder=6.55,
            left_arm=5.58,
            left_hand=0.35,
            right_shoulder=6.55,
            right_arm=5.58,
            right_hand=0.35,
            left_thigh=2.12,
            left_leg=1.7,
            left_foot=1.54,
            right_thigh=2.12,
            right_leg=1.7,
            right_foot=1.54,
        ),
    },
    "bra+panty, turtle neck, goose down-with hood, ski pants, sports socks, sports shoes": {
        "whole_body": 2.75,
        "local_body_part": JOS3BodyParts(
            head=3.52,
            neck=3.52,
            chest=12.62,
            back=3.99,
            pelvis=5.05,
            left_shoulder=6.2,
            left_arm=5.73,
            left_hand=0.53,
            right_shoulder=6.2,
            right_arm=5.73,
            right_hand=0.53,
            left_thigh=2.11,
            left_leg=1.81,
            left_foot=1.58,
            right_thigh=2.11,
            right_leg=1.81,
            right_foot=1.58,
        ),
    },
    "bra+panty, turtle neck, goose down-with hood and gloves, ski pants, sports socks, sports shoes": {
        "whole_body": 3.27,
        "local_body_part": JOS3BodyParts(
            head=3.92,
            neck=3.92,
            chest=16.13,
            back=4.47,
            pelvis=5.71,
            left_shoulder=7.12,
            left_arm=5.37,
            left_hand=2.54,
            right_shoulder=7.12,
            right_arm=5.37,
            right_hand=2.54,
            left_thigh=2.14,
            left_leg=1.82,
            left_foot=1.61,
            right_thigh=2.14,
            right_leg=1.82,
            right_foot=1.61,
        ),
    },
    "briefs, socks, T-shirt, half pants, sneakers": {
        "whole_body": 0.53,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.5,
            back=1.13,
            pelvis=1.21,
            left_shoulder=0.39,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.39,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.94,
            left_leg=0.07,
            left_foot=0.62,
            right_thigh=0.94,
            right_leg=0.07,
            right_foot=0.62,
        ),
    },
    "briefs, undershirt, sports t-shirts, sports shorts": {
        "whole_body": 0.7,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.9,
            back=1.76,
            pelvis=2.14,
            left_shoulder=0.48,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.48,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=1.18,
            left_leg=0.0,
            left_foot=0.01,
            right_thigh=1.18,
            right_leg=0.0,
            right_foot=0.01,
        ),
    },
    "briefs, socks, polo shirt, long pants, sneakers": {
        "whole_body": 0.54,
        "local_body_part": JOS3BodyParts(
            head=0.02,
            neck=0.02,
            chest=0.52,
            back=0.97,
            pelvis=1.12,
            left_shoulder=0.34,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.34,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.79,
            left_leg=0.56,
            left_foot=0.65,
            right_thigh=0.79,
            right_leg=0.56,
            right_foot=0.65,
        ),
    },
    "briefs, under shirt, long-sleeved shirt, long pants": {
        "whole_body": 1.01,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.2,
            back=1.65,
            pelvis=2.29,
            left_shoulder=0.98,
            left_arm=0.78,
            left_hand=0.03,
            right_shoulder=0.98,
            right_arm=0.78,
            right_hand=0.03,
            left_thigh=1.46,
            left_leg=0.62,
            left_foot=0.02,
            right_thigh=1.46,
            right_leg=0.62,
            right_foot=0.02,
        ),
    },
    "briefs, socks, undershirt, short-sleeved shirt, long pants, belt, shoes": {
        "whole_body": 0.72,
        "local_body_part": JOS3BodyParts(
            head=0.01,
            neck=0.01,
            chest=0.8,
            back=1.4,
            pelvis=1.55,
            left_shoulder=0.54,
            left_arm=0.0,
            left_hand=0.0,
            right_shoulder=0.54,
            right_arm=0.0,
            right_hand=0.0,
            left_thigh=0.89,
            left_leg=0.64,
            left_foot=0.99,
            right_thigh=0.89,
            right_leg=0.64,
            right_foot=0.99,
        ),
    },
    "briefs, socks, undershirt, long-sleeved shirt, long pants, belt": {
        "whole_body": 0.77,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=0.86,
            back=1.45,
            pelvis=1.54,
            left_shoulder=0.82,
            left_arm=0.6,
            left_hand=0.01,
            right_shoulder=0.82,
            right_arm=0.6,
            right_hand=0.01,
            left_thigh=0.9,
            left_leg=0.66,
            left_foot=0.64,
            right_thigh=0.9,
            right_leg=0.66,
            right_foot=0.64,
        ),
    },
    "briefs, socks, undershirt, long-sleeved shirt, jacket, long pants, belt, shoes": {
        "whole_body": 1.39,
        "local_body_part": JOS3BodyParts(
            head=0.02,
            neck=0.02,
            chest=2.13,
            back=2.28,
            pelvis=3.04,
            left_shoulder=1.8,
            left_arm=1.54,
            left_hand=0.15,
            right_shoulder=1.8,
            right_arm=1.54,
            right_hand=0.15,
            left_thigh=1.33,
            left_leg=0.69,
            left_foot=0.97,
            right_thigh=1.33,
            right_leg=0.69,
            right_foot=0.97,
        ),
    },
    "briefs, socks, undershirt, work jacket, work pants, safety shoes": {
        "whole_body": 0.8,
        "local_body_part": JOS3BodyParts(
            head=0.0,
            neck=0.0,
            chest=1.25,
            back=1.39,
            pelvis=1.78,
            left_shoulder=0.84,
            left_arm=0.71,
            left_hand=0.08,
            right_shoulder=0.84,
            right_arm=0.71,
            right_hand=0.08,
            left_thigh=0.65,
            left_leg=0.59,
            left_foot=1.12,
            right_thigh=0.65,
            right_leg=0.59,
            right_foot=1.12,
        ),
    },
}


if __name__ == "__main__":
    print(show_out_param_docs())
