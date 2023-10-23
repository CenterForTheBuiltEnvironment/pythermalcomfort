from pythermalcomfort.models import zhang_sensation_comfort
from pprint import pprint

# -------------------------------------------
# This model is often used with a thermal physiology model.
# However, this function can be used when you have physiological data and wants to use Zhang's comfort model.
# For example, this function can be used when you want to predict thermal sensation or comfort
# but you only have skin temperature or core temperature data measured by a subject experiment
# or when you have skin temperature data from a thermal manikin".
#
# If you have a time series of skin temperature data, calculate and enter the time derivative of skin temperature.
# If you want to do a steady state simulation, change 0 to it.
# The time derivative of core temperature is set to be 0 because it does change due to human thermoregulation
# except in an extremely hot or cold environment.
# -------------------------------------------

dict_results = zhang_sensation_comfort(
    t_skin_local={
        "head": 34.3,
        "neck": 34.6,
        "chest": 34.1,
        "back": 34.3,
        "pelvis": 34.3,
        "left_shoulder": 33.2,
        "left_arm": 33.6,
        "left_hand": 33.4,
        "right_shoulder": 33.2,
        "right_arm": 33.6,
        "right_hand": 33.4,
        "left_thigh": 33.3,
        "left_leg": 31.8,
        "left_foot": 32.3,
        "right_thigh": 33.3,
        "right_leg": 31.8,
        "right_foot": 32.3,
    },
    dt_skin_local_dt={
        "head": 0.01,
        "neck": 0.01,
        "chest": 0.01,
        "back": 0.01,
        "pelvis": 0.01,
        "left_shoulder": 0.01,
        "left_arm": 0.01,
        "left_hand": 0.01,
        "right_shoulder": 0.01,
        "right_arm": 0.01,
        "right_hand": 0.01,
        "left_thigh": 0.01,
        "left_leg": 0.01,
        "left_foot": 0.01,
        "right_thigh": 0.01,
        "right_leg": 0.01,
        "right_foot": 0.01,
    },
    dt_core_local_dt=0,
    options={
        "sensation_scale": "7-point",
    },
)

pprint(dict_results)
