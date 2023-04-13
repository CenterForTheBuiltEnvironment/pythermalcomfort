# -*- coding: utf-8 -*-
"""
This code defines a dictionary called ALL_OUT_PARAMS that contains information about various output parameters
related to human body properties, heat exchange, and environmental conditions.

It also includes a function called show_outparam_docs() that generates a formatted string with the documentation
of the output parameters.

The show_outparam_docs() function uses text wrapping to create a readable documentation string
for both regular output parameters and extra output parameters.

It sorts the parameters alphabetically by key and formats each line with the parameter's name, meaning, and unit.
The resulting documentation string can be displayed or printed for user reference.
"""
import textwrap

ALL_OUT_PARAMS = {
    "Age": {"ex_output": True, "meaning": "Age", "suffix": None, "unit": "years"},
    "BFava_foot": {
        "ex_output": True,
        "meaning": "AVA blood flow rate of one foot",
        "suffix": None,
        "unit": "L/h",
    },
    "BFava_hand": {
        "ex_output": True,
        "meaning": "AVA blood flow rate of one hand",
        "suffix": None,
        "unit": "L/h",
    },
    "BFcr": {
        "ex_output": True,
        "meaning": "Core blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "BFfat": {
        "ex_output": True,
        "meaning": "Fat blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "BFms": {
        "ex_output": True,
        "meaning": "Muscle blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "BFsk": {
        "ex_output": True,
        "meaning": "Skin blood flow rate (each body part)",
        "suffix": "Body name",
        "unit": "L/h",
    },
    "BSA": {
        "ex_output": True,
        "meaning": "Body surface area (each body part)",
        "suffix": "Body name",
        "unit": "m2",
    },
    "CO": {
        "ex_output": False,
        "meaning": "Cardiac output (the sum of the whole blood flow)",
        "suffix": None,
        "unit": "L/h",
    },
    "CycleTime": {
        "ex_output": False,
        "meaning": "The counts of executing one cycle calculation",
        "suffix": None,
        "unit": "-",
    },
    "Emax": {
        "ex_output": True,
        "meaning": "Maximum evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Esk": {
        "ex_output": True,
        "meaning": "Evaporative heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Esweat": {
        "ex_output": True,
        "meaning": "Evaporative heat loss from the skin by only sweating (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Fat": {"ex_output": True, "meaning": "Body fat rate", "suffix": None, "unit": "%"},
    "Height": {"ex_output": True, "meaning": "Body height", "suffix": None, "unit": "m"},
    "Icl": {
        "ex_output": True,
        "meaning": "Clothing insulation (each body part)",
        "suffix": "Body name",
        "unit": "clo",
    },
    "LHLsk": {
        "ex_output": True,
        "meaning": "Latent heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Mbasecr": {
        "ex_output": True,
        "meaning": "Core heat production by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Mbasefat": {
        "ex_output": True,
        "meaning": "Fat heat production by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Mbasems": {
        "ex_output": True,
        "meaning": "Muscle heat production by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Mbasesk": {
        "ex_output": True,
        "meaning": "Skin heat production by basal metabolism (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Met": {
        "ex_output": False,
        "meaning": "Total heat production of the whole body",
        "suffix": None,
        "unit": "W",
    },
    "Mnst": {
        "ex_output": True,
        "meaning": "Core heat production by non-shivering thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "ModTime": {
        "ex_output": False,
        "meaning": "Simulation times",
        "suffix": None,
        "unit": "sec",
    },
    "Mshiv": {
        "ex_output": True,
        "meaning": "Core or muscle heat production by shivering thermogenesis (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Mwork": {
        "ex_output": True,
        "meaning": "Core or muscle heat production by work (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Name": {
        "ex_output": True,
        "meaning": "Name of the model",
        "suffix": None,
        "unit": "-",
    },
    "PAR": {
        "ex_output": True,
        "meaning": "Physical activity ratio",
        "suffix": None,
        "unit": "-",
    },
    "Qcr": {
        "ex_output": True,
        "meaning": "Core total heat production (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Qfat": {
        "ex_output": True,
        "meaning": "Fat total heat production (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Qms": {
        "ex_output": True,
        "meaning": "Muscle total heat production (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Qsk": {
        "ex_output": True,
        "meaning": "Skin total heat production (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "RES": {
        "ex_output": False,
        "meaning": "Heat loss by respiration",
        "suffix": None,
        "unit": "W",
    },
    "RESlh": {
        "ex_output": True,
        "meaning": "Latent heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "RESsh": {
        "ex_output": True,
        "meaning": "Sensible heat loss by respiration (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "RH": {
        "ex_output": True,
        "meaning": "Relative humidity (each body part)",
        "suffix": "Body name",
        "unit": "%",
    },
    "Ret": {
        "ex_output": True,
        "meaning": "Total clothing evaporative heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "m2.kPa/W",
    },
    "Rt": {
        "ex_output": True,
        "meaning": "Total clothing heat resistance (each body part)",
        "suffix": "Body name",
        "unit": "m2.K/W",
    },
    "SHLsk": {
        "ex_output": True,
        "meaning": "Sensible heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Setptcr": {
        "ex_output": True,
        "meaning": "Set point skin temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Setptsk": {
        "ex_output": True,
        "meaning": "Set point core temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Sex": {
        "ex_output": True,
        "meaning": "Sex",
        "suffix": None,
        "unit": "-",
    },
    "THLsk": {
        "ex_output": False,
        "meaning": "Total heat loss from the skin (each body part)",
        "suffix": "Body name",
        "unit": "W",
    },
    "Ta": {
        "ex_output": True,
        "meaning": "Air temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tar": {
        "ex_output": True,
        "meaning": "Arterial temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tcb": {
        "ex_output": True,
        "meaning": "Central blood temperature",
        "suffix": None,
        "unit": "oC",
    },
    "Tcr": {
        "ex_output": False,
        "meaning": "Core temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tfat": {
        "ex_output": True,
        "meaning": "Fat temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tms": {
        "ex_output": True,
        "meaning": "Muscle temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "To": {
        "ex_output": True,
        "meaning": "Operative temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tr": {
        "ex_output": True,
        "meaning": "Mean radiant temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tsk": {
        "ex_output": False,
        "meaning": "Skin temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "TskMean": {
        "ex_output": False,
        "meaning": "Mean skin temperature",
        "suffix": None,
        "unit": "oC",
    },
    "Tsve": {
        "ex_output": True,
        "meaning": "Superficial vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Tve": {
        "ex_output": True,
        "meaning": "Vein temperature (each body part)",
        "suffix": "Body name",
        "unit": "oC",
    },
    "Va": {
        "ex_output": True,
        "meaning": "Air velocity (each body part)",
        "suffix": "Body name",
        "unit": "m/s",
    },
    "Weight": {
        "ex_output": True,
        "meaning": "Body weight",
        "suffix": None,
        "unit": "kg",
    },
    "Wet": {
        "ex_output": False,
        "meaning": "Skin wettedness (each body part)",
        "suffix": "Body name",
        "unit": "-",
    },
    "WetMean": {
        "ex_output": False,
        "meaning": "Mean skin wettedness",
        "suffix": None,
        "unit": "-",
    },
    "Wle": {
        "ex_output": False,
        "meaning": "Weight loss by the evaporation and respiration of the whole body",
        "suffix": None,
        "unit": "g/sec",
    },
    "dt": {
        "ex_output": False,
        "meaning": "Time step",
        "suffix": None,
        "unit": "sec",
    },
    "PythermalcomfortVersion": {
        "ex_output": False,
        "meaning": "Version of pythermalcomfort",
        "suffix": None,
        "unit": "-",
    },
}


def show_outparam_docs():
    """
    Show the documentation of the output parameters.

    Returns
    -------
    docstirng : str
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
    print(show_outparam_docs())
