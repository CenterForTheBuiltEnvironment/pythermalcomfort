import pandas as pd
import re

def convert_and_print_local_clo_values_from_csv_to_dict(csv_name):
    """

    Parameters
    ----------
    csv_name : file path that you want to convert (it should be in the same folder as this function)

    Returns
    -------
    local_clo_dict : a dictionary including local clothing insulation values as well as that for the whole body

    Notes
    -----
    The 17 body segments correspond to the JOS-3 model.
    """
    # Read the Excel file
    df = pd.read_csv(csv_name)

    # Create an empty dictionary
    local_clo_dict = {}

    # Add data from each row to the dictionary
    for index, row in df.iterrows():
        # Get the name of the clothing combination as the key
        key = row['clothing_ensemble']
        # Create a dictionary as the value
        value = {'whole_body': row['whole_body'], 'local_body_part': {}}
        # Add data from each column to the Local body dictionary
        for col in df.columns[2:]:
            # Get the name of the body part
            body_part = col
            # Get the clo value for the local body part and add it to the Local body dictionary
            clo = row[col]
            value['local_body_part'][body_part] = clo
        # Add the data to the dictionary
        local_clo_dict[key] = value

    return local_clo_dict

def add_prompt_to_code(code: str, prompt: str = ">>> ") -> str:
    lines = code.strip().split("\n")
    result = []
    for line in lines:
        if re.match(r"^\s*#", line):  # If it's a comment line
            result.append(line)
        else:
            result.append(prompt + line)
    return "\n".join(result)

sample_code = """
# Build a model and set a body built
# Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.
model = JOS3(
    height=1.7,
    weight=60,
    fat=20,
    age=30,
    sex="male",
    bmr_equation="japanese",
    bsa_equation="fujimoto",
    ex_output="all",
)

# Set environmental conditions such as air temperature, mean radiant temperature using the setter methods.
# Set the first condition
# Environmental parameters can be input as int, float, list, dict, numpy array format.
model.tdb = 28  # Air temperature [oC]
model.tr = 30  # Mean radiant temperature [oC]
model.rh = 40  # Relative humidity [%]
model.v = np.array( # Air velocity [m/s]
    [
        0.2,  # head
        0.4,  # neck
        0.4,  # chest
        0.1,  # back
        0.1,  # pelvis
        0.4,  # left shoulder
        0.4,  # left arm
        0.4,  # left hand
        0.4,  # right shoulder
        0.4,  # right arm
        0.4,  # right hand
        0.1,  # left thigh
        0.1,  # left leg
        0.1,  # left foot
        0.1,  # right thigh
        0.1,  # right leg
        0.1,  # right foot
    ]
)
model.clo = local_clo_typical_ensembles["briefs, socks, undershirt, work jacket, work pants, safety shoes"]["local_body_part"]

# par should be input as int, float.
model.par = 1.2  # Physical activity ratio [-], assuming a sitting position
# posture should be input as int (0, 1, or 2) or str ("standing", "sitting" or "lying").
# (0="standing", 1="sitting" or 2="lying")
model.posture = "sitting"  # Posture [-], assuming a sitting position

# Run JOS-3 model
model.simulate(
    times=30,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.to = 20  # Change operative temperature
model.v = { # Air velocity [m/s], assuming to use a desk fan
    'head' : 0.2,
    'neck' : 0.4,
    'chest' : 0.4,
    'back': 0.1,
    'pelvis' : 0.1,
    'left_shoulder' : 0.4,
    'left_arm' : 0.4,
    'left_hand' : 0.4,
    'right_shoulder' : 0.4,
    'right_arm' : 0.4,
    'right_hand' : 0.4,
    'left_thigh' : 0.1,
    'left_leg' : 0.1,
    'left_foot' : 0.1,
    'right_thigh' : 0.1,
    'right_leg' : 0.1,
    'right_foot' : 0.1
    }
# Run JOS-3 model
model.simulate(
    times=60,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.tdb = 30  # Change air temperature [oC]
model.tr = 35  # Change mean radiant temperature [oC]
# Run JOS-3 model
model.simulate(
    times=30,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df[["t_skin_mean", "t_skin_head", "t_skin_chest", "t_skin_left_hand"]].plot()  # Plot time series of local skin temperature.
plt.legend(["Mean", "Head", "Chest", "Left hand"])  # Reset the legends
plt.ylabel("Skin temperature [oC]")  # Set y-label as 'Skin temperature [oC]'
plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
plt.savefig(os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example2_skin_temperatures.png"))  # Save plot at the current directory
plt.show()  # Show the plot

# Exporting the results as csv
model.to_csv(os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example2 (all output).csv"))
"""
print(convert_and_print_local_clo_values_from_csv_to_dict(csv_name='local_clo_summary.csv'))
print(add_prompt_to_code(sample_code))
