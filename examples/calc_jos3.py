import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pythermalcomfort.models import JOS3
from pythermalcomfort.jos3_functions.utilities import local_clo_typical_ensembles

# Make "jos3_example_example" directory in the current directory
directory_name = "jos3_output_example"
current_directory = os.getcwd()
jos3_example_directory = os.path.join(current_directory, directory_name)
if not os.path.exists(jos3_example_directory):
    os.makedirs(jos3_example_directory)

# -------------------------------------------
# EXAMPLE 1 (simple simulation)
# -------------------------------------------

# Build a model and set a body built
# Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.
model = JOS3(height=1.7, weight=60, age=30)

# Set the first phase
model.to = 28  # Operative temperature [°C]
model.rh = 40  # Relative humidity [%]
model.v = 0.2  # Air velocity [m/s]
model.par = 1.2  # Physical activity ratio [-]
model.simulate(60)  # Exposure time = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.to = 20  # Change only operative temperature
model.simulate(60)  # Additional exposure time = 60 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.t_skin_mean.plot()  # Plot time series of mean skin temperature.
plt.ylabel("Mean skin temperature [°C]")  # Set y-label as 'Mean skin temperature [°C]'
plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
plt.savefig(
    os.path.join(jos3_example_directory, "jos3_example1_mean_skin_temperature.png")
)  # Save plot at the current directory
plt.show()  # Show the plot

# Exporting the results as csv
model.to_csv(os.path.join(jos3_example_directory, "jos3_example1 (default output).csv"))

# Print the BMR value using the getter
print("BMR=", model.bmr)
print("Body name list: ", model.body_names)

# -------------------------------------------
# EXAMPLE 2 (detail simulation)
# -------------------------------------------

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
model.tdb = 28  # Air temperature [°C]
model.tr = 30  # Mean radiant temperature [°C]
model.rh = 40  # Relative humidity [%]
model.v = np.array(  # Air velocity [m/s]
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
model.clo = local_clo_typical_ensembles[
    "briefs, socks, undershirt, work jacket, work pants, safety shoes"
]["local_body_part"]

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
model.v = {  # Air velocity [m/s], assuming to use a desk fan
    "head": 0.2,
    "neck": 0.4,
    "chest": 0.4,
    "back": 0.1,
    "pelvis": 0.1,
    "left_shoulder": 0.4,
    "left_arm": 0.4,
    "left_hand": 0.4,
    "right_shoulder": 0.4,
    "right_arm": 0.4,
    "right_hand": 0.4,
    "left_thigh": 0.1,
    "left_leg": 0.1,
    "left_foot": 0.1,
    "right_thigh": 0.1,
    "right_leg": 0.1,
    "right_foot": 0.1,
}
# Run JOS-3 model
model.simulate(
    times=60,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.tdb = 30  # Change air temperature [°C]
model.tr = 35  # Change mean radiant temperature [°C]
# Run JOS-3 model
model.simulate(
    times=30,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df[
    ["t_skin_mean", "t_skin_head", "t_skin_chest", "t_skin_left_hand"]
].plot()  # Plot time series of local skin temperature.
plt.legend(["Mean", "Head", "Chest", "Left hand"])  # Reset the legends
plt.ylabel("Skin temperature [°C]")  # Set y-label as 'Skin temperature [°C]'
plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
plt.savefig(
    os.path.join(jos3_example_directory, "jos3_example2_skin_temperatures.png")
)  # Save plot at the current directory
plt.show()  # Show the plot

# Exporting the results as csv
model.to_csv(os.path.join(jos3_example_directory, "jos3_example2 (all output).csv"))
