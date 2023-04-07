import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pythermalcomfort.models import JOS3

# Make "jos3_example" directory in the current directory
directory_name = "jos3_output_example"
CURRENT_DIRECTORY = os.getcwd()
JOS3_EXAMPLE_DIRECTORY = os.path.join(CURRENT_DIRECTORY, directory_name)
if not os.path.exists(JOS3_EXAMPLE_DIRECTORY):
    os.makedirs(JOS3_EXAMPLE_DIRECTORY)

# -------------------------------------------
# EXAMPLE 1 (simple simulation)
# -------------------------------------------

# Build a model and set a body built
# Create an instance of the JOS3 class with optional body parameters such as body height, weight, age, sex, etc.
model = JOS3(height=1.7, weight=60, age=30)

# Set the first phase
model.To = 28  # Operative temperature [oC]
model.RH = 40  # Relative humidity [%]
model.Va = 0.2  # Air velocity [m/s]
model.PAR = 1.2  # Physical activity ratio [-]
model.simulate(60)  # Exposure time = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.To = 20  # Change only operative temperature
model.simulate(60)  # Additional exposure time = 60 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df.TskMean.plot()  # Plot time series of mean skin temperature.
plt.ylabel("Mean skin temperature [oC]")  # Set y-label as 'Mean skin temperature [oC]'
plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
plt.savefig(
    os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example1_mean_skin_temperature.png")
)  # Save plot at the current directory
plt.show()  # Show the plot

# Exporting the results as csv
model.to_csv(os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example1 (default output).csv"))

# Print the BMR value using the getter
print('BMR=', model.BMR)

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
model.Ta = 28  # Air temperature [oC]
model.Tr = 30  # Mean radiant temperature [oC]
model.RH = 40  # Relative humidity [%]
model.Va = 0.2  # Air velocity [m/s]
model.PAR = 1.2  # Physical activity ratio [-], assuming a sitting position
model.posture = "sitting"  # Posture [-], assuming a sitting position
model.Icl = np.array(
    [  # Clothing insulation [clo]
        0.00,  # Head
        0.00,  # Neck
        1.14,  # Chest
        0.84,  # Back
        1.04,  # Pelvis
        0.84,  # Left-Shoulder
        0.42,  # Left-Arm
        0.00,  # Left-Hand
        0.84,  # Right-Shoulder
        0.42,  # Right-Arm
        0.00,  # Right-Hand
        0.58,  # Left-Thigh
        0.62,  # Left-Leg
        0.82,  # Left-Foot
        0.58,  # Right-Thigh
        0.62,  # Right-Leg
        0.82,  # Right-Foot
    ]
)
# Run JOS-3 model
model.simulate(
    times=30,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.To = 20  # Change operative temperature
model.Va = np.array(
    [  # Air velocity [m/s], assuming to use a desk fan
        0.2,  # Head
        0.4,  # Neck
        0.4,  # Chest
        0.1,  # Back
        0.1,  # Pelvis
        0.4,  # Left-Shoulder
        0.4,  # Left-Arm
        0.4,  # Left-Hand
        0.4,  # Right-Shoulder
        0.4,  # Right-Arm
        0.4,  # Right-Hand
        0.1,  # Left-Thigh
        0.1,  # Left-Leg
        0.1,  # Left-Foot
        0.1,  # Right-Thigh
        0.1,  # Right-Leg
        0.1,  # Right-Foot
    ]
)
# Run JOS-3 model
model.simulate(
    times=60,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 60 [loops] * 60 [sec] = 60 [min]

# Set the next condition (You only need to change the parameters that you want to change)
model.Ta = 30  # Change air temperature [oC]
model.Tr = 35  # Change mean radiant temperature [oC]
# Run JOS-3 model
model.simulate(
    times=30,  # Number of loops of a simulation
    dtime=60,  # Time delta [sec]. The default is 60.
)  # Additional exposure time = 30 [loops] * 60 [sec] = 30 [min]

# Show the results
df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
df[["TskMean", "TskHead", "TskChest", "TskLHand"]].plot()  # Plot time series of local skin temperature.
plt.ylabel("Skin temperature [oC]")  # Set y-label as 'Skin temperature [oC]'
plt.xlabel("Time [min]")  # Set x-label as 'Time [min]'
plt.savefig(os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example2_local_skin_temperatures.png"))  # Save plot at the current directory
plt.show()  # Show the plot

# Exporting the results as csv
model.to_csv(os.path.join(JOS3_EXAMPLE_DIRECTORY, "jos3_example2 (all output).csv"))
