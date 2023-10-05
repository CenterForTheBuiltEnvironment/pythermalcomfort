import pandas as pd
import matplotlib.pyplot as plt
import os
from pythermalcomfort import atcs

# Make "atcs_output_example" directory in the current directory
directory_name = "atcs_output_example"
current_directory = os.getcwd()
atcs_example_directory = os.path.join(current_directory, directory_name)
if not os.path.exists(atcs_example_directory):
    os.makedirs(atcs_example_directory)

# -------------------------------------------
# EXAMPLE (simple simulation)
# The following code simulates human thermal sensation in transient environment.
# The simulated people is exposed to a warm environment and goes to cool environment.
# This model can simulate the sudden changes in thermal sensations we usually experience
# when the environmental step change happens.
# -------------------------------------------

# Build a model and set a body built
# Create an instance of the class with optional body parameters such as body height, weight, age, sex, etc.
model = atcs.ATCS(height=1.7, weight=60, age=30)

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
df = pd.DataFrame(model.dict_results())
df["sensation_overall"].plot()
plt.ylabel("Overall thermal sensation [-]")
plt.xlabel("Time [min]")
plt.savefig(os.path.join(atcs_example_directory, "atcs_example_local_sensations.png"))
plt.show()

# Exporting the results as csv
model.to_csv(os.path.join(atcs_example_directory, "atcs_example1 (default output).csv"))
