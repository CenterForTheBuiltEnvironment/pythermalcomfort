import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from pythermalcomfort.utilities import operative_tmp
from pythermalcomfort.models import adaptive_ashrae
from pythermalcomfort.charts import adaptive_chart


ashrae_measurements = pd.read_csv(
    "https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/raw/refs/heads/master/v2.1.0/db_measurements_v2.1.0.csv.gz",
    low_memory=False,
).reset_index(drop=True)

adaptive_subset = ashrae_measurements[["ta", "tr", "t_out", "vel"]].dropna()
print(adaptive_subset.head(5))

adaptive_subset["top"] = operative_tmp(
    tdb=adaptive_subset["ta"], tr=adaptive_subset["tr"], v=adaptive_subset["vel"]
)

adaptive_results = adaptive_ashrae(
    tdb=adaptive_subset["ta"],
    tr=adaptive_subset["tr"],
    t_running_mean=adaptive_subset["t_out"],  # ! simplified due to lack of data
    v=adaptive_subset["vel"],
    units="SI",
)

adaptive_subset["adaptive_acceptability_80%"] = adaptive_results["acceptability_80"]
adaptive_subset["adaptive_acceptability_90%"] = adaptive_results["acceptability_90"]

df = adaptive_subset.sample(500, random_state=40)

fig = adaptive_chart(df, show_summary=True)
fig.show()
