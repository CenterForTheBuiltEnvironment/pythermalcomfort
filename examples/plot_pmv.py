import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from pythermalcomfort.charts import pmv_chart

ashrae_measurements = pd.read_csv(
    "https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/raw/refs/heads/master/v2.1.0/db_measurements_v2.1.0.csv.gz",
    low_memory=False,
).reset_index(drop=True)

ashrae_measurements = ashrae_measurements.rename(columns={"ta": "tdb"})

pmv_subset = ashrae_measurements[["tdb", "rh"]].dropna()

df = pmv_subset.sample(500, random_state=40)

pmv_inputs = {
    "met": 1.1,
    "clo": 0.5,
    "tr": 23,
    "v": 0.1,
}

fig = pmv_chart(df, pmv_inputs, show_summary=True)
fig.show()
