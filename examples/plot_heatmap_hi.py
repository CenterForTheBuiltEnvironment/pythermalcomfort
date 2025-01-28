import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythermalcomfort.models import heat_index_rothfusz
from pythermalcomfort.charts import heatmap
from pythermalcomfort.utilities import epw_to_dataframe

weather_data = epw_to_dataframe(
    epw_file_path="test_data/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMYx.2009-2023.epw",
)

hi_results = heat_index_rothfusz(
    tdb=weather_data["Dry Bulb Temperature"],
    rh=weather_data["Relative Humidity"],
)

weather_data["HI"] = hi_results["hi"]
weather_data["categorical"] = hi_results["stress_category"]

fig = heatmap(
    df=weather_data, var="HI", global_local="global", si_ip="si", show_summary=True
)
fig.show()
