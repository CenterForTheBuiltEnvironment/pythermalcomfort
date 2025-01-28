import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pythermalcomfort.models import utci
from pythermalcomfort.charts import heatmap
from pythermalcomfort.utilities import epw_to_dataframe

weather_data = epw_to_dataframe(
    epw_file_path="test_data/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMYx.2009-2023.epw",
)

utci_results = utci(
    tdb=weather_data["Dry Bulb Temperature"],
    tr=weather_data["Dry Bulb Temperature"],
    v=weather_data["Wind Speed"],
    rh=weather_data["Relative Humidity"],
    units="si",
)

weather_data["UTCI"] = utci_results["utci"]
weather_data["categorical"] = utci_results["stress_category"]


fig = heatmap(
    weather_data, "UTCI", global_local="global", si_ip="si", show_summary=True
)
fig.show()
