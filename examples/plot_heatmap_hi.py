import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from epw import epw

from pythermalcomfort.models import heat_index
from pythermalcomfort.charts import heatmap


weather_file = epw()  # https://github.com/building-energy/epw
weather_file.read(
    "test_data/USA_AZ_Phoenix-Sky.Harbor.Intl.AP.722780_TMYx.2009-2023.epw"
)

weather_data = weather_file.dataframe

weather_data["Hour"] = weather_data["Hour"] - 1
weather_data["Year"] = 2019

weather_data["DateTime"] = pd.to_datetime(
    weather_data[["Year", "Month", "Day", "Hour"]]
)
weather_data["Month Name"] = weather_data["DateTime"].dt.strftime("%b")


weather_data["HI"] = weather_data.apply(
    lambda row: (
        heat_index(tdb=row["Dry Bulb Temperature"], rh=row["Relative Humidity"])["hi"]
        if row["Dry Bulb Temperature"] >= 27
        else None
    ),
    axis=1,
)


def categorize_heat_index(x, si_ip):

    if si_ip.lower() == "ip":
        x = (x - 32) * 5 / 9

    if 27 <= x < 32:
        return "Caution"
    elif 32 <= x < 41:
        return "Extreme<br>Caution"
    elif 41 <= x < 54:
        return "Danger"
    elif x >= 54:
        return "Extreme<br>Danger"
    else:
        return "No Risk"


weather_data["categorical"] = weather_data["HI"].apply(
    categorize_heat_index, si_ip="si"
)


fig = heatmap(weather_data, "HI", global_local="global", si_ip="si", show_summary=True)
fig.show()
