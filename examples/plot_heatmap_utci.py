import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from epw import epw

from pythermalcomfort.models import utci
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


weather_data["UTCI"] = weather_data.apply(
    lambda row: (
        utci(
            tdb=row["Dry Bulb Temperature"],
            tr=row["Dry Bulb Temperature"],
            v=row["Wind Speed"],
            rh=row["Relative Humidity"],
        )["utci"]
    ),
    axis=1,
)


def categorize_utci(x, si_ip):

    if si_ip.lower() == "ip":
        x = (x - 32) * 5 / 9

    if x < -40:
        return "Extreme<br>Cold Stress"
    elif -40 <= x < -27:
        return "Very Strong<br>Cold Stress"
    elif -27 <= x < -13:
        return "Strong<br>Cold Stress"
    elif -13 <= x < 0:
        return "Moderate<br>Cold Stress"
    elif 0 <= x < 9:
        return "Slight<br>Cold Stress"
    elif 9 <= x < 26:
        return "No Thermal<br>Stress"
    elif 26 <= x < 32:
        return "Moderate<br>Heat Stress"
    elif 32 <= x < 38:
        return "Strong<br>Heat Stress"
    elif 38 <= x < 46:
        return "Very Strong<br>Heat Stress"
    elif x >= 46:
        return "Extreme<br>Heat Stress"
    else:
        return "Out of<br>Range"


weather_data["categorical"] = weather_data["UTCI"].apply(categorize_utci, si_ip="si")


fig = heatmap(
    weather_data, "UTCI", global_local="global", si_ip="si", show_summary=True
)
fig.show()
