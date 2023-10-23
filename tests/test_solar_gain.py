from pythermalcomfort.models import solar_gain
import json
import requests

# get file containing validation tables
url = "https://raw.githubusercontent.com/FedericoTartarini/validation-data-comfort-models/main/validation_data.json"
resp = requests.get(url)
reference_tables = json.loads(resp.text)


def test_solar_gain():
    for table in reference_tables["reference_data"]["solar_gain"]:
        for entry in table["data"]:
            inputs = entry["inputs"]
            outputs = entry["outputs"]
            sg = solar_gain(
                inputs["alt"],
                inputs["sharp"],
                inputs["I_dir"],
                inputs["t_sol"],
                inputs["f_svv"],
                inputs["f_bes"],
                inputs["asa"],
                inputs["posture"],
            )
            assert sg["erf"] == outputs["erf"]
            assert sg["delta_mrt"] == outputs["t_rsw"]
