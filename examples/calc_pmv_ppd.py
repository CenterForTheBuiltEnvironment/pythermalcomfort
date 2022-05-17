from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

# input variables
tdb = 27  # dry bulb air temperature, [$^{\circ}$C]
tr = 25  # mean radiant temperature, [$^{\circ}$C]
v = 0.3  # average air speed, [m/s]
rh = 50  # relative humidity, [%]
activity = "Typing"  # participant's activity description
garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

met = met_typical_tasks[activity]  # activity met, [met]
icl = sum(
    [clo_individual_garments[item] for item in garments]
)  # calculate total clothing insulation

# calculate the relative air velocity
vr = v_relative(v=v, met=met)
# calculate the dynamic clothing insulation
clo = clo_dynamic(clo=icl, met=met)

# calculate PMV in accordance with the ASHRAE 55 2020
results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")

# print the results
print(results)

# print PMV value
print(f"pmv={results['pmv']}, ppd={results['ppd']}%")

# for users who want to use the IP system
results_ip = pmv_ppd(tdb=77, tr=77, vr=0.6, rh=50, met=1.1, clo=0.5, units="IP")
print(results_ip)

# If you want you can also pass an array of inputs
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

ta = df["tdb"].values
tr = df["tr"].values
vel = df["v"].values
rh = df["rh"].values
met = df["met"].values
clo = df["clo"].values

v_rel = v_relative(vel, met)
clo_d = clo_dynamic(clo, met)
results = pmv_ppd(ta, tr, v_rel, rh, met, clo_d, 0, "ashrae", "SI")

df["vr"] = v_rel
df["clo_d"] = clo_d
df["pmv"] = results["pmv"]
df["ppd"] = results["ppd"]

print(df.head())

# uncomment the following line if you want to save the data to .csv file
# df.to_csv('results.csv')

# This method is extremely fast and can perform a lot of calculations in very little time
from pythermalcomfort.models import pmv_ppd
import numpy as np
import time

iterations = 10000
tdb = np.empty(iterations)
tdb.fill(25)
met = np.empty(iterations)
met.fill(1.5)

v_rel = v_relative(0.1, met)
clo_d = clo_dynamic(1, met)

# ASHRAE PMV
start = time.time()
pmv_ppd(tdb=tdb, tr=23, vr=v_rel, rh=40, met=1.2, clo=clo_d, standard="ashrae")
end = time.time()
print(end - start)

# ISO PMV
start = time.time()
pmv_ppd(tdb=tdb, tr=23, vr=v_rel, rh=40, met=1.2, clo=clo_d, standard="iso")
end = time.time()
print(end - start)
