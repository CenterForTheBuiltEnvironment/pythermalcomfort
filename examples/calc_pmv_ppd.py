from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments

# input variables
tdb = 27  # dry-bulb air temperature, [$^{\circ}$C]
tr = 25  # mean radiant temperature, [$^{\circ}$C]
v = 0.1  # average air velocity, [m/s]
rh = 50  # relative humidity, [%]
activity = "Typing"  # participant's activity description
garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

met = met_typical_tasks[activity]  # activity met, [met]
icl = sum([clo_individual_garments[item] for item in garments])  # calculate total clothing insulation

# calculate the relative air velocity
vr = v_relative(v=v, met=met)

# calculate PMV in accordance with the ASHRAE 55 2017
results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=icl, standard="ASHRAE")

# print the results
print(results)

# print PMV value
print(f"pmv={results['pmv']}, ppd={results['ppd']}%")

# for users who wants to use the IP system
results_ip = pmv_ppd(tdb=77, tr=77, vr=0.4, rh=50, met=1.2, clo=0.5, units="IP")
print(results_ip)

# the following code can be used to iterate over a Pandas DataFrame and calculate the reults
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative
import os

df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

import time
start = time.time()

df['PMV'] = None
df['PPD'] = None

for index, row in df.iterrows():
    vr = v_relative(v=row['v'], met=row['met'])
    results = pmv_ppd(tdb=row['tdb'], tr=row['tr'], vr=vr, rh=row['rh'], met=row['met'], clo=row['clo'], standard="ashrae")
    df.loc[index, 'PMV'] = results['pmv']
    df.loc[index, 'PPD'] = results['ppd']

print(df)

end = time.time()

print(end - start)

# uncomment the following line if you want to save the data to .csv file
# df.to_csv('results.csv')

# The code above has the following limitations:
# * it can be slow if you have a large dataframe (i.e. more than 1000 rows)
# * if the Cooling Effect cannot be calculated then the code throws an error.

# For example to calculate the results for 7 entries the above function takes 0.022 s while the one below 0.0045 s

# A possible solution to the above mentioned problems is presented below:
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative
import os
import time

df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

start = time.time()

ta = df["tdb"].values
tr = df["tr"].values
vel = df["v"].values
rh = df["rh"].values
met = df["met"].values
clo = df["clo"].values

results = []

for ix in range(df.shape[0]):

    _vr = v_relative(vel[ix], met[ix])

    try:
        _pmv_ppd = pmv_ppd(
            ta[ix],
            tr[ix],
            _vr,
            rh[ix],
            met[ix],
            clo[ix],
            standard="ashrae",
            units="SI",
            )
        _pmv = _pmv_ppd["pmv"]
        _ppd = _pmv_ppd["ppd"]
    except:
        _pmv, _ppd = [9999, 9999]
    results.append({"pmv": _pmv, "ppd": _ppd, "vr": _vr})

# split the pmv column in two since currently contains both pmv and ppd values
df_ = pd.DataFrame(results)
df = pd.concat([df, df_], axis=1, sort=False)
df.loc[df.ppd == 9999, ["pmv", "ppd"]] = None

end = time.time()

print(end - start)

# Finally one other alternative to improve the speed of the code above is to use numpy vectorize as shown below
# Please note that this code will break if the PMV value cannot be calcualted
import pandas as pd
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative
import numpy as np

df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

import time
start = time.time()

ta = df["tdb"].values
tr = df["tr"].values
vel = df["v"].values
rh = df["rh"].values
met = df["met"].values
clo = df["clo"].values

v_rel=np.vectorize(v_relative)(vel,met)
results=np.vectorize(pmv_ppd)(ta,tr,v_rel,rh,met,clo,0,"ashrae","SI")

# split the pmv column in two since currently contains both pmv and ppd values
df_ = pd.DataFrame(results)
df = pd.concat([df, df_], axis=1, sort=False)

end = time.time()

print(end - start)
