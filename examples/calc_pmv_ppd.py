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

import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "/examples/template-SI.csv")

df['PMV'] = None
df['PPD'] = None

for index, row in df.iterrows():
    vr = v_relative(v=row['v'], met=row['met'])
    results = pmv_ppd(tdb=row['tdb'], tr=row['tr'], vr=vr, rh=row['rh'], met=row['met'], clo=row['clo'], standard="ashrae")
    df.loc[index, 'PMV'] = results['pmv']
    df.loc[index, 'PPD'] = results['ppd']

print(df)
df.to_csv('results.csv')
