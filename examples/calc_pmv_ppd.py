from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.psychrometrics import v_relative

# measured air velocity
v = 0.1
met = 1.2

v_r = v_relative(v=v, met=met)
print(v_r)

# calculate PMV in accordance with the ASHRAE 55 2017
results = pmv_ppd(tdb=27, tr=25, vr=v_r, rh=50, met=met, clo=0.5, wme=0, standard="ISO")

# print the results
print(results)

# print PMV value
print(results['pmv'])

# for users who wants to use the IP system
results_ip = pmv_ppd(tdb=77, tr=77, vr=0.4, rh=50, met=1.2, clo=0.5, units="IP")
print(results_ip)
