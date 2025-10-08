import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pythermalcomfort.models import two_nodes_gagge

rh_array = np.arange(0, 100, 1)
tdb_array = np.arange(-10, 46, 1)
results = []
for rh in rh_array:
    for tdb in tdb_array:
        result = two_nodes_gagge(
            tdb=tdb,
            tr=tdb,
            v=0.1,
            rh=rh,
            met=1.2,
            clo=0.5,
            wme=0,
            body_surface_area=1.8258,
            p_atm=101325,
            position="standing",
            max_skin_blood_flow=90,
            round_output=True,
            max_sweating=500,
            w_max=False,
        )
        results.append(
            {
                "tdb": tdb,
                "rh": rh,
                "set": result.set,
                "et": result.et,
                "pmv_gagge": result.pmv_gagge,
                "pmv_set": result.pmv_set,
                "disc": result.disc,
                "t_sens": result.t_sens,
                "w": result.w,
                "t_core": result.t_core,
                "t_skin": result.t_skin,
                "e_skin": result.e_skin,
                "e_rsw": result.e_rsw,
                "e_max": result.e_max,
                "q_sensible": result.q_sensible,
                "q_skin": result.q_skin,
                "q_res": result.q_res,
                "m_bl": result.m_bl,
                "m_rsw": result.m_rsw,
                "w_max": result.w_max,
                "e_max-e_rsw": result.e_max - result.e_rsw,
            }
        )
df = pd.DataFrame(results)
# disc = 4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff)
# values that are okay are e_comfort, e_diff, w_max, w
# potential values to check are e_rsw, e_max
# df_p = df.pivot(index="rh", columns="tdb", values="e_max-e_rsw")
df_p = df.pivot(index="rh", columns="tdb", values="disc")
df_p = df_p.sort_index(ascending=False)
sns.heatmap(df_p, vmax=5)  # , annot=True, fmt=".0f")
plt.show()
