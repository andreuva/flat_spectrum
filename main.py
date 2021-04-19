from atom import ESE
from RTcoefs import RTcoefs
from conditions import conditions, state
import parameters as pm

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

for itteration in tqdm(range(cdt.max_iter)):
    st.new_itter()

    for i, point in enumerate(cdt.zz):
        for j, ray in enumerate(cdt.rays):

            if ray.is_downward():
                z = -i - 1
                step = -1
            else:
                z = i
                step = 1

            eps, KK = RT_coeficients.getRTcoefs(st.atomic[z], ray)

            st.radiation[z].sumStokes(ray)

    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
