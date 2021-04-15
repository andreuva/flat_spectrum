from ESE import ESE
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

    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
