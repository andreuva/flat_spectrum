from RTcoefs import RTcoefs
from conditions import conditions, state
import parameters as pm

import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from tqdm import tqdm

cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

for itteration in tqdm(range(cdt.max_iter)):
    st.new_itter()

    for i, point in enumerate(cdt.zz[1:-1], start=1):
        for j, ray in enumerate(cdt.rays):

            cent_limb_coef = 1

            if ray.is_downward():
                z = -i - 1
                step = -1
            else:
                if i == 1:
                    cent_limb_coef = 1 - 0.62 + 0.2 + 0.62*np.cos(ray.inc) - 0.2*np.cos(ray.inc)**2
                z = i
                step = 1
            
            eps, KK = RT_coeficients.getRTcoefs(st.atomic[z], ray)
            _ , KKm = RT_coeficients.getRTcoefs(st.atomic[z - step], ray)

            tauMO = (st.tau[z] - st.tau[z - step])/np.cos(ray.inc)

            psim = (1 - np.exp(-tauMO)*(1 + tauMO))/(tauMO)
            psio = (np.exp(-tauMO) + tauMO - 1)/(tauMO)
            
            wm = (2 - np.exp(-tauMO)*(tauMO**2 + 2*tauMO + 2))/(tauMO**2)
            wo = 1 - 2*(np.exp(-tauMO) + tauMO - 1)/(tauMO**2)
            wc = 2*(tauMO - 2 + np.exp(-tauMO)*(tauMO + 2))/(tauMO**2)

            cm = 1

            k_1_inv = (cdt.Id_tens + psio*KKm)
            k_1 = k_1_inv
            k_2 = (np.exp(-tauMO) - psim * KK)
            k_2 = np.einsum("ijb, jkb -> ikb", k_1, k_2)
            kt = np.einsum("ijk, jk -> ik", k_2, st.radiation[z - step].stokes)
            st.radiation[z].stokes = kt*cent_limb_coef + \
                                     wm*st.atomic[z - step].source + \
                                     wo*st.atomic[z].source + wc*cm*pm.I_units
            

            st.radiation[z].sumStokes(ray)   

    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
