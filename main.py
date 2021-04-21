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

    for i, point in enumerate(cdt.zz[1:], start=1):
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
            
            taumo = (st.tau[z] - st.tau[z - step])/np.cos(ray.inc)

            psim = (1 - np.exp(-taumo)*(1 + taumo))/(taumo)
            psio = (np.exp(-taumo) + taumo - 1)/(taumo)
            
            wm = (2 - np.exp(-taumo)*(taumo**2 + 2*taumo + 2))/(taumo**2)
            wo = 1 - 2*(np.exp(-taumo) + taumo - 1)/(taumo**2)
            wc = 2*(taumo - 2 + np.exp(-taumo)*(taumo + 2))/(taumo**2)
            
            cm = 1

            for k in range(4):
                st.radiation[z].stokes[k] = (np.exp(-taumo) - psim * KK[k][k])* \
                                            st.radiation[z - step].stokes[k]*cent_limb_coef + \
                                            wm*st.atomic[z - step].source[k] + \
                                            wo*st.atomic[z].source[k] + wc*cm*pm.I_units

            st.radiation[z].sumStokes(ray)   

    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
