# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state
import parameters as pm

# Import needed libraries
import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initializating the conditions, state and RT coefficients
cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter)):
    # Reset the internal state for a new itteration
    st.new_itter()

    # go through all the points (besides 0 and -1 for being IC)
    for i, point in enumerate(cdt.zz[1:-1], start=1):
        # go through all the rays in the cuadrature
        for j, ray in enumerate(cdt.rays):

            # If not specified the CL variation is not computed
            cent_limb_coef = 1

            # If the ray is downward start for the last point downward
            if ray.is_downward():
                z = -i - 1
                step = -1
            else:
                # If we are in the first point, compute the CL for the IC (z=0)
                if i == 1:
                    cent_limb_coef = 1 - 0.62 + 0.2 + 0.62*np.cos(ray.inc) - 0.2*np.cos(ray.inc)**2
                z = i
                step = 1

            # Compute the RT coeficients for the current and last points (for solving RTE)
            eps, KK = RT_coeficients.getRTcoefs(st.atomic[z], ray)
            _ , KKm = RT_coeficients.getRTcoefs(st.atomic[z - step], ray)

            # Obtain the optical thicknes between the points in this ray and compute
            # BESSER coeficients to solve RTE (Jiri Stepan and Trujillo Bueno A&A 557 2013)
            tauMO = np.abs((st.tau[z] - st.tau[z - step])/np.cos(ray.inc))

            psim = (1 - np.exp(-tauMO)*(1 + tauMO))/(tauMO)
            psio = (np.exp(-tauMO) + tauMO - 1)/(tauMO)

            wm = (2 - np.exp(-tauMO)*(tauMO**2 + 2*tauMO + 2))/(tauMO**2)
            wo = 1 - 2*(np.exp(-tauMO) + tauMO - 1)/(tauMO**2)
            wc = 2*(tauMO - 2 + np.exp(-tauMO)*(tauMO + 2))/(tauMO**2)

            cm = 1

            k_1_inv = (cdt.Id_tens + psio*KKm)

            # Inverting the matrices K^-1 for all the wavelenghts
            k_1 = np.zeros_like(k_1_inv)
            for i in range(cdt.nus_N):
                k_1[:,:,i] = np.linalg.solve(k_1_inv[:,:,i], cdt.identity)
            k_2 = (np.exp(-tauMO) - psim * KK)
            # Multipling matrices of all wavelengths with at once (eq 7 and 8)
            k_2 = np.einsum("ijb, jkb -> ikb", k_1, k_2)
            kt = np.einsum("ijk, jk -> ik", k_2, st.radiation[z - step].stokes)
            # Bring all together to compute the new stokes parameters
            st.radiation[z].stokes = kt*cent_limb_coef + \
                                     wm*st.atomic[z - step].source + \
                                     wo*st.atomic[z].source + wc*cm*pm.I_units

            # Adding the ray contribution to the Jqq's
            st.radiation[z].sumStokes(ray)   

    # Update the MRC and check wether we reached convergence
    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
