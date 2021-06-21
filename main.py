# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters as pm
from solver import BESSER, LinSC
from plot_utils import plot_quadrature, plot_z_profile

# Import needed libraries
import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.seterr(all='raise')

# Initializating the conditions, state and RT coefficients
cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

interesting_ray = np.empty((cdt.z_N, 2))

plot_quadrature(cdt)

# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    plot_z_profile(cdt, st)

    # go through all the points (besides 0 and -1 for being IC)
    for j, ray in enumerate(tqdm(cdt.rays, desc=f'propagating rays', leave=False)):
        # go through all the rays in the cuadrature
        for i in range(cdt.z_N):

            if j == 1:
                interesting_ray[i, 0] = st.radiation[i].stokes[0][50].value

            # If the ray is downward start for the last point downward
            if ray.is_downward:
                z = -i - 1
                step = -1
            else:
                z = i
                step = 1

            # If we are in the boundaries, compute the CL for the IC (z=0)
            cent_limb_coef = 1
            lineal = False
            if i == 0:
                cent_limb_coef = ray.clv

                if ray.is_downward:
                    point_M = point(st.space_atom, st.space_rad,         cdt.zf+cdt.dz)
                else:
                    point_M = point(st.sun_atom,   st.sun_rad,           cdt.z0-cdt.dz)
                point_O = point(st.atomic[z],      st.radiation[z],      cdt.zz[z])
                point_P = point(st.atomic[z+step], st.radiation[z+step], cdt.zz[z+step])
            elif i == (len(cdt.zz) - 1):
                point_M = point(st.atomic[z-step], st.radiation[z-step], cdt.zz[z-step])
                point_O = point(st.atomic[z],      st.radiation[z],      cdt.zz[z])
                point_P = False
                lineal = True
            else:
                point_M = point(st.atomic[z-step], st.radiation[z-step], cdt.zz[z-step])
                point_O = point(st.atomic[z],      st.radiation[z],      cdt.zz[z])
                point_P = point(st.atomic[z+step], st.radiation[z+step], cdt.zz[z+step])

            # Compute the RT coeficients for the current and last points (for solving RTE)
            sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)
            sf_m, kk_m = RT_coeficients.getRTcoefs(point_M.atomic, ray, cdt)

            if not lineal:
                sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)
                BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, ray, cdt, cent_limb_coef)
            else:
                LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt)

            # Adding the ray contribution to the Jqq's
            # point_O.radiation.check_I()
            if j == 1:
                interesting_ray[i, 1] = st.radiation[i].stokes[0][50].value

            point_O.radiation.sumStokes(ray)

    # Update the MRC and check wether we reached convergence
    st.update_mrc(cdt, itteration)

    plot_z_profile(cdt, st)

    if (st.mrc.max() < pm.tolerance):
        break
