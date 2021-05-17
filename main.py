# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point, plot_quadrature
import parameters as pm
from solver import BESSER, LinSC

# Import needed libraries
import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initializating the conditions, state and RT coefficients
cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

# plot_quadrature(cdt)

# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    # go through all the points (besides 0 and -1 for being IC)
    for i in tqdm(range(len(cdt.zz)), desc=f'solve RT in the {itteration} itteration', leave=False):
        # go through all the rays in the cuadrature
        for j, ray in enumerate(cdt.rays):

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
                    point_M = point(st.sun_atom,   st.sun_rad,           cdt.z0*0)
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
            point_O.radiation.sumStokes(ray)

    # Update the MRC and check wether we reached convergence
    st.update_mrc(cdt)
    if (st.mrc.max() < pm.tolerance):
        break
