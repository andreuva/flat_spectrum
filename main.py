# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
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


theta = [ray.inc.value*np.sin(ray.az.value) for ray in cdt.rays]
r = np.ones_like(theta)
r0 = np.zeros_like(r)

plt.figure()
plt.subplot(projection="aitoff")

inclinations = np.array([ray.inc.value for ray in cdt.rays])
azimuts = np.array([ray.az.value for ray in cdt.rays])

plt.plot(inclinations - 90, azimuts, 'o')
plt.grid(True)
plt.show()

ax = plt.subplot(111, projection='polar')

for i in range(cdt.rays_N):
    ax.plot([theta[i], theta[i]], [0, 1], 'b', linewidth=0.5)

ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    # go through all the points (besides 0 and -1 for being IC)
    for i in tqdm(range(len(cdt.zz)), desc=f'solve RT in the {itteration} itteration', leave=False):
        # go through all the rays in the cuadrature
        for j, ray in enumerate(cdt.rays):

            # If the ray is downward start for the last point downward
            if ray.is_downward():
                z = -i - 1
                step = -1
            else:
                z = i
                step = 1

            # If we are in the boundaries, compute the CL for the IC (z=0)
            cent_limb_coef = 1
            lineal = False
            if i == 0:
                cent_limb_coef = ray.clv(cdt.z0, cdt.alpha, cdt.theta_crit)

                if ray.is_downward():
                    point_M = point(st.space_atom, st.space_rad,         cdt.zf+cdt.dz)
                else:
                    point_M = point(st.sun_atom,   st.sun_rad,           0)
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
            sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray)
            sf_m, kk_m = RT_coeficients.getRTcoefs(point_M.atomic, ray)

            if not lineal:
                sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray)
                BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, ray, cdt, cent_limb_coef)
            else:
                LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt)

            # Adding the ray contribution to the Jqq's
            point_O.radiation.sumStokes(ray)

    # Update the MRC and check wether we reached convergence
    st.update_mrc()
    if (st.mrc.max() < pm.tolerance):
        break
