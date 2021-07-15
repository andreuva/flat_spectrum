# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters as pm
from solver import BESSER, LinSC
from plot_utils import *

# Import needed libraries
import numpy as np
import pickle
from astropy import units as u
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.seterr(all='raise')

# Initializating the conditions, state and RT coefficients
cdt = conditions(pm)
RT_coeficients = RTcoefs(cdt.nus)
st = state(cdt)

if not os.path.exists(pm.dir):
    os.makedirs(pm.dir)

datadir = pm.dir + 'out'
if not os.path.exists(datadir):
    os.makedirs(datadir)
datadir = pm.dir + 'out/'

plot_quadrature(cdt, directory=pm.dir)

# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    # go through all the points (besides 0 and -1 for being IC)
    for j, ray in enumerate(tqdm(cdt.rays, desc='propagating rays', leave=False)):
        tau_tot = np.array([0])
        source = np.array([])
        emisivity = np.array([])
        absortivity = np.array([])
        # go through all the rays in the cuadrature
        for i in range(cdt.z_N):

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
                # cent_limb_coef = ray.clv
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
            em_o, ab_o, sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)
            _, _, sf_m, kk_m = RT_coeficients.getRTcoefs(point_M.atomic, ray, cdt)

            source = np.append(source, sf_o[0][int(cdt.nus_N/2)].value)
            emisivity = np.append(emisivity, em_o)
            absortivity = np.append(absortivity, ab_o)

            if not lineal:
                _, _, sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)
                tau_tot = BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, ray, cdt, tau_tot, cent_limb_coef)
            else:
                LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt)

            # Adding the ray contribution to the Jqq's
            point_O.radiation.sumStokes(ray)

            if i == 0:
                simetry = point_O.radiation.jqq
                file = open(datadir + f"simetry_{itteration}_{round(ray.inc.value)}_{round(ray.az.value)}.pkl", "wb")
                pickle.dump(simetry, file)
                file.close()

        subfix = f'_itt{itteration}'
        # subfix = f'_ray_inc_{round(ray.inc.value, 1)}_az_{round(ray.az.value, 1)}'
        plot_z_profile(cdt, st, nu=int(cdt.nus_N/2), directory=pm.dir + 'plots_core_norm' + subfix)
        plot_z_profile(cdt, st, directory=pm.dir + 'plots_norm' + subfix)
        plot_stokes_im(cdt, st, directory=pm.dir + 'plots_norm' + subfix)
        plot_z_profile(cdt, st, nu=int(cdt.nus_N/2), norm=False, directory=pm.dir + 'plots_core' + subfix)
        plot_z_profile(cdt, st, norm=False, directory=pm.dir + 'plots_prof' + subfix)
        plot_stokes_im(cdt, st, norm=False, directory=pm.dir + 'plots_prof' + subfix)
        plot_quantity(cdt, cdt.zz, tau_tot, names=['Z (CGS)', r'$\tau$'], directory=pm.dir + 'plots' + subfix)
        plot_quantity(cdt, cdt.zz, source, names=['Z (CGS)', r'$Sf_I$'], directory=pm.dir + 'plots' + subfix)
        plot_quantity(cdt, cdt.zz, emisivity, names=['Z (CGS)', r'$\varepsilon_I$'], directory=pm.dir + 'plots' + subfix)
        plot_quantity(cdt, cdt.zz, absortivity, names=['Z (CGS)', r'$\eta_I$'], directory=pm.dir + 'plots' + subfix)

    rad_jqq = {}
    for i in range(cdt.z_N):
        rad_jqq[i] = st.radiation[i].jqq

    file = open(unique_filename(datadir, "jqq", 'csv'), "wb")
    pickle.dump(rad_jqq, file)
    file.close()

    J_KQ = np.zeros((3, 5, cdt.nus_N)) + 0j
    J_iKQ = {}
    for i in range(cdt.z_N):
        for K in [0, 1, 2]:
            for Q in range(-K, K+1):
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        J_KQ[K][Q+K] += ((-1)**(1-q) * np.sqrt(3*(2*K + 1)) * st.atomic[i].jsim.j3(1, 1, K, q, -qp, -Q) *
                                         st.radiation[i].jqq[q][qp].value)

                J_iKQ[i] = J_KQ

    file = open(unique_filename(datadir, "J_KQ", 'pkl'), "wb")
    pickle.dump(J_iKQ, file)
    file.close()

    # Update the MRC and check wether we reached convergence
    st.update_mrc(cdt, itteration)

    glob_pop = np.array([st.atomic[i].rho for i in range(cdt.z_N)])
    np.savetxt(unique_filename(datadir, "rho_qq", 'csv'), glob_pop)

    rho_KQ = np.zeros((3, 5)) + 0j
    rho_iKQ = {}
    for i in range(cdt.z_N):
        for K in [0, 1, 2]:
            for Q in range(-K, K+1):
                for M in [-1, 0, 1]:
                    for Mp in [-1, 0, 1]:
                        rho_KQ[K][Q+K] += ((-1)**(1-M) * np.sqrt(2*K + 1) * st.atomic[i].jsim.j3(1, 1, K, M, -Mp, Q) *
                                           st.atomic[i].rho_call(1, 1, M, Mp))

                rho_iKQ[i] = rho_KQ

    file = open(unique_filename(datadir, "rho_KQ", 'pkl'), "wb")
    pickle.dump(rho_iKQ, file)
    file.close()

    # file = open("data.pkl", "rb")
    # output = pickle.load(file)
    # print(output)
    # file.close()

    if (st.mrc.max() < pm.tolerance):
        break
