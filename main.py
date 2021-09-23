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

# Measure time
# import time,sys

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

# ''
# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    '''
    ####
    # DEBUG
    ####
    start = time.time()
    mod = 'SCreorder'
    f1 = open('debugKS-'+mod, 'w')
    f2 = open('debugI-'+mod, 'w')
    '''

    # Go through all the rays in the cuadrature
    for j, ray in enumerate(tqdm(cdt.rays, desc='propagating rays', leave=False)):

        '''
        # DEBUG for Voigt and getRTcoefs
        NN = 50
        mod = 'original'
        mod = 'savevoigt'
        mod = 'rtcoefedit'
        mod = 'savevoigtrtcoefedit'
        start = time.time()
        for ii in range(NN):
            z = cdt.z_N//2
            point_D = point(st.atomic[z],st.radiation[z],cdt.zz[z])
            em, ab, sf, kk = RT_coeficients.getRTcoefs(point_D.atomic, ray, cdt)
            if ii == 0:
                f = open('debugI-'+mod, 'w')
                f.write('em: ')
                f.write('{0:16.8e}\n'.format(em))
                f.write('ab: ')
                f.write('{0:16.8e}\n'.format(ab))
                f.write('KK,sf\n')
                for mm in range(sf.shape[1]):
                    for jj in range(4):
                        fmt = '{0:1d} {1:3d} -- {2:16.8e} {3:16.8e}'
                        fmt += '{4:16.8e} {5:16.8e} {6:16.8e}\n'
                        f.write(fmt.format(
                                      jj,mm,kk[jj,0][mm],kk[jj,1][mm], \
                                      kk[jj,2][mm],kk[jj,3][mm], \
                                      sf[jj][mm]))
                f.close()
            if ii == NN-1:
                f = open('debugF-'+mod, 'w')
                f.write('em: ')
                f.write('{0:16.8e}\n'.format(em))
                f.write('ab: ')
                f.write('{0:16.8e}\n'.format(ab))
                f.write('KK,sf\n')
                for mm in range(sf.shape[1]):
                    for jj in range(4):
                        fmt = '{0:1d} {1:3d} -- {2:16.8e} {3:16.8e}'
                        fmt += '{4:16.8e} {5:16.8e} {6:16.8e}\n'
                        f.write(fmt.format(
                                      jj,mm,kk[jj,0][mm],kk[jj,1][mm], \
                                      kk[jj,2][mm],kk[jj,3][mm], \
                                      sf[jj][mm]))
                f.close()
        end = time.time()
        f = open('time-'+mod, 'w')
        f.write('{0}'.format(end-start))
        f.close()
        sys.exit()
        '''

        # Initialize arrays for later plots
        tau_tot = np.array([0])
        source = np.array([])
        emisivity = np.array([])
        absortivity = np.array([])

        # Reset lineal and set cent_limb_coef
        cent_limb_coef = 1
        lineal = False

        # Define limits in height index and direction
        if ray.is_downward:
            step = -1
            iz0 = -1
            iz1 = -cdt.z_N - 1
        else:
            step = 1
            iz0 = 0
            iz1 = cdt.z_N

        # Deal with very first point computing first and second
        z = iz0
        if iz0 == -1:
            point_O = point(st.space_atom, st.space_rad, cdt.zf+cdt.dz)
        elif iz0 == 0:
            point_O = point(st.sun_atom, st.sun_rad, cdt.z0-cdt.dz)
        em_o, ab_o, sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)
        point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
        em_p, ab_p, sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

        # Go through all the points (besides 0 and -1 for being IC)
        for z in range(iz0, iz1, step):

            # Shift data
            point_M = point_O
            point_O = point_P
            em_m = em_o
            ab_m = ab_o
            sf_m = sf_o
            kk_m = kk_o
            em_o = em_p
            ab_o = ab_p
            sf_o = sf_p
            kk_o = kk_p

            # If we are in the boundaries, compute the CL for the IC (z=0)
            cent_limb_coef = 1
            if z == iz1 - step:
                point_P = False
                lineal = True
            else:
                point_P = point(st.atomic[z+step], st.radiation[z+step], cdt.zz[z+step])

            # Compute the RT coeficients for plots
            source = np.append(source, sf_o[0][int(cdt.nus_N/2)].value)
            emisivity = np.append(emisivity, em_o)
            absortivity = np.append(absortivity, ab_o)

            # Compute the RT coeficients for the next point (for solving RTE)
            if not lineal:
                em_p, ab_p, sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)
                tau_tot = BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, ray, cdt, tau_tot, cent_limb_coef)
            else:
                LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt)

            # Adding the ray contribution to the Jqq's
            point_O.radiation.sumStokes(ray)

            '''
            ####
            # DEBUG
            ####
            sf = sf_o
            kk = kk_o
            f1.write('{0:2d} {1:2d}\n'.format(j,z))
            f2.write('{0:2d} {1:2d}\n'.format(j,z))
            for mm in range(sf.shape[1]):
                for jj in range(4):
                    fmt = '{0:1d} {1:3d} -- {2:16.8e} {3:16.8e} '
                    fmt += '{4:16.8e} {5:16.8e} {6:16.8e}\n'
                    f1.write(fmt.format(
                                  jj,mm,kk[jj,0][mm],kk[jj,1][mm], \
                                  kk[jj,2][mm],kk[jj,3][mm], \
                                  sf[jj][mm]))
                fmt = '{0:3d} -- {1:16.8e} {2:16.8e} '
                fmt += '{3:16.8e} {4:16.8e} '
                f2.write(fmt.format(mm,point_O.radiation.stokes[0][mm], \
                                       point_O.radiation.stokes[1][mm], \
                                       point_O.radiation.stokes[2][mm], \
                                       point_O.radiation.stokes[3][mm]))
            '''

            if z == iz0:
                jqq_base = point_O.radiation.jqq
                file = open(datadir + f"jqq_base_{itteration}_{round(ray.inc.value)}_{round(ray.az.value)}.pkl", "wb")
                pickle.dump(jqq_base, file)
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

    '''
    ####
    # DEBUG
    ####
    f1.close()
    f2.close()
    end = time.time()
    f = open('time-'+mod, 'w')
    f.write('{0}'.format(end-start))
    f.close()
    sys.exit()

    rad_jqq = {}
    for i in range(cdt.z_N):
        rad_jqq[i] = st.radiation[i].jqq

    file = open(unique_filename(datadir, "jqq", 'pkl'), "wb")
    pickle.dump(rad_jqq, file)
    file.close()

    glob_pop = np.array([st.atomic[i].rho for i in range(cdt.z_N)])
    np.savetxt(unique_filename(datadir, "rho_qq_before", 'csv'), glob_pop)

    # Update the MRC and check wether we reached convergence
    st.update_mrc(cdt, itteration)

    glob_pop = np.array([st.atomic[i].rho for i in range(cdt.z_N)])
    np.savetxt(unique_filename(datadir, "rho_qq_after", 'csv'), glob_pop)

    if (st.mrc.max() < pm.tolerance):
        print('\n----------------------------------')
        print(f'FINISHED WITH A TOLERANCE OF {st.mrc.max()}')
        print('----------------------------------')
        break
    '''

'''
# Start the main loop for the Lambda iteration
for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    # Reset the internal state for a new itteration
    st.new_itter()

    ####
    # DEBUG
    ####
    start = time.time()
    mod = 'original'
    f1 = open('debugKS-'+mod, 'w')
    f2 = open('debugI-'+mod, 'w')

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

            ####
            # DEBUG
            ####
            sf = sf_o
            kk = kk_o
            f1.write('{0:2d} {1:2d}\n'.format(j,z))
            f2.write('{0:2d} {1:2d}\n'.format(j,z))
            for mm in range(sf.shape[1]):
                for jj in range(4):
                    fmt = '{0:1d} {1:3d} -- {2:16.8e} {3:16.8e} '
                    fmt += '{4:16.8e} {5:16.8e} {6:16.8e}\n'
                    f1.write(fmt.format(
                                  jj,mm,kk[jj,0][mm],kk[jj,1][mm], \
                                  kk[jj,2][mm],kk[jj,3][mm], \
                                  sf[jj][mm]))
                fmt = '{0:3d} -- {1:16.8e} {2:16.8e} '
                fmt += '{3:16.8e} {4:16.8e} '
                f2.write(fmt.format(mm,point_O.radiation.stokes[0][mm], \
                                       point_O.radiation.stokes[1][mm], \
                                       point_O.radiation.stokes[2][mm], \
                                       point_O.radiation.stokes[3][mm]))

            if i == 0:
                jqq_base = point_O.radiation.jqq
                file = open(datadir + f"jqq_base_{itteration}_{round(ray.inc.value)}_{round(ray.az.value)}.pkl", "wb")
                pickle.dump(jqq_base, file)
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
        plot_quantity(cdt, cdt.zz, absortivity, names=['Z (CGS)', r'$eta_I$'], directory=pm.dir + 'plots' + subfix)

    ####
    # DEBUG
    ####
    f1.close()
    f2.close()
    end = time.time()
    f = open('time-'+mod, 'w')
    f.write('{0}'.format(end-start))
    f.close()
    sys.exit()

    rad_jqq = {}
    for i in range(cdt.z_N):
        rad_jqq[i] = st.radiation[i].jqq

    file = open(unique_filename(datadir, "jqq", 'pkl'), "wb")
    pickle.dump(rad_jqq, file)
    file.close()

    glob_pop = np.array([st.atomic[i].rho for i in range(cdt.z_N)])
    np.savetxt(unique_filename(datadir, "rho_qq_before", 'csv'), glob_pop)

    # Update the MRC and check wether we reached convergence
    st.update_mrc(cdt, itteration)

    glob_pop = np.array([st.atomic[i].rho for i in range(cdt.z_N)])
    np.savetxt(unique_filename(datadir, "rho_qq_after", 'csv'), glob_pop)

    if (st.mrc.max() < pm.tolerance):
        print('\n----------------------------------')
        print(f'FINISHED WITH A TOLERANCE OF {st.mrc.max()}')
        print('----------------------------------')
        break
#'''
