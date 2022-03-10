# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters as pm
from solver import BESSER, LinSC_old, BESSER_old
from plot_utils import *
import constants as c
from iopy import io_saverho,io_saverad

# Import needed libraries
import numpy as np
import pickle, struct, sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Measure time
# import time,sys

# np.seterr(all='raise')

def main(cdt=conditions(pm)):
    """ Main code
    """

    # Initializating state and RT coefficients
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)
    st = state(cdt)

    # Create path for output files
    if not os.path.exists(pm.dir):
        os.makedirs(pm.dir)
    datadir = pm.dir + 'out'
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    datadir = pm.dir + 'out/'

    # Clean
    for fil in os.listdir(datadir):
        if not os.path.isdir(datadir+fil):
            os.remove(datadir+fil)

    # Plot quadrature
    # plot_quadrature(cdt, directory=cdt.datadir)

    # verbose
    verbose = False

    # Load data
    # ''
    # TODO TODO TODO TODO TODO

    # Not loading
    # f = open(datadir+'MRC', 'w')
    # f.close()

    # Start the main loop for the Lambda iteration
    #for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    for itteration in range(cdt.max_iter):


        # print(f'Starting iteration {itteration+1}')

        # Reset the internal state for a new itteration
        st.new_itter()

        # Go through all the rays in the cuadrature
        # for j, ray in enumerate(cdt.rays):
        for j, ray in enumerate(tqdm(cdt.rays, desc=f'propagating rays itteration {itteration}')):

            # Initialize optical depth
            tau = np.zeros((cdt.nus_N))

            # Reset lineal and set cent_limb_coef
            cent_limb_coef = 1
            lineal = False
            tau_tot = [0.]

            # Define limits in height index and direction
            if ray.is_downward:
                step = -1
                iz0 = -1
                iz1 = -cdt.z_N - 1
            else:
                step = 1
                iz0 = 0
                iz1 = cdt.z_N

            # verbose
            if verbose:
                print(f'\nPropagating ray {j} {ray.inc} {ray.az}: {iz0}--{iz1}:{step}')

            # Deal with very first point computing first and second
            z = iz0

            # Allocate point
            point_O = point(st.atomic[z], st.radiation[z], cdt.zz[z])

            # If top boundary
            if iz0 == -1:

                # Set Stokes
                point_O.setradiationas(st.space_rad)

                # verbose
                if verbose:
                    print(f'Defined first point at top boundary')

            # If bottom boundary
            elif iz0 == 0:

                # Set Stokes
                point_O.setradiationas(st.sun_rad[j])

                # verbose
                if verbose:
                    print(f'Defined first point at bottom boundary')

                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

                # verbose
                if verbose:
                    print(f'Added Jqq contribution at point {z}')

            # Get RT coefficients at initial point
            sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)

            # verbose
            if verbose:
                print(f'Got RT coefficients at point {z}')

            # Get next point and its RT coefficients
            z += step
            point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
            sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

            # verbose
            if verbose:
                print(f'Got RT coefficients at point {z}')

            # Go through all the points (besides 0 and -1 for being IC)
            for z in range(iz0+step, iz1, step):

                # Shift data
                point_M = point_O
                point_O = point_P
                sf_m = sf_o
                kk_m = kk_o
                sf_o = sf_p
                kk_o = kk_p
                kk_p = None
                sf_p = None

                # verbose
                if verbose:
                    print(f'New current point {z}')

                # If we are in the boundaries, compute the CL for the IC (z=0)
                cent_limb_coef = 1
                if z == iz1 - step:

                    point_P = False
                    lineal = True

                    # verbose
                    if verbose:
                        print(f'Last linear point {z}')

                else:

                    point_P = point(st.atomic[z+step], st.radiation[z+step], \
                                    cdt.zz[z+step])

                    # verbose
                    if verbose:
                        print(f'Prepare next point {z+step}')

                    # Compute the RT coeficients for the next point (for solving RTE)
                    sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

                    # verbose
                    if verbose:
                        print(f'Got RT coefficients at point {z+step}')

                # Propagate
                tau_tot = BESSER(point_M, point_O, point_P, \
                                 sf_m, sf_o, sf_p, \
                                 kk_m, kk_o, kk_p, \
                                 ray, cdt, tau_tot, not lineal, \
                                 tau, cent_limb_coef)

                # verbose
                if verbose:
                    print(f'Besser {z-step}-{z}-{z+step}')

                # If last point
                # if lineal:

                    # If ray going out
                    # if not ray.is_downward:

                        # Store last Stokes parameters
                        # f = open(datadir + f'stokes_{itteration:03d}_{j:02d}', 'wb')
                        # N_nus = point_O.radiation.nus.size
                        # f.write(struct.pack('i',N_nus))
                        # f.write(struct.pack('d'*N_nus,*point_O.radiation.nus))
                        # for i in range(4):
                        #     f.write(struct.pack('d'*N_nus, \
                        #                         *point_O.radiation.stokes[i]))
                        # f.close()

                        # Store optical depth
                        # f = open(datadir + f'tau_{itteration:03d}_{j:02d}', 'w')
                        # N_nus = point_O.radiation.nus.size
                        # f.write(f'{N_nus}\n')
                        # for nu,ta in zip(point_O.radiation.nus,tau):
                        #     f.write(f'{1e7*c.c/nu:25.16f}  {ta:23.16e}\n')
                        # f.close()

                # Add to Jqq
                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

                # verbose
                if verbose:
                    print(f'Added Jqq contribution at point {z}')

        # Update the MRC and check wether we reached convergence
        st.update_mrc(cdt, itteration)

        # Write into file
        f = open(datadir+f'MRC','a')
        f.write(f'{itteration:4d}   {st.mrc_p:14.8e}  {st.mrc_c:14.8e}\n')
        f.close()

        # If converged
        if (st.mrc_p < cdt.tolerance_p and st.mrc_c < cdt.tolerance_c):
            print('\n----------------------------------')
            print(f'FINISHED WITH A TOLERANCE OF {st.mrc_p};{st.mrc_c}')
            print('----------------------------------')
            break
        elif itteration == cdt.max_iter-1:
            print('\n----------------------------------')
            print(f'FINISHED DUE TO MAXIMUM ITERATIONS {cdt.max_iter}')
            print('----------------------------------')

    # Save the rho quantities
    # io_saverho(datadir,st.atomic)
    # io_saverad(datadir,st.atomic)

    # Remove unused boundaries
    for j in cdt.rays:
        st.sun_rad.pop(0)

    # Go through all the rays in the emergent directions
    outputs = []
    # for j, ray in enumerate(tqdm(cdt.orays, desc='emerging rays', leave=False)):
    for j, ray in enumerate(cdt.orays):

        # Initialize optical depth
        tau = np.zeros((cdt.nus_N))

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
        point_O = point(st.atomic[z], st.radiation[z], cdt.zz[z])
        z = iz0
        if iz0 == -1:
            point_O.setradiationas(st.space_rad)
        elif iz0 == 0:
            point_O.setradiationas(st.sun_rad[j])

        # Get RT coefficients at initial point
        sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)

        # Get next point and its RT coefficients
        z += step
        point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
        sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

        # Go through all the points (besides 0 and -1 for being IC)
        for z in range(iz0, iz1, step):

            # Shift data
            point_M = point_O
            point_O = point_P
            sf_m = sf_o
            kk_m = kk_o
            sf_o = sf_p
            kk_o = kk_p
            kk_p = None
            sf_p = None

            # If we are in the boundaries, compute the CL for the IC (z=0)
            cent_limb_coef = 1
            if z == iz1 - step:
                point_P = False
                lineal = True
            else:
                point_P = point(st.atomic[z+step], st.radiation[z+step], \
                                cdt.zz[z+step])
                # Compute the RT coeficients for the next point (for solving RTE)
                sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

            # Transfer
            tau_tot = BESSER(point_M, point_O, point_P, \
                             sf_m, sf_o, sf_p, \
                             kk_m, kk_o, kk_p, \
                             ray, cdt, tau_tot, not lineal, \
                             tau, cent_limb_coef)

        # Store last Stokes parameters
        # f = open(datadir + f'stokes_{j:02d}', 'wb')
        f = open(datadir + f'stokes_{j:02d}.out', 'w')
        N_nus = point_O.radiation.nus.size
        # f.write(struct.pack('i',N_nus))
        # f.write(struct.pack('d'*N_nus,*point_O.radiation.nus))
        # for i in range(4):
        #     f.write(struct.pack('d'*N_nus,*point_O.radiation.stokes[i]))
        f.write(f'Number of frequencies:\t{N_nus}\n')
        f.write(f'frequencies(cgs)\t\tI\t\tQ\t\tU\t\tV\n')
        f.write('----------------------------------------------------------\n')
        for i in range(N_nus):
            f.write(f'{point_O.radiation.nus[i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[0][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[1][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[2][i]:25.16e}\t' + \
                    f'{point_O.radiation.stokes[3][i]:25.16e}\n')
        f.close()

        f = open(datadir + f'tau_{j:02d}.out', 'w')
        N_nus = point_O.radiation.nus.size
        f.write(f'Number of wavelengths:\t{N_nus}\n')
        f.write(f'wavelengths(nm)\ttau\n')
        f.write('----------------------------------------------------------\n')
        for nu,ta in zip(point_O.radiation.nus,tau):
            f.write(f'{1e7*c.c/nu:25.16f}  {ta:23.16e}\n')
        f.close()

        # add the ray, wavelegnths, taus, and stokes to a variable and output it
        outputs.append([ray, point_O.radiation.nus, tau, point_O.radiation.stokes])
    return outputs

if __name__ == '__main__':
    main()
