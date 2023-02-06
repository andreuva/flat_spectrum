# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters_comparison as pm
from solver import BESSER, LinSC_old, BESSER_old
from plot_utils import *
import constants as c
from iopy import io_saverho,io_saverad

# Import needed libraries
import numpy as np
import struct, sys
from tensors import Jqq_to_JKQ
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import pickle as pkl

# Measure time
# import time,sys

# np.seterr(all='raise')

def main(pm=pm, disable_display=False):
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
    # plot_quadrature(cdt, directory=datadir)

    # verbose
    verbose = False

    # Load data
    # ''
    # TODO TODO TODO TODO TODO

    # Not loading
    f = open(datadir+'MRC', 'w')
    f.write(f'Itteration  ||  Max. Rel. change\n')
    f.write('-'*50 + '\n')
    f.close()

    for KK in range(3):
        for QQ in range(KK+1):
            # Write the JKQ of each height into a file
            prof_real = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_real[0,:] = 1e7*c.c/st.radiation[0].nus
            prof_imag = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_imag[0,:] = 1e7*c.c/st.radiation[0].nus
            np.savetxt(datadir+f'real_JK{KK}{QQ}_start.out', prof_real)
            np.savetxt(datadir+f'imag_JK{KK}{QQ}_start.out', prof_imag)
            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(15,20), dpi=150)
            # for lay, heigt in enumerate(st.radiation):
            #     real = Jqq_to_JKQ(heigt.jqq, cdt.JS)[KK][QQ].real
            #     imag = Jqq_to_JKQ(heigt.jqq, cdt.JS)[KK][QQ].imag
            #     prof_real[1+lay, :] = real
            #     prof_imag[1+lay, :] = imag
            #     ax1.plot(1e7*c.c/st.radiation[0].nus, real, label=f'{lay:02d}', color=f'C{lay}')
            #     ax2.plot(1e7*c.c/st.radiation[0].nus, imag, color=f'C{lay}')
            # # remove white space between x-axis of the two plots
            # plt.subplots_adjust(hspace=0)
            # ax1.set_title(fr'$J^{KK}_{QQ}$ real and imaginary part')
            # ax1.set_ylabel(fr'$J^{KK}_{QQ}$ real part')
            # ax1.legend()
            # ax2.set_ylabel(fr'$J^{KK}_{QQ}$ imag part')
            # ax2.set_xlabel(r'$\lambda$ (nm)')
            # plt.tight_layout()
            # plt.savefig(datadir+f'JK{KK}{QQ}_start.pdf')
            # plt.close()

    # Start the main loop for the Lambda iteration
    #for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    for itteration in range(cdt.max_iter):
        # print(f'Starting iteration {itteration+1}')

        # Reset the internal state for a new itteration
        st.new_itter()

        # Go through all the rays in the cuadrature
        # for j, ray in enumerate(cdt.rays):
        for j, ray in enumerate(tqdm(cdt.rays, desc=f'propagating rays itteration {itteration}', disable=disable_display)):

            # Initialize optical depth
            tau = np.zeros((cdt.nus_N))

            # Reset lineal
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
                                 tau)

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
    # jqq_list = glob(f'{cdt.dir}jqq_*.pkl')
    # jqq_list.sort()
    # with open(jqq_list[-1], 'rb') as file:
        # jqq = pkl.load(file)
        # components = list(jqq.keys())
        # JKQ_1_bc = Jqq_to_JKQ(jqq[components[0]], cdt.JS)
        # JKQ_2_bc = Jqq_to_JKQ(jqq[components[1]], cdt.JS)
  
    for KK in range(3):
        for QQ in range(KK+1):
            # Write the JKQ of each height into a file
            prof_real = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_real[0,:] = 1e7*c.c/st.radiation[0].nus
            prof_imag = np.zeros((1 + cdt.z_N,len(st.radiation[0].nus)))
            prof_imag[0,:] = 1e7*c.c/st.radiation[0].nus
            # Also the integrated ones
            if st.atomic[0].atom.lines[0].especial:
                int_real_red = np.zeros((cdt.z_N))
                int_imag_red = np.zeros((cdt.z_N))
                int_real_blue = np.zeros((cdt.z_N))
                int_imag_blue = np.zeros((cdt.z_N))
            else:
                int_real = np.zeros((cdt.z_N))
                int_imag = np.zeros((cdt.z_N))
            # fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(15,20), dpi=150)
            for lay, heigt in enumerate(st.radiation):
                real = Jqq_to_JKQ(heigt.jqq, cdt.JS)[KK][QQ].real
                imag = Jqq_to_JKQ(heigt.jqq, cdt.JS)[KK][QQ].imag
                prof_real[1+lay, :] = real
                prof_imag[1+lay, :] = imag

            for lay, heigt in enumerate(st.atomic):
                if heigt.atom.lines[0].especial:
                    red, blue = st.atomic[0].atom.lines[0].resos[0], st.atomic[0].atom.lines[0].resos[1]
                    int_real_red[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq[red], cdt.JS)[KK][QQ].real
                    int_imag_red[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq[red], cdt.JS)[KK][QQ].imag
                    int_real_blue[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq[blue], cdt.JS)[KK][QQ].real
                    int_imag_blue[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq[blue], cdt.JS)[KK][QQ].imag
                else:
                    int_real[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq, cdt.JS)[KK][QQ].real
                    int_imag[lay] = Jqq_to_JKQ(heigt.atom.lines[0].jqq, cdt.JS)[KK][QQ].imag

                # ax1.plot(1e7*c.c/st.radiation[0].nus, real, label=f'{lay:02d}', color=f'C{lay}')
                # ax2.plot(1e7*c.c/st.radiation[0].nus, imag, color=f'C{lay}')
            # ax1.axhline(JKQ_1_bc[KK][QQ].real, label='JKQ_1')
            # ax1.axhline(JKQ_2_bc[KK][QQ].real, label='JKQ_2')
            # ax2.axhline(JKQ_1_bc[KK][QQ].imag, label='JKQ_1')
            # ax2.axhline(JKQ_2_bc[KK][QQ].imag, label='JKQ_2')
            # remove white space between x-axis of the two plots
            # plt.subplots_adjust(hspace=0)
            # ax1.set_title(fr'$J^{KK}_{QQ}$ real and imaginary part')
            # ax1.set_ylabel(fr'$J^{KK}_{QQ}$ real part')
            # ax1.legend()
            # ax2.set_ylabel(fr'$J^{KK}_{QQ}$ imag part')
            # ax2.set_xlabel(r'$\lambda$ (nm)')
            # plt.tight_layout()
            # plt.savefig(datadir+f'JK{KK}{QQ}_finished.pdf')
            # plt.close()
            np.savetxt(datadir+f'real_JK{KK}{QQ}_finished.out', prof_real)
            np.savetxt(datadir+f'imag_JK{KK}{QQ}_finished.out', prof_imag)
            if st.atomic[0].atom.lines[0].especial:
                np.savetxt(datadir+f'int_real_red_JK{KK}{QQ}_finished.out', int_real_red)
                np.savetxt(datadir+f'int_imag_red_JK{KK}{QQ}_finished.out', int_imag_red)
                np.savetxt(datadir+f'int_real_blue_JK{KK}{QQ}_finished.out', int_real_blue)
                np.savetxt(datadir+f'int_imag_blue_JK{KK}{QQ}_finished.out', int_imag_blue)
            else:
                np.savetxt(datadir+f'int_real_JK{KK}{QQ}_finished.out', int_real)
                np.savetxt(datadir+f'int_imag_JK{KK}{QQ}_finished.out', int_imag)

    # Remove unused boundaries
    for j in cdt.rays:
        st.sun_rad.pop(0)

    # Go through all the rays in the emergent directions
    outputs = []
    # for j, ray in enumerate(tqdm(cdt.orays, desc='emerging rays', leave=False)):
    for j, ray in enumerate(cdt.orays):

        # Initialize optical depth
        tau = np.zeros((cdt.nus_N))

        # Reset lineal
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
            point_O.setradiationas(st.osun_rad[j])

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
                             tau)

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
        outputs.append([ray, point_O.radiation.nus, cdt.zz, tau, point_O.radiation.stokes])

    return outputs

if __name__ == '__main__':
    main()
