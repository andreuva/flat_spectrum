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
import pickle,struct,sys
import matplotlib.pyplot as plt
from tqdm import tqdm

# Measure time
# import time,sys

# np.seterr(all='raise')

def main():
    """ Main code
    """

    # Initializating the conditions, state and RT coefficients
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

    #############
    # Debug SEE #
    #############
   #if True:
    if False:

        # Import extras
        from physical_functions import jsymbols
        JS = jsymbols()

        '''
        def fffu(Jl,Ml,Ju,Mu):
            ww = JS.j3(Ju,Jl,1,-Mu,Ml,Mu-Ml)
            return (2.*Ju+1.)*ww*ww
        def fffl(Jl,Ml,Ju,Mu):
            ww = JS.j3(Ju,Jl,1,-Mu,Ml,Mu-Ml)
            return (2.*Jl+1.)*ww*ww

        jls = [1]
        jus = [2,1,0]
        for jl in jls:
            mls = np.linspace(-jl,jl,2*jl+1,endpoint=True)
            for mll in mls:
                ml = int(round(mll))
                for ju in jus:
                    mus = np.linspace(-ju,ju,2*ju+1,endpoint=True)
                    for muu in mus:
                        mu = int(round(muu))
                        if np.absolute(mu-ml) > 1:
                            continue
                        print(f'({jl:1d}{ml:2d},{ju:1d}{mu:2d}) = ' + \
                              f'{fffu(jl,ml,ju,mu)} {fffl(jl,ml,ju,mu)}')
        sys.exit()
        '''

        # Define a point
        pointC = point(st.atomic[1], st.radiation[1], cdt.zz[1])

        # Ad-hoc radiation field
        JKQ = {0: {0: 3e0 + 0j}, \
               1: {0: 0. + 0.j, \
                   1: 0. + 0.j}, \
               2: {0: 0. + 0.j, \
                   1: 0. + 0.j, \
                   2: 0. + 0.j}}

        # Get negative Q
        for K in range(3):
            for Q in range(1,K+1):
                if Q == 1:
                    ss = -1.0
                elif Q == 2:
                    ss = 1.0
                JKQ[K][-Q] = ss*np.conjugate(JKQ[K][Q])

        # Add units to JKQ
        for K in range(3):
            for Q in range(-K,K+1):
                JKQ[K][Q] = JKQ[K][Q]

        # Get Jqq (assuming Helium)
        line = pointC.atomic.atom.lines[0]

        # For each q
        for qq in range(-1,2):

            # Factor for q
            f1 = JS.sign(1+qq)

            # Initialize jqq
            line.jqq[qq] = {}

            # For each q'
            for qp in range(-1,2):

                # Initialize
                line.jqq[qq][qp] = (0. + 0.j)

                # For each K
                for K in range(3):

                    # K factor
                    f2 = f1*np.sqrt((2.*K+1.)/3.)

                    # Get Q from 3J
                    Q = qq - qp

                    # Control Q
                    if np.absolute(Q) > K:
                        continue

                    # Contribution
                    contr = f2*JS.j3(1,1,K,qq,-qp,-Q)*JKQ[K][Q]

                    # Add contribution
                    line.jqq[qq][qp] += contr

        # Ad-hoc Einstein
       #line.A_ul /= line.A_ul
       #line.B_ul /= line.B_ul
       #line.B_lu = line.B_ul * (pointC.atomic.atom.terms[1].g/ \
       #                         pointC.atomic.atom.terms[0].g)
       #line.B_ul *= 0.
       #for i in range(20):
       #    print('NO STIMULATED')
       #line.B_lu /= line.B_lu

        # Summon SEE
        pointC.atomic.solveESE(st.radiation[1],cdt)

        # Exit debug
        sys.exit('Debugging SEE')
    #############
    # Debug SEE #
    #############

    # Plot quadrature
   #plot_quadrature(cdt, directory=pm.dir)

    # Debug
    debug = False

    # Load data
    # ''
    # TODO TODO TODO TODO TODO

    # Not loading
    f = open(datadir+'MRC', 'w')
    f.close()

    # Start the main loop for the Lambda iteration
   #for itteration in tqdm(range(cdt.max_iter), desc='Lambda itteration progress'):
    for itteration in range(cdt.max_iter):

        # Debug
        if debug:
            print(f'Starting iteration {itteration+1}')

        # Reset the internal state for a new itteration
        st.new_itter()

        # Go through all the rays in the cuadrature
       #for j, ray in enumerate(tqdm(cdt.rays, desc='propagating rays', leave=False)):
        for j, ray in enumerate(cdt.rays):

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

            # Debug
            if debug:
                print(f'\nPropagating ray {j} {ray.inc} {ray.az}: {iz0}--{iz1}:{step}')

            # Deal with very first point computing first and second
            z = iz0

            # Allocate point
            point_O = point(st.atomic[z], st.radiation[z], cdt.zz[z])

            # If top boundary
            if iz0 == -1:

                # Set Stokes
                point_O.setradiationas(st.space_rad)

                # Debug
                if debug:
                    print(f'Defined first point at top boundary')

            # If bottom boundary
            elif iz0 == 0:

                # Set Stokes
                point_O.setradiationas(st.sun_rad[j])

                # Debug
                if debug:
                    print(f'Defined first point at bottom boundary')

                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

                # Debug
                if debug:
                    print(f'Added Jqq contribution at point {z}')

            # Get RT coefficients at initial point
            sf_o, kk_o = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)

            # Debug
            if debug:
                print(f'Got RT coefficients at point {z}')

            # Get next point and its RT coefficients
            z += step
            point_P = point(st.atomic[z], st.radiation[z], cdt.zz[z])
            sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

            # Debug
            if debug:
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

                # Debug
                if debug:
                    print(f'New current point {z}')

                # If we are in the boundaries, compute the CL for the IC (z=0)
                cent_limb_coef = 1
                if z == iz1 - step:

                    point_P = False
                    lineal = True

                    # Debug
                    if debug:
                        print(f'Last linear point {z}')

                else:

                    point_P = point(st.atomic[z+step], st.radiation[z+step], \
                                    cdt.zz[z+step])

                    # Debug
                    if debug:
                        print(f'Prepare next point {z+step}')

                    # Compute the RT coeficients for the next point (for solving RTE)
                    sf_p, kk_p = RT_coeficients.getRTcoefs(point_P.atomic, ray, cdt)

                    # Debug
                    if debug:
                        print(f'Got RT coefficients at point {z+step}')

                # Propagate
                tau_tot = BESSER(point_M, point_O, point_P, \
                                 sf_m, sf_o, sf_p, \
                                 kk_m, kk_o, kk_p, \
                                 ray, cdt, tau_tot, not lineal, \
                                 tau, cent_limb_coef)

                # Debug
                if debug:
                    print(f'Besser {z-step}-{z}-{z+step}')

                # If last point
                if lineal:

                    # If ray going out
                    if not ray.is_downward:

                        # Store last Stokes parameters
                        f = open(datadir + f'stokes_{itteration:03d}_{j:02d}', 'wb')
                        N_nus = point_O.radiation.nus.size
                        f.write(struct.pack('i',N_nus))
                        f.write(struct.pack('d'*N_nus,*point_O.radiation.nus))
                        for i in range(4):
                            f.write(struct.pack('d'*N_nus, \
                                                *point_O.radiation.stokes[i]))
                        f.close()

                        # Store optical depth
                        f = open(datadir + f'tau_{itteration:03d}_{j:02d}', 'w')
                        N_nus = point_O.radiation.nus.size
                        f.write(f'{N_nus}\n')
                        for nu,ta in zip(point_O.radiation.nus,tau):
                            f.write(f'{1e7*c.c/nu:25.16f}  {ta:23.16e}\n')
                        f.close()

                # Add to Jqq
                point_O.sumStokes(ray,cdt.nus_weights,cdt.JS)

                # Debug
                if debug:
                    print(f'Added Jqq contribution at point {z}')

        # Update the MRC and check wether we reached convergence
        st.update_mrc(cdt, itteration)

        # Write into file
        f = open(datadir+f'MRC','a')
        f.write(f'{itteration:4d}   {st.mrc_p:14.8e}  {st.mrc_c:14.8e}\n')
        f.close()

        # If converged
        if (st.mrc_p < pm.tolerance_p and st.mrc_c < pm.tolerance_c):
            print('\n----------------------------------')
            print(f'FINISHED WITH A TOLERANCE OF {st.mrc_p};{st.mrc_c}')
            print('----------------------------------')
            break
        elif itteration == cdt.max_iter-1:
            print('\n----------------------------------')
            print(f'FINISHED DUE TO MAXIMUM ITERATIONS {cdt.max_iter}')
            print('----------------------------------')

    # Save the rho quantities
    io_saverho(datadir,st.atomic)
    io_saverad(datadir,st.atomic)

    # Remove unused boundaries
    for j in cdt.rays:
        st.sun_rad.pop(0)

    # Go through all the rays in the emergent directions
   #for j, ray in enumerate(tqdm(cdt.orays, desc='emerging rays', leave=False)):
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
        f = open(datadir + f'stokes_{j:02d}', 'wb')
        N_nus = point_O.radiation.nus.size
        f.write(struct.pack('i',N_nus))
        f.write(struct.pack('d'*N_nus,*point_O.radiation.nus))
        for i in range(4):
            f.write(struct.pack('d'*N_nus,*point_O.radiation.stokes[i]))
        f.close()

        f = open(datadir + f'tau_{j:02d}', 'w')
        N_nus = point_O.radiation.nus.size
        f.write(f'{N_nus}\n')
        for nu,ta in zip(point_O.radiation.nus,tau):
            f.write(f'{1e7*c.c/nu:25.16f}  {ta:23.16e}\n')
        f.close()

if __name__ == '__main__':
    main()
