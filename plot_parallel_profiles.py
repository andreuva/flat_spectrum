import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import pickle as pkl
from tqdm import tqdm
import time, os
from glob import glob
import parameters_parallel as pm

from allen import Allen_class
import constants as cts
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0, TKQ


# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rcParams['font.size'] = 10


# Function that computes the unit 3D vector from the inclination and the azimuth
def ang_to_cart(inc, az):
    return np.array([np.sin(inc)*np.cos(az), np.sin(inc)*np.sin(az), np.cos(inc)])


# Function that computes the spherical coordinates vector from the cartesian one
def cart_to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2),z)
    phi = np.arctan2(y,x)
    return np.array([r, theta, phi])


# Function that returns a gaussian profile given a x, mean and std
def gaussian(x, mu, std):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(std, 2.)))


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=np.array([0, 0, 0]), especial=True, jqq=None):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, pm.initial_equilibrium, 0, especial=especial)

    if especial:
        # Retrieve the different components of the line profile
        components = list(atoms.atom.lines[0].jqq.keys())

        # Initialize the jqq and construct the dictionary
        atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

        # reset the jqq to zero to construct from there the radiation field with the JKQ
        atoms.reset_jqq(cdt.nus_N)
        atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(JKQ_1, cdt.JS)
        atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(JKQ_2, cdt.JS)
    else:
        # Initialize the jqq and construct the dictionary
        atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)
        atoms.atom.lines[0].jqq = JKQ_to_Jqq(JKQ_1, cdt.JS)

    # print(atoms.atom.lines[0].jqq)
    if jqq is not None:
        atoms.atom.lines[0].jqq = jqq
    #     atoms.atom.lines[0].jqq = atoms.atom.lines[0].jqq = {276733332165635.5: {-1: {-1: (8.375176071590015e-06+0j), 0: (1.1855266276444219e-19+5.2353657675870114e-18j), 1: (6.347751887308007e-07-2.8554737086374053e-13j)},0: {-1: 0j, 0: (9.00950985634052e-06+0j),1: (-1.1855266276444219e-19-5.235372385031912e-18j)}, 1: {-1: 0j, 0: 0j, 1: (8.374099225151616e-06+0j)}}, 
    #                                                          276764094706172.62:{-1: {-1: (6.094236834279895e-06+0j), 0: (-3.805888397243992e-18-1.685474405888409e-16j), 1: (4.4084743129272864e-07+3.580508415744089e-12j)}, 0: {-1: 0j, 0: (6.5419934862266725e-06+0j), 1: (3.805888397243992e-18+1.6854745051500825e-16j)}, 1: {-1: 0j, 0: 0j, 1: (6.105864267347921e-06+0j)}}}

    # Solve the ESE
    atoms.solveESE(None, cdt)

    # select the ray direction as the otuput ray
    ray = cdt.orays[0]

    # Compute the RT coeficients for a given ray
    sf, kk = RT_coeficients.getRTcoefs(atoms, ray, cdt)

    # Compute the emision coefficients from the Source functions
    profiles = {}
    profiles['nus'] = cdt.nus
    profiles['eps_I'] = sf[0]*(kk[0][0] + cts.vacuum)
    profiles['eps_Q'] = sf[1]*(kk[0][0] + cts.vacuum)
    profiles['eps_U'] = sf[2]*(kk[0][0] + cts.vacuum)
    profiles['eps_V'] = sf[3]*(kk[0][0] + cts.vacuum)

    # retrieve the absorption coefficients from the K matrix
    profiles['eta_I'] = kk[0][0]
    profiles['eta_Q'] = kk[0][1]*(kk[0][0] + cts.vacuum)
    profiles['eta_U'] = kk[0][2]*(kk[0][0] + cts.vacuum)
    profiles['eta_V'] = kk[0][3]*(kk[0][0] + cts.vacuum)
    profiles['rho_Q'] = kk[1][0]*(kk[0][0] + cts.vacuum)
    profiles['rho_U'] = kk[1][1]*(kk[0][0] + cts.vacuum)
    profiles['rho_V'] = kk[1][2]*(kk[0][0] + cts.vacuum)

    return profiles, profiles['nus'], cdt.rays, cdt.orays


def phi_o_calc(tau):
    if tau < 0.0001:
        return  tau/2 - tau**2/6 + tau**3/24
    else:
        return 1 - (1-np.exp(-tau))/tau


def phi_m_calc(tau):
    if tau < 0.0001:
        return - tau/2 + tau**2/6 - tau**3/24 + tau - tau**2/2 + tau**3/6 - tau**4/24
    else:
        return (1-np.exp(-tau))/tau - np.exp(-tau)


if __name__ == '__main__':

    folders = sorted(glob(f'{pm.basedir}tau_*'))

    print(f'{len(folders)} itterations found')

    with open(f'{pm.basedir}zz.pkl', 'rb') as f:
        zzs = pkl.load(f)

    with open(f'{pm.basedir}tau.pkl', 'rb') as f:
        taus = pkl.load(f)

    with open(f'{pm.basedir}BB.pkl', 'rb') as f:
        BBs = pkl.load(f)

    with open(f'{pm.basedir}stokes.pkl', 'rb') as f:
        stokes = pkl.load(f)
    
    stokes_analytical = []

    for i, _ in enumerate(folders):
        folder = f'{pm.basedir}tau_{i % 12}_BB_{BBs[i]}'
        pm.dir = folder
        print('--'*30)
        print(f'Loading {folder}')
        print('wich corresponds grid  (B,tau)   =    ({},{})'.format(i//12, i%12))
        wave_nlte, tau_nlte = np.loadtxt(f'{folder}/out/tau_00.out', skiprows=3, unpack=True)
        tau_nlte = taus[i]
        I_nlte, Q_nlte, U_nlte, V_nlte = stokes[i]

        wave_nlte = wave_nlte*1e-9
        nus_nlte = 299792458/wave_nlte
        tau_nlte_max = tau_nlte.max()

        # create the conditions to compute the JKQ and profiles
        B_spherical = np.array([BBs[i], 90*np.pi/180, 0])
        velocity = np.array(pm.velocity)

        print(f'Computing analitical profiles for {folder}')
        print(f'with parameters:\n B = {B_spherical} \n tau_max = {tau_nlte_max}')
        print('--'*30 + '\n')

        # plot the profiles for the test
        # fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # ax[0, 0].plot(wave_nlte, I_nlte, label='NLTE')
        # ax[0, 1].plot(wave_nlte, Q_nlte/I_nlte, label='NLTE')
        # ax[1, 0].plot(wave_nlte, U_nlte/I_nlte, label='NLTE')
        # ax[1, 1].plot(wave_nlte, V_nlte/I_nlte, label='NLTE')
        # plt.suptitle('Stokes parameters normalized to I')
        # plt.show()

        ###############################################################################################

        # Get Allen class instance and gamma angles
        Allen = Allen_class()
        Allen.get_gamma(pm.z0)

        nu_1 = 2.76733e14
        nu_2 = 2.76764e14
        wave_1 = 299792458/nu_1
        wave_2 = 299792458/nu_2
        gaussian_width = 7e9
        gaussian_1_height = 1e-1
        gaussian_2_height = gaussian_1_height/7

        ###############################################################################################
        #                   Test the computation of the JKQ and the profiles                          #
        ###############################################################################################
        # create the JKQ
        JKQ_1 = construct_JKQ_0()
        JKQ_2 = construct_JKQ_0()

        # compute the JKQ with ad-hoc values
        JKQ_1[0][0] = 1e-4
        JKQ_2[0][0] = 1e-6

        # compute the profiles for the test
        _, nus_cp, rays, _ = compute_profile(JKQ_1, JKQ_1, B=B_spherical, pm=pm, especial=pm.especial)

        # compute the wavelength from the frequencies
        wave_cp = 299792458/nus_cp

        # compute the index of each of the peaks
        nu_peak_1_indx = np.abs(nus_cp - nu_1).argmin()
        nu_peak_2_indx = np.abs(nus_cp - nu_2).argmin()

        ###############################################################################################

        # compute the gaussian profiles for each component
        gauss_1 = gaussian(nus_cp, nu_1, gaussian_width)*gaussian_1_height
        gauss_2 = gaussian(nus_cp, nu_2, gaussian_width)*gaussian_2_height
        gauss_1_norm = gauss_1/np.trapz(gauss_1, nus_cp)
        gauss_2_norm = gauss_2/np.trapz(gauss_2, nus_cp)

        # compute the JKQ taking into account the background and velocity
        JKQ_1 = construct_JKQ_0()
        JKQ_2 = construct_JKQ_0()
        for K in range(3):
            for Q in range(0,K+1):
                for ray in rays:
                    background = Allen.get_radiation(nus_cp)*Allen.get_clv(ray, nus_cp)

                    if ray.rinc < np.pi/2:
                        continue

                    JKQ_1[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                    JKQ_2[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight

                # integrate over nus
                JKQ_1[K][Q] = np.trapz(JKQ_1[K][Q]*gauss_1_norm, nus_cp)
                JKQ_2[K][Q] = np.trapz(JKQ_2[K][Q]*gauss_2_norm, nus_cp)

        JKQ_1[2][-2] =      np.conjugate(JKQ_1[2][2])
        JKQ_1[2][-1] = -1.0*np.conjugate(JKQ_1[2][1])
        JKQ_1[1][-1] = -1.0*np.conjugate(JKQ_1[1][1])

        JKQ_2[2][-2] =      np.conjugate(JKQ_2[2][2])
        JKQ_2[2][-1] = -1.0*np.conjugate(JKQ_2[2][1])
        JKQ_2[1][-1] = -1.0*np.conjugate(JKQ_2[1][1])

        # compute the profiles with the new JKQs
        profiles, nus_cp, rays, orays = compute_profile(JKQ_1, JKQ_2, pm, B_spherical, especial=pm.especial)

        ################   SEMI-ANALYTICAL RADIATIVE TRANSFER  #################
        # identity matrix
        Ident = np.identity(4)

        # optical depth profile (background with peak at 1.0)
        tau_prof = tau_nlte_max*profiles['eta_I']/np.abs(profiles['eta_I']).max()

        # plot the optical depth profile
        # print('plot the optical depth profile')
        # plt.plot(wave_cp, tau_prof)
        # plt.plot(wave_nlte, tau_nlte, '--')
        # plt.show()

        Kp = [[profiles['eta_I']*0                               , profiles['eta_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_U']/(profiles['eta_I'] + cts.vacuum), profiles['eta_V']/(profiles['eta_I'] + cts.vacuum)],
            [profiles['eta_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0                               , profiles['rho_V']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_U']/(profiles['eta_I'] + cts.vacuum)],
            [profiles['eta_U']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_V']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0                               , profiles['rho_Q']/(profiles['eta_I'] + cts.vacuum)],
            [profiles['eta_V']/(profiles['eta_I'] + cts.vacuum), profiles['rho_U']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0]]
        Kp = np.array(Kp)

        # Radiation coming from the underlying medium entering the slab
        background = Allen.get_radiation(nus_cp)*Allen.get_clv(orays[0],nus_cp)
        Isun = np.array([background, 0*background, 0*background, 0*background])

        # Source function computed with the new JKQs
        SS = np.array([profiles['eps_I']/(profiles['eta_I'] + cts.vacuum), profiles['eps_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eps_U']/(profiles['eta_I'] + cts.vacuum), profiles['eps_V']/(profiles['eta_I'] + cts.vacuum)])

        # Initialize the intensities and an auxiliary matrix to invert
        M_inv = np.zeros((4,4,profiles['nus'].size))
        II = np.zeros((4,profiles['nus'].size))
        # go through the different frequencies
        for i in range(len(nus_cp)):
            # retrieve the optical depth at the current frequency
            tau = tau_prof[i]
            # compute the phi coefficients at the current frequency
            phi_m = phi_m_calc(tau)
            phi_0 = phi_o_calc(tau)

            # compute the matrix M and invert it
            M_inv[:,:,i] = np.linalg.inv(phi_0*Kp[:,:,i] + Ident)
            # Do the matrix multiplications to obtain the stokes parameters
            AA = np.einsum('ij,j->i', (np.exp(-tau)*Ident - phi_m*Kp[:,:,i]), Isun[:,i])
            DD = (phi_m + phi_0)*SS[:,i]
            II[:,i] = np.einsum('ij,j->i',M_inv[:,:,i],(AA + DD))


        ################  "SAVE" FINAL RESULTS  #################

        stokes_analytical.append([II[0], II[1], II[2], II[3]])


        ################   PLOT THE FINAL RESULTS  #################

    len_w = len(wave_cp)
    p1 = int(len_w/8)
    p3 = int(p1*7)
    nn = 1.000293
    wave = wave_cp/1e-9/nn

    ticks = [wave[nu_peak_1_indx], wave[nu_peak_2_indx]]
    labels = [f'{wave[nu_peak_1_indx]:.2f}', f'{wave[nu_peak_2_indx]:.2f}']

    # separate the stokes parameters in their own lists
    stokes = np.array(stokes)
    I_nlte = stokes[:,0,:]
    I_nlte_grid = np.resize(I_nlte, (30,12,len_w))
    Q_nlte = stokes[:,1,:]
    Q_nlte_grid = np.resize(Q_nlte, (30,12,len_w))
    U_nlte = stokes[:,2,:]
    U_nlte_grid = np.resize(U_nlte, (30,12,len_w))
    V_nlte = stokes[:,3,:]
    V_nlte_grid = np.resize(V_nlte, (30,12,len_w))

    stokes_analytical = np.array(stokes_analytical)
    I_analytical = stokes_analytical[:,0,:]
    I_analytical_grid = np.resize(I_analytical, (30,12,len_w))
    Q_analytical = stokes_analytical[:,1,:]
    Q_analytical_grid = np.resize(Q_analytical, (30,12,len_w))
    U_analytical = stokes_analytical[:,2,:]
    U_analytical_grid = np.resize(U_analytical, (30,12,len_w))
    V_analytical = stokes_analytical[:,3,:]
    V_analytical_grid = np.resize(V_analytical, (30,12,len_w))

    B_grid = np.resize(np.array(BBs), (30,12))
    tau_grid = np.resize(np.array(taus).max(axis=-1), (30,12))

    figure, axis = plt.subplots(nrows=30, ncols=12, figsize=(30, 20), dpi=200)
    for i in tqdm(range(30), desc='Plotting', ncols=50):
        for j in range(12):
            axis[i, j].plot(wave[p1:p3], I_nlte_grid[i,j,p1:p3]/I_nlte_grid[i,j,0], label='NLTE')
            axis[i, j].plot(wave[p1:p3], I_analytical_grid[i,j,p1:p3]/I_nlte_grid[i,j,0], label='analytical')
            axis[i, j].set_yticks([])
            # axis[i, j].set_xticks(ticks, labels)
            axis[i, j].set_xticks([])
            axis[i, j].set_ylim(0.1, 1.1)
            axis[i, j].set_title(f'B = {B_grid[i,j]:.2f} G, tau = {tau_grid[i,j]:.2f}',
                                 fontsize=8)

    # figure.tight_layout()
    figure.savefig(f'{pm.basedir}comparison_I.png')
    plt.close()
    # plt.show()

    figure, axis = plt.subplots(nrows=30, ncols=12, figsize=(30, 20), dpi=200)
    for i in tqdm(range(30), desc='Plotting', ncols=50):
        for j in range(12):
            axis[i, j].plot(wave[p1:p3], -Q_nlte_grid[i,j,p1:p3]/I_nlte_grid[i,j,p1:p3]*100, label='NLTE')
            axis[i, j].plot(wave[p1:p3], Q_analytical_grid[i,j,p1:p3]/I_analytical[i*12+j,p1:p3]*100, label='slab')
            axis[i, j].set_yticks([])
            # axis[i, j].set_xticks(ticks, labels)
            axis[i, j].set_xticks([])
            axis[i, j].set_ylim(-2.5, 2.5)
            axis[i, j].set_title(f'B = {B_grid[i,j]:.2f} G, tau = {tau_grid[i,j]:.2f}',
                                 fontsize=8)

    axis[0,0].legend()
    # figure.tight_layout()
    figure.savefig(f'{pm.basedir}comparison_Q.png')
    plt.close()
    # plt.show()

    # plot a 2d histogram of the Q errors in red and blue components
    I_2d_error_r = ((I_nlte_grid - I_analytical_grid)/I_nlte_grid)[:,:,nu_peak_1_indx]*100
    I_2d_error_b = ((I_nlte_grid - I_analytical_grid)/I_nlte_grid)[:,:,nu_peak_2_indx]*100

    # print the images of the errors in the red and blue components
    plt.figure(figsize=(20,10), dpi=120)
    plt.subplot(1,2,1)
    # show the image to be squared
    plt.imshow(I_2d_error_r, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)
    plt.xticks(np.arange(len(tau_grid[0,:]))[::2],[f'{tau:.2f}' for tau in tau_grid[0,::2]])
    plt.yticks(np.arange(len(B_grid[:,0]))[::2],B_grid[::2,0])
    # plt.colorbar()
    plt.title(r'$\frac{I_{NLTE} - I_{analit}}{I_{NLTE}}$ --> % in $\nu_{red}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.subplot(1,2,2)
    plt.imshow(I_2d_error_b, cmap='RdBu_r', aspect='auto',  vmin=-10, vmax=10)
    plt.xticks(np.arange(len(tau_grid[0,:]))[::2],[f'{tau:.2f}' for tau in tau_grid[0,::2]])
    plt.yticks(np.arange(len(B_grid[:,0]))[::2],B_grid[::2,0])
    plt.colorbar()
    plt.title(r'$\frac{I_{NLTE} - I_{analit}}{I_{NLTE}}$ --> % in $\nu_{blue}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.tight_layout()
    plt.savefig(f'{pm.basedir}I_error.png')
    plt.close()

    Q_2d_error_r = (Q_nlte_grid/I_nlte_grid - Q_analytical_grid/I_analytical_grid)[:,:,nu_peak_1_indx]*100
    Q_2d_error_b = (Q_nlte_grid/I_nlte_grid - Q_analytical_grid/I_analytical_grid)[:,:,nu_peak_2_indx]*100

    # same but for Q
    plt.figure(figsize=(20,10), dpi=120)
    plt.subplot(1,2,1)
    plt.imshow(Q_2d_error_r, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
    plt.xticks(np.arange(len(tau_grid[0,:]))[::2],[f'{tau:.2f}' for tau in tau_grid[0,::2]])
    plt.yticks(np.arange(len(B_grid[:,0]))[::2],B_grid[::2,0])
    # plt.colorbar()
    plt.title(r'$\frac{Q_{NLTE}}{I_{NLTE}} - \frac{Q_{ANAL.}}{I_{ANAL.}}$ --> % in $\nu_{red}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.subplot(1,2,2)
    plt.imshow(Q_2d_error_b, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
    plt.xticks(np.arange(len(tau_grid[0,:]))[::2], [f'{tau:.2f}' for tau in tau_grid[0,::2]])
    plt.yticks(np.arange(len(B_grid[:,0]))[::2], B_grid[::2,0])
    plt.colorbar()
    plt.title(r'$\frac{Q_{NLTE}}{I_{NLTE}} - \frac{Q_{ANAL.}}{I_{ANAL.}}$ --> % in $\nu_{blue}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.tight_layout()
    plt.savefig(f'{pm.basedir}Q_error.png')
    plt.close()

    # PLOT OF THE STOKES PARAMETERS
    # plt.figure(figsize=(10,3.5), dpi=120)
    # plt.subplot(1,2,1)
    # plt.plot(wave[p1:p3], I_nlte[p1:p3] , linewidth=2, color=cm.plasma(0/10.0), label=fr'Self-consistent NLTE')
    # plt.plot(wave[p1:p3], II[0,p1:p3], linewidth=2, color=cm.plasma(8/10.0), label=fr'Constant property slab')
    # # plt.ylim(0, (I_nlte/I_nlte[0]).max()*1.1)
    # plt.legend(loc='lower left')
    # plt.title(r'$I$')
    # plt.xlabel('Wavelength [nm]')
    # plt.xticks(ticks, labels)

    # plt.subplot(1,2,2)
    # plt.plot(wave[p1:p3], Q_nlte[p1:p3] , linewidth=2, color=cm.plasma(0/10.0), label=fr'Self-consistent NLTE')
    # plt.plot(wave[p1:p3], II[1,p1:p3] , linewidth=2, color=cm.plasma(8/10.0), label=fr'Const. slab')
    # # plt.legend()
    # plt.title(r'$Q$')
    # plt.xlabel('Wavelength [nm]')
    # plt.xticks(ticks, labels)

    # plt.tight_layout()
    # # plt.savefig('3_3_2.pdf')
    # plt.show()

    # PLOT OF THE STOKES PARAMETERS
    # plt.figure(figsize=(15,5), dpi=180)
    # plt.subplot(1,2,1)
    # plt.plot(wave[p1:p3], I_nlte[p1:p3]/I_nlte[0], linewidth=2, color=cm.plasma(0/10.0), label=fr'Self-consistent NLTE')
    # plt.plot(wave[p1:p3], II[0,p1:p3]/I_nlte[0], linewidth=2, color=cm.plasma(8/10.0), label=fr'Constant property slab')
    # plt.ylim(0, (I_nlte/I_nlte[0]).max()*1.1)
    # plt.legend(loc='lower left')
    # plt.ylabel(r'$I/I_c$')
    # plt.xlabel('Wavelength [nm]')
    # plt.xticks(ticks, labels)

    # plt.subplot(1,2,2)
    # plt.plot(wave[p1:p3], Q_nlte[p1:p3]/I_nlte[p1:p3]*100 , linewidth=2, color=cm.plasma(0/10.0), label=fr'Self-consistent NLTE')
    # plt.plot(wave[p1:p3], -II[1,p1:p3]/I_nlte[p1:p3]*100 , linewidth=2, color=cm.plasma(8/10.0), label=fr'Const. slab')
    # # plt.legend()
    # plt.ylabel(r'$Q/I_c$ (%)')
    # plt.ylim(-1.5, 1.5)
    # plt.xlabel('Wavelength [nm]')
    # plt.xticks(ticks, labels)

    # plt.tight_layout()
    # plt.savefig(f'{folder}.png')
    # plt.show()
