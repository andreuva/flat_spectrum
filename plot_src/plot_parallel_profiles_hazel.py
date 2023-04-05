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


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#matplotlib.rcParams['font.size'] = 10


class param_obj:     
    # constructor
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


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
        for line in atoms.atom.lines:
            line.initialize_profiles_first(cdt.nus_N)

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

    # rotate the radiation field
    atoms.rotate_Jqq(cdt.JS)
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

    folders = sorted(glob(f'{pm.basedir}sample_*'))

    print(f'{len(folders)} itterations found')

    with open(f'{pm.basedir}tau.pkl', 'rb') as f:
        taus = pkl.load(f)

    with open(f'{pm.basedir}params.pkl', 'rb') as f:
        params = pkl.load(f)

    with open(f'{pm.basedir}stokes.pkl', 'rb') as f:
        stokes = pkl.load(f)

    taus = np.array(taus)
    BBs = np.array([p['B'] for p in params])
    z0 = np.array([p['z0'] for p in params])
    zf = np.array([p['zf'] for p in params])
    temp = np.array([p['temp'] for p in params])
    mu = np.array([p['ray_out'][0][0] for p in params])
    stokes = np.array(stokes)

    stokes_analytical = []

    for i, folder in enumerate(folders):
        if not os.path.isdir(folder):
            continue

        param = param_obj(params[i])
        param.dir = folder
        print('--'*50)
        print(f'Loading {folder}')
        wave_nlte, tau_nlte = np.loadtxt(f'{folder}/out/tau_00.out', skiprows=3, unpack=True)
        tau_nlte = taus[i]
        I_nlte, Q_nlte, U_nlte, V_nlte = stokes[i]

        wave_nlte = wave_nlte*1e-9
        nus_nlte = 299792458/wave_nlte
        tau_nlte_max = tau_nlte.max()

        # create the conditions to compute the JKQ and profiles
        B_spherical = np.array([BBs[i], 90*np.pi/180, 0])
        velocity = np.array(param.velocity)

        print(f'Computing analitical profiles for {folder}')
        print(f'with parameters:\n B = {B_spherical} \n tau_max = {tau_nlte_max}')
        print('--'*50 + '\n')

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
        Allen.get_gamma(param.z0)

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
        _, nus_cp, rays, _ = compute_profile(JKQ_1, JKQ_1, B=B_spherical, pm=param, especial=param.especial)

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
        profiles, nus_cp, rays, orays = compute_profile(JKQ_1, JKQ_2, param, B_spherical, especial=param.especial)

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


    stokes_analytical = np.array(stokes_analytical)

    ################   PLOT THE FINAL RESULTS  #################

    len_w = len(wave_cp)
    p1 = int(len_w/8)
    p3 = int(p1*7)
    nn = 1.000293
    wave = wave_cp/1e-9/nn


    len_w_nlte = len(wave_nlte)
    # replicate the wave points in the analytical solution to match the NLTE solution
    # (i.e remove the D3 points and center in 1083.0)
    p1_nlte = np.argmin(np.abs(wave_nlte - wave_cp[p1]))
    p3_nlte = np.argmin(np.abs(wave_nlte - wave_cp[p3]))
    wave_nlte = wave_nlte/1e-9/nn

    ticks = [wave[nu_peak_1_indx], wave[nu_peak_2_indx]]
    labels = [f'{wave[nu_peak_1_indx]:.2f}', f'{wave[nu_peak_2_indx]:.2f}']

    # show a subsample of the stokes profiles (I,Q,U,V)
    for ind, st in enumerate(['I','Q','U','V']):
        plt.figure(figsize=(20, 20))
        for pl, i in enumerate(np.random.randint(0, len(stokes)-1, 9)):
            plt.subplot(3, 3, pl+1)
            norm = stokes[i, 0, 0] if ind==0 else stokes[i, 0, p1:p3]
            norm_analytical = stokes_analytical[i, 0, 0] if ind==0 else stokes_analytical[i, 0, p1:p3]
            plt.plot(wave[p1:p3], stokes[i, ind, p1:p3]/norm)
            plt.plot(wave[p1:p3], stokes_analytical[i, ind, p1:p3]/norm_analytical, '--')
            plt.xlabel('Wavelength [nm]')
            plt.xticks(ticks, labels)
            plt.ylabel(f'{st}/I')
            plt.title(fr'$\tau = $ {taus[i].max():.2f} B = {BBs[i]:.0f} G, mu = {mu[i]:.2f}, T = {temp[i]:.0f} K')
        # plt.tight_layout()
        plt.savefig(f'{pm.basedir}stokes_{st}_profiles.png')
        plt.close()