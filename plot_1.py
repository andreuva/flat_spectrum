from tokenize import Special
import plot_1_params as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0, TKQ
from plot_utils import plot_4_profiles, plot_quantity

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import time


# Function that computes the unit 3D vector from the inclination and the azimuth
def ang_to_cart(inc, az):
    return np.array([np.sin(inc)*np.cos(az), np.sin(inc)*np.sin(az), np.cos(inc)])


# Function that computes the spherical coordinates vector from the cartesian one
def cart_to_ang(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2),z)
    phi = np.arctan2(y,x)
    return np.array([r, theta, phi])


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos( costheta )
    x = np.sin( theta) * np.cos( phi )
    y = np.sin( theta) * np.sin( phi )
    z = np.cos( theta )
    return np.array([x,y,z])


# Function that returns a gaussian profile given a x, mean and std
def gaussian(x, mu, std):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(std, 2.)))


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=np.array([0, 0, 0]), especial=True):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0, especial=especial)
    # Retrieve the different components of the line profile
    components = list(atoms.atom.lines[0].jqq.keys())

    # Initialize the jqq and construct the dictionary
    atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

    # reset the jqq to zero to construct from there the radiation field with the JKQ
    atoms.reset_jqq(cdt.nus_N)

    if especial:
        atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(JKQ_1, cdt.JS)
        atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(JKQ_2, cdt.JS)
    else:
        atoms.atom.lines[0].jqq = JKQ_to_Jqq(JKQ_1, cdt.JS)

    # Solve the ESE
    atoms.solveESE(None, cdt)

    # select the ray direction as the otuput ray
    ray = cdt.orays[0]

    # Compute the RT coeficients for a given ray
    sf, kk = RT_coeficients.getRTcoefs(atoms, ray, cdt)

    # Compute the emision coefficients from the Source functions
    profiles = {}
    profiles['nus'] = cdt.nus
    profiles['eps_I'] = sf[0]*kk[0][0]
    profiles['eps_Q'] = sf[1]*kk[0][0]
    profiles['eps_U'] = sf[2]*kk[0][0]
    profiles['eps_V'] = sf[3]*kk[0][0]

    # retrieve the absorption coefficients from the K matrix
    profiles['eta_I'] = kk[0][0]
    profiles['eta_Q'] = kk[0][1]*kk[0][0]
    profiles['eta_U'] = kk[0][2]*kk[0][0]
    profiles['eta_V'] = kk[0][3]*kk[0][0]
    profiles['rho_Q'] = kk[1][0]*kk[0][0]
    profiles['rho_U'] = kk[1][1]*kk[0][0]
    profiles['rho_V'] = kk[1][2]*kk[0][0]

    return profiles, profiles['nus'], cdt.rays


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
    # create the conditions to compute the JKQ and profiles
    B_xyz = np.array([0, 0, 0])
    B_spherical = cart_to_ang(B_xyz[0], B_xyz[1], B_xyz[2])
    velocity_magnitude = 0e4  # m/s
    n_samples = 1000
    especial = True

    # define the parameters that will construct the background radiation field
    # to later compute the JKQ en both components

    # TO DO: compute the real solar J00
    continium = 14e-3
    gaussian_width = 7e9
    nu_1 = 2.76733e14
    nu_2 = 2.76764e14
    gaussian_1_height = 7e-3
    gaussian_2_height = 1e-3

    tau_max = 1
    tau_continium = 0
    sun_ilum = False
    flat_spectrum = True
    background_type = 'absorption'

    ###############################################################################################
    #                   Test the computation of the JKQ and the profiles                          #
    ###############################################################################################
    # create the JKQ
    JKQ_1 = construct_JKQ_0()
    JKQ_2 = construct_JKQ_0()

    # compute the JKQ with ad-hoc values
    JKQ_1[0][0] = 1e-2
    JKQ_2[0][0] = 1e-4

    # compute the profiles for the test
    _, nus, rays = compute_profile(JKQ_1, JKQ_1, B=B_spherical, pm=pm, especial=especial)

    # compute the wavelength from the frequencies
    wave = 299792458/nus

    # compute the index of each of the peaks
    nu_peak_1_indx = np.abs(nus - nu_1).argmin()
    nu_peak_2_indx = np.abs(nus - nu_2).argmin()

    ###############################################################################################

    # compute the gaussian profiles for each component
    background_1 = gaussian(nus, nu_1, gaussian_width)*gaussian_1_height
    background_2 = gaussian(nus, nu_2, gaussian_width)*gaussian_2_height
    background_1_norm = background_1/np.trapz(background_1, nus)
    background_2_norm = background_2/np.trapz(background_2, nus)

    # compute the total background (absorption + emission)
    background_absorption = -background_1 -background_2 + continium
    background_emision = background_1 + background_2 + continium
    if background_type == 'absorption':
        background = background_absorption*0 + continium
    else:
        background = background_emision*0 + continium

    # plot the background ilumination
    # print('plot the background ilumination')
    # plot_quantity(wave, background, [r'$\nu$', r'$I_b$'], mode='show')

    profiles_samples = []
    intensities_samples = []
    parameters = []

    for samp in tqdm(range(n_samples)):
        velocity = random_three_vector()*velocity_magnitude
        B_spherical = np.array([np.logspace(-1, 4, n_samples, True)[samp], np.pi/4, np.pi/4])

        # compute the JKQ taking into account the background and velocity
        JKQ_1 = construct_JKQ_0()
        JKQ_2 = construct_JKQ_0()
        for K in range(3):
            for Q in range(0,K+1):
                for ray in rays:
                    vlos = np.dot(ang_to_cart(ray.inc, ray.az), velocity)
                    nus_p = nus*(1+vlos/299792458)
                    background_dop = np.interp(nus, nus_p, background)
                    if flat_spectrum:
                        background_dop = np.ones_like(background_dop)*background_dop.mean()

                    if np.cos(ray.inc) < -0.8:
                        JKQ_1[K][Q] += background_dop*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                        JKQ_2[K][Q] += background_dop*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight

                # integrate over nus
                JKQ_1[K][Q] = np.trapz(JKQ_1[K][Q]*background_1_norm, nus)
                JKQ_2[K][Q] = np.trapz(JKQ_2[K][Q]*background_2_norm, nus)

        JKQ_1[2][-2] =      np.conjugate(JKQ_1[2][2])
        JKQ_1[2][-1] = -1.0*np.conjugate(JKQ_1[2][1])
        JKQ_1[1][-1] = -1.0*np.conjugate(JKQ_1[1][1])

        JKQ_2[2][-2] =      np.conjugate(JKQ_2[2][2])
        JKQ_2[2][-1] = -1.0*np.conjugate(JKQ_2[2][1])
        JKQ_2[1][-1] = -1.0*np.conjugate(JKQ_2[1][1])

        # compute the profiles with the new JKQs
        profiles, nus, rays = compute_profile(JKQ_1, JKQ_2, pm, B_spherical, especial=especial)

        ################   SEMI-ANALYTICAL RADIATIVE TRANSFER  #################

        # identity matrix
        Ident = np.identity(4)

        # optical depth profile (background with peak at 1.0)
        tau_prof = (background_1/gaussian_1_height + background_2/gaussian_1_height)*tau_max + tau_continium

        # plot the optical depth profile
        # print('plot the optical depth profile')
        # plot_quantity(wave, tau_prof, [r'$\nu$', r'$\tau$'], mode='show')

        Kp = [[profiles['eta_I']*0                , profiles['eta_Q']/profiles['eta_I'], profiles['eta_U']/profiles['eta_I'], profiles['eta_V']/profiles['eta_I']],
            [profiles['eta_Q']/profiles['eta_I'], profiles['eta_I']*0                , profiles['rho_V']/profiles['eta_I'],-profiles['rho_U']/profiles['eta_I']],
            [profiles['eta_U']/profiles['eta_I'],-profiles['rho_V']/profiles['eta_I'], profiles['eta_I']*0                , profiles['rho_Q']/profiles['eta_I']],
            [profiles['eta_V']/profiles['eta_I'], profiles['rho_U']/profiles['eta_I'],-profiles['rho_Q']/profiles['eta_I'], profiles['eta_I']*0]]
        Kp = np.array(Kp)

        # Radiation coming from the underlying medium entering the slab
        Isun = np.array([background, 0*background, 0*background, 0*background])
        if not sun_ilum:
            Isun = Isun*0
        # Source function computed with the new JKQs
        SS = np.array([profiles['eps_I']/profiles['eta_I'], profiles['eps_Q']/profiles['eta_I'], profiles['eps_U']/profiles['eta_I'], profiles['eps_V']/profiles['eta_I']])

        # Initialize the intensities and an auxiliary matrix to invert
        M_inv = np.zeros((4,4,profiles['nus'].size))
        II = np.zeros((4,profiles['nus'].size))
        # go through the different frequencies
        for i in range(len(nus)):
            # retrieve the optical depth at the current frequency
            tau = tau_prof[i]
            # compute the phi coefficients at the current frequency
            phi_m = phi_m_calc(tau)
            phi_0 = phi_o_calc(tau)

            # compute the matrix M and invert it
            M_inv[:,:,i] = np.linalg.inv(phi_m*Kp[:,:,i] + Ident)
            # Do the matrix multiplications to obtain the stokes parameters
            AA = np.einsum('ij,j->i', (np.exp(-tau)*Ident - phi_m*Kp[:,:,i]), Isun[:,i])
            DD = (phi_m + phi_0)*SS[:,i]
            II[:,i] = np.einsum('ij,j->i',M_inv[:,:,i],(AA + DD))

        profiles_samples.append(profiles)
        intensities_samples.append(II)
        parameters.append([velocity, B_spherical])

    ################   SAVE THE FINAL RESULTS  #################
    # compute the ratio of the peaks
    print('compute the ratio of the peaks')
    ratio_Q = np.array([intensity[1][nu_peak_1_indx]/intensity[1][nu_peak_2_indx] for intensity in intensities_samples])
    ratio_U = np.array([intensity[2][nu_peak_1_indx]/intensity[2][nu_peak_2_indx] for intensity in intensities_samples])
    ratio_V = np.array([intensity[3][nu_peak_1_indx]/intensity[3][nu_peak_2_indx] for intensity in intensities_samples])

    intensities_samples = np.array(intensities_samples)
    parameters = np.array(parameters)

    # save the computed profiles in a pickle file
    print('save the computed profiles')
    timestr = time.strftime("%Y%m%d_%H%M%S")
    module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    save_dict = {'profiles':profiles_samples, 'intensities':intensities_samples,
                 'parameters':parameters, 'pm':module_to_dict(pm), 'ratio_Q':ratio_Q, 'ratio_U':ratio_U, 'ratio_V':ratio_V}
    if especial:
        with open(f'plot_1/plot_1_{timestr}.pkl', 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        with open(f'plot_1/plot_1_{timestr}_mt.pkl', 'wb') as f:
            pkl.dump(save_dict, f)