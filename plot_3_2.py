from allen import Allen_class
import params_plot_3_2 as pm
from conditions import conditions
import matplotlib.cm as cm
import constants as cts
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0, TKQ
from plot_utils import plot_4_profiles, plot_quantity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 16
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
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=np.array([0, 0, 0]), especial=True, jqq=None):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0, especial=especial)

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
    # create the conditions to compute the JKQ and profiles
    B_xyz = np.array([0, 0, 0])
    B_spherical = cart_to_ang(B_xyz[0], B_xyz[1], B_xyz[2])
    velocity_magnitude = 3e4  # m/s
    velocity = random_three_vector()*velocity_magnitude
    velocity = np.array([0, 3e4, 0])     # np.array(pm.velocity)
    especial = True
    datadir ='output_vel_anis_ray_1.0_90.0__B_0.0_v_30100.0'
    pm.dir = datadir + '/'

    wave_imp, tau_imp = np.loadtxt(f'{datadir}/out/tau_00.out', skiprows=3, unpack=True)
    tau_max = 2 # tau_imp.max()

    # define the parameters that will construct the background radiation field
    # to later compute the JKQ en both components

    # Get Allen class instance and gamma angles
    Allen = Allen_class()
    Allen.get_gamma(pm.z0)

    nn = 1.000293
    nu_1 = 2.76733e14
    nu_2 = 2.76764e14
    gaussian_width = 7e9
    gaussian_1_height = 1e-1
    gaussian_2_height = gaussian_1_height/7

    sun_ilum = True
    flat_spectrum = False
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
    _, nus, rays, _ = compute_profile(JKQ_1, JKQ_1, B=B_spherical, pm=pm, especial=especial)

    # compute the wavelength from the frequencies
    wave = 299792458/nus

    # compute the index of each of the peaks
    nu_peak_1_indx = np.abs(nus - nu_1).argmin()
    nu_peak_2_indx = np.abs(nus - nu_2).argmin()

    ###############################################################################################

    # compute the gaussian profiles for each component
    gauss_1 = gaussian(nus, nu_1, gaussian_width)*gaussian_1_height
    gauss_2 = gaussian(nus, nu_2, gaussian_width)*gaussian_2_height
    gauss_1_norm = gauss_1/np.trapz(gauss_1, nus)
    gauss_2_norm = gauss_2/np.trapz(gauss_2, nus)

    # compute the JKQ taking into account the background and velocity
    JKQ_1 = construct_JKQ_0()
    JKQ_2 = construct_JKQ_0()
    for K in range(3):
        for Q in range(0,K+1):
            for ray in rays:
                background = Allen.get_radiation(nus)*Allen.get_clv(ray,nus)
                background = background - (gauss_1 + gauss_2)*background.max()*6.5

                vlos = np.dot(ang_to_cart(ray.inc, ray.az), velocity)
                nus_p = nus*(1+vlos/299792458)

                if ray.rinc < np.pi/2:
                    continue
                else:
                    background_dop = np.interp(nus, nus_p, background)

                    # print(ray.inc, ray.az)
                    # fig, (ax1,ax2,ax3) = plt.subplots(3,1, sharex='col')
                    # ax1.plot(wave, background_dop)
                    # ax1.plot(wave, background)
                    # ax2.plot(wave, gauss_1_norm)
                    # ax2.plot(wave, gauss_2_norm)
                    # ax3.plot(wave, background_dop*gauss_1_norm)
                    # ax3.plot(wave, background_dop*gauss_2_norm)
                    # plt.show()

                JKQ_1[K][Q] += background_dop*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                JKQ_2[K][Q] += background_dop*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight

            # integrate over nus
            JKQ_1[K][Q] = np.trapz(JKQ_1[K][Q]*gauss_1_norm, nus)
            JKQ_2[K][Q] = np.trapz(JKQ_2[K][Q]*gauss_2_norm, nus)

    JKQ_1[2][-2] =      np.conjugate(JKQ_1[2][2])
    JKQ_1[2][-1] = -1.0*np.conjugate(JKQ_1[2][1])
    JKQ_1[1][-1] = -1.0*np.conjugate(JKQ_1[1][1])

    JKQ_2[2][-2] =      np.conjugate(JKQ_2[2][2])
    JKQ_2[2][-1] = -1.0*np.conjugate(JKQ_2[2][1])
    JKQ_2[1][-1] = -1.0*np.conjugate(JKQ_2[1][1])

    # compute the profiles with the new JKQs
    profiles, nus, rays, orays = compute_profile(JKQ_1, JKQ_2, pm, B_spherical, especial=especial)

    ################   SEMI-ANALYTICAL RADIATIVE TRANSFER  #################
    # identity matrix
    Ident = np.identity(4)

    # optical depth profile (background with peak at 1.0)
    # FULL NLTE TRANSFER
    # Q and U comparable
    tau_prof = tau_max*profiles['eta_I']/np.abs(profiles['eta_I']).max()

    # plot the optical depth profile
    print('plot the optical depth profile')
    plot_quantity(wave, tau_prof, [r'$\nu$', r'$\tau$'], mode='show')
    plot_quantity(wave, tau_prof-tau_imp, [r'$\nu$', r'$\tau - \tau_{NLTE}$'], mode='show')

    Kp = [[profiles['eta_I']*0                               , profiles['eta_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_U']/(profiles['eta_I'] + cts.vacuum), profiles['eta_V']/(profiles['eta_I'] + cts.vacuum)],
          [profiles['eta_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0                               , profiles['rho_V']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_U']/(profiles['eta_I'] + cts.vacuum)],
          [profiles['eta_U']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_V']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0                               , profiles['rho_Q']/(profiles['eta_I'] + cts.vacuum)],
          [profiles['eta_V']/(profiles['eta_I'] + cts.vacuum), profiles['rho_U']/(profiles['eta_I'] + cts.vacuum),-profiles['rho_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_I']*0]]
    Kp = np.array(Kp)

    # Radiation coming from the underlying medium entering the slab
    background = Allen.get_radiation(nus)*Allen.get_clv(rays[0],nus)
    background = background - (gauss_1 + gauss_2)*background.max()*6.5
    Isun = np.array([background, 0*background, 0*background, 0*background])
    if not sun_ilum:
        Isun = Isun*0
    # Source function computed with the new JKQs
    SS = np.array([profiles['eps_I']/(profiles['eta_I'] + cts.vacuum), profiles['eps_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eps_U']/(profiles['eta_I'] + cts.vacuum), profiles['eps_V']/(profiles['eta_I'] + cts.vacuum)])

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
        M_inv[:,:,i] = np.linalg.inv(phi_0*Kp[:,:,i] + Ident)
        # Do the matrix multiplications to obtain the stokes parameters
        AA = np.einsum('ij,j->i', (np.exp(-tau)*Ident - phi_m*Kp[:,:,i]), Isun[:,i])
        DD = (phi_m + phi_0)*SS[:,i]
        II[:,i] = np.einsum('ij,j->i',M_inv[:,:,i],(AA + DD))


    ################   SAVE THE FINAL RESULTS  #################
    intensities_samples = np.array(II)
    parameters = np.array([velocity, B_spherical])

    # save the computed profiles in a pickle file
    print('save the computed profiles')
    timestr = time.strftime("%Y%m%d_%H%M%S")
    module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}
    save_dict = {'profiles':profiles, 'intensities':intensities_samples,
                 'parameters':parameters, 'pm':module_to_dict(pm)}

    # PLOT OF THE STOKES PARAMETERS
    stokes_fil = np.loadtxt(f'{datadir}/out/stokes_00.out', skiprows=3)
    nus = stokes_fil[:,0]
    wave = 299792458/nus
    stokes_fil = stokes_fil[:,1:].transpose()

    ################   PLOT THE FINAL RESULTS  #################
    print('plot the final results')
    title = f'$v_x$={velocity[0]/1000:1.2f} km/s,\t $v_y$={velocity[1]/1000:1.2f} km/s,\t $v_z$={velocity[2]/1000:1.2f} km/s'+\
            '\n'+ fr' LOS:  $\mu$ = {pm.ray_out[0][0]:1.2f} $\phi$ = {pm.ray_out[0][1]:1.2f}'
    plot_4_profiles(wave, profiles['eps_I']/profiles['eps_I'].max()*100,
                          profiles['eps_Q']/profiles['eps_I'].max()*100,
                          profiles['eps_U']/profiles['eps_I'].max()*100,
                          profiles['eps_V']/profiles['eps_I'].max()*100,
                    title=r'$\epsilon$  normaliced  in  % (B=0)'+'\n'+title, save=False, show=True, directory=f'plot_1', name=f'eps_{timestr}')
    plot_4_profiles(wave, II[0,:]*100/II[0,:].max(), II[1,:]*100/II[0,:].max(), II[2,:]*100/II[0,:].max(), II[3,:]*100/II[0,:].max(),
                    save=False, show=True)
    plot_4_profiles(wave, stokes_fil[0]*100/stokes_fil[0].max(), stokes_fil[1]*100/stokes_fil[0].max(), stokes_fil[2]*100/stokes_fil[0].max(), stokes_fil[3]*100/stokes_fil[0].max(),
                    save=False, show=True)

    wave = wave*1e9/nn
    ticks = [wave[nu_peak_1_indx], wave[nu_peak_2_indx]]
    labels = [f'{wave[nu_peak_1_indx]:.2f}', f'{wave[nu_peak_2_indx]:.2f}']

    length = len(wave)
    p1 = int(length/8)
    p3 = int(p1*7)

    # plot the intensity, Q and U polarization as a function of the wavelength
    plt.figure(figsize=(13,4), dpi=200)
    plt.subplot(1,3,1)
    plt.plot(wave[p1:p3], II[0,p1:p3]/II[0,:].max(), label='I', color=cm.plasma(0))
    plt.ylabel(r'$I/I_{c}$')
    plt.xlabel('Wavelength [nm]')
    plt.xticks(ticks, labels)

    plt.subplot(1,3,2)
    plt.plot(wave[p1:p3], II[1,p1:p3]*100/II[0,:].max(), label='Q', color=cm.plasma(0))
    plt.ylabel(r'$Q/I_{c} (\%)$')
    plt.xlabel('Wavelength [nm]')
    plt.xticks(ticks, labels)

    plt.subplot(1,3,3)
    plt.plot(wave[p1:p3], II[2,p1:p3]*100/II[0,:].max(), label='U', color=cm.plasma(0))
    plt.ylabel(r'$U/I_{c} (\%)$')
    plt.xlabel('Wavelength [nm]')
    plt.xticks(ticks, labels)
    plt.tight_layout()
    plt.savefig('3_2.pdf')
    plt.show()
