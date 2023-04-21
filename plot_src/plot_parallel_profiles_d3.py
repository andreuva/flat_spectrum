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
def compute_profile(JKQ, pm=pm, B=np.array([0, 0, 0]), especial=True, jqq=None):
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
        atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(JKQ[0][0], cdt.JS)
        atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(JKQ[0][1], cdt.JS)
        atoms.atom.lines[1].jqq = JKQ_to_Jqq(JKQ[1], cdt.JS)
        atoms.atom.lines[2].jqq = JKQ_to_Jqq(JKQ[2], cdt.JS)
        atoms.atom.lines[3].jqq = JKQ_to_Jqq(JKQ[3], cdt.JS)

    else:
        # Initialize the jqq and construct the dictionary
        for i,line in enumerate(atoms.atom.lines):
            line.initialize_profiles_first(cdt.nus_N)
            if i == 0:
                line.jqq = JKQ_to_Jqq(JKQ[i][0], cdt.JS)
            else:
                line.jqq = JKQ_to_Jqq(JKQ[i], cdt.JS)

    # print(atoms.atom.lines[0].jqq)
    if jqq is not None:
        for i, line in enumerate(atoms.atom.lines):
            line.jqq = jqq[i]

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

        ###############################################################################################
        # Get Allen class instance and gamma angles
        Allen = Allen_class()
        Allen.get_gamma(param.z0)

        ###############################################################################################
        #                   Test the computation of the JKQ and the profiles                          #
        ###############################################################################################
        # create the JKQ
        JKQ_0r = construct_JKQ_0()
        # compute the JKQ with ad-hoc values
        JKQ_0r[0][0] = 1e-4
        # compute the profiles for the test
        _, nus_cp, rays, _ = compute_profile([[JKQ_0r, JKQ_0r], JKQ_0r, JKQ_0r, JKQ_0r], B=B_spherical, pm=param, especial=param.especial)

        # compute the wavelength from the frequencies
        wave_cp = 299792458/nus_cp

        # compute the index of each of the peaks
        nu_0r = 2.76733e14
        nu_0b = 2.76764e14
        nu_1 = 770725260863620.6
        nu_2 = 424202775298533.06
        nu_3 = 510086259106490.94

        wave_0r = 299792458/nu_0r
        wave_0b = 299792458/nu_0b
        wave_1 = 299792458/nu_1
        wave_2 = 299792458/nu_2
        wave_3 = 299792458/nu_3

        nu_peak_0r_indx = np.abs(nus_cp - wave_0r).argmin()
        nu_peak_0b_indx = np.abs(nus_cp - wave_0b).argmin()
        nu_peak_1_indx = np.abs(nus_cp - wave_1).argmin()
        nu_peak_2_indx = np.abs(nus_cp - wave_2).argmin()
        nu_peak_3_indx = np.abs(nus_cp - wave_3).argmin()

        ###############################################################################################

        # compute the gaussian profiles for each component
        gaussian_width = 7e9
        gaussian_0r_height = 1e-1
        gaussian_0b_height = gaussian_0r_height/7

        gauss_0r = gaussian(nus_cp, nu_0r, gaussian_width)*gaussian_0r_height
        gauss_0b = gaussian(nus_cp, nu_0b, gaussian_width)*gaussian_0b_height
        gauss_1 = gaussian(nus_cp, nu_1, gaussian_width)*gaussian_0r_height
        gauss_2 = gaussian(nus_cp, nu_2, gaussian_width)*gaussian_0r_height
        gauss_3 = gaussian(nus_cp, nu_3, gaussian_width)*gaussian_0r_height

        gauss_0r_norm = gauss_0r/np.trapz(gauss_0r, nus_cp)
        gauss_0b_norm = gauss_0b/np.trapz(gauss_0b, nus_cp)
        gauss_1_norm = gauss_1/np.trapz(gauss_1, nus_cp)
        gauss_2_norm = gauss_2/np.trapz(gauss_2, nus_cp)
        gauss_3_norm = gauss_3/np.trapz(gauss_3, nus_cp)

        # compute the JKQ taking into account the background and velocity
        JKQ_0r = construct_JKQ_0()
        JKQ_0b = construct_JKQ_0()
        JKQ_1 = construct_JKQ_0()
        JKQ_2 = construct_JKQ_0()
        JKQ_3 = construct_JKQ_0()
        for K in range(3):
            for Q in range(0,K+1):
                for ray in rays:
                    background = Allen.get_radiation(nus_cp)*Allen.get_clv(ray, nus_cp)

                    if ray.rinc < np.pi/2:
                        continue
                    
                    JKQ_0r[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                    JKQ_0b[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                    JKQ_1[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                    JKQ_2[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight
                    JKQ_3[K][Q] += background*TKQ(0,K,Q,ray.rinc,ray.raz)*ray.weight

                # integrate over nus
                JKQ_0r[K][Q] = np.trapz(JKQ_0r[K][Q]*gauss_0r_norm, nus_cp)
                JKQ_0b[K][Q] = np.trapz(JKQ_0b[K][Q]*gauss_0b_norm, nus_cp)
                JKQ_1[K][Q] = np.trapz(JKQ_1[K][Q]*gauss_1_norm, nus_cp)
                JKQ_2[K][Q] = np.trapz(JKQ_2[K][Q]*gauss_2_norm, nus_cp)
                JKQ_3[K][Q] = np.trapz(JKQ_3[K][Q]*gauss_3_norm, nus_cp)

        JKQ_0r[2][-2] =      np.conjugate(JKQ_0r[2][2])
        JKQ_0r[2][-1] = -1.0*np.conjugate(JKQ_0r[2][1])
        JKQ_0r[1][-1] = -1.0*np.conjugate(JKQ_0r[1][1])

        JKQ_0b[2][-2] =      np.conjugate(JKQ_0b[2][2])
        JKQ_0b[2][-1] = -1.0*np.conjugate(JKQ_0b[2][1])
        JKQ_0b[1][-1] = -1.0*np.conjugate(JKQ_0b[1][1])

        JKQ_1[2][-2] =      np.conjugate(JKQ_1[2][2])
        JKQ_1[2][-1] = -1.0*np.conjugate(JKQ_1[2][1])
        JKQ_1[1][-1] = -1.0*np.conjugate(JKQ_1[1][1])

        JKQ_2[2][-2] =      np.conjugate(JKQ_2[2][2])
        JKQ_2[2][-1] = -1.0*np.conjugate(JKQ_2[2][1])
        JKQ_2[1][-1] = -1.0*np.conjugate(JKQ_2[1][1])

        JKQ_3[2][-2] =      np.conjugate(JKQ_3[2][2])
        JKQ_3[2][-1] = -1.0*np.conjugate(JKQ_3[2][1])
        JKQ_3[1][-1] = -1.0*np.conjugate(JKQ_3[1][1])

        # compute the profiles with the new JKQs
        profiles, nus_cp, rays, orays = compute_profile([[JKQ_0r, JKQ_0b], JKQ_1, JKQ_2, JKQ_3], param, B_spherical, especial=param.especial)

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

        Kp = [[profiles['eta_I']*0                             , profiles['eta_Q']/(profiles['eta_I'] + cts.vacuum), profiles['eta_U']/(profiles['eta_I'] + cts.vacuum), profiles['eta_V']/(profiles['eta_I'] + cts.vacuum)],
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

    nu_1 = 5.100897681891647e14
    nu_2 = 5.100597552934561e14
    wave_1 = 299792458/nu_1
    wave_2 = 299792458/nu_2

    nu_peak_1_indx = np.abs(nus_cp - nu_1).argmin()
    nu_peak_2_indx = np.abs(nus_cp - nu_2).argmin()

    ################   PLOT THE FINAL RESULTS  ################# 

    len_w = len(wave_cp) 
    p1 = nu_peak_1_indx - 120 
    p3 = nu_peak_2_indx + 150 
    nn = 1.000293 
    wave = wave_cp/1e-9/nn 

    ################   PLOT THE FINAL RESULTS  #################

    len_w = len(wave_cp)
    len_w_nlte = len(wave_nlte)
    # replicate the wave points in the analytical solution to match the NLTE solution
    # (i.e remove the D3 points and center in 1083.0)
    # p1_nlte = np.argmin(np.abs(wave_nlte - wave_cp[p1]))
    # p3_nlte = np.argmin(np.abs(wave_nlte - wave_cp[p3]))
    nus_nlte = 299792458/wave_nlte
    wave_nlte = wave_nlte/1e-9/nn

    ticks = [wave[nu_peak_1_indx], wave[nu_peak_2_indx]]
    labels = [f'{wave[nu_peak_1_indx]:.2f}', f'{wave[nu_peak_2_indx]:.2f}']

    num_BBs = 22 # len(BBs)
    num_taus = 4 # len(tau_prof)

    # separate the stokes parameters in their own lists
    I_nlte = stokes[:,0,:]
    I_nlte_grid = np.resize(I_nlte, (num_BBs,num_taus,len_w_nlte))
    Q_nlte = stokes[:,1,:]
    Q_nlte_grid = np.resize(Q_nlte, (num_BBs,num_taus,len_w_nlte))
    U_nlte = stokes[:,2,:]
    U_nlte_grid = np.resize(U_nlte, (num_BBs,num_taus,len_w_nlte))
    V_nlte = stokes[:,3,:]
    V_nlte_grid = np.resize(V_nlte, (num_BBs,num_taus,len_w_nlte))

    stokes_analytical = np.array(stokes_analytical)
    I_analytical = stokes_analytical[:,0,:]
    I_analytical_grid = np.resize(I_analytical, (num_BBs,num_taus,len_w))
    Q_analytical = stokes_analytical[:,1,:]
    Q_analytical_grid = np.resize(Q_analytical, (num_BBs,num_taus,len_w))
    U_analytical = stokes_analytical[:,2,:]
    U_analytical_grid = np.resize(U_analytical, (num_BBs,num_taus,len_w))
    V_analytical = stokes_analytical[:,3,:]
    V_analytical_grid = np.resize(V_analytical, (num_BBs,num_taus,len_w))

    B_grid = np.resize(BBs, (num_BBs,num_taus))
    tau_grid = np.resize(taus.max(axis=-1), (num_BBs,num_taus))

    figure, axis = plt.subplots(nrows=num_BBs, ncols=num_taus, figsize=(10, 40), dpi=200)
    for i in tqdm(range(num_BBs), desc='Plotting', ncols=50):
        for j in range(num_taus):
            axis[i, j].plot(wave[p1:p3], I_nlte_grid[i,j,p1:p3]/I_nlte_grid[i,j,0], color=cm.plasma(0/10.0), label='NLTE')
            axis[i, j].plot(wave[p1:p3], I_analytical_grid[i,j,p1:p3]/I_analytical_grid[i,j,0], color=cm.plasma(8/10.0), label='analytical')
            axis[i, j].axvline(x=wave[nu_peak_1_indx], color='r', linestyle='--', linewidth=0.5)
            axis[i, j].axvline(x=wave[nu_peak_2_indx], color='b', linestyle='--', linewidth=0.5)
            axis[i, j].set_yticks([])
            # axis[i, j].set_xticks(ticks, labels)
            axis[i, j].set_xticks([])
            axis[i, j].set_ylim(0.7, 1.05)
            axis[i, j].set_title(f'B = {B_grid[i,j]:.2f} G, tau = {tau_grid[i,j]:.2f}',
                                 fontsize=8)

    figure.tight_layout()
    figure.savefig(f'{pm.basedir}comparison_I.pdf')
    plt.close()
    # plt.show()

    figure, axis = plt.subplots(nrows=num_BBs, ncols=num_taus, figsize=(10, 40), dpi=200)
    for i in tqdm(range(num_BBs), desc='Plotting', ncols=50):
        for j in range(num_taus):
            axis[i, j].plot(wave[p1:p3], Q_nlte_grid[i,j,p1:p3]/I_nlte_grid[i,j,p1:p3]*100, color=cm.plasma(0/10.0), label='NLTE')
            axis[i, j].plot(wave[p1:p3], Q_analytical_grid[i,j,p1:p3]/I_analytical_grid[i,j,p1:p3]*100, color=cm.plasma(8/10.0), label='slab')
            axis[i, j].axvline(x=wave[nu_peak_1_indx], color='r', linestyle='--', linewidth=0.5)
            axis[i, j].axvline(x=wave[nu_peak_2_indx], color='b', linestyle='--', linewidth=0.5)
            axis[i, j].set_yticks([])
            # axis[i, j].set_xticks(ticks, labels)
            axis[i, j].set_xticks([])
            axis[i, j].set_ylim(-0.25, 0.25)
            axis[i, j].set_title(f'B = {B_grid[i,j]:.2f} G, tau = {tau_grid[i,j]:.2f}',
                                 fontsize=8)

    axis[0,0].legend()
    figure.tight_layout()
    figure.savefig(f'{pm.basedir}comparison_Q.pdf')
    plt.close()
    # plt.show()

    # plot a 2d histogram of the Q errors in red and blue components
    I_2d_error_r = (I_nlte_grid - I_analytical_grid)[:,:,nu_peak_1_indx]/I_nlte_grid[:,:,0] *100
    I_2d_error_b = (I_nlte_grid - I_analytical_grid)[:,:,nu_peak_2_indx]/I_nlte_grid[:,:,0] *100

    # print the images of the errors in the red and blue components
    plt.figure(figsize=(20,10), dpi=120)
    plt.subplot(1,2,1)
    # show the image to be squared
    plt.imshow(I_2d_error_r, cmap='RdBu_r', aspect='auto')#, vmin=-10, vmax=10)
    plt.xticks(np.arange(len(tau_grid[0,:])),[f'{tau:.2f}' for tau in tau_grid[0,:]])
    plt.yticks(np.arange(len(B_grid[:,0])),B_grid[:,0])
    plt.colorbar()
    plt.title(r'$\frac{I_{NLTE} - I_{analit}}{I_{NLTE}}$ --> % in $\nu_{red}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.subplot(1,2,2)
    plt.imshow(I_2d_error_b, cmap='RdBu_r', aspect='auto')#,  vmin=-10, vmax=10)
    plt.xticks(np.arange(len(tau_grid[0,:])),[f'{tau:.2f}' for tau in tau_grid[0,:]])
    plt.yticks(np.arange(len(B_grid[:,0])),B_grid[:,0])
    plt.colorbar()
    plt.title(r'$\frac{I_{NLTE} - I_{analit}}{I_{NLTE}}$ --> % in $\nu_{blue}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.tight_layout()
    plt.savefig(f'{pm.basedir}I_error.pdf')
    plt.close()

    Q_2d_error_r = (Q_nlte_grid/I_nlte_grid - Q_analytical_grid/I_analytical_grid)[:,:,nu_peak_1_indx]*100
    Q_2d_error_b = (Q_nlte_grid/I_nlte_grid - Q_analytical_grid/I_analytical_grid)[:,:,nu_peak_2_indx]*100

    # same but for Q
    plt.figure(figsize=(20,10), dpi=120)
    plt.subplot(1,2,1)
    plt.imshow(Q_2d_error_r, cmap='RdBu_r', aspect='auto' , vmin=-0.1, vmax=0.1)
    plt.xticks(np.arange(len(tau_grid[0,:])),[f'{tau:.2f}' for tau in tau_grid[0,:]])
    plt.yticks(np.arange(len(B_grid[:,0])),B_grid[:,0])
    plt.colorbar()
    plt.title(r'$\frac{Q_{NLTE}}{I_{NLTE}} - \frac{Q_{ANAL.}}{I_{ANAL.}}$ --> % in $\nu_{red}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.subplot(1,2,2)
    plt.imshow(Q_2d_error_b, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    plt.xticks(np.arange(len(tau_grid[0,:])), [f'{tau:.2f}' for tau in tau_grid[0,:]])
    plt.yticks(np.arange(len(B_grid[:,0])), B_grid[:,0])
    plt.colorbar()
    plt.title(r'$\frac{Q_{NLTE}}{I_{NLTE}} - \frac{Q_{ANAL.}}{I_{ANAL.}}$ --> % in $\nu_{blue}$')
    plt.xlabel(fr'$\tau$')
    plt.ylabel(fr'$B$')
    plt.tight_layout()
    plt.savefig(f'{pm.basedir}Q_error.pdf')
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
    # plt.savefig(f'{folder}.pdf')
    # plt.show()

    fs = np.load(f'{pm.basedir[:-1]}_fs/data_for_plots.npz')

    # plot 4 panels with diferent tau values where the x axis is the magnetic field
    taus_indexes = [0, 1, 2, -1]
    figure, axis = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 6.67), dpi=150)
    plt.subplots_adjust(hspace=4e-2, wspace=2e-2)
    # plt.figure(figsize=(11,11), dpi=150)
    for i,ax in enumerate(axis.flatten()):
        # ax = plt.subplot(2,2,i+1)
        # red line
        ax.plot(B_grid[:,0], (Q_nlte_grid/I_nlte_grid)[:,taus_indexes[i],nu_peak_1_indx]*100,
                 '-', label=r'$\nu_{red}$, S-C NLTE', color=cm.plasma(8/10.0))
        ax.plot(fs['B_grid'][:,0], (fs['Q_nlte_grid']/fs['I_nlte_grid'])[:,taus_indexes[i],fs['nu_peak_1_indx']]*100,
                '--', label=r'$\nu_{red}$, S-C NLTE, F.S', color=cm.plasma(8/10.0))
        ax.plot(B_grid[:,0], (Q_analytical_grid/I_analytical_grid)[:,taus_indexes[i],nu_peak_1_indx]*100,
                 ':', label=r'$\nu_{red}$, C.P. slab', color=cm.plasma(8/10.0))
        # blue line
        ax.plot(B_grid[:,0], (Q_nlte_grid/I_nlte_grid)[:,taus_indexes[i],nu_peak_2_indx]*100,
                 '-', label=r'$\nu_{blue}$, S-C NLTE', color=cm.plasma(0/10.0))
        ax.plot(fs['B_grid'][:,0], (fs['Q_nlte_grid']/fs['I_nlte_grid'])[:,taus_indexes[i],fs['nu_peak_2_indx']]*100,
                 '--', label=r'$\nu_{blue}$, S-C NLTE, F.S', color=cm.plasma(0/10.0))
        ax.plot(B_grid[:,0], (Q_analytical_grid/I_analytical_grid)[:,taus_indexes[i],nu_peak_2_indx]*100,
                 ':', label=r'$\nu_{blue}$, C.P. slab', color=cm.plasma(0/10.0))

        if i == 2 or i == 3:
            ax.set_xlabel(fr'$B$ (G)')
        # put the ticks and label of the y axis to the right in the right plots
        if i==1 or i==3:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(fr'$Q/I$ (%)')
        else:
            ax.set_ylabel(fr'$Q/I$ (%)')

        ax.set_xscale('log')
        ax.set_xlim(9e-3, 7e2)
        # plt.ylim(-0.02, 0.041)
        # plt.yscale('log')
        if i==0:
            ax.legend(loc= 'best', bbox_to_anchor=(0.4, 0.8))

        _, ymax = ax.get_ylim()
        ax.text(2e-2, ymax*0.9, fr'$\tau$ = {tau_grid[0,taus_indexes[i]]:.2f}', {})
    # plt.tight_layout()
    plt.savefig(f'{pm.basedir}Q_I_vs_B_taus.pdf')
    # plt.close()
    plt.show()


    # save the data for the last plots to add it to the fs plot

    np.savez(f'{pm.basedir}data_for_plots.npz',
             I_nlte_grid=I_nlte_grid, Q_nlte_grid=Q_nlte_grid, U_nlte_grid=U_nlte_grid, V_nlte_grid=V_nlte_grid,
             I_analytical_grid=I_analytical_grid, Q_analytical_grid=Q_analytical_grid, U_analytical_grid=U_analytical_grid, V_analytical_grid=V_analytical_grid,
             B_grid=B_grid, tau_grid=tau_grid, wave=wave, nu_peak_1_indx=nu_peak_1_indx, nu_peak_2_indx=nu_peak_2_indx)


    # plot ratios of red and blue components with tau for each approximation
    plt.figure(figsize=(15,7), dpi=150)
    line = [':', '--', '-']
    for ii,B in enumerate([0.1,1,10]):
        # search for the index of the magnetic field
        B_indx = np.argmin(np.abs(B_grid[:,0]-B))
        # plot the ratio of the intensities for the two peaks for that magnetic field
        I_ratio_nlte = I_nlte_grid[B_indx,:,nu_peak_1_indx]/I_nlte_grid[B_indx,:,nu_peak_2_indx]
        I_ratio_fs = fs['I_nlte_grid'][B_indx,:,fs['nu_peak_1_indx']]/fs['I_nlte_grid'][B_indx,:,fs['nu_peak_2_indx']]
        I_ratio_analytical = I_analytical_grid[B_indx,:,nu_peak_1_indx]/I_analytical_grid[B_indx,:,nu_peak_2_indx]
        
        Q_ratio_nlte = Q_nlte_grid[B_indx,:,nu_peak_1_indx]/Q_nlte_grid[B_indx,:,nu_peak_2_indx]
        Q_ratio_fs = fs['Q_nlte_grid'][B_indx,:,fs['nu_peak_1_indx']]/fs['Q_nlte_grid'][B_indx,:,fs['nu_peak_2_indx']]
        Q_ratio_analytical = Q_analytical_grid[B_indx,:,nu_peak_1_indx]/Q_analytical_grid[B_indx,:,nu_peak_2_indx]

        U_ratio_nlte = U_nlte_grid[B_indx,:,nu_peak_1_indx]/U_nlte_grid[B_indx,:,nu_peak_2_indx]
        U_ratio_fs = fs['U_nlte_grid'][B_indx,:,fs['nu_peak_1_indx']]/fs['U_nlte_grid'][B_indx,:,fs['nu_peak_2_indx']]
        U_ratio_analytical = U_analytical_grid[B_indx,:,nu_peak_1_indx]/U_analytical_grid[B_indx,:,nu_peak_2_indx]

        V_ratio_nlte = V_nlte_grid[B_indx,:,nu_peak_1_indx]/V_nlte_grid[B_indx,:,nu_peak_2_indx]
        V_ratio_fs = fs['V_nlte_grid'][B_indx,:,fs['nu_peak_1_indx']]/fs['V_nlte_grid'][B_indx,:,fs['nu_peak_2_indx']]
        V_ratio_analytical = V_analytical_grid[B_indx,:,nu_peak_1_indx]/V_analytical_grid[B_indx,:,nu_peak_2_indx]

        plt.subplot(1,2,1)
        plt.plot(tau_grid[0,:], I_ratio_nlte, line[ii]+'r' ,label='S-C NLTE B='+str(B))
        plt.plot(tau_grid[0,:], I_ratio_fs, line[ii]+'b', label='S-C NLTE, F.S. B='+str(B))
        plt.plot(tau_grid[0,:], I_ratio_analytical, line[ii]+'k', label='C.P. slab B='+str(B))

        plt.subplot(1,2,2)
        plt.plot(tau_grid[0,:], Q_ratio_nlte, line[ii]+'r', label='S-C NLTE')
        plt.plot(tau_grid[0,:], Q_ratio_fs, line[ii]+'b', label='S-C NLTE, F.S.')
        plt.plot(tau_grid[0,:], Q_ratio_analytical, line[ii]+'k', label='C.P. slab')

    plt.subplot(1,2,1)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$I_{red}/I_{blue}$')
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$Q_{red}/Q_{blue}$')
    plt.suptitle('Ratios of red and blue components')
    # plt.tight_layout()
    plt.savefig(f'{pm.basedir}ratios.pdf')
    plt.show()
    # plt.close()
