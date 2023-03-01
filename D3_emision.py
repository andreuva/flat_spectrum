# Especific modules
import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0
import constants as cts
from allen import Allen_class

# General modules
import numpy as np
import matplotlib.pyplot as plt


def build_parameters(self, b, x_r, Bx, By, Bz, T):
    """ 
    Build the parameters for the model
        b: height of the point from the center of the sun in the plane of the sky
        x_r: LOS distance from the center of the star (plane of the sky)
        Bx: x component of the magnetic field in global coordinates
        By: y component of the magnetic field in global coordinates
        Bz: z component of the magnetic field in global coordinates
        T: temperature of the plasma
    output:
        pm: parameters object with the parameters of the model, Jqq of the
            point computed with Allen and rotated magnetic field
    """

    # compute height over the surface and angle from the plane of the sky
    h = np.sqrt(b**2 + x_r**2)
    mu = b/h
    delt = -np.arcsin(x_r/h)

    # make a rotation of the magnetic field to to have the z in the solar radial direction
    # this is equivalent to rotate the reference frame using e_y an angle of delt
    Bx_new = Bx*np.cos(delt) + Bz*np.sin(delt)
    By_new = By
    Bz_new = -Bx*np.sin(delt) + Bz*np.cos(delt)

    B_mod = np.sqrt(Bx_new**2 + By_new**2 + Bz_new**2)
    if B_mod == 0:
        B_inc = 0.0
    else:
        B_inc = np.arccos(Bz_new/B_mod)
    B_az = np.arctan2(By_new, Bx_new)

    pm.Bx_global = Bx
    pm.By_global = By
    pm.Bz_global = Bz

    pm.Bx = Bx_new
    pm.By = By_new
    pm.Bz = Bz_new

    pm.B = B_mod
    pm.B_inc = B_inc
    pm.B_az = B_az
    B = np.array([pm.B, pm.B_inc, pm.B_az])

    pm.b = b
    pm.x = x_r
    pm.h = np.sqrt(pm.x**2 + pm.b**2)
    pm.mu = pm.x/pm.h
    pm.chi = 0
    pm.ray_out = [[pm.mu, pm.chi]]

    pm.z0 = pm.h
    pm.zf = pm.z0 + pm.z0*1e-3
    # amplitude of the profile
    pm.temp = T

    self.Allen = Allen_class()
    # Initialize the conditions objects given the initial set of parameters
    cdt = conditions(pm)
    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0, especial=False)
    # Retrieve the resonance wavelength of each line in the atom
    self.nus = []
    for line in atoms.atom.lines:
        self.nus.append(line.nu)

    # construct the JKQ dictionary
    # Get Allen gamma angles
    self.Allen.get_gamma(pm.z0)

    pm.JKQ = []
    pm.wnu = np.zeros(len(self.nus))
    for i,nu in enumerate(self.nus):
        pm.JKQ.append(construct_JKQ_0())
        pm.wnu[i], pm.JKQ[-1][0][0], pm.JKQ[-1][2][0] = self.Allen.get_anisotropy(nu, pm.z0)

    return pm


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(pm=pm, especial=False, jqq=None):
    """
    Compute the profile of the emission of a point in the solar atmosphere
        pm: parameters object with the parameters of the model
        especial: if True, the jqq is computed for both components of the 1083 line
    output:
        profiles: dictionary with the different components of the line emission profiles
    """
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    B = np.array([pm.B, pm.B_inc, pm.B_az])
    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0, especial=especial)

    if especial:
        # Retrieve the different components of the line profile
        components = list(atoms.atom.lines[0].jqq.keys())

        # Initialize the jqq and construct the dictionary
        atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

        for line in atoms.atom.lines:
            line.initialize_profiles_first(cdt.nus_N)

        # reset the jqq to zero to construct from there the radiation field with the JKQ
        atoms.reset_jqq(cdt.nus_N)

        atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(pm.JKQ[0], cdt.JS)
        atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(pm.JKQ[0], cdt.JS)
        for i,line in enumerate(atoms.atom.lines[1:]):
            line.jqq = JKQ_to_Jqq(pm.JKQ[i+1], cdt.JS)

    else:
        # Initialize the jqq and construct the dictionary
        for line in atoms.atom.lines:
            line.initialize_profiles_first(cdt.nus_N)
        for i,line in enumerate(atoms.atom.lines):
            line.jqq = JKQ_to_Jqq(pm.JKQ[i], cdt.JS)

    # print(atoms.atom.lines[0].jqq)
    if jqq is not None:
        atoms.atom.lines[0].jqq = jqq

    # Solve the ESE
    atoms.solveESE(None, cdt)

    # select the ray direction as the otuput ray
    ray = cdt.orays[0]

    # Compute the RT coeficients for a given ray
    sf, kk = RT_coeficients.getRTcoefs(atoms, ray, cdt)

    # Compute the emision coefficients from the Source functions
    profiles = {}
    profiles['nus'] = cdt.nus.copy()
    profiles['eps_I'] = sf[0]*(kk[0][0] + cts.vacuum)
    profiles['eps_Q'] = -sf[1]*(kk[0][0] + cts.vacuum)
    profiles['eps_U'] = -sf[2]*(kk[0][0] + cts.vacuum)
    profiles['eps_V'] = sf[3]*(kk[0][0] + cts.vacuum)

    return profiles


if __name__ == '__main__':
    # Computing the parameters object with the given parameters
    print('Building the parameters object')
    print('-'*80 + '\n')

    # INTRODUCE THE PARAMETERS HERE (b, x_r, Bx, By, Bz, T)
    """ 
    Parameters:
        b: height of the point from the center of the sun in the plane of the sky
        x_r: LOS distance from the center of the star (plane of the sky)
        Bx: x component of the magnetic field in global coordinates
        By: y component of the magnetic field in global coordinates
        Bz: z component of the magnetic field in global coordinates
        T: temperature of the plasma
    """
    params = build_parameters(pm, b=2*cts.R_sun, x_r=2*cts.R_sun, Bx=1, By=1, Bz=1, T=1e4)

    # construct the dictionary of parameters to print
    parameters = {'b': pm.b, 'x': pm.x, 'h':pm.z0, 'T':pm.temp, 'mu':pm.mu,
                  'Bx_global':pm.Bx_global, 'By_global':pm.By_global, 'Bz_global':pm.Bz_global,
                  'Bx':pm.Bx, 'By':pm.By, 'Bz':pm.Bz,
                  'B':pm.B, 'B_inc':pm.B_inc*180/np.pi, 'B_az':pm.B_az*180/np.pi,
                  'JKQ':pm.JKQ}

    # compute the profile
    print('Computing the emission profiles')
    profiles = compute_profile(pm=params)

    print('Computed the emission profiles for full He I with the following parameters:')
    print('-'*80 + '\n')
    for key, value in parameters.items():
        if key == 'JKQ':
            print(f'{key:<25}: {value}')
        else:
            print(f'{key:<25}: {value:.4e}')

    # plot the profiles (I, Q, U, V) with all the lines (D3, 1083...)
    plt.figure()
    plt.plot(profiles['nus'], profiles['eps_I'], label='I')
    plt.plot(profiles['nus'], profiles['eps_Q'], label='Q')
    plt.plot(profiles['nus'], profiles['eps_U'], label='U')
    plt.plot(profiles['nus'], profiles['eps_V'], label='V')
    plt.legend()
    plt.show()
