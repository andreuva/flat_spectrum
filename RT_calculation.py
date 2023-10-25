# Especific modules
import parameters_jkq as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0

# General modules
import os
import pickle as pkl
import numpy as np


if __name__ == '__main__':

    # if not os.path.exists(pm.dir):
    #     os.makedirs(pm.dir)
    
    pm.JKQr = construct_JKQ_0()
    pm.JKQr[0][0] = 10**np.random.uniform(-6, -4)
    pm.JKQr[1][0] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]
    pm.JKQr[2][0] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]

    pm.JKQr[1][1] = (np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j)
    pm.JKQr[2][1] = (np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j)
    pm.JKQr[2][2] = (np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j)

    pm.JKQr[2][-2] =      np.conjugate(pm.JKQr[2][2])
    pm.JKQr[2][-1] = -1.0*np.conjugate(pm.JKQr[2][1])
    pm.JKQr[1][-1] = -1.0*np.conjugate(pm.JKQr[1][1])

    pm.JKQb = construct_JKQ_0()
    pm.JKQb[0][0] = 10**np.random.uniform(-6, -4)
    pm.JKQb[1][0] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]
    pm.JKQb[2][0] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]

    pm.JKQb[1][1] = (np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j)
    pm.JKQb[2][1] = (np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j)
    pm.JKQb[2][2] = (np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j)

    pm.JKQb[2][-2] =      np.conjugate(pm.JKQb[2][2])
    pm.JKQb[2][-1] = -1.0*np.conjugate(pm.JKQb[2][1])
    pm.JKQb[1][-1] = -1.0*np.conjugate(pm.JKQb[1][1])

    jqq = None

    pm.mu = np.random.uniform(-1,1)
    pm.chi = np.random.uniform(0,np.pi)
    # ray direction (will change with each itteration to cover all the possible cases)
    pm.ray_out = [[pm.mu, pm.chi]]

    pm.B = 1000
    pm.B_inc = np.arccos(np.random.uniform(0, 1))*180/np.pi
    pm.B_az = np.random.uniform(0, 360)

    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    B = np.array([pm.B, pm.B_inc*np.pi/180, pm.B_az*np.pi/180])
    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0, especial=pm.especial)

    if pm.especial:
        # Retrieve the different components of the line profile
        components = list(atoms.atom.lines[0].jqq.keys())

        # Initialize the jqq and construct the dictionary
        atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

        # reset the jqq to zero to construct from there the radiation field with the JKQ
        atoms.reset_jqq(cdt.nus_N)
        atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(pm.JKQr, cdt.JS)
        atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(pm.JKQb, cdt.JS)
    else:
        # Initialize the jqq and construct the dictionary
        atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)
        atoms.atom.lines[0].jqq = JKQ_to_Jqq(pm.JKQr, cdt.JS)

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
    profiles['eps_I'] = sf[0]# *(kk[0][0] + cts.vacuum)
    profiles['eps_Q'] = sf[1]# *(kk[0][0] + cts.vacuum)
    profiles['eps_U'] = sf[2]# *(kk[0][0] + cts.vacuum)
    profiles['eps_V'] = sf[3]# *(kk[0][0] + cts.vacuum)

    # retrieve the absorption coefficients from the K matrix
    profiles['eta_I'] = kk[0][0]
    profiles['eta_Q'] = kk[0][1]# *(kk[0][0] + cts.vacuum)
    profiles['eta_U'] = kk[0][2]# *(kk[0][0] + cts.vacuum)
    profiles['eta_V'] = kk[0][3]# *(kk[0][0] + cts.vacuum)
    profiles['rho_Q'] = kk[1][0]# *(kk[0][0] + cts.vacuum)
    profiles['rho_U'] = kk[1][1]# *(kk[0][0] + cts.vacuum)
    profiles['rho_V'] = kk[1][2]# *(kk[0][0] + cts.vacuum)

    parameters = {'JKQr':pm.JKQr, 'JKQb':pm.JKQb,
                'B':pm.B, 'B_inc':pm.B_inc, 'B_az':pm.B_az,
                'mu':pm.ray_out[0][0], 'chi':pm.ray_out[0][1],
                'a_voigt':pm.a_voigt, 'temp':pm.temp,
                'n_dens':pm.n_dens, 'v_dop':pm.v_dop}

    with open(f'../IMP_JIRI/imp/tests/data/profiles_jkq.pkl', 'wb') as f:
        pkl.dump(profiles, f)

    with open(f'../IMP_JIRI/imp/tests/data/parameters_jkq.pkl', 'wb') as f:
        pkl.dump(parameters, f)

