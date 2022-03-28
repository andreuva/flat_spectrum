import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0

import numpy as np
from matplotlib import pyplot as plt


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=[0,0,0]):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    # print_params(cdt, B)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0)
    # Retrieve the different components of the line profile
    components = list(atoms.atom.lines[0].jqq.keys())

    # Initialize the jqq and construct the dictionary
    atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

    # reset the jqq to zero to construct from there the radiation field with the JKQ
    atoms.reset_jqq(cdt.nus_N)

    atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(JKQ_1, cdt.JS)
    atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(JKQ_2, cdt.JS)

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

    return profiles



# B field will change with each itteration to cover all the possible cases
B = [0,0,0]

# construct the JKQ dictionary
JKQ = construct_JKQ_0()
JKQ[0][0] = 1e-8
JKQ[1][0] = 1e-10
JKQ[2][0] = 1e-12

# compute the profiles
profiles = compute_profile(JKQ, JKQ, pm, B)
nus = profiles['nus']

plt.figure()
plt.plot(nus, profiles['eps_I'], label='eps_I')
plt.show()