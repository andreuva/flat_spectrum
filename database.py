# Especific modules
import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, Jqq_to_JKQ
# General modules
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm


# Function to check the conditions that will be used in the Rtcoefs module
def print_params(cdts=conditions(pm), B=np.zeros(3)):
    ray = cdts.orays[1]
    # print the parameters of the conditions instance
    print('------------------------------------------------------')
    print(f'computed ray:\ninclination={ray.inc}\nazimut={ray.az}\n')
    print(f'v_dop_0={cdts.v_dop_0:1.3e}')
    print(f'v_dop={cdts.v_dop:1.3e}')
    print(f'a_voigt={cdts.a_voigt:1.3e}\n')
    print(f'T={cdts.temp:1.3e}')
    print(f'n_dens={cdts.n_dens:1.3e}\n')
    print(f'zn={cdts.z_N}')
    print(f'B=[{B[0]:1.3e}, {B[1]:1.3e}, {B[2]:1.3e}]')
    print('------------------------------------------------------\n')


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(pm=pm, B=np.zeros(3)):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    print_params(cdt, B)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, B, cdt.temp, cdt.JS, True, 0)
    # Retrieve the different components of the line profile
    components = list(atoms.atom.lines[0].jqq.keys())

    # Initialize the jqq and construct the dictionary
    atoms.atom.lines[0].initialize_profiles_first(cdt.nus_N)

    # reset the jqq to zero to construct from there the radiation field with the JKQ
    atoms.reset_jqq(cdt.nus_N)

    # Set the JKQ individually for each component
    JKQ = Jqq_to_JKQ(atoms.atom.lines[0].jqq[components[0]], cdt.JS)
    JKQ[0][0] = JKQ[0][0]*0 + 1e-9
    JKQ[1][0] = JKQ[0][0]*0 + 1e-8
    atoms.atom.lines[0].jqq[components[0]] = JKQ_to_Jqq(JKQ, cdt.JS)

    JKQ = Jqq_to_JKQ(atoms.atom.lines[0].jqq[components[1]], cdt.JS)
    JKQ[0][0] = JKQ[0][0]*0 + 1e-9
    JKQ[1][0] = JKQ[0][0]*0 + 1e-8
    atoms.atom.lines[0].jqq[components[1]] = JKQ_to_Jqq(JKQ, cdt.JS)

    # Solve the ESE
    atoms.solveESE(None, cdt)

    # select the ray direction as the otuput ray
    ray = cdt.orays[1]

    # Compute the RT coeficients for a given ray
    sf, kk = RT_coeficients.getRTcoefs(atoms, ray, cdt)

    # Compute the emision coefficients from the Source functions
    profiles = {}
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

    return cdt.nus, profiles


if __name__ == '__main__':
    
    # keeping the loop in just 1 itteration for testing purposes
    # for i in tqdm(range(1,10)):
    for i in range(1,2):
        B=np.array([i,10/i,i*2])
        pm.zn = pm.zn+1
        nus, profiles = compute_profile(B=B, pm=pm)

        # Plot the emision coefficients
        plt.plot(nus, profiles['eps_I'], label='eps_I')
        plt.plot(nus, profiles['eps_Q'], label='eps_Q')
        plt.plot(nus, profiles['eps_U'], label='eps_U')
        plt.plot(nus, profiles['eps_V'], label='eps_V')
        plt.xlabel(r'$\nu$ [cm$^{-1}$]')
        plt.ylabel(r'$\epsilon$')
        plt.title(r'$\epsilon$ vs $\nu$')
        plt.legend()
        plt.show()

        # Plot the absorption coefficients
        plt.plot(nus, profiles['eta_I'], label='eta_I')
        plt.plot(nus, profiles['eta_Q'], label='eta_Q')
        plt.plot(nus, profiles['eta_U'], label='eta_U')
        plt.plot(nus, profiles['eta_V'], label='eta_V')
        plt.xlabel(r'$\nu$ [cm$^{-1}$]')
        plt.ylabel(r'$\rho$')
        plt.title(r'$\rho$ vs $\nu$')
        plt.legend()
        plt.show()

        # Plot the absorption coefficients
        plt.plot(nus, profiles['rho_Q'], label='rho_Q')
        plt.plot(nus, profiles['rho_U'], label='rho_U')
        plt.plot(nus, profiles['rho_V'], label='rho_V')
        plt.xlabel(r'$\nu$ [cm$^{-1}$]')
        plt.ylabel(r'$\rho$')
        plt.title(r'$\rho$ vs $\nu$')
        plt.legend()
        plt.show()
