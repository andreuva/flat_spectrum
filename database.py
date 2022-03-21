# Especific modules
from doctest import master
import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0

# General modules
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def new_parameters(pm):
    # B field will change with each itteration to cover all the possible cases
    B = np.random.normal(10, 100)
    while B < 0:
        B = np.random.normal(10, 100)

    mu = np.random.uniform(0,1)
    chi = np.random.uniform(0,180)
    # ray direction (will change with each itteration to cover all the possible cases)
    pm.ray_out = [[mu, chi]]
    pm.a_voigt = 0.
    pm.v_dop = 5.0*1e5
    pm.temp = 8.665251563142749e3

    # construct the JKQ dictionary
    JKQ = construct_JKQ_0()
    JKQ[0][0] = np.random.lognormal(-8, 4)
    JKQ[1][-1] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[1][0] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[1][1] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[2][2] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[2][1] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[2][0] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[2][1] = np.random.uniform(0, 0.2)*JKQ[0][0]
    JKQ[2][2] = np.random.uniform(0, 0.2)*JKQ[0][0]

    return JKQ, JKQ, B, pm


# Function to check the conditions that will be used in the Rtcoefs module
def print_params(cdts=conditions(pm), B=0):
    ray = cdts.orays[0]
    # print the parameters of the conditions instance
    print('------------------------------------------------------')
    print(f'B={B:1.3e} G')
    print(f'computed ray:\ninclination={ray.inc}\nazimut={ray.az}\n')
    print(f'v_dop_0={cdts.v_dop_0:1.3e}')
    print(f'v_dop={cdts.v_dop:1.3e}')
    print(f'a_voigt={cdts.a_voigt:1.3e}\n')
    print(f'T={cdts.temp:1.3e}')
    print('------------------------------------------------------\n')


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=0):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    print_params(cdt, B)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # Initialize the ESE object and computing the initial populations (equilibrium = True)
    atoms = ESE(cdt.v_dop, cdt.a_voigt, [B,0,0], cdt.temp, cdt.JS, True, 0)
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


def master_work():
    pass


def slave_work():
    pass


def test_results(pm):
    print("Only one node active", flush=True)

    plt.figure(1, (10, 10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\epsilon_I$')
    plt.title(r'$\epsilon_I$ vs $\nu$')

    plt.figure(2, (10, 10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\epsilon$')
    plt.title(r'$\epsilon_{Q,U,V}$ vs $\nu$')

    plt.figure(3, (10, 10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\eta_I$')
    plt.title(r'$\eta_I$ vs $\nu$')

    plt.figure(4, (10, 10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\eta_{Q,U,V}$')
    plt.title(r'$\eta_{Q,U,V}$ vs $\nu$')

    plt.figure(5, (10, 10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\rho$')
    plt.title(r'$\rho$ vs $\nu$')

    # for i in tqdm(range(5)):
    for i in range(5):

        JKQ_1, JKQ_2, B, pm = new_parameters(pm)
        # compute the profile
        profiles = compute_profile(JKQ_1, JKQ_2, B=B, pm=pm)

        nus = profiles['nus']

        # Plot the emision coefficients
        plt.plot(nus, profiles['eps_I'], label='eps_I', figure=plt.figure(1), color = 'orange')
        plt.plot(nus, profiles['eps_Q'], label='eps_Q', figure=plt.figure(2), color='green')
        plt.plot(nus, profiles['eps_U'], label='eps_U', figure=plt.figure(2), color='blue')
        plt.plot(nus, profiles['eps_V'], label='eps_V', figure=plt.figure(2), color = 'orange')

        # Plot the absorption coefficients
        plt.plot(nus, profiles['eta_I'], label='eta_I', figure=plt.figure(3), color = 'orange')
        plt.plot(nus, profiles['eta_Q'], label='eta_Q', figure=plt.figure(4), color='green')
        plt.plot(nus, profiles['eta_U'], label='eta_U', figure=plt.figure(4), color='blue')
        plt.plot(nus, profiles['eta_V'], label='eta_V', figure=plt.figure(4), color = 'orange')

        # Plot the absorption coefficients
        plt.plot(nus, profiles['rho_Q'], label='rho_Q', figure=plt.figure(5), color='green')
        plt.plot(nus, profiles['rho_U'], label='rho_U', figure=plt.figure(5), color='blue')
        plt.plot(nus, profiles['rho_V'], label='rho_V', figure=plt.figure(5), color = 'orange')
    plt.show()


if __name__ == '__main__':

    # Initialize the MPI environment
    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank  # get current process id
    size = comm.size  # total number of processes
    status = MPI.Status()   # get MPI status object

    # Initialize the random number generator
    # seed = int(time.time())
    seed = 777 # Jackpot because we are going to be lucky :)
    np.random.seed(seed)

    if size > 1:
        print(f"Node {rank}/{size} active", flush=True)
        if rank == 0:
            master_work()
        else:
            slave_work()
    else:
        test_results(pm)
