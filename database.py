# Especific modules
from asyncio import tasks
import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0

# General modules
from mpi4py import MPI
import os, time, argparse
from enum import IntEnum
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback


# Define the class to store the MPI tags
class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


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


# Function to dump the data to files
def dump_data(profiles, parameters, filename, task_index=None):
    with open(f'{filename}profiles.pkl', 'wb') as f:
        pkl.dump(profiles[0:task_index], f)
    
    with open(f'{filename}parameters.pkl', 'wb') as f:
        pkl.dump(parameters[0:task_index], f)


def new_parameters(pm, test=False):

    if test:
        pm.a_voigt = 1e-10
        JKQ = construct_JKQ_0()
        JKQ[0][0] = 1e-8
        JKQ[1][0] = 1e-10
        JKQ[2][0] = 1e-12
        B = 100
    else:
        # B field will change with each itteration to cover all the possible cases
        B = np.random.normal(10, 100)
        while B < 0:
            B = np.random.normal(1, 400)

        mu = np.random.uniform(-1,1)
        chi = np.random.uniform(0,180)
        # ray direction (will change with each itteration to cover all the possible cases)
        pm.ray_out = [[mu, chi]]
        # amplitude of the profile
        pm.a_voigt = np.random.choice(np.logspace(0,1e-6,10000)) #  1e-6 to 0.
        pm.temp = np.random.lognormal(3.5, 2) # 1e1 to 10e5

        # construct the JKQ dictionary
        JKQ = construct_JKQ_0()
        JKQ[0][0] = np.random.lognormal(-8, 4)
        JKQ[1][0] = np.random.uniform(-0.2, 0.2)*JKQ[0][0]
        JKQ[2][0] = np.random.uniform(-0.2, 0.2)*JKQ[0][0]

        JKQ[1][1] = np.random.uniform(-0.2, 0.2)*JKQ[0][0] + np.random.uniform(-0.2, 0.2)*JKQ[0][0]*1j
        JKQ[2][1] = np.random.uniform(-0.2, 0.2)*JKQ[0][0] + np.random.uniform(-0.2, 0.2)*JKQ[0][0]*1j
        JKQ[2][2] = np.random.uniform(-0.2, 0.2)*JKQ[0][0] + np.random.uniform(-0.2, 0.2)*JKQ[0][0]*1j

        JKQ[2][-2] =      np.conjugate(JKQ[2][2])
        JKQ[2][-1] = -1.0*np.conjugate(JKQ[2][1])
        JKQ[1][-1] = -1.0*np.conjugate(JKQ[1][1])

    return JKQ, JKQ, B, pm


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(JKQ_1, JKQ_2, pm=pm, B=0):
    # Initialize the conditions and rtcoefs objects
    # given a set of parameters via the parameters_rtcoefs module
    cdt = conditions(pm)
    # print_params(cdt, B)
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


def master_work(nsamples, filename, write_frequency=100):

    # Index of the task to keep track of each job
    task_status = [0]*nsamples
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    # Initialize the variables to store the results
    profiles = [None]*nsamples
    parameters = [None]*nsamples

    with tqdm(total=nsamples, ncols=100) as pbar:
        while closed_workers < num_workers:

            # Recieve the ready signal from the workers
            dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            # If the worker is ready to compute
            if tag == tags.READY:
                # Look for the first task that is not done
                try:
                    task_index = task_status.index(0)
                except ValueError:
                    # If all the tasks are done, send the terminate signal to the workers
                    task_index = -1

                # Send the task to the worker
                if task_index >= 0:
                    comm.send(task_index, dest=source, tag=tags.START)
                    task_status[task_index] = 1
                else:
                    # If the task is -1 (all the tasks done), send the kill signal
                    comm.send(None, dest=source, tag=tags.EXIT)
                    closed_workers += 1

            if tag == tags.DONE:
                # If the worker is done, store the results
                task_index = dataReceived['index']
                success = dataReceived['success']
                
                if success:
                    profiles[task_index] = dataReceived['profiles']
                    parameters[task_index] = dataReceived['parameters']
                    task_status[task_index] = 1
                    pbar.update(1)
                else:
                    task_status[task_index] = 0

            # If the number of itterations is multiple with the write frequency dump the data
            if (pbar.n / write_frequency == pbar.n // write_frequency):
                # Dump the data
                dump_data(profiles, parameters, filename, pbar.n)
        
    # Once all the workers are done, dump the data
    print('\n'+'#'*100)
    print('All the workers are done')
    print('#'*100)
    print("Master finishing")
    print("Dumping data")
    print('#'*100)
    dump_data(profiles, parameters, filename)
    


def slave_work(pm):

    module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}

    while True:
        # send the ready signal to the master
        comm.send(None, dest=0, tag=tags.READY)

        # Receive the task index or the kill signal from the master
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        # if the signal is to start the computation, compute the task
        if tag == tags.START:
            # store the index of the received task
            task_index = dataReceived
            success = True
            try:
                # create a new parameters
                JKQ_1, JKQ_2, B, pm = new_parameters(pm)
                # compute the profile
                profiles = compute_profile(JKQ_1, JKQ_2, B=B, pm=pm)
                pm.B = B
                pm.JKQ_1 = JKQ_1
                pm.JKQ_2 = JKQ_2
            except:
                success = False
                print('-'*50)
                print("Error in the computation of the profile")
                print("The parameters are:")
                print("JKQ_1:", JKQ_1)
                print("JKQ_2:", JKQ_2)
                print("B:", B)
                print("pm:", pm)
                print("The error is:")
                print(traceback.format_exc())
                print('-'*50)
                profiles = None

            parameters = module_to_dict(pm)

            # Send the results to the master
            dataToSend = {'index': task_index, 'success': success, 'profiles': profiles, 
                          'parameters': parameters}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
        
        # If the master is sending the kill signal exit
        elif tag == tags.EXIT:
            return


def create_test_amplitudes(pm):
    plt.figure(1, (10,10), 150)
    plt.xlabel(r'$\nu$ [cm$^{-1}$]')
    plt.ylabel(r'$\epsilon_I$')
    plt.title(r'$\epsilon_I$ vs $\nu$')

    module_to_dict = lambda module: {k: getattr(module, k) for k in dir(module) if not k.startswith('_')}

    JKQ_1, JKQ_2, B, pm = new_parameters(pm)
    pm_min = module_to_dict(pm)
    profiles_min = compute_profile(JKQ_1, JKQ_2, B=B, pm=pm)

    pm.a_voigt = pm.a_voigt*2
    pm_mean = module_to_dict(pm)
    profiles = compute_profile(JKQ_1, JKQ_2, B=B, pm=pm)

    pm.a_voigt = pm.a_voigt*2
    pm_max = module_to_dict(pm)
    profiles_max = compute_profile(JKQ_1, JKQ_2, B=B, pm=pm)

    nus = profiles['nus']

    plt.plot(nus, profiles_min['eta_I'], 'b-', label='min')
    plt.plot(nus, profiles_max['eta_I'], 'r-', label='max')
    plt.plot(nus, profiles['eta_I'], 'k-', label='mean')
    plt.legend()

    # if the folder tests does not exist create it
    if not os.path.exists('tests/'):
        os.makedirs('tests/')
    dump_data([profiles_min, profiles, profiles_max], [pm_min, pm_mean, pm_max], 'tests/test_amplitudes_')

    plt.savefig('tests/eta_amplitudes_test.png')
    plt.show()


def test_results(pm):
    print("\nOnly one node active", flush=True)

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
        plt.plot(nus, profiles['eps_V'], label='eps_V', figure=plt.figure(2), color = 'red')

        # Plot the absorption coefficients
        plt.plot(nus, profiles['eta_I'], label='eta_I', figure=plt.figure(3), color = 'orange')
        plt.plot(nus, profiles['eta_Q'], label='eta_Q', figure=plt.figure(4), color='green')
        plt.plot(nus, profiles['eta_U'], label='eta_U', figure=plt.figure(4), color='blue')
        plt.plot(nus, profiles['eta_V'], label='eta_V', figure=plt.figure(4), color = 'red')

        # Plot the absorption coefficients
        plt.plot(nus, profiles['rho_Q'], label='rho_Q', figure=plt.figure(5), color='green')
        plt.plot(nus, profiles['rho_U'], label='rho_U', figure=plt.figure(5), color='blue')
        plt.plot(nus, profiles['rho_V'], label='rho_V', figure=plt.figure(5), color = 'red')
    plt.show()


if __name__ == '__main__':

    # Initialize the MPI environment
    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.rank  # get current process id
    size = comm.size  # total number of processes
    status = MPI.Status()   # get MPI status object
    print(f"\nNode {rank+1}/{size} active", flush=False, end='')

    # Initialize the random number generator
    # seed = int(time.time())
    seed = 777*rank # Jackpot because we are going to be lucky :)
    np.random.seed(seed)

    if size > 1:
        if rank == 0:
            parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
            parser.add_argument('--n', '--nmodels', default=1000, type=int, metavar='NMODELS', help='Number of models')
            parser.add_argument('--f', '--freq', default=100, type=int, metavar='FREQ', help='Frequency of model write')
            parser.add_argument('--sav', '--savedir', default=f'data_{time.strftime("%Y%m%d_%H%M%S")}/', metavar='SAVEDIR', help='directory for output files')

            parsed = vars(parser.parse_args())

            if not os.path.exists(parsed['sav']):
                os.makedirs(parsed['sav'])
            # else:
            #     os.system(f'rm -rf {parsed["sav"]}')

            master_work(parsed['n'], parsed['sav'], parsed['f'])
        else:
            slave_work(pm)
    else:
        create_test_amplitudes(pm)
        test_results(pm)
