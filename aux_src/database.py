# Especific modules
import parameters_rtcoefs as pm
from conditions import conditions
from RTcoefs import RTcoefs
from atom import ESE
from tensors import JKQ_to_Jqq, construct_JKQ_0
import constants as cts

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


# Function to dump the data to files
def dump_data(profiles, parameters, filename, task_index=None):
    with open(f'{filename}profiles.pkl', 'wb') as f:
        pkl.dump(profiles[0:task_index], f)
    
    with open(f'{filename}parameters.pkl', 'wb') as f:
        pkl.dump(parameters[0:task_index], f)


def new_parameters(pm):

    # B field will change with each itteration to cover all the possible cases
    pm.B = np.random.normal(10, 100)
    while pm.B < 0:
        pm.B = np.random.normal(10, 100)

    pm.B_inc = np.arccos(np.random.uniform(0, 1))*180/np.pi
    pm.B_az = np.random.uniform(0, 360)

    pm.mu = np.random.uniform(-1,1)
    pm.chi = np.random.uniform(0,np.pi)
    # ray direction (will change with each itteration to cover all the possible cases)
    pm.ray_out = [[pm.mu, pm.chi]]
    # amplitude of the profile
    pm.a_voigt = 10**np.random.uniform(-1, -10) #  1e-6 to 0.
    pm.temp = 10**np.random.uniform(3., 5.)

    # construct the JKQ dictionary
    pm.JKQr = construct_JKQ_0()
    pm.JKQr[0][0] = np.random.lognormal(-5, 1.5)
    pm.JKQr[1][0] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]
    pm.JKQr[2][0] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]

    pm.JKQr[1][1] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j
    pm.JKQr[2][1] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j
    pm.JKQr[2][2] = np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQr[0][0]*1j

    pm.JKQr[2][-2] =      np.conjugate(pm.JKQr[2][2])
    pm.JKQr[2][-1] = -1.0*np.conjugate(pm.JKQr[2][1])
    pm.JKQr[1][-1] = -1.0*np.conjugate(pm.JKQr[1][1])

    pm.JKQb = construct_JKQ_0()
    pm.JKQb[0][0] = np.random.lognormal(-5, 1.5)
    pm.JKQb[1][0] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]
    pm.JKQb[2][0] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]

    pm.JKQb[1][1] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j
    pm.JKQb[2][1] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j
    pm.JKQb[2][2] = np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0] + np.random.uniform(-0.2, 0.2)*pm.JKQb[0][0]*1j

    pm.JKQb[2][-2] =      np.conjugate(pm.JKQb[2][2])
    pm.JKQb[2][-1] = -1.0*np.conjugate(pm.JKQb[2][1])
    pm.JKQb[1][-1] = -1.0*np.conjugate(pm.JKQb[1][1])

    return pm


# Wraper to compute the Rtcoefs module with a given parameters
def compute_profile(pm=pm, especial=True, jqq=None):
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
                pm = new_parameters(pm)
                # compute the profile
                profiles = compute_profile(pm=pm)
            except:
                success = False
                print('-'*50)
                print("Error in the computation of the profile")
                print("The parameters are:")
                print("JKQ_red:", pm.JKQr)
                print("JKQ_blue:", pm.JKQb)
                print("B:", pm.B)
                print("pm:", pm)
                print("The error is:")
                print(traceback.format_exc())
                print('-'*50)
                profiles = None

            parameters = {'JKQr':pm.JKQr, 'JKQb':pm.JKQb,
                          'B':pm.B, 'B_inc':pm.B_inc, 'B_az':pm.B_az,
                          'mu':pm.mu, 'chi':pm.chi,
                          'a_voigt':pm.a_voigt, 'temp':pm.temp}

            # Send the results to the master
            dataToSend = {'index': task_index, 'success': success, 'profiles': profiles, 
                          'parameters': parameters}
            comm.send(dataToSend, dest=0, tag=tags.DONE)
        
        # If the master is sending the kill signal exit
        elif tag == tags.EXIT:
            return


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

    if rank == 0:
        parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
        parser.add_argument('-n', '--nmodels', default=1000, type=int, metavar='NMODELS', help='Number of models')
        parser.add_argument('-f', '--freq', default=100, type=int, metavar='FREQ', help='Frequency of model write')
        parser.add_argument('-s', '--savedir', default=f'database', metavar='SAVEDIR', help='directory for output files')
        parser.add_argument('-i', '--id', default=f'', metavar='ID', help='identifier for the database name')

        parsed = vars(parser.parse_args())
        parsed['savedir'] = os.path.join(parsed['savedir'], f'data_{parsed["id"]}_{time.strftime("%Y%m%d_%H%M%S")}/')

        if not os.path.exists(parsed['savedir']):
            os.makedirs(parsed['savedir'])

        master_work(parsed['nmodels'], parsed['savedir'], parsed['freq'])
    else:
        slave_work(pm)
