from enum import IntEnum
from main import main
import parameters_parallel as pm
import numpy as np
from mpi4py import MPI
import os, argparse
from tqdm import tqdm
import traceback, time
import pickle as pkl


# Define the class to store the MPI tags
class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


# Function to dump the data to files
def dump_data(zz, tau, BB, stokes, filename, task_index=None):
    with open(f'{filename}zz.pkl', 'wb') as f:
        pkl.dump(zz[0:task_index], f)
    
    with open(f'{filename}tau.pkl', 'wb') as f:
        pkl.dump(tau[0:task_index], f)
    
    with open(f'{filename}BB.pkl', 'wb') as f:
        pkl.dump(BB[0:task_index], f)

    with open(f'{filename}stokes.pkl', 'wb') as f:
        pkl.dump(stokes[0:task_index], f)


def new_parameters(pm, npoints, index):

    # Define the parameters for the grid in tau and B
    # the idea is to first vary tau for a fixed B and then move the B
    # for that we need to have the number of points in tau and B and the index
    pm.B = [0, 0.01, 0.025, 0.05, 0.08, 0.15, 0.3, 0.5, 0.75, 0.9, 1.1,
            2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 50.0, 100.0, 250.0, 400.0,  500.0]
    pm.z0 = pm.z0
    pm.zf = pm.z0 + np.array([1, 50, 100, 200])*1e5

    # reduced grid for testing
    # pm.B = [0, 0.1, 1, 10, 100]
    # pm.zf = pm.z0 + np.array([25, 50, 100, 200, 300])*1e5

    b_index = index // len(pm.zf)
    t_index = index % len(pm.zf)

    pm.B = pm.B[b_index]
    pm.zf = pm.zf[t_index]
    pm.dir = f'{pm.basedir}tau_{t_index}_BB_{pm.B}_{time.strftime("%Y%m%d-%H%M%S")}/'
    return pm


def master_work(npoints):

    # Index of the task to keep track of each job
    task_status = [0]*npoints
    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    # Initialize the variables to store the results
    tau = [None]*npoints
    stokes = [None]*npoints
    zz = [None]*npoints
    BB = [None]*npoints

    with tqdm(total=npoints, ncols=100, disable=True) as pbar:
        while closed_workers < num_workers:
            # Recieve the ready signal from the workers
            print('\n')
            print('-'*50)
            print('waiting data from the workers')
            dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            # If the worker is ready to compute
            if tag == tags.READY:
                print('worker {} is ready to compute'.format(source))
                # Look for the first task that is not done
                try:
                    task_index = task_status.index(0)
                except ValueError:
                    # If all the tasks are done, send the terminate signal to the workers
                    task_index = -1
                
                # Send the task to the worker
                if task_index >= 0:
                    print('sending task {}/{} to worker {}'.format(task_index, npoints, source))
                    print('wich corresponds grid  (B,tau)   =    ({},{})'.format(task_index//8, task_index%8))
                    comm.send(task_index, dest=source, tag=tags.START)
                    task_status[task_index] = 1
                else:
                    print('sending terminate signal to worker {}'.format(source))
                    # If the task is -1 (all the tasks done), send the kill signal
                    comm.send(None, dest=source, tag=tags.EXIT)
                    closed_workers += 1

            if tag == tags.DONE:
                # If the worker is done, store the results
                task_index = dataReceived['index']
                success = dataReceived['success']
                
                if success:
                    tau[task_index] = dataReceived['tau']
                    stokes[task_index] = dataReceived['stokes']
                    zz[task_index] = dataReceived['zz']
                    BB[task_index] = dataReceived['BB']
                    task_status[task_index] = 1
                    pbar.update(1)
                else:
                    task_status[task_index] = 0

            # If the number of itterations is multiple with the write frequency dump the data
            if (task_index / 10 == task_index // 10):
                # Dump the data
                print('saving data')
                dump_data(zz, tau, BB, stokes, pm.basedir, pbar.n)

    # Once all the workers are done, dump the data
    print('\n'+'#'*50)
    print('All the workers are done')
    print('#'*50)
    print("Master finishing")
    print("Dumping data")
    print('#'*50)
    dump_data(zz, tau, BB, stokes, pm.basedir)


def slave_work(npoints, pm = pm):

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
                pm = new_parameters(pm, npoints, task_index)
                # compute the profile
                out = main(pm, disable_display=True)
                ray, nus, zz, tau, stokes = out[0]

            except:
                success = False
                print('-'*50)
                print("Error in the computation of the profile")
                print("The parameters are:")
                print("pm:", module_to_dict(pm))
                print("The error is:")
                print(traceback.format_exc())
                print('-'*50)
                ray, nus, zz, tau, stokes, pm.B = [None]*5

            # Send the results to the master
            dataToSend = {'index': task_index, 'success': success, 
                          'stokes': stokes, 'tau': tau, 'ray': ray, 'nus': nus, 'zz': zz, 'BB': pm.B}
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

    parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
    parser.add_argument('--n', '--npoints', default=176, type=int, metavar='NPOINTS', help='Number of points')
    parsed = vars(parser.parse_args())

    if rank == 0:
        
        # Create the directory to store the different runs if it does not exist
        if not os.path.exists(pm.basedir):
            os.makedirs(pm.basedir)

        master_work(parsed['n'])
    else:
        slave_work(parsed['n'])
