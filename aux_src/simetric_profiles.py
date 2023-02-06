from enum import IntEnum
from main import main
import parameters as pm
from conditions import conditions
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import os, argparse
from tqdm import tqdm
import traceback
import pickle as pkl


# Define the class to store the MPI tags
class tags(IntEnum):
    """ Class to define the state of a worker.
    It inherits from the IntEnum class """
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3


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


# Function to dump the data to files
def dump_data(tau, stokes, rays, velocities, Bs, filename, task_index=None):
    with open(f'{filename}tau.pkl', 'wb') as f:
        pkl.dump(tau[0:task_index], f)

    with open(f'{filename}stokes.pkl', 'wb') as f:
        pkl.dump(stokes[0:task_index], f)

    with open(f'{filename}rays.pkl', 'wb') as f:
        pkl.dump(rays[0:task_index], f)

    with open(f'{filename}velocities.pkl', 'wb') as f:
        pkl.dump(velocities[0:task_index], f)
    
    with open(f'{filename}Bs.pkl', 'wb') as f:
        pkl.dump(Bs[0:task_index], f)


def new_parameters(pm, npoints, index):

    # ray direction (will change with each itteration to cover all the possible cases)
    mu = np.random.uniform(0, np.pi/2)
    mu = np.cos(mu)
    chi = np.random.uniform(0, 180)
    pm.ray_out = [[mu, chi]]

    pm.B_inc = np.random.uniform(0, 90)
    pm.B_az = mu = np.random.uniform(0, 180)

    # pm.velocity = random_three_vector()*2.5e4

    pm.dir = f'{pm.basedir}mu_{pm.ray_out[0][0]}_az_{pm.ray_out[0][1]}'+\
             f'_bmu_{pm.B_inc}_baz_{pm.B_az}_vx_{pm.velocity[0]}_vy_{pm.velocity[1]}_vz_{pm.velocity[2]}/'

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
    rays = [None]*npoints
    velocities = [None]*npoints
    Bs = [None]*npoints

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
                    rays[task_index] = dataReceived['ray']
                    velocities[task_index] = dataReceived['velocity']
                    Bs[task_index] = (dataReceived['B_inc'], dataReceived['B_az'])
                    task_status[task_index] = 1
                    pbar.update(1)
                else:
                    task_status[task_index] = 0

            # If the number of itterations is multiple with the write frequency dump the data
            if (task_index / 10 == task_index // 10):
                # Dump the data
                print('saving data')
                dump_data(tau, stokes, rays, velocities, Bs, pm.basedir, pbar.n)

    # Once all the workers are done, dump the data
    print('\n'+'#'*50)
    print('All the workers are done')
    print('#'*50)
    print("Master finishing")
    print("Dumping data")
    print('#'*50)
    dump_data(tau, stokes, rays, velocities, Bs, pm.basedir)


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
                ray, nus, _, tau, stokes = out[0]

            except:
                success = False
                print('-'*50)
                print("Error in the computation of the profile")
                print("The parameters are:")
                print("pm:", module_to_dict(pm))
                print("The error is:")
                print(traceback.format_exc())
                print('-'*50)
                ray, nus, _, tau, stokes = [None]*5

            # Send the results to the master
            dataToSend = {'index': task_index, 'success': success, 
                          'stokes': stokes, 'tau': tau, 'ray': ray, 'nus': nus,
                          'velocity': pm.velocity, 'B_inc': pm.B_inc, 'B_az': pm.B_az}
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
    parser.add_argument('--n', '--npoints', default=250, type=int, metavar='NPOINTS', help='Number of points')
    parsed = vars(parser.parse_args())

    if rank == 0:
        master_work(parsed['n'])
    else:
        slave_work(parsed['n'])
