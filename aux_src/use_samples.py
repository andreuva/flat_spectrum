import pickle as pkl
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

datadir = 'output_hazel_samples/'

# read the stokes, tau and params files
with open(f'{datadir}stokes.pkl', 'rb') as f:
    stokes = pkl.load(f)

with open(f'{datadir}tau.pkl', 'rb') as f:
    taus = pkl.load(f)

with open(f'{datadir}params.pkl', 'rb') as f:
    params = pkl.load(f)

# read all the folders in the output directory
folders = glob(f'{datadir}sample_*')

# get the arrays of each parameter in the samples
z0 = np.array([p['z0'] for p in params])
zf = np.array([p['zf'] for p in params])
temp = np.array([p['temp'] for p in params])
mu = np.array([p['ray_out'][0][0] for p in params])
B = np.array([p['B'] for p in params])
nus = np.array([p['nus'] for p in params])

stokes = np.array(stokes)
taus = np.array(taus)

# show a subsample of the stokes profiles (I,Q,U,V)
for ind, st in enumerate(['I','Q','U','V']):
    plt.figure(figsize=(20, 20))
    for pl, i in enumerate(np.random.randint(0, len(stokes)-1, 9)):
        plt.subplot(3, 3, pl+1)
        norm = stokes[i, 0, 0] if ind==0 else stokes[i, 0, :]
        plt.plot(nus[i], stokes[i, ind, :]/norm)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(f'{st}/I')
        plt.title(fr'$\tau = $ {taus[i].max():.2f} B = {B[i]:.0f} G, mu = {mu[i]:.2f}, T = {temp[i]:.0f} K')
    # plt.tight_layout()
    plt.savefig(f'{datadir}stokes_{st}_profiles.png')
    plt.close()



# get the distribution of the parameters in the samples
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.hist(z0/1e5, bins=20)
plt.xlabel('z0 [km]')
plt.subplot(2, 2, 2)
plt.hist(zf/1e5, bins=20)
plt.xlabel('zf [km]')
plt.subplot(2, 2, 3)
plt.hist(temp, bins=20)
plt.xlabel('temp [K]')
plt.subplot(2, 2, 4)
plt.hist(B, bins=20)
plt.xlabel('B [G]')
plt.tight_layout()
plt.savefig(f'{datadir}parameter_distribution.png')
plt.close()