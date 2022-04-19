from random import choices
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
from plot_utils import plot_4_profiles
from physics_test import ang_to_cart, cart_to_ang
from glob import glob

# load the data for the separated components
n_plots = 5
files = glob('plot_1/plot_1_*.pkl')
files.sort()
last_file = files[0]
print(f'Loading {last_file}')
timestr =  last_file.split('_')[-2] + '_' + last_file.split('_')[-1].split('.')[0]
with open(last_file, 'rb') as f:
    data = pkl.load(f)

intensities_samples = data['intensities']
profiles_samples = data['profiles']
parameters = data['parameters']
ratio_Q = data['ratio_Q']
ratio_U = data['ratio_U']
nus = data['profiles'][0]['nus']
pm = data['pm']

wave = 299792458/nus
nu_1 = 2.76733e14
nu_2 = 2.76764e14

nu_peak_1_indx = np.abs(nus - nu_1).argmin()
nu_peak_2_indx = np.abs(nus - nu_2).argmin()

# plot a sample of the final profiles
# if the total number is more than 100 then take a random sample
if len(profiles_samples)>n_plots:
    plot_selection = np.linspace(0, len(profiles_samples)-1, n_plots, dtype=int)
else:
    plot_selection = np.arange(len(profiles_samples))

# for num, intensity in enumerate(choices(intensities, k=5)):
for index in plot_selection:
    profile = profiles_samples[index]
    velocity, B_spherical = parameters[index]
    title = f'B = {B_spherical[0]:1.2f} G \t' + fr'$\theta$={B_spherical[1]*180/np.pi:1.2f},'+'\t'+fr' $\phi$={B_spherical[2]*180/np.pi:1.2f}'+\
            '\n'+ fr' LOS:  $\mu$ = {pm["ray_out"][0][0]:1.2f} $\phi$ = {pm["ray_out"][0][1]:1.2f}'
    plot_4_profiles(nus, profile['eps_I'], profile['eps_Q'], profile['eps_U'], profile['eps_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'eps_{index}_{timestr}', eps=True)
    plot_4_profiles(nus, profile['eta_I'], profile['eta_Q'], profile['eta_U'], profile['eta_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'eta_{index}_{timestr}', eps=True)
    plot_4_profiles(nus, profile['eta_I'], profile['rho_Q'], profile['rho_U'], profile['rho_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'rho_{index}_{timestr}', eps=True)

# load the data for the multiterm case
n_plots = 5
files = glob('plot_1/plot_1_*mt.pkl')
files.sort()
last_file = files[0]
print(f'Loading {last_file}')
timestr =  last_file.split('_')[-2] + '_' + last_file.split('_')[-1].split('.')[0]
with open(last_file, 'rb') as f:
    data_mt = pkl.load(f)

intensities_samples_mt = data_mt['intensities']
profiles_samples_mt = data_mt['profiles']
parameters_mt = data_mt['parameters']
ratio_Q_mt = data_mt['ratio_Q']
ratio_U_mt = data_mt['ratio_U']
pm_mt = data_mt['pm']

# for num, intensity in enumerate(choices(intensities, k=5)):
for index in plot_selection:
    profile = profiles_samples_mt[index]
    velocity, B_spherical = parameters_mt[index]
    title = f'B = {B_spherical[0]:1.2f} G \t' + fr'$\theta$={B_spherical[1]*180/np.pi:1.2f},'+'\t'+fr' $\phi$={B_spherical[2]*180/np.pi:1.2f}'+\
            '\n'+ fr' LOS:  $\mu$ = {pm["ray_out"][0][0]:1.2f} $\phi$ = {pm["ray_out"][0][1]:1.2f}'
    plot_4_profiles(nus, profile['eps_I'], profile['eps_Q'], profile['eps_U'], profile['eps_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'eps_{index}_{timestr}', eps=True)
    plot_4_profiles(nus, profile['eta_I'], profile['eta_Q'], profile['eta_U'], profile['eta_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'eta_{index}_{timestr}', eps=True)
    plot_4_profiles(nus, profile['eta_I'], profile['rho_Q'], profile['rho_U'], profile['rho_V'], title=title,
                    save=True, show=False, directory=f'plot_1', name=f'rho_{index}_{timestr}', eps=True)

difference_eps = np.zeros((len(profiles_samples), 4, len(nus)))
difference_eta = np.zeros((len(profiles_samples), 4, len(nus)))
difference_rho = np.zeros((len(profiles_samples), 3, len(nus)))
for index, (profile, profile_mt) in enumerate(zip(profiles_samples, profiles_samples_mt)):
    difference_eps[index, 0] = (profile_mt['eps_I'] - profile['eps_I'])/(profile['eps_I']+1e-20) * 100
    difference_eps[index, 1] = (profile_mt['eps_Q'] - profile['eps_Q'])/(profile['eps_Q']+1e-20) * 100
    difference_eps[index, 2] = (profile_mt['eps_U'] - profile['eps_U'])/(profile['eps_U']+1e-20) * 100
    difference_eps[index, 3] = (profile_mt['eps_V'] - profile['eps_V'])/(profile['eps_V']+1e-20) * 100

    difference_eta[index, 0] = (profile_mt['eta_I'] - profile['eta_I'])/(profile['eta_I']+1e-20) * 100
    difference_eta[index, 1] = (profile_mt['eta_Q'] - profile['eta_Q'])/(profile['eta_Q']+1e-20) * 100
    difference_eta[index, 2] = (profile_mt['eta_U'] - profile['eta_U'])/(profile['eta_U']+1e-20) * 100
    difference_eta[index, 3] = (profile_mt['eta_V'] - profile['eta_V'])/(profile['eta_V']+1e-20) * 100

    difference_rho[index, 0] = (profile_mt['rho_Q'] - profile['rho_Q'])/(profile['rho_Q']+1e-20) * 100
    difference_rho[index, 1] = (profile_mt['rho_U'] - profile['rho_U'])/(profile['rho_U']+1e-20) * 100
    difference_rho[index, 2] = (profile_mt['rho_V'] - profile['rho_V'])/(profile['rho_V']+1e-20) * 100

# plot the differences
plt.imshow(difference_eps[:, 0, :], aspect='auto', origin='lower', cmap='plasma')
plt.colorbar(label='% difference')
# set the ticks in x as the wavelengths
plt.xticks(range(len(wave))[::50], [f'{val*1e9:4.2f}' for val in wave[::50]])
plt.xlabel('Wavelength (nm)')
# set the ticks in y as the magnetic field values
plt.yticks(plot_selection, [f'{B[0]:4.2f}' for _,B in parameters_mt[plot_selection]])
plt.ylabel('Magnetic field (G)')
plt.show()