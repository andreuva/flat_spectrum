from random import choices
import numpy as np
import pickle as pkl
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from plot_utils import plot_4_profiles
from physics_test import ang_to_cart, cart_to_ang
from glob import glob

# load the data for the separated components
n_plots = 5
files = glob('plot_1/plot_1_*.pkl')
files.sort()
last_file = files[-2]
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
    plot_4_profiles(nus, profile['eps_I'], profile['eps_Q'], profile['eps_U'], profile['eps_V'], title=r'$\epsilon$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'eps_{index}_{timestr}')
    plot_4_profiles(nus, profile['eta_I'], profile['eta_Q'], profile['eta_U'], profile['eta_V'], title=r'$\eta$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'eta_{index}_{timestr}')
    plot_4_profiles(nus, profile['eta_I'], profile['rho_Q'], profile['rho_U'], profile['rho_V'], title=r'$\rho$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'rho_{index}_{timestr}')

# load the data for the multiterm case
n_plots = 5
files = glob('plot_1/plot_1_*.pkl')
files.sort()
last_file = files[-1]
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
    plot_4_profiles(nus, profile['eps_I'], profile['eps_Q'], profile['eps_U'], profile['eps_V'], title=r'$\epsilon$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'eps_{index}_{timestr}')
    plot_4_profiles(nus, profile['eta_I'], profile['eta_Q'], profile['eta_U'], profile['eta_V'], title=r'$\eta$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'eta_{index}_{timestr}')
    plot_4_profiles(nus, profile['eta_I'], profile['rho_Q'], profile['rho_U'], profile['rho_V'], title=r'$\rho$'+'\n'+title,
                    save=True, show=False, directory=f'plot_1', name=f'rho_{index}_{timestr}')

difference_eps = np.zeros((len(profiles_samples), 4, len(nus)))
difference_eta = np.zeros((len(profiles_samples), 4, len(nus)))
difference_rho = np.zeros((len(profiles_samples), 3, len(nus)))
for index, (profile, profile_mt) in enumerate(zip(profiles_samples, profiles_samples_mt)):
    difference_eps[index, 0] = (profile_mt['eps_I'] - profile['eps_I'])/profile['eta_I'].max() * 100
    difference_eps[index, 1] = (profile_mt['eps_Q'] - profile['eps_Q'])/profile['eta_I'].max() * 100
    difference_eps[index, 2] = (profile_mt['eps_U'] - profile['eps_U'])/profile['eta_I'].max() * 100
    difference_eps[index, 3] = (profile_mt['eps_V'] - profile['eps_V'])/profile['eta_I'].max() * 100

    difference_eta[index, 0] = (profile_mt['eta_I'] - profile['eta_I'])/profile['eta_I'].max() * 100
    difference_eta[index, 1] = (profile_mt['eta_Q'] - profile['eta_Q'])/profile['eta_I'].max() * 100
    difference_eta[index, 2] = (profile_mt['eta_U'] - profile['eta_U'])/profile['eta_I'].max() * 100
    difference_eta[index, 3] = (profile_mt['eta_V'] - profile['eta_V'])/profile['eta_I'].max() * 100

    difference_rho[index, 0] = (profile_mt['rho_Q'] - profile['rho_Q'])/profile['eta_I'].max() * 100
    difference_rho[index, 1] = (profile_mt['rho_U'] - profile['rho_U'])/profile['eta_I'].max() * 100
    difference_rho[index, 2] = (profile_mt['rho_V'] - profile['rho_V'])/profile['eta_I'].max() * 100

# plot a panel with 4 images of the difference between the two profiles
fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
mag_fields_ticks = np.linspace(0, len(profiles_samples)-1, 15, dtype=int)
for i, ax in enumerate(axs.flatten()):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(difference_eps[:,i,:], aspect='auto', origin='lower', cmap='bwr',  vmin=-np.abs(difference_eps[:,i,:].max()), vmax=np.abs(difference_eps[:,i,:].max()))
    fig.colorbar(im, cax=cax, orientation='vertical', label='% difference')
    ax.set_xticks(range(len(wave))[::60], [f'{val*1e9:4.2f}' for val in wave[::60]])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks(mag_fields_ticks, [f'{B[0]:4.2f}' for _,B in parameters_mt[mag_fields_ticks]])
    ax.set_ylabel('Magnetic field (G)')
    ax.set_title(f'$\epsilon_{i+1}$')

fig.suptitle(fr'Difference between the two profiles, $\epsilon$')
fig.tight_layout()
plt.savefig(f'plot_1/difference_eps_{timestr}.png')
plt.close()

# plot a panel with 4 images of the difference between the two profiles
fig, axs = plt.subplots(2, 2, figsize=(20, 20), dpi=100)
mag_fields_ticks = np.linspace(0, len(profiles_samples)-1, 15, dtype=int)
for i, ax in enumerate(axs.flatten()):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(difference_eta[:,i,:], aspect='auto', origin='lower', cmap='bwr', vmin=-np.abs(difference_eta[:,i,:].max()), vmax=np.abs(difference_eta[:,i,:].max()))
    fig.colorbar(im, cax=cax, orientation='vertical', label='% difference')
    ax.set_xticks(range(len(wave))[::60], [f'{val*1e9:4.2f}' for val in wave[::60]])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks(mag_fields_ticks, [f'{B[0]:4.2f}' for _,B in parameters_mt[mag_fields_ticks]])
    ax.set_ylabel('Magnetic field (G)')
    ax.set_title(f'$\eta_{i+1}$')

fig.suptitle(fr'Difference between the two profiles, $\eta$')
fig.tight_layout()
plt.savefig(f'plot_1/difference_eta_{timestr}.png')
plt.close()

# plot a panel with 4 images of the difference between the two profiles
fig, axs = plt.subplots(1, 3, figsize=(23, 7), dpi=100)
mag_fields_ticks = np.linspace(0, len(profiles_samples)-1, 15, dtype=int)
for i, ax in enumerate(axs.flatten()):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(difference_rho[:,i,:], aspect='auto', origin='lower', cmap='bwr', vmin=-np.abs(difference_rho[:,i,:].max()), vmax=np.abs(difference_rho[:,i,:].max()))
    fig.colorbar(im, cax=cax, orientation='vertical', label='% difference')
    ax.set_xticks(range(len(wave))[::60], [f'{val*1e9:4.2f}' for val in wave[::60]])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_yticks(mag_fields_ticks, [f'{B[0]:4.2f}' for _,B in parameters_mt[mag_fields_ticks]])
    ax.set_ylabel('Magnetic field (G)')
    ax.set_title(fr'$\rho_{i+1}$')

fig.suptitle(fr'Difference between the two profiles, $\rho$')
fig.tight_layout()
plt.savefig(f'plot_1/difference_rho_{timestr}.png')
plt.close()

mag_fields_mt = np.array([B[0] for _,B in parameters_mt])
mag_fields = np.array([B[0] for _,B in parameters])

eps_I_red = np.array([profile['eps_I'][nu_peak_1_indx] for profile in profiles_samples])
eps_I_red_mt = np.array([profile_mt['eps_I'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eps_I_blue = np.array([profile['eps_I'][nu_peak_2_indx] for profile in profiles_samples])
eps_I_blue_mt = np.array([profile_mt['eps_I'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eps_Q_red = np.array([profile['eps_Q'][nu_peak_1_indx] for profile in profiles_samples])
eps_Q_red_mt = np.array([profile_mt['eps_Q'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eps_Q_blue = np.array([profile['eps_Q'][nu_peak_2_indx] for profile in profiles_samples])
eps_Q_blue_mt = np.array([profile_mt['eps_Q'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eps_U_red = np.array([profile['eps_U'][nu_peak_1_indx] for profile in profiles_samples])
eps_U_red_mt = np.array([profile_mt['eps_U'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eps_U_blue = np.array([profile['eps_U'][nu_peak_2_indx] for profile in profiles_samples])
eps_U_blue_mt = np.array([profile_mt['eps_U'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eps_V_red = np.array([profile['eps_V'][nu_peak_1_indx] for profile in profiles_samples])
eps_V_red_mt = np.array([profile_mt['eps_V'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eps_V_blue = np.array([profile['eps_V'][nu_peak_2_indx] for profile in profiles_samples])
eps_V_blue_mt = np.array([profile_mt['eps_V'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eta_I_red = np.array([profile['eta_I'][nu_peak_1_indx] for profile in profiles_samples])
eta_I_red_mt = np.array([profile_mt['eta_I'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eta_I_blue = np.array([profile['eta_I'][nu_peak_2_indx] for profile in profiles_samples])
eta_I_blue_mt = np.array([profile_mt['eta_I'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eta_Q_red = np.array([profile['eta_Q'][nu_peak_1_indx] for profile in profiles_samples])
eta_Q_red_mt = np.array([profile_mt['eta_Q'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eta_Q_blue = np.array([profile['eta_Q'][nu_peak_2_indx] for profile in profiles_samples])
eta_Q_blue_mt = np.array([profile_mt['eta_Q'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eta_U_red = np.array([profile['eta_U'][nu_peak_1_indx] for profile in profiles_samples])
eta_U_red_mt = np.array([profile_mt['eta_U'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eta_U_blue = np.array([profile['eta_U'][nu_peak_2_indx] for profile in profiles_samples])
eta_U_blue_mt = np.array([profile_mt['eta_U'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])

eta_V_red = np.array([profile['eta_V'][nu_peak_1_indx] for profile in profiles_samples])
eta_V_red_mt = np.array([profile_mt['eta_V'][nu_peak_1_indx] for profile_mt in profiles_samples_mt])
eta_V_blue = np.array([profile['eta_V'][nu_peak_2_indx] for profile in profiles_samples])
eta_V_blue_mt = np.array([profile_mt['eta_V'][nu_peak_2_indx] for profile_mt in profiles_samples_mt])


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(mag_fields, eps_I_red, f'C0', label=r'$\epsilon$ red')
plt.plot(mag_fields_mt, eps_I_red_mt, f':C0', label=r'$\epsilon$ red (mt)')
plt.plot(mag_fields, eps_I_blue, f'C1', label=r'$\epsilon$ blue')
plt.plot(mag_fields_mt, eps_I_blue_mt, f':C1', label=r'$\epsilon$ blue (mt)')
plt.title(fr'$\epsilon_I$')
plt.xlabel(r'$B$')
plt.legend()
plt.subplot(2,2,2)
plt.plot(mag_fields, eps_Q_red/eps_I_red, f'C0')
plt.plot(mag_fields_mt, eps_Q_red_mt/eps_I_red_mt, f':C0')
plt.plot(mag_fields, eps_Q_blue/eps_I_blue, f'C1')
plt.plot(mag_fields_mt, eps_Q_blue_mt/eps_I_blue_mt, f':C1')
plt.title(fr'$\epsilon_Q/\epsilon_I$')
plt.xlabel(r'$B$')
plt.subplot(2,2,3)
plt.plot(mag_fields, eps_U_red/eps_I_red, f'C0')
plt.plot(mag_fields_mt, eps_U_red_mt/eps_I_red_mt, f':C0')
plt.plot(mag_fields, eps_U_blue/eps_I_blue, f'C1')
plt.plot(mag_fields_mt, eps_U_blue_mt/eps_I_blue_mt, f':C1')
plt.title(fr'$\epsilon_U/\epsilon_I$')
plt.xlabel(r'$B$')
plt.subplot(2,2,4)
plt.plot(mag_fields, eps_V_red/eps_I_red, f'C0')
plt.plot(mag_fields_mt, eps_V_red_mt/eps_I_red_mt, f':C0')
plt.plot(mag_fields, eps_V_blue/eps_I_blue, f'C1')
plt.plot(mag_fields_mt, eps_V_blue_mt/eps_I_blue_mt, f':C1')
plt.title(fr'$\epsilon_V/\epsilon_I$')
plt.xlabel(r'$B$')

plt.suptitle('B '+title[15:])
plt.tight_layout()
plt.savefig(f'plot_1/peaks_ratio_eps_{timestr}.png')
# plt.show()
plt.close()

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(mag_fields, eta_I_red, f'C0', label=r'$\eta$ red')
plt.plot(mag_fields_mt, eta_I_red_mt, f':C0', label=r'$\eta$ red (mt)')
plt.plot(mag_fields, eta_I_blue, f'C1', label=r'$\eta$ blue')
plt.plot(mag_fields_mt, eta_I_blue_mt, f':C1', label=r'$\eta$ blue (mt)')
plt.title(fr'$\eta_I$')
plt.xlabel(r'$B$')
plt.legend()
plt.subplot(2,2,2)
plt.plot(mag_fields, eta_Q_red/eta_I_red, f'C0')
plt.plot(mag_fields_mt, eta_Q_red_mt/eta_I_red_mt, f':C0')
plt.plot(mag_fields, eta_Q_blue/eta_I_blue, f'C1')
plt.plot(mag_fields_mt, eta_Q_blue_mt/eta_I_blue_mt, f':C1')
plt.title(fr'$\eta_Q/\eta_I$')
plt.xlabel(r'$B$')
plt.subplot(2,2,3)
plt.plot(mag_fields, eta_U_red/eta_I_red, f'C0')
plt.plot(mag_fields_mt, eta_U_red_mt/eta_I_red_mt, f':C0')
plt.plot(mag_fields, eta_U_blue/eta_I_blue, f'C1')
plt.plot(mag_fields_mt, eta_U_blue_mt/eta_I_blue_mt, f':C1')
plt.title(fr'$\eta_U/\eta_I$')
plt.xlabel(r'$B$')
plt.subplot(2,2,4)
plt.plot(mag_fields, eta_V_red/eta_I_red, f'C0')
plt.plot(mag_fields_mt, eta_V_red_mt/eta_I_red_mt, f':C0')
plt.plot(mag_fields, eta_V_blue/eta_I_blue, f'C1')
plt.plot(mag_fields_mt, eta_V_blue_mt/eta_I_blue_mt, f':C1')
plt.title(fr'$\eta_V/\eta_I$')
plt.xlabel(r'$B$')

plt.suptitle('B '+title[15:])
plt.tight_layout()
plt.savefig(f'plot_1/peaks_ratio_eta_{timestr}.png')
# plt.show()
plt.close()

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(mag_fields, difference_eps[:,0,nu_peak_1_indx], f'C1', label=r'diff $\epsilon$ red')
plt.plot(mag_fields, difference_eps[:,0,nu_peak_2_indx], f'C0', label=r'diff $\epsilon$ blue')
plt.title(r'$\frac{\epsilon_{I mt} - \epsilon_I}{MAX(\epsilon_{I})} \%$')
plt.xlabel(r'$B$')
plt.legend()
plt.subplot(2,2,2)
plt.plot(mag_fields, difference_eps[:,1,nu_peak_1_indx], f'C1')
plt.plot(mag_fields, difference_eps[:,1,nu_peak_2_indx], f'C0')
plt.title(r'$\frac{\epsilon_{Q mt} - \epsilon_Q}{MAX(\epsilon_{I})} \%$')
plt.xlabel(r'$B$')
plt.subplot(2,2,3)
plt.plot(mag_fields, difference_eps[:,2,nu_peak_1_indx], f'C1')
plt.plot(mag_fields, difference_eps[:,2,nu_peak_2_indx], f'C0')
plt.title(r'$\frac{\epsilon_{U mt} - \epsilon_U}{MAX(\epsilon_{I})} \%$')
plt.xlabel(r'$B$')
plt.subplot(2,2,4)
plt.plot(mag_fields, difference_eps[:,3,nu_peak_1_indx], f'C1')
plt.plot(mag_fields, difference_eps[:,3,nu_peak_2_indx], f'C0')
plt.title(r'$\frac{\epsilon_{V mt} - \epsilon_V}{MAX(\epsilon_{I})} \%$')
plt.xlabel(r'$B$')

plt.suptitle('B '+title[15:])
plt.tight_layout()
plt.savefig(f'plot_1/difference_peaks_eps_{timestr}.png')
# plt.show()
plt.close()