from random import choices
import numpy as np
import pickle as pkl
import time
import matplotlib.pyplot as plt
from plot_utils import plot_4_profiles
from physics_test import ang_to_cart, cart_to_ang

# load the data
flat_spectrum = False
sun_ilum = False
background_type = 'absorption'

with open(f'plots_ratio_fs_False_sunilum_False_fil_100K/fs_{flat_spectrum}_sunilum_{sun_ilum}_20220409_021156.pkl', 'rb') as f:
    data = pkl.load(f)

intensities_samples = data['intensities']
profiles_samples = data['profiles']
parameters = data['parameters']
ratio_Q = data['ratio_Q']
ratio_U = data['ratio_U']
nus = data['profiles'][0]['nus']
pm = data['pm']

nu_1 = 2.76733e14
nu_2 = 2.76764e14

nu_peak_1_indx = np.abs(nus - nu_1).argmin()
nu_peak_2_indx = np.abs(nus - nu_2).argmin()

timestr = time.strftime("%Y%m%d_%H%M%S")
print('plot the ratio histogram')
bins = np.linspace(-100,100,2000)
plt.hist(ratio_Q, bins=bins, label='Q', alpha=0.3)
plt.hist(ratio_U, bins=bins, label='U', alpha=0.3)
plt.legend()
plt.savefig('ratio_Q_U_'+f'fs_{flat_spectrum}_sunilum_{sun_ilum}_{timestr}'+'.png')
plt.show()
plt.close()

# mask the intensities with positives values
if sun_ilum:
    indx_selected = np.where(ratio_U>0)[0]
else:
    indx_selected = np.where(ratio_U<0)[0]

intensities_selected = intensities_samples[indx_selected]
params_selected = parameters[indx_selected]

print(f'{len(intensities_selected)} selected ratio U (blue/red)')
if sun_ilum:
    print('The selection is from the U(R)/U(B) > 0')
else:
    print('The selection is from the U(R)/U(B) < 0')

print('-'*50)
print(f'mean intensity of the selection I = {np.mean(intensities_selected[:,0,:]):1.2e}')
print(f'mean intensity signal U (blue/red > 0) = {np.mean(intensities_selected[:,2,:]):1.2e}')
print(f'mean intensity total population I = {np.mean(intensities_samples[:,0,:]):1.2e}')
print(f'mean intensity total signal U  = {np.mean(intensities_samples[:,2,:]):1.2e}')
print('-'*50)

# plot a sample of the final profiles
# if the total number is more than 100 then take a random sample
if len(intensities_selected)>100:
    plot_selected = np.random.choice(len(intensities_selected), 100, replace=False)
else:
    plot_selected = np.arange(len(intensities_selected))

# for num, intensity in enumerate(choices(intensities_selected, k=5)):
for num, intensity in enumerate(intensities_selected[plot_selected]):
    velocity, B_spherical = params_selected[num]
    title = f'v = [{velocity[0]/1e3:1.2f}, {velocity[1]/1e3:1.2f}, {velocity[2]/1e3:1.2f}] km/s \n B = {B_spherical[0]:1.2f} G \t '+\
            fr'$\theta$={B_spherical[1]*180/np.pi:1.2f},'+'\t'+fr' $\phi$={B_spherical[2]*180/np.pi:1.2f}'+\
            '\n'+ fr' LOS:  $\mu$ = {pm["ray_out"][0][0]:1.2f} $\phi$ = {pm["ray_out"][0][1]:1.2f}'+\
            '\n'+ r' $I_{sun}$'+f'= {sun_ilum} \t {background_type}'
    plot_4_profiles(nus, intensity[0], intensity[1]/intensity[0].max(), intensity[2]/intensity[0].max(), intensity[3]/intensity[0].max(), title=title,
                    save=True, show=False, directory=f'plots_ratio_norm_fs_{flat_spectrum}_sunilum_{sun_ilum}_{timestr}', name=f'S_{num}')

# plot the distribution of magnetic fields and velocities
velocities = np.array([cart_to_ang(*param[0]) for param in params_selected])
B_spherical = np.array([param[1] for param in params_selected])
print('plot the distribution of magnetic fields and velocities')
plt.figure(figsize=(10,5))
plt.plot(np.cos(velocities[:,1]), velocities[:,2]*180/np.pi, '.', alpha=0.2, label='v')
plt.plot(np.cos(B_spherical[:,1]), B_spherical[:,2]*180/np.pi, '.', alpha=0.2, label='B')
plt.ylabel(r'$\phi$ (deg)')
plt.xlabel(r'$\mu$')
plt.title(f'B and v distribution')
plt.legend()
plt.savefig('B_V_'+f'fs_{flat_spectrum}_sunilum_{sun_ilum}_{timestr}'+'.png')
plt.show()
plt.close()
