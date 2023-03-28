import numpy as np
import matplotlib.pyplot as plt
from scipy.io import readsav


data = readsav('para_andreu.sav')

lam = data['lam']
I_obs = data['perfil_obs']
I_anal = data['perfil_syn']

datadir = 'output_hazel_comp_20230328-085952'
# datadir = ''

wave, tau = np.loadtxt(f'{datadir}/out/tau_00.out', skiprows=3, unpack=True)

print('tau max', tau.max())
plt.plot(wave, tau)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Optical depth')
plt.show()

nus, I_nlte, Q_nlte, U_nlte, V_nlte = np.loadtxt(f'{datadir}/out/stokes_00.out', skiprows=3, unpack=True)
norm = I_nlte.max()
II_nlte = np.array([I_nlte, -Q_nlte, -U_nlte, V_nlte])/norm

wave_nus = 299792458/nus
nn = 1.000293
wave_nus = wave_nus/1e-9/nn

nu_1 = 2.76733e14
nu_2 = 2.76764e14
wave_1 = 299792458/nu_1/1e-9/nn
wave_2 = 299792458/nu_2/1e-9/nn

length = len(wave_nus)
p1 = int(length/8)
p3 = int(p1*7)

II_nlte_plot = II_nlte[:,p1:p3]
wave_plot = wave_nus[p1:p3]

"""
PARAMETERS OF THE INVERSION:
Altura de la capa: 6000 km
LOS 98 grados == -0.13 mu
B = 22.8 G
B_inc = 149.0 grados
B_az = -0.1 grados
# v_dop = 5.0 km/s
vtherm = 6.4 km/s
vmacro = 0.41 km/s (0 en NLTE)
a_voigt = 0.19
tau = 1.8 (1.8 en NLTE)
"""


# make a 4 panel plot with the observed and synthetic profiles
fig, axes = plt.subplots(2, 2, figsize=(10,10))
labels = [r'$I/I_{max}$', r'$Q/I_{max}$', r'$U/I_{max}$', r'$V/I_{max}$']
for i,ax in enumerate(axes.flatten()):
    ax.plot(lam/10 + wave_2, I_obs[:,i], label='Observed')
    ax.plot(lam/10 + wave_2, I_anal[:,i], label='Hazel')
    ax.plot(wave_plot, II_nlte_plot[i], label='Synthetic')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel(labels[i])
    # vertical lines at the transition wavelengths
    ax.axvline(wave_1, color='r', linestyle='--')
    ax.axvline(wave_2, color='b', linestyle='--')
ax.legend()

plt.tight_layout()
plt.savefig('comparison_hazel_inversion.png')
plt.show()
