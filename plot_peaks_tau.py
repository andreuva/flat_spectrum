from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle as pkl
from glob import glob
import parameters_peaks_vs_tau as pm
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 12

with open(f'{pm.basedir}zz.pkl', 'rb') as f:
    zz = pkl.load(f)

with open(f'{pm.basedir}tau.pkl', 'rb') as f:
    tau = pkl.load(f)

with open(f'{pm.basedir}stokes.pkl', 'rb') as f:
    stokes = pkl.load(f)

folders = glob(f'{pm.basedir}index*')
print(f'{len(folders)} itterations found')

wave_imp, _ = np.loadtxt(f'{folders[-1]}/out/tau_00.out', skiprows=3, unpack=True)

wave_imp = wave_imp*1e-9
nus = 299792458/wave_imp

nu_1 = 2.76733e14
nu_2 = 2.76764e14

wave_1 = 299792458/nu_1
wave_2 = 299792458/nu_2

nu_peak_1_indx = np.abs(nus - nu_1).argmin()
nu_peak_2_indx = np.abs(nus - nu_2).argmin()

tau_max = np.array([tt.max() for tt in tau])
intensity_red = np.array([profile[0][nu_peak_1_indx] for profile in stokes])
intensity_blue = np.array([profile[0][nu_peak_2_indx] for profile in stokes])
continuum = stokes[0][0][0]

taus = np.array([0.5, 1, 5, 10, 20])
indices = np.array([np.argmin(np.abs(tau_max - tt)) for tt in taus])

len_w = len(wave_imp)
p1 = int(len_w/8)
p3 = int(p1*7)
nn = 1.000293

ticks = [wave_imp[nu_peak_1_indx]/1e-9/nn, wave_imp[nu_peak_2_indx]/1e-9/nn]
labels = [f'{wave_imp[nu_peak_1_indx]/1e-9/nn:.2f}', f'{wave_imp[nu_peak_2_indx]/1e-9/nn:.2f}']
color_codes = ['#d9480f', '#5c940d', '#1864ab', '#ae3ec9', '#e03131']

plt.figure(figsize=(11,3.5), dpi=120)
plt.subplot(1,2,1)
for jj, ii in enumerate(indices):
    plt.plot(wave_imp[p1:p3]/1e-9/nn, stokes[ii][0][p1:p3]/continuum, color=cm.plasma(jj/len(taus)), label=fr'$\tau = {taus[jj]}$')

plt.arrow(wave_1/1e-9/nn, 1, 0, intensity_red[indices[0]]/continuum - 0.96, color='r', width=1e-4, head_width=5e-3, head_length=3e-2)
plt.arrow(wave_2/1e-9/nn, 1, 0, intensity_blue[indices[0]]/continuum - 0.96, color='b', width=1e-4, head_width=5e-3, head_length=3e-2)
plt.hlines(1, wave_imp[p1]/1e-9/nn, wave_imp[p3]/1e-9/nn, linestyle='--', color='k', label='continuum')
plt.ylim(0, 1*1.05)
plt.xlabel('Wavelength [nm]')
plt.ylabel(r'$I/I_c$')
plt.legend()
plt.xticks(ticks, labels)

analytical_tau, analytical_intensity = np.loadtxt(f'{pm.basedir}peaks_tau_analytical.txt', unpack=True)
ir_ib = (intensity_red-continuum)/(intensity_blue-continuum)
interp_consistent = np.interp(analytical_tau, tau_max, ir_ib)

ax = plt.subplot(1,2,2)
ln1 = ax.plot(tau_max, (intensity_red-continuum)/(intensity_blue-continuum), color=cm.plasma(0/(len(taus)-1)), label='self-consistent NLTE')
ln2 = ax.plot(analytical_tau, analytical_intensity, color=cm.plasma(4/(len(taus))), label='analytical')
# ratio betwen peaks (consisten-analytical)/consistent
ax2 = ax.twinx()
ln3 = ax2.plot(analytical_tau, (interp_consistent-analytical_intensity)/ interp_consistent, color='k', linestyle='--', label='ratio')

ax.set_ylim (0, 8)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel('$I_r/I_b$')
ax2.set_ylabel(r'$\frac{I_r^c-I_r^a}{I_r^c}$')

lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='right')
plt.tight_layout()
plt.savefig('I_peaks_tau.pdf')
plt.show()
