import matplotlib.pyplot as plt
import numpy as np
import parameters_velocity as params
import matplotlib.cm as cm


# PLOT OF THE STOKES PARAMETERS
stokes_fil = np.loadtxt(f'{params.dir}out/stokes_00.out', skiprows=3)
nus = stokes_fil[:,0]
nn = 1.000293
wave = 299792458/nus/nn/1e-9
stokes_fil = stokes_fil[:,1:].transpose()

# compute the index of each of the peaks
nu_1 = 2.76733e14
nu_2 = 2.76764e14
nu_peak_1_indx = np.abs(nus - nu_1).argmin()
nu_peak_2_indx = np.abs(nus - nu_2).argmin()
ticks = [wave[nu_peak_1_indx], wave[nu_peak_2_indx]]
labels = [f'{wave[nu_peak_1_indx]:.2f}', f'{wave[nu_peak_2_indx]:.2f}']

len = len(nus)
p1 = int(len/8)
p3 = int(p1*7)


plt.figure(figsize=(3.5,10), dpi=120)
plt.subplot(3,1,1)
plt.plot(wave[p1:p3], stokes_fil[0][p1:p3], color=cm.plasma(0/10.0), label=fr'$I$')
plt.xlabel('Wavelength [nm]')
plt.xticks(ticks, labels)
plt.ylabel(r'$I$')

plt.subplot(3,1,2)
plt.plot(wave[p1:p3], stokes_fil[1][p1:p3]*100/stokes_fil[0].max(), color=cm.plasma(0/10.0), label=fr'$Q$')
plt.xlabel('Wavelength [nm]')
plt.xticks(ticks, labels)
plt.ylabel(r'$Q/I_{cont}$ %')

plt.subplot(3,1,3)
plt.plot(wave[p1:p3], stokes_fil[2][p1:p3]*100/stokes_fil[0].max(), color=cm.plasma(0/10.0), label=fr'$U$')
plt.xlabel('Wavelength [nm]')
plt.xticks(ticks, labels)
plt.ylabel(r'$U/I_{cont}$ %')

plt.tight_layout()
plt.savefig('3_3.png')
plt.show()
