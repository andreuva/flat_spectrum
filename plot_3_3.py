import matplotlib.pyplot as plt
import numpy as np
import parameters_JKQ as params

JK00_fil = np.loadtxt('output_compar_B_10.0_90.0_0.0_20220512-122617/out/real_JK00_finished.out')
JK20_fil = np.loadtxt('output_compar_B_10.0_90.0_0.0_20220512-122617/out/real_JK20_finished.out')

heights = np.linspace(params.z0, params.zf, params.zn)
wave = JK00_fil[0,:]
JK00_fil = JK00_fil[1:,:]
JK20_fil = JK20_fil[1:,:]

plt.figure(figsize=(20,12), dpi=300)
plt.subplot(1,3,1)
for i,jkq in enumerate(JK00_fil/JK00_fil[0,0]):
    if i//2 == 0:
        continue
    plt.plot(wave, jkq, color=f'C{i}', label=f'z={heights[i]/1e8:1.4f} Mm')
for i,jkq in enumerate(JK20_fil/JK00_fil[0,0]):
    plt.plot(wave, jkq, '--', color=f'C{i}')
plt.legend()
plt.ylabel(r'$J^K_Q$')
plt.xlabel('Wavelength [nm]')


# PLOT OF THE STOKES PARAMETERS
stokes_fil = np.loadtxt('output_compar_B_10.0_90.0_0.0_20220512-122617/out/stokes_00.out', skiprows=3)
stokes_prom = np.loadtxt('output_compar_B_10.0_45.0_0.0_20220621-125328/out/stokes_00.out', skiprows=3)

stokes_fil = stokes_fil[:,1:].transpose()
stokes_prom = stokes_prom[:,1:].transpose()

plt.subplot(1,3,2)
plt.plot(wave, stokes_fil[0]/stokes_fil[0,0] , color=f'C{i}', label=fr'$I$ filament')
plt.plot(wave, stokes_prom[0]/stokes_fil[0,0] , '--', color=f'C{i}', label=fr'$I$ prominence')
plt.legend()
plt.ylabel(r'$I$')
plt.xlabel('Wavelength [nm]')

plt.subplot(1,3,3)
polar = ['Q', 'U', 'V']
for i,stokes in enumerate(stokes_fil[1:]/stokes_fil[0,0]):
    plt.plot(wave, stokes*100, color=f'C{i}', label=fr'${polar[i-1]}$')
for i,stokes in enumerate(stokes_prom[1:]/stokes_fil[0,0]):
    plt.plot(wave, stokes*100, '--', color=f'C{i}')
plt.legend()
plt.ylabel(r'$Q,U,V$ normaliced to $I$ %')
plt.xlabel('Wavelength [nm]')

plt.savefig('3_3.png')
plt.show()
