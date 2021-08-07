import pickle
import numpy as np
from glob import glob
import parameters as pm
import matplotlib.pyplot as plt
from physical_functions import jsymbols
from plot_utils import save_or_show

# Definition of the files to load
directory = '20210807-210338_plots_ndens_1.0_dz_10.0/out/'
prefix = 'jqq_base*'
files = sorted(glob(directory + prefix))
looked_for = '_0_16_'

colors = ['coral', 'cornflowerblue', 'violet', 'seagreen', 'gray', 'gold']

jqq_prefix = "jqq_[0-9].pkl"
jqq_filenames = sorted(glob(directory + jqq_prefix))
rho_qq_prefix = "rho_qq_after_*"
rho_qq_filenames = sorted(glob(directory + rho_qq_prefix))

# Loading the data
jqq_base_rays = []
for filename in files:
    print('opening and reading file:', filename)
    file = open(filename, "rb")
    jqq_base_rays.append(pickle.load(file))
    file.close()

jqq_itt_z = []
for path in jqq_filenames:
    file = open(path, "rb")
    jqq_itt_z.append(pickle.load(file))
    file.close()

rho_itt_z_qq = []
for path in rho_qq_filenames:
    rho_itt_z_qq.append(np.loadtxt(path, dtype=np.complex_))

for i in range(len(jqq_base_rays)):
    if looked_for in files[i]:
        print('----------------------------------------------')
        print(sorted(glob(directory + prefix))[i], '\n\tMean of jqq,s:\n')
        for q in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                print(f'J{q}{qp} = {jqq_base_rays[i][q][qp].real.value.mean():1.2e}', end='\t')
            print('')

#  Compute the JKQ and rhoKQ

jsymb = jsymbols()
nus_weights = np.ones_like(jqq_itt_z[0][0][0][0].value)
nus_weights[0] = 0.5
nus_weights[-1] = 0.5

J_KQ_bar_itt = np.zeros((len(jqq_itt_z), len(jqq_itt_z[0]), 3, 5)) + 0j
for it, jqq_z in enumerate(jqq_itt_z):
    for i in range(len(jqq_itt_z[0])):
        for K in [0, 1, 2]:
            for Q in range(-K, K+1):
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        J_KQ_nu = ((-1)**(1+q) * np.sqrt(3*(2*K + 1)) * jsymb.j3(1, 1, K, q, -qp, -Q) *
                                   jqq_z[i][q][qp].value)
                        J_KQ_bar_itt[it, i, K, Q+K] += np.sum(J_KQ_nu*nus_weights)

rho_up_KQ_itt = np.zeros((len(rho_itt_z_qq), len(rho_itt_z_qq[0]), 3, 5)) + 0j
rho_low_KQ_itt = np.zeros((len(rho_itt_z_qq), len(rho_itt_z_qq[0]))) + 0j
for it, rho_qq in enumerate(rho_itt_z_qq):
    for i in range(len(rho_qq)):
        for K in [0, 1, 2]:
            for Q in range(-K, K+1):
                for M in [-1, 0, 1]:
                    for Mp in [-1, 0, 1]:
                        index_up = 1 + (M+1)*3 + (Mp+1)
                        rho_up_KQ_itt[it, i, K, Q+K] += ((-1)**(1-M) * np.sqrt(2*K + 1) * jsymb.j3(1, 1, K, M, -Mp, Q) *
                                                         rho_qq[i][index_up])
        rho_low_KQ_itt[it, i] += (rho_qq[i][0])

print('------------------------------------')
print(' MAKING PLOTS ')
print('------------------------------------')

# Plots for JKQ
for K in [0, 1, 2]:
    for Q in range(-K, K+1):
        plt.figure(figsize=(20, 20), dpi=300)
        for i, J_KQ_bar in enumerate(J_KQ_bar_itt):
            normalization = 1
            if K != 0 or Q != 0:
                normalization = J_KQ_bar[:, 0, 0]

            plt.plot(J_KQ_bar[:, K, Q+K].real/normalization.real,
                     color=colors[i],
                     label=f'$Real(J^{K}_{Q})$'+f' itt = {i}')
        for i, J_KQ_bar in enumerate(J_KQ_bar_itt):
            plt.plot(J_KQ_bar[:, K, Q+K].imag,
                     '-.', color=colors[i],
                     label=f'$Imag(J^{K}_{Q})$'+f' itt = {i}')
        plt.legend()
        plt.xlabel('z')
        if type(normalization) != int:
            plt.ylabel(f'$J_{Q}^{K} / J_0^0$')
            plt.title(f'$J_{Q}^{K}$ profile normaliced')
            save_or_show('save', f'J{K}{Q}_norm', directory + 'plots/')
        else:
            plt.ylabel(f'$J^{K}_{Q}$')
            plt.title(f'$J^{K}_{Q}$ profile')
            save_or_show('save', f'J{K}{Q}', directory + 'plots/')

# Plots for rhoKQ
for K in [0, 1, 2]:
    for Q in range(-K, K+1):
        plt.figure(figsize=(20, 20), dpi=300)
        for i, rho_up_KQ in enumerate(rho_up_KQ_itt):
            normalization = 1
            if K != 0 or Q != 0:
                normalization = rho_up_KQ[:, 0, 0]
            else:
                plt.plot(rho_low_KQ_itt[i, :].real,
                         '--', linewidth=2, color=colors[i],
                         label=f'$Real(rho^{K}_{Q})$ LOW'+f' itt = {i}')
                plt.plot(rho_low_KQ_itt[i, :].imag,
                         ':', linewidth=2, color=colors[i],
                         label=f'$Im(rho^{K}_{Q})$ LOW'+f' itt = {i}')

            plt.plot(rho_up_KQ[:, K, Q+K].real/normalization.real,
                     color=colors[i],
                     label=f'$Real(rho^{K}_{Q})$'+f' itt = {i}')
        for i, rho_up_KQ in enumerate(rho_up_KQ_itt):
            plt.plot(rho_up_KQ[:, K, Q+K].imag,
                     '-.', color=colors[i],
                     label=f'$Imag(rho^{K}_{Q})$'+f' itt = {i}')

        plt.legend()
        plt.xlabel('z')
        if type(normalization) != int:
            plt.ylabel(f'$rho_{Q}^{K} / rho_0^0$')
            plt.title(f'$rho_{Q}^{K}$ profile normaliced')
            save_or_show('save', f'rho{K}{Q}_norm', directory + 'plots/')
        else:
            plt.ylabel(f'$rho^{K}_{Q}$')
            plt.title(f'$rho^{K}_{Q}$ profile')
            save_or_show('save', f'rho{K}{Q}', directory + 'plots/')

print('------------------------------------')
print(' FINISHED ')
print('------------------------------------')
