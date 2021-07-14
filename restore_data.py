import pickle
import numpy as np
from glob import glob

directory = '20210714-122758_plots_ndens_1.0_dz_10.0/'

jqq_rays = []
for filename in sorted(glob(directory + 'sim*')):
    print('open file:', filename)
    file = open(filename, "rb")
    jqq_rays.append(pickle.load(file))
    file.close()

file = open(directory + "out/jqq_1.pkl", "rb")
jqq = pickle.load(file)
file.close()

file = open(directory + "out/J_KQ_1.pkl", "rb")
J_KQ = pickle.load(file)
file.close()

file = open(directory + "out/rho_KQ_1.pkl", "rb")
rho_KQ = pickle.load(file)
file.close()

rho_qq = np.loadtxt(directory + "out/rho_qq_1.csv")

for i, jqq in enumerate(jqq_rays):
    print(sorted(glob('20210714-122758_plots_ndens_1.0_dz_10.0/sim*'))[i], '\n')
    print(f'{jqq[-1][-1].real.value.mean():1.2e}\t {jqq[0][0].real.value.mean():1.2e}\t {jqq[1][1].real.value.mean():1.2e}')
    print(f'{jqq[0][-1].real.value.mean():1.2e}\t {jqq[-1][0].real.value.mean():1.2e}')
    print(f'{jqq[0][1].real.value.mean():1.2e}\t  {jqq[1][0].real.value.mean():1.2e}')
    print(f'{jqq[-1][1].real.value.mean():1.2e}\t {jqq[1][-1].real.value.mean():1.2e}')
    print('----------------------------------------------')

for K in [0, 1, 2]:
    for Q in range(-K, K+1):
        print(f'J^{K}_{Q} = {J_KQ[0][K][Q][int(len(J_KQ[0][K][Q])/2)]:1.2e}')
