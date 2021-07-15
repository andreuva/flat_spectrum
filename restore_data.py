import pickle
import numpy as np
from glob import glob
from atom import ESE

ese = ESE(None, None, None, None, None, None)

directory = '20210715-084452_plots_ndens_1.0_dz_10.0/out/'

jqq_rays = []
for filename in sorted(glob(directory + 'sim*')):
    print('open file:', filename)
    file = open(filename, "rb")
    jqq_rays.append(pickle.load(file))
    file.close()

file = open(directory + "jqq_1.pkl", "rb")
jqq = pickle.load(file)
file.close()

file = open(directory + "J_KQ_1.pkl", "rb")
J_KQ = pickle.load(file)
file.close()

file = open(directory + "rho_KQ_1.pkl", "rb")
rho_KQ = pickle.load(file)
file.close()

rho_qq = np.loadtxt(directory + "rho_qq_1.csv")

for i, jqq in enumerate(jqq_rays):
    if sorted(glob(directory + 'sim*'))[i][-14:-9] == '0_16_':
        print('----------------------------------------------')
        print(sorted(glob(directory + 'sim*'))[i], '\n')
        print(f'{jqq[-1][-1].real.value.mean():1.2e}\t {jqq[0][0].real.value.mean():1.2e}\t {jqq[1][1].real.value.mean():1.2e}')
        print(f'{jqq[0][-1].real.value.mean():1.2e}\t {jqq[-1][0].real.value.mean():1.2e}')
        print(f'{jqq[0][1].real.value.mean():1.2e}\t  {jqq[1][0].real.value.mean():1.2e}')
        print(f'{jqq[-1][1].real.value.mean():1.2e}\t {jqq[1][-1].real.value.mean():1.2e}')
        print('')
        Jp_KQ = np.zeros((3, 5, len(jqq[-1][1]))) + 0j
        # J_iKQ = {}
        for K in [0, 1, 2]:
            for Q in range(-K, K+1):
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        Jp_KQ[K][Q+K] += ((-1)**(1-q) * np.sqrt(3*(2*K + 1)) * ese.jsim.j3(1, 1, K, q, -qp, -Q) *
                                          jqq[q][qp].value)

                # J_iKQ[i] = Jp_KQ
                print(f'J^{K}_{Q} = {Jp_KQ[K][Q+K][int(len(Jp_KQ[K][Q+K])/2)]:1.2e}')

print('------------------------------------')
for K in [0, 1, 2]:
    for Q in range(-K, K+1):
        print(f'J^{K}_{Q} = {J_KQ[0][K][Q+K][int(len(J_KQ[0][K][Q+K])/2)]:1.2e}')
