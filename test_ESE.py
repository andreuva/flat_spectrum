import numpy as np
from atom import ESE
from rad import RTE
import parameters as pm
from conditions import conditions
from physical_functions import Tqq

from astropy import constants as c


cdts = conditions(pm)

atom = ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, 0, cdts.temp)
rad = RTE(cdts.nus, cdts.v_dop)
rad.make_IC(cdts.nus)

for j, ray in enumerate(cdts.rays):
    rad.sumStokes(ray)

mrc = atom.solveESE(rad, cdts)

solve = np.real(atom.ESE)

# print('')
# for i in range(len(solve)):
#     print(f'Row {i}\t', end='')
#     for j in range(len(solve)):
#         if solve[i][j] >= 0:
#             print(' ', end='')
#         print(f'{solve[i][j]:.2E} ', end='')
#     print(f'')

print("Finished")
