import numpy as np
from atom import ESE
from rad import RTE
import parameters as pm
from conditions import conditions
from physical_functions import Tqq
from astropy import units as u

cdts = conditions(pm)
anis_frac = 0.5

atom = ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, 0, cdts.temp, equilibrium=True)
rad = RTE(cdts.nus, cdts.v_dop)
rad.make_IC(cdts.nus, cdts.temp)

for j, ray in enumerate(cdts.rays):
    rad.sumStokes(ray)

rad.resetRadiation()

for q in [-1, 0, 1]:
    for qp in [-1, 0, 1]:
        rad.jqq[q][qp] += rad.stokes[0]*((-1)**(1+q) * np.sqrt(1/3) * atom.jsim.j3(1, 1, 0, q, -qp, 0) +
                                         # (-1)**(1+q) *                atom.jsim.j3(1, 1, 1, q, -qp, 0)*anis_frac)*u.sr
                                         (-1)**(1+q) * np.sqrt(5/3) * atom.jsim.j3(1, 1, 2, q, -qp, 0)*anis_frac)*u.sr

mrc = atom.solveESE(rad, cdts)

solve = np.real(atom.ESE)

print('')
for i in range(len(solve)):
    print(f'Row {i}\t', end='')
    for j in range(len(solve)):
        if solve[i][j] >= 0:
            print(' ', end='')
        print(f'{solve[i][j].value:.2E} ', end='')
    print(f'')

print('\nPopulations after the ESE solution:\n', np.array_str(atom.rho, precision=3))

rho_K0 = np.zeros(3)
for K in [0, 1, 2]:
    for M in [-1, 0, 1]:
        for Mp in [-1, 0, 1]:
            rho_K0[K] += ((-1)**(1-M) * np.sqrt(2*K + 1) * atom.jsim.j3(1, 1, K, M, -Mp, 0)*atom.rho_call(1, 1, M, Mp))

    rho_K0[K] += ((-1)**(-M) * np.sqrt(2*K + 1) * atom.jsim.j3(0, 0, K, 0, 0, 0)*atom.rho_call(0, 0, 0, 0))

print("Finished")
