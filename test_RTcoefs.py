import numpy as np
from atom import ESE
from RTcoefs import RTcoefs
import parameters as pm
from conditions import conditions
from astropy.modeling.models import BlackBody as bb
import matplotlib.pyplot as plt

cdts = conditions(pm)
RT_coeficients = RTcoefs(cdts.nus)
# st = state(cdt)
atom = ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, 0, cdts.temp)

ray = cdts.rays[4]

em, abs, Source_func, KK = RT_coeficients.getRTcoefs(atom, ray, cdts)
Bw = bb(temperature=cdts.temp)(cdts.nus)

print('')
for i in range(len(KK)):
    print(f'Row {i}\t', end='')
    for j in range(len(KK)):
        if KK[i][j].mean() >= 0:
            print(' ', end='')
        print(f'{KK[i][j].mean().value:.2E} ', end='')
    print(f'')
