import parameters as pm
from physical_functions import Tqq, jsymbols

import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from astropy import constants as cts
from astropy import units as unt
# from py3nj import wigner3j


class RTcoefs:
    """ Radiative transfer coefficients.
        Just one instance of the class needs to be created in the program.
    """

    def __init__(self, nus):
        """
            nus: array of the line frequencies
            return value: None
        """
        self.jsim = jsymbols()

    def getRTcoefs(self, ese, ray, cdts):
        """ Provides the 4-vector of epsilon and the 4x4 K-matrix for the point
            with given ESE state and ray direction.
            ese: the local instance of the ESE class
            ray: object with .theta and .chi variables defining the ray
            of propagation direction
            return value: [S (source function vector in frequencies), K (4x4 list of
            vectors in frequencies)]
        """

        # Eq 7.10 of LL04
        eta_a = [0, 0, 0, 0]
        eta_s = [0, 0, 0, 0]
        rho_a = [0, 0, 0, 0]
        rho_s = [0, 0, 0, 0]

        for i in range(4):

            eta_as_rho_as = []

            pool = Pool(processes=8)

            func = partial(parallel_RTcoefs, self.jsim, ese, ray, cdts, i)
            eta_as_rho_as = pool.map(func, range(len(ese.atom.dens_elmnt)))
            pool.close()
            pool.join()

            sum_etaa = eta_as_rho_as[:][0]
            sum_etas = eta_as_rho_as[:][1]
            sum_rhoa = eta_as_rho_as[:][2]
            sum_rhos = eta_as_rho_as[:][3]

            eta_a[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens * sum(sum_etaa)
            eta_s[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens * sum(sum_etas)
            rho_a[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens * sum(sum_rhoa)
            rho_s[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens * sum(sum_rhos)

        eta = [et_a - et_s for et_a, et_s in zip(eta_a, eta_s)]
        rho = [ro_a - ro_s for ro_a, ro_s in zip(rho_a, rho_s)]
        KK = np.array([[eta[0],  eta[1],  eta[2],  eta[3]],
                       [eta[1],  eta[0],  rho[3], -rho[2]],
                       [eta[2], -rho[3],  eta[0],  rho[1]],
                       [eta[3],  rho[2], -rho[1],  eta[0]]])

        eps = [2*cts.h.cgs*cdts.nus**3/(cts.c.cgs**2)*et_s for et_s in eta_s]
        SS = np.array([ep.value/(eta[0].value+1e-30) for ep in eps]) * pm.I_units

        # Just for debuging purposes overwrite KK and SS discarding the previous
        KK = np.ones((4, 4, pm.wn))*1e-10
        SS = np.ones((4, pm.wn))*pm.I_units*1e-10

        return SS, KK


def parallel_RTcoefs(jsim, ese, ray, cdts, i, low):

    sum_etaa = 0
    sum_etas = 0
    sum_rhoa = 0
    sum_rhos = 0

    l_lev = ese.atom.dens_elmnt[low]
    Ll = l_lev[0]
    Ml = l_lev[-2]
    Mlp = l_lev[-1]
    Jl = l_lev[-3]

    for line in ese.atom.lines:

        if Ll not in line.levels:
            continue

        for up, u_lev in enumerate(ese.atom.dens_elmnt):

            Lu = u_lev[0]
            Mu = u_lev[-2]
            Mup = u_lev[-1]
            Ju = u_lev[-3]

            if Lu not in line.levels:
                continue

            for q in [-1, 0, 1]:
                for qp in [-1, 0, 1]:

                    sum_etaa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                 jsim.j6(Ju, Jl, 1, -Mu, Ml, -q) *
                                 jsim.j6(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                 np.real(Tqq(q, qp, i, ray.inc.value, ray.az.value)*ese.rho[low] *
                                 cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                    sum_rhoa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                 jsim.j6(Ju, Jl, 1, -Mu, Ml, -q) *
                                 jsim.j6(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                 np.imag(Tqq(q, qp, i, ray.inc.value, ray.az.value)*ese.rho[low] *
                                 cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                    sum_etas += (3*(2*Ju + 1)*line.B_ul *
                                 jsim.j6(Ju, Jl, 1, -Mu, Ml, -q) *
                                 jsim.j6(Ju, Jl, 1, -Mup, Ml, -qp) *
                                 np.real(Tqq(q, qp, i, ray.inc.value, ray.az.value)*ese.rho[up] *
                                 cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                    sum_rhos += (3*(2*Ju + 1)*line.B_ul *
                                 jsim.j6(Ju, Jl, 1, -Mu, Ml, -q) *
                                 jsim.j6(Ju, Jl, 1, -Mup, Ml, -qp) *
                                 np.imag(Tqq(q, qp, i, ray.inc.value, ray.az.value)*ese.rho[up] *
                                 cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

    return [sum_etaa, sum_etas, sum_rhoa, sum_rhos]
