import parameters as pm
from physical_functions import Tqq, jsymbols

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as cts
from astropy import units as unt
# from py3nj import wigner3j


class RTcoefs:
    """ Radiative transfer coefficients.
        Just one instance of the class needs to be created in the program.
    """
    jsim = jsymbols()

    def __init__(self, nus):
        """
            nus: array of the line frequencies
            return value: None
        """
        pass

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
            sum_mluq = 0
            sum_mulq = 0
            for q in [-1, 0, 1]:
                for q_p in [-1, 0, 1]:
                    for M_l in cdts.atomic_model.transitions[0].lower.M:
                        for M_u in cdts.atomic_model.transitions[0].upper.M:
                            sum_mluq = sum_mluq + 3*(self.jsim.j6(cdts.atomic_model.transitions[0].upper.J,
                                                                  cdts.atomic_model.transitions[0].lower.J, 1, -M_u, M_l, -q) *
                                                     self.jsim.j6(cdts.atomic_model.transitions[0].upper.J,
                                                                  cdts.atomic_model.transitions[0].lower.J, 1, -M_u, M_l, -q_p))
                            sum_mluq = sum_mluq + (Tqq(q, q_p, i, ray.inc.value, ray.az.value) * ese.rho_l *
                                                   cdts.voigt_profile(cdts.a_voigt, cdts.atomic_model.transitions[0], M_u, M_l, cdts.B))

                            sum_mulq = sum_mulq + 3*(self.jsim.j6(cdts.atomic_model.transitions[0].upper.J,
                                                                  cdts.atomic_model.transitions[0].lower.J, 1, -M_u, M_l, -q) *
                                                     self.jsim.j6(cdts.atomic_model.transitions[0].upper.J,
                                                                  cdts.atomic_model.transitions[0].lower.J, 1, -M_u, M_l, -q_p))
                            sum_mulq = sum_mulq + (Tqq(q, q_p, i, ray.inc.value, ray.az.value) * ese.rho_u *
                                                   cdts.voigt_profile(cdts.a_voigt, cdts.atomic_model.transitions[0],  M_u, M_l, cdts.B))

            eta_a[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                (2*cdts.atomic_model.transitions[0].lower.J + 1) * cdts.atomic_model.transitions[0].B_lu * np.real(sum_mluq)

            eta_s[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                (2*cdts.atomic_model.transitions[0].upper.J + 1) * cdts.atomic_model.transitions[0].B_ul * np.real(sum_mulq)

            rho_a[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                (2*cdts.atomic_model.transitions[0].lower.J + 1) * cdts.atomic_model.transitions[0].B_lu * np.imag(sum_mluq)

            rho_s[i] = cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                (2*cdts.atomic_model.transitions[0].upper.J + 1) * cdts.atomic_model.transitions[0].B_ul * np.imag(sum_mulq)

        eta = [et_a - et_s for et_a, et_s in zip(eta_a, eta_s)]
        rho = [ro_a - ro_s for ro_a, ro_s in zip(rho_a, rho_s)]
        eps = [2*cts.h.cgs*cdts.nus**3/(cts.c.cgs**2)*et_s for et_s in eta_s]
        SS = [ep/eta[0] for ep in eps]
        KK = np.array([[eta[0],  eta[1],  eta[2],  eta[3]],
                       [eta[1],  eta[0],  rho[3], -rho[2]],
                       [eta[2], -rho[3],  eta[0],  rho[1]],
                       [eta[3],  rho[2], -rho[1],  eta[0]]])

        KK = np.zeros((4, 4, pm.wn))
        SS = np.zeros((4, pm.wn))*pm.I_units

        return SS, KK
