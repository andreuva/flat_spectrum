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
            for trans in ese.atom.transitions:
                sum_MlMlpMuqq = 0
                sum_MuMupMlqq = 0
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        for Ml in trans.lower.M:
                            for Mu in trans.upper.M:
                                for Mlp in trans.lower.M:

                                    sum_MlMlpMuqq += 3*(-1)**(Ml-Mlp) *\
                                                        (self.jsim.j6(trans.upper.J,
                                                                      trans.lower.J, 1, -Mu, Ml, -q) *
                                                         self.jsim.j6(trans.upper.J,
                                                                      trans.lower.J, 1, -Mu, Mlp, -qp))
                                    sum_MlMlpMuqq += (Tqq(q, qp, i, ray.inc.value, ray.az.value) *
                                                      trans.lower.rho[Ml+trans.lower.J, Mlp+trans.lower.J] *
                                                      cdts.voigt_profile(cdts.a_voigt, trans, Mu, Ml, cdts.B))
                                for Mup in trans.upper.M:

                                    sum_MuMupMlqq += 3*(self.jsim.j6(trans.upper.J,
                                                                     trans.lower.J, 1, -Mu, Ml, -q) *
                                                        self.jsim.j6(trans.upper.J,
                                                                     trans.lower.J, 1, -Mup, Ml, -qp))
                                    sum_MuMupMlqq += (Tqq(q, qp, i, ray.inc.value, ray.az.value) *
                                                      trans.upper.rho[Mup+trans.upper.J, Mu+trans.upper.J] *
                                                      cdts.voigt_profile(cdts.a_voigt, trans,  Mu, Ml, cdts.B))

                eta_a[i] += cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                    (2*trans.lower.J + 1) * trans.B_lu * np.real(sum_MlMlpMuqq)

                eta_s[i] += cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                    (2*trans.upper.J + 1) * trans.B_ul * np.real(sum_MuMupMlqq)

                rho_a[i] += cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                    (2*trans.lower.J + 1) * trans.B_lu * np.imag(sum_MlMlpMuqq)

                rho_s[i] += cts.h.cgs*cdts.nus/(4*np.pi) * cdts.n_dens *\
                    (2*trans.upper.J + 1) * trans.B_ul * np.imag(sum_MuMupMlqq)

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
