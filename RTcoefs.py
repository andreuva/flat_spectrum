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
        eta_a = np.zeros((4, cdts.nus_N)) * 1/(unt.cm)
        eta_s = np.zeros((4, cdts.nus_N)) * 1/(unt.cm)
        rho_a = np.zeros((4, cdts.nus_N)) * 1/(unt.cm)
        rho_s = np.zeros((4, cdts.nus_N)) * 1/(unt.cm)

        # eta_a_LTE = 0
        # eta_s_LTE = 0
        # emision = 0
        # for q in [-1, 0, 1]:
        #     eta_a_LTE += cts.h.cgs*cdts.nus.cgs/(4*np.pi)*cdts.n_dens*ese.atom.lines[0].B_lu *\
        #         3*self.jsim.j3(1, 0, 1, q, 0, -q)**2 *\
        #         np.real(ese.rho[0]*cdts.voigt_profile(ese.atom.lines[0], -q, 0, cdts.B.value) *
        #                 Tqq(q, q, 0, ray.inc.to('rad').value, ray.az.to('rad').value))

        #     for qp in [-1, 0, 1]:
        #         eta_s_LTE += cts.h.cgs*cdts.nus.cgs/(4*np.pi)*cdts.n_dens*ese.atom.lines[0].B_ul *\
        #             9*self.jsim.j3(1, 0, 1, q, 0, -q)*self.jsim.j3(1, 0, 1, qp, 0, -qp) *\
        #             np.real(ese.rho_call(1, 1, -qp, -q) * cdts.voigt_profile(ese.atom.lines[0], -q, 0, cdts.B.value) *
        #                     Tqq(q, qp, 0, ray.inc.to('rad').value, ray.az.to('rad').value))
        #         emision += cts.h.cgs*cdts.nus.cgs/(4*np.pi)*cdts.n_dens*ese.atom.lines[0].A_lu *\
        #             9*self.jsim.j3(1, 0, 1, q, 0, -q)*self.jsim.j3(1, 0, 1, qp, 0, -qp) *\
        #             np.real(ese.rho_call(1, 1, -qp, -q) * cdts.voigt_profile(ese.atom.lines[0], -q, 0, cdts.B.value) *
        #                     Tqq(q, qp, 0, ray.inc.to('rad').value, ray.az.to('rad').value))

        for i in range(4):

            sum_etaa = 0
            sum_etas = 0
            sum_rhoa = 0
            sum_rhos = 0

            for line in ese.atom.lines:

                Ll = line.levels[0]
                Jl = line.jlju[0]
                Lu = line.levels[1]
                Ju = line.jlju[1]

                for _, _, Ml, Mlp in ese.atom.levels[Ll].MMp:

                    for Mu in ese.atom.levels[Lu].M:
                        q = int(Ml - Mu)
                        qp = int(Mlp - Mu)

                        sum_etaa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                     np.real(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho_call(Ll, Jl, Ml, Mlp) *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                        sum_rhoa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                     np.imag(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho_call(Ll, Jl, Ml, Mlp) *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                for _, _, Mu, Mup in ese.atom.levels[Lu].MMp:

                    for Ml in ese.atom.levels[Ll].M:
                        q = int(Ml - Mu)
                        qp = int(Ml - Mup)

                        sum_etas += (3*(2*Ju + 1)*line.B_ul *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mup, Ml, -qp) *
                                     np.real(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho_call(Lu, Ju, Mup, Mu) *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                        sum_rhos += (3*(2*Ju + 1)*line.B_ul *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mup, Ml, -qp) *
                                     np.imag(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho_call(Lu, Ju, Mup, Mu) *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

            eta_a[i, :] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_etaa
            eta_s[i, :] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_etas
            rho_a[i, :] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_rhoa
            rho_s[i, :] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_rhos

        eta = eta_a - eta_s
        rho = rho_a - rho_s

        if np.any(eta[0] < 0):
            print("Warning: eta_I < 0")

        KK = np.array([[eta[0],  eta[1],  eta[2],  eta[3]],
                       [eta[1],  eta[0],  rho[3], -rho[2]],
                       [eta[2], -rho[3],  eta[0],  rho[1]],
                       [eta[3],  rho[2], -rho[1],  eta[0]]])*eta.unit

        eps = 2*cts.h.cgs*cdts.nus.cgs**3/(cts.c.cgs**2)*eta_s
        SS = eps/(eta[0]+1e-30*eta[0].unit) / unt.s / unt.Hz / unt.sr

        EM = eps[0].value
        ABS = eta[0].value

        # plt.plot(eps[0])
        # plt.plot(emision)
        # plt.show()
        # plt.plot(eta_a[0])
        # plt.plot(eta_a_LTE)
        # plt.show()
        # plt.plot(eta_s[0])
        # plt.plot(eta_s_LTE)
        # plt.show()

        return EM, ABS, SS, KK
