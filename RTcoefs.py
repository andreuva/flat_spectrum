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
        eta_a = [0, 0, 0, 0]
        eta_s = [0, 0, 0, 0]
        rho_a = [0, 0, 0, 0]
        rho_s = [0, 0, 0, 0]

        for i in range(4):

            sum_etaa = 0
            sum_etas = 0
            sum_rhoa = 0
            sum_rhos = 0

            for low, l_lev in enumerate(ese.atom.dens_elmnt):

                Ll = l_lev[0]
                Ml = l_lev[-2]
                Mlp = l_lev[-1]
                Jl = l_lev[-3]

                for line in ese.atom.lines:

                    if not Ll == line.levels[0]:
                        continue

                    for up, u_lev in enumerate(ese.atom.dens_elmnt):

                        Lu = u_lev[0]
                        Mu = u_lev[-2]
                        Mup = u_lev[-1]
                        Ju = u_lev[-3]

                        if not Lu == line.levels[1]:
                            continue

                        q = int(Ml - Mu)
                        qp = int(Mlp - Mu)

                        sum_etaa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                     np.real(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho[low] *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                        sum_rhoa += (3*(-1)**(Ml - Mlp)*(2*Jl + 1)*line.B_lu *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Mlp, -qp) *
                                     np.imag(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho[low] *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                        # plt.plot(cdts.nus, cdts.voigt_profile(line, Mu, Ml, cdts.B.value))
                        # plt.show()
                        qp = int(Ml - Mup)

                        sum_etas += (3*(2*Ju + 1)*line.B_ul *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mup, Ml, -qp) *
                                     np.real(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho[up] *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

                        sum_rhos += (3*(2*Ju + 1)*line.B_ul *
                                     self.jsim.j3(Ju, Jl, 1, -Mu, Ml, -q) *
                                     self.jsim.j3(Ju, Jl, 1, -Mup, Ml, -qp) *
                                     np.imag(Tqq(q, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*ese.rho[up] *
                                     cdts.voigt_profile(line, Mu, Ml, cdts.B.value)))

            eta_a[i] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_etaa
            eta_s[i] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_etas
            rho_a[i] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_rhoa
            rho_s[i] = cts.h.cgs*cdts.nus.cgs/(4*np.pi) * cdts.n_dens * sum_rhos

        eta = [et_a - et_s for et_a, et_s in zip(eta_a, eta_s)]

        if np.any(eta[0] < 0):
            print("Warning: eta_I < 0")

        rho = [ro_a - ro_s for ro_a, ro_s in zip(rho_a, rho_s)]
        KK = np.array([[eta[0],  eta[1],  eta[2],  eta[3]],
                       [eta[1],  eta[0],  rho[3], -rho[2]],
                       [eta[2], -rho[3],  eta[0],  rho[1]],
                       [eta[3],  rho[2], -rho[1],  eta[0]]])

        eps = [2*cts.h.cgs*cdts.nus.cgs**3/(cts.c.cgs**2)*et_s for et_s in eta_s]
        SS = np.array([ep.value/(eta[0].value+1e-30) for ep in eps]) * pm.I_units

        EM = eps[0][79].value
        ABS = eta[0][79].value

        # Just for debuging purposes overwrite KK and SS discarding the previous
        # KK = np.ones((4, 4, pm.wn))*1e-10
        # SS = np.ones((4, pm.wn))*pm.I_units*1e-10

        return EM, ABS, SS, KK
