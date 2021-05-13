import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.modeling.models import BlackBody as bb
from physical_functions import Tqq, jsymbols


class level():
    """Class that defines the energy level of the atomic model"""
    def __init__(self, energy, JJ, g):
        self.E = energy
        self.g = g
        self.J = JJ
        self.M = [ml for ml in range(-self.J, self.J+1)]
        self.num_M = 2*self.J + 1
        self.rho = np.ones((self.num_M, self.num_M))


class transition():
    def __init__(self, level_u, level_d, Alu):
        self.upper = level_u
        self.lower = level_d

        self.wavelength = 1/(level_u.E - level_d.E)
        self.energy = c.h.cgs * c.c.cgs / self.wavelength
        self.nu = self.energy/c.h.cgs

        self.A_lu = Alu
        self.B_lu = Alu * c.c.cgs**3/(2*c.h.cgs*self.nu**3)
        self.B_ul = self.B_lu * self.upper.g/self.lower.g

        self.dJ = level_u.J - level_d.J


class HeI_1083():
    """
    Class to acces the atomic model
    """
    def __init__(self):
        self.levels = [level(169_086.8428979/u.cm, 1, 3),      # MNIST data for HeI levels (2 level atom)
                       level(159_855.9743297/u.cm, 0, 3)]

        self.transitions = [transition(*self.levels, 1.0216e+07)]


class ESE:
    """ A class that stores the atomic state and needs to be constantly updated
        during the Lambda iterations by providing the Stokes parameters.
        After every Lambda iteration, solveESE() needs to be called.
        It is assumed that only one spectral line is involved in the problem.
        This class needs to be instantiated at every grid point.
    """

    jsim = jsymbols()

    def __init__(self, v_dop, a_voigt, nus, nus_weights, B):
        """
            nus_weights: array of the frequency quadrature weights
            B: object of the magnetic field vector with xyz components (gauss)
            return value: None
        """

        self.atom = HeI_1083()

        self.rho_dim = 0
        for lev in self.atom.levels:
            lev.initial_N = self.rho_dim*1   # Initial position of the density level submatrix
            self.rho_dim += lev.num_M

        self.rho = np.zeros((self.rho_dim, self.rho_dim))
        self.rho_vec = []
        for lev in self.atom.levels:
            for i in range(self.rho_dim):
                for j in range(self.rho_dim):
                    if lev.initial_N <= i < lev.initial_N + lev.num_M and\
                       lev.initial_N <= j < lev.initial_N + lev.num_M:
                        self.rho[i, j] = 1
                        self.rho_vec.append(self.rho[i, j])

        self.rho_vec = np.array(self.rho_vec)
        self.rho_elements = len(self.rho_vec)

    def solveESE(self, rad):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """
        self.Coeffs = np.zeros((self.rho_elements, self.rho_elements))
        self.Coeffs[-1, :] = np.ones(self.rho_elements)

        self.Indep = np.zeros(self.rho_elements)
        self.Indep[-1] = 1

        return 1


# Eq 7.9 from LL04 for the SEE coeficients
def TA(J, M, Mp, Jl, Ml, Mlp, JJ, Blu):
    sum_qq = 0
    for q in [-1, 0, 1]:
        for qp in [-1, 0, 1]:
            sum_qq = 3*(-1)**(Ml - Mlp)*(jsim.j6(J, Jl, 1, -M,  Ml, -q) *
                                         jsim.j6(J, Jl, 1, -Mp, Mlp, -qp) *
                                         JJ[q][qp])
    return (2*Jl + 1)*Blu*sum_qq


def TE(J, M, Mp, Ju, Mu, Mup, Aul):
    sum_q = 0
    for q in [-1, 0, 1]:
        sum_q = (-1)**(Mu - Mup)*(jsim.j6(Ju, J, 1, -Mup, Mp, -q) *
                                  jsim.j6(Ju, J, 1, -Mu,  M, -q))
    return (2*Ju + 1)*Aul*sum_q


def TS(J, M, Mp, Ju, Mu, Mup, JJ, Bul):
    sum_qq = 0
    for q in [-1, 0, 1]:
        for qp in [-1, 0, 1]:
            sum_qq += 3*(-1)**(Mp - M)*(jsim.j6(Ju, J, 1, -Mup, Mp, -q) *
                                        jsim.j6(Ju, J, 1, -Mu,  M, -qp) *
                                        JJ[q][qp])
    return (2*Ju + 1)*Bul*sum_qq


def RA(J, M, Mp, JJ):
    sum_u = 0
    for up in up_levels:
        sum_qqMu = 0
        for q in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                for Mu in up.M:
                    sum_qqMu += 3*(-1)**(M - Mp)*(jsim.j6(Ju, J, 1, -Mu, M, -q) *
                                                  jsim.j6(Ju, J, 1, -Mu, Mp, -qp) *
                                                  JJ[q][qp])
        sum_u += (2*J+1)*Blu*sum_qqMu
    return 0.5*sum_u


def RE(J, M, Mp):
    sum_l = 0
    if M == Mp:
        for low in lower_levels:
            sum_l += Aul
    return 0.5*sum_l


def RS(J, M, Mp, JJ):
    sum_l = 0
    for low in lower_levels:
        sum_qqMl = 0
        for q in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                for Ml in low.M:
                    sum_qqMl += 3*(jsim.j6(J, Jl, 1, -M,  Ml, -q) *
                                   jsim.j6(J, Jl, 1, -Mp, Ml, -qp) *
                                   JJ[q][qp])
        sum_u += (2*J+1)*Bul*sum_qqMl
    return 0.5*sum_u
