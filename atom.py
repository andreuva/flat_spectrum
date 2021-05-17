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

        self.MMp = []
        for M in range(-self.J, self.J+1):
            for Mp in range(-self.J, self.J+1):
                self.MMp.append([self.E, self.J, M, Mp])


class line():
    """Class that defines the lines of the atomic model"""
    def __init__(self, levels, line_levels, Alu):

        self.levels = line_levels

        self.wavelength = 1/(levels[line_levels[1]].E - levels[line_levels[0]].E)
        self.energy = c.h.cgs * c.c.cgs / self.wavelength
        self.nu = self.energy/c.h.cgs

        self.A_lu = Alu
        self.B_lu = Alu * c.c.cgs**3/(2*c.h.cgs*self.nu**3)
        self.B_ul = self.B_lu * levels[line_levels[1]].g/levels[line_levels[0]].g

        self.dJ = levels[line_levels[1]].J - levels[line_levels[0]].J


class HeI_1083():
    """
    Class to acces the atomic model
    """
    def __init__(self):
        levels = [level(169_086.8428979/u.cm, 1, 3),      # MNIST data for HeI levels (2 level atom)
                  level(159_855.9743297/u.cm, 0, 3)]

        indx = np.argsort([lev.E.value for lev in levels])
        self.levels = []
        for i, ord in enumerate(indx):
            self.levels.append(levels[ord])

        self.lines = [line(self.levels, (0, 1), 1.0216e+07),
                      ]

        self.dens_elmnt = []
        for i, lev in enumerate(self.levels):
            for comb in lev.MMp:
                self.dens_elmnt.append([i, *comb])

        self.line_elmnt = []
        for i, ln in enumerate(self.lines):
            self.line_elmnt.append([i, *ln.levels])


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
        self.rho = [0 for i in self.atom.dens_elmnt]
        self.N_rho = len(self.rho)
        self.ESE = np.zeros((self.N_rho, self.N_rho)).astype('complex128')

    def solveESE(self, rad):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """

        for i, p_lev in enumerate(self.atom.dens_elmnt):
            self.ESE[i] = np.zeros_like(self.ESE[i])

            Li = p_lev[0]
            M = p_lev[-2]
            Mp = p_lev[-1]

            for line in self.atom.lines:
                if Li not in line.levels:
                    continue

                for j, q_lev in enumerate(self.atom.dens_elmnt):

                    Lj = q_lev[0]
                    if Lj not in line.levels:
                        continue

                    N = q_lev[-2]
                    Np = q_lev[-1]

                    if Lj > Li:
                        # calculate the TE(q -> p) and add it to self.ESE[i][j]
                        self.ESE[i][j] = self.ESE[i][j] + 1
                    elif Lj < Li:
                        # calculate the TA(q -> p) and add it to self.ESE[i][j]
                        self.ESE[i][j] = self.ESE[i][j] + 1
                    elif Lj == Li:
                        pass    # calculate the RA and RE
                    else:
                        print("Error in the ESE matrix calculation")
                        exit()

            nu_L = 1.3996e6*np.linalg.norm(B.value)     # Eq 3.10 LL04 Larmor freq
            self.ESE[i][i] = self.ESE[i][i] - 2j*np.pi*(M - Mp)*nu_L*self.atom.levels[Li].g

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
