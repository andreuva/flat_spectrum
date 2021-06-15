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
        self.M = []
        for M in range(-self.J, self.J+1):
            self.M.append(M)
            for Mp in range(-self.J, self.J+1):
                self.MMp.append([self.E, self.J, M, Mp])


class line():
    """Class that defines the lines of the atomic model"""
    def __init__(self, levels, line_levels, Alu):

        self.levels = line_levels

        self.gl = levels[line_levels[0]].g
        self.gu = levels[line_levels[1]].g

        self.wavelength = 1/(levels[line_levels[1]].E - levels[line_levels[0]].E)
        self.energy = c.h.cgs * c.c.cgs / self.wavelength
        self.nu = self.energy/c.h.cgs

        self.A_lu = Alu
        self.B_lu = Alu * (c.c.cgs**2/(2*c.h.cgs*self.nu**3)).value
        self.B_ul = self.B_lu * (levels[line_levels[1]].g/levels[line_levels[0]].g)

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

    def __init__(self, v_dop, a_voigt, nus, nus_weights, B, T):
        """
            nus_weights: array of the frequency quadrature weights
            B: object of the magnetic field vector with xyz components (gauss)
            return value: None
        """

        self.atom = HeI_1083()
        self.rho = np.zeros(len(self.atom.dens_elmnt))
        self.populations = 0

        for i, lev in enumerate(self.atom.dens_elmnt):

            Ll = lev[0]
            Ml = lev[-2]
            Mlp = lev[-1]

            if Mlp == Ml:   # If it's a population compute the Boltzman LTE ratio
                self.rho[i] = np.exp(-c.h.cgs*c.c.cgs*self.atom.levels[Ll].E/c.k_B.cgs/T) / \
                              len(self.atom.levels[Ll].M)
                self.populations += self.rho[i]

        self.rho = self.rho/self.populations
        self.N_rho = len(self.rho)
        self.coherences = self.N_rho - self.populations
        self.ESE = np.zeros((self.N_rho, self.N_rho)).astype('complex128')

    def solveESE(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """

        for i, p_lev in enumerate(self.atom.dens_elmnt):
            self.ESE[i] = np.zeros_like(self.ESE[i])

            Li = p_lev[0]
            M = p_lev[-2]
            Mp = p_lev[-1]
            JJ = p_lev[-3]

            for line in self.atom.lines:
                if Li not in line.levels:
                    continue

                for j, q_lev in enumerate(self.atom.dens_elmnt):

                    Lj = q_lev[0]
                    N = q_lev[-2]
                    Np = q_lev[-1]
                    Jp = q_lev[-3]

                    if Lj not in line.levels:
                        continue

                    if Lj > Li:
                        # calculate the TE(q -> p) and add it to self.ESE[i][j]
                        self.ESE[i][j] = self.ESE[i][j] + TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu)
                    elif Lj < Li:
                        # calculate the TA(q -> p) and add it to self.ESE[i][j]
                        self.ESE[i][j] = self.ESE[i][j] + TA(self, JJ, M, Mp, Jp, N, Np, rad, line.B_lu, line.nu)
                    elif Lj == Li:
                        # calculate the RA and RE
                        if M == N:
                            self.ESE[i][j] = self.ESE[i][j] - (RA(self, Li, JJ, Mp, Np, rad) +
                                                               RE(self, Li, JJ, Np, Mp) +
                                                               RS(self, Li, JJ, Np, Mp, rad))
                        if Mp == Np:
                            self.ESE[i][j] = self.ESE[i][j] - (RA(self, Li, JJ, N, M, rad) +
                                                               RE(self, Li, JJ, M, N) +
                                                               RS(self, Li, JJ, M, N, rad))
                        else:
                            continue
                    else:
                        print("Error in the ESE matrix calculation")
                        exit()

            nu_L = 1.3996e6*np.linalg.norm(cdt.B.value)     # Eq 3.10 LL04 Larmor freq
            self.ESE[i][i] = self.ESE[i][i] - 2j*np.pi*(M - Mp)*nu_L*self.atom.levels[Li].g

        indep = np.zeros(self.N_rho)
        indep[0] = 1

        for i, lev in enumerate(self.atom.dens_elmnt):
            Ml = lev[-2]
            Mlp = lev[-1]
            if Mlp == Ml:
                self.ESE[0, i] = 1

        rho_n = np.linalg.solve(np.real(self.ESE), indep)
        change = np.abs(rho_n - self.rho)/np.abs((rho_n + 1e-40))
        self.rho = rho_n.copy()

        # Printo the ESE matrix
        solve = np.real(self.ESE)
        # print('')
        # for i in range(len(solve)):
        #     print(f'Row {i}\t', end='')
        #     for j in range(len(solve)):
        #         if solve[i][j] >= 0:
        #             print(' ', end='')
        #         print(f'{solve[i][j]:.2E} ', end='')
        #     print(f'= {indep[i]}')

        # Check for the populations to be > 0 and to be normaliced
        suma = 0
        for i, lev in enumerate(self.atom.dens_elmnt):
            Ll = lev[0]
            Ml = lev[-2]
            Mlp = lev[-1]
            JJ = lev[-3]
            if Mlp == Ml:
                suma += self.rho[i]
                if self.rho[i] < 0:
                    print(f"Warning: Negative population of the level: L={Ll},J={JJ}, M={Ml},M'={Mlp}")
                    input("Press Enter to continue...")

        if not 0.98 < suma < 1.02:
            print("Warning: Not normaliced populations in this itteration")
            input("Press Enter to continue...")

        # print('----------------------')

        return np.max(change)


# Eq 7.9 from LL04 for the SEE coeficients
def TA(ESE, J, M, Mp, Jl, Ml, Mlp, rad, Blu, nu):
    sum_qq = 0
    for q in [-1, 0, 1]:
        for qp in [-1, 0, 1]:
            sum_qq += 3*(-1)**(Ml - Mlp)*(ESE.jsim.j3(J, Jl, 1, -M,  Ml, -q) *
                                          ESE.jsim.j3(J, Jl, 1, -Mp, Mlp, -qp) *
                                          rad.Jqq_nu(q, qp, nu))
    return (2*Jl + 1)*Blu*sum_qq


def TE(ESE, J, M, Mp, Ju, Mu, Mup, Aul):
    sum_q = 0
    for q in [-1, 0, 1]:
        sum_q += (-1)**(Mu - Mup)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                                   ESE.jsim.j3(Ju, J, 1, -Mu,  M, -q))
    return (2*Ju + 1)*Aul*sum_q


def TS(ESE, J, M, Mp, Ju, Mu, Mup, rad, Bul, nu):
    sum_qq = 0
    for q in [-1, 0, 1]:
        for qp in [-1, 0, 1]:
            sum_qq += 3*(-1)**(Mp - M)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                                        ESE.jsim.j3(Ju, J, 1, -Mu,  M, -qp) *
                                        rad.Jqq_nu(q, qp, nu))
    return (2*Ju + 1)*Bul*sum_qq


def RA(ESE, Li, J, M, Mp, rad):

    sum_u = 0
    for k, k_lev in enumerate(ESE.atom.levels):
        for line in ESE.atom.lines:

            Lk = k
            Jk = k_lev.J
            Mus = k_lev.M

            if Li not in line.levels or Lk not in line.levels:
                continue

            if Lk > Li:
                sum_qqMu = 0
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        for Mu in Mus:
                            sum_qqMu += 3*(-1)**(M - Mp)*(ESE.jsim.j3(Jk, J, 1, -Mu, M, -q) *
                                                          ESE.jsim.j3(Jk, J, 1, -Mu, Mp, -qp) *
                                                          rad.Jqq_nu(q, qp, line.nu))
                sum_u += (2*J+1)*line.B_lu*sum_qqMu

    return 0.5*sum_u


def RE(ESE, Li, J, M, Mp):

    # RE(\alpha J M M') = 1/2 * \delta_{M M'} * \sum_{\alpha_l J_l} A(\alpha J -> \alpha_l J_l)

    sum_l = 0
    if M == Mp:
        for k, k_lev in enumerate(ESE.atom.levels):
            Lk = k
            if Lk < Li:
                for line in ESE.atom.lines:
                    if Li not in line.levels or Lk not in line.levels:
                        continue
                    else:
                        sum_l += line.A_lu

    return 0.5*sum_l


def RS(ESE, Li, J, M, Mp, rad):

    sum_l = 0
    for k, k_lev in enumerate(ESE.atom.levels):
        for line in ESE.atom.lines:

            Lk = k
            Jk = k_lev.J
            Mls = k_lev.M

            if Li not in line.levels or Lk not in line.levels:
                continue

            if Lk < Li:
                sum_qqMl = 0
                for q in [-1, 0, 1]:
                    for qp in [-1, 0, 1]:
                        for Ml in Mls:
                            sum_qqMl += 3*(ESE.jsim.j3(J, Jk, 1, -M, Ml, -q) *
                                           ESE.jsim.j3(J, Jk, 1, -Mp, Ml, -qp) *
                                           rad.Jqq_nu(q, qp, line.nu))

                sum_l += (2*J+1)*line.B_ul*sum_qqMl

    return 0.5*sum_l
