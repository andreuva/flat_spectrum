import numpy as np
import os
import scipy.linalg as linalg
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

        self.MMp_indep = []
        for M in range(-self.J, self.J+1):
            for Mp in range(-self.J, self.J+1):
                if Mp == M:
                    self.MMp_indep.append([self.J, M, Mp, False])
                elif Mp > M:
                    self.MMp_indep.append([self.J, M, Mp, False])
                    self.MMp_indep.append([self.J, M, Mp, True])


class line():
    """Class that defines the lines of the atomic model"""
    def __init__(self, levels, line_levels, jlju, Alu):

        self.levels = line_levels
        self.jlju = jlju

        self.gl = levels[line_levels[0]].g
        self.gu = levels[line_levels[1]].g

        self.wavelength = 1/(levels[line_levels[1]].E - levels[line_levels[0]].E)
        self.energy = c.h.cgs * c.c.cgs / self.wavelength
        self.nu = self.energy/c.h.cgs

        self.A_lu = Alu
        self.B_lu = Alu * (c.c.cgs**2/(2*c.h.cgs*self.nu**3))
        self.B_ul = self.B_lu * (levels[line_levels[1]].g/levels[line_levels[0]].g)

        self.dJ = levels[line_levels[1]].J - levels[line_levels[0]].J


class HeI_1083():
    """
    Class to acces the atomic model
    """
    def __init__(self):
        levels = [level(169_086.8428979/u.cm, 1, 3),      # MNIST data for HeI levels (2 level atom)
                  level(159_855.9743297/u.cm, 0, 1)]

        indx = np.argsort([lev.E.value for lev in levels])
        self.levels = []
        for i, ord in enumerate(indx):
            self.levels.append(levels[ord])

        self.lines = [line(self.levels, (0, 1), (0, 1), 1.0216e+07 / u.s), ]

        self.dens_elmnt = []
        for i, lev in enumerate(self.levels):
            for comb in lev.MMp:
                self.dens_elmnt.append([i, *comb])

        self.dens_elmnt_indep = []
        for i, lev in enumerate(self.levels):
            for comb in lev.MMp_indep:
                self.dens_elmnt_indep.append([i, *comb])

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

    def __init__(self, v_dop, a_voigt, nus, nus_weights, B, T, equilibrium=False):
        """
            nus_weights: array of the frequency quadrature weights
            B: object of the magnetic field vector with xyz components (gauss)
            return value: None
        """

        self.nus_weights = nus_weights
        self.B = B
        self.atom = HeI_1083()
        self.rho = np.zeros(len(self.atom.dens_elmnt)).astype('complex128')
        self.populations = 0

        if equilibrium:
            for i, lev in enumerate(self.atom.dens_elmnt):

                Ll = lev[0]
                Ml = lev[-2]
                Mlp = lev[-1]

                if Mlp == Ml:   # If it's a population compute the Boltzman LTE ratio
                    self.rho[i] = np.exp(-c.h.cgs*c.c.cgs*self.atom.levels[Ll].E/c.k_B.cgs/T) / \
                                  len(self.atom.levels[Ll].M)
                    self.populations += self.rho[i]

            self.rho = self.rho/self.populations
        else:
            self.rho[0] = 1

        self.N_rho = len(self.rho)
        self.coherences = self.N_rho - self.populations
        self.ESE_indep = np.zeros((self.N_rho, self.N_rho)) / u.s

    def solveESE(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """

        for i, p_lev in enumerate(self.atom.dens_elmnt_indep):
            self.ESE_indep[i] = np.zeros_like(self.ESE_indep[i])

            Li = p_lev[0]
            JJ = p_lev[1]
            M = p_lev[2]
            Mp = p_lev[3]
            imag_row = p_lev[4]

            for line in self.atom.lines:
                if Li not in line.levels:
                    continue

                for j, q_lev in enumerate(self.atom.dens_elmnt_indep):

                    Lj = q_lev[0]
                    Jp = q_lev[1]
                    N = q_lev[2]
                    Np = q_lev[3]
                    imag_col = q_lev[4]

                    if Lj not in line.levels:
                        continue

                    if Lj > Li:
                        if Np == N:
                            if not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.real(TS(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + TE(self, JJ, M, Mp, Jp, N, N, line.A_lu)
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.imag(TS(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))

                        elif Np > M:
                            if imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu)
                                                                               - TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                            elif not imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))
                            elif not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (TE(self, JJ, M, Mp, Jp, Np, N, line.A_lu)
                                                                               + TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))
                            else:
                                print('ERROR IN FILLING ESE MATRIX')
                                exit()

                    elif Lj < Li:
                        if Np == N:
                            if not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.real(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.imag(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))

                        elif Np > N:

                            if imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                            elif not imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))
                            elif not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                            else:
                                print('ERROR IN FILLING ESE MATRIX')
                                exit()

                    elif Lj == Li:
                        # calculate the RA and RE
                        if M == N:
                            if Np == M:
                                if not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))

                            elif Np > M:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))

                            elif Np < M:

                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))

                        if Mp == Np:
                            if N == Mp:
                                if not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))

                            elif N > Mp:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))

                            elif N < Mp:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))

                    else:
                        print("Error in the ESE matrix calculation")
                        exit()

            nu_L = 1.3996e6*np.linalg.norm(cdt.B.value)     # Eq 3.10 LL04 Larmor freq
            self.ESE_indep[i][i] = self.ESE_indep[i][i] + 2*np.pi*(M - Mp)*nu_L*self.atom.levels[Li].g

        indep = np.zeros(self.N_rho)/u.s
        indep[0] = 1/u.s

        for i, lev in enumerate(self.atom.dens_elmnt_indep):
            Ml = lev[2]
            Mlp = lev[3]
            if Mlp == Ml:
                self.ESE_indep[0, i] = 1/u.s
            else:
                self.ESE_indep[0, i] = 0/u.s

        # Print the ESE matrix to a file
        extension = 'txt'
        ind = 0
        dir = 'checks/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        path = dir + 'ESE_matrix' + f'_{ind}.' + extension
        while os.path.exists(path):
            ind += 1
            path = dir + 'ESE_matrix' + f'_{ind}.' + extension

        solve = np.real(self.ESE_indep.value)
        rows = len(solve)
        cols = len(solve[0])
        with open(path, 'w') as f:
            for i in range(rows):
                print(f'Row {i+1}\t', end='', file=f)
                for j in range(cols):
                    if solve[i][j] >= 0:
                        print(' ', end='', file=f)
                    print(f'{solve[i][j]:.2E}', end='', file=f)
                print(f'= {indep[i].value}', file=f)

        # LU = linalg.lu_factor(self.ESE)
        # rho_n = linalg.lu_solve(LU, indep)
        rho_n = linalg.solve(self.ESE_indep, indep)
        # Construct the full rho (complex)
        rho_comp = np.zeros_like(self.rho).astype('complex128')

        indexes = []
        for i, lev in enumerate(self.atom.dens_elmnt_indep):
            ll = lev[0]
            JJ = lev[1]
            M = lev[2]
            Mp = lev[3]
            imag = lev[4]

            index = self.atom.dens_elmnt.index([ll, self.atom.levels[ll].E, JJ, M, Mp])
            indexes.append(index)
            if not imag:
                rho_comp[index] += rho_n[i]
            else:
                rho_comp[index] += 1j*rho_n[i]

        for index in indexes:
            lev = self.atom.dens_elmnt[index]
            ll = lev[0]
            JJ = lev[-3]
            M = lev[-2]
            Mp = lev[-1]
            op_index = self.atom.dens_elmnt.index([ll, self.atom.levels[ll].E, JJ, Mp, M])
            rho_comp[op_index] = np.conjugate(rho_comp[index])

        change = np.abs(rho_comp - self.rho)/np.abs((rho_comp + 1e-40))
        self.rho = rho_comp.copy()

        path = dir + 'rho_array' + f'_{ind}.' + extension
        with open(path, 'w') as f:
            print('RHO\n----------\n', file=f)
            for i in range(rows):
                print(f'{self.rho[i]}', file=f)

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
                    print(f"with population of rho={self.rho[i]}")
                    # input("Press Enter to continue...")

        if not 0.98 < suma < 1.02:
            print("Warning: Not normaliced populations in this itteration")
            print(f"With the sum of the populations rho = {suma}")
            # input("Press Enter to continue...")

        # print('----------------------')
        # raise SystemExit(0)

        return np.max(change)

    def rho_call(self, lev, JJ, M, Mp):
        index = self.atom.dens_elmnt.index([lev, self.atom.levels[lev].E, JJ, M, Mp])
        return self.rho[index]


# Eq 7.9 from LL04 for the SEE coeficients
def TA(ESE, J, M, Mp, Jl, Ml, Mlp, rad, line, cdt):
    # Applied selection rules to remove sumation
    q = int(Ml - M)
    qp = int(Mlp - Mp)

    sum_qq = 3*(-1)**(Ml - Mlp)*(ESE.jsim.j3(J, Jl, 1, -M,  Ml, -q) *
                                 ESE.jsim.j3(J, Jl, 1, -Mp, Mlp, -qp) *
                                 rad.Jqq_nu(cdt, line, q, qp, M, Ml, ESE.B, ESE.nus_weights))
    return (2*Jl + 1)*line.B_lu*sum_qq


def TE(ESE, J, M, Mp, Ju, Mu, Mup, Aul):
    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    sum_q = (-1)**(Mu - Mup)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                              ESE.jsim.j3(Ju, J, 1, -Mu,  M, -q))
    return (2*Ju + 1)*Aul*sum_q


def TS(ESE, J, M, Mp, Ju, Mu, Mup, rad, line, cdt):
    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    qp = int(M - Mu)

    sum_qq = 3*(-1)**(Mp - M)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                               ESE.jsim.j3(Ju, J, 1, -Mu,  M, -qp) *
                               rad.Jqq_nu(cdt, line, q, qp, Mu, M, ESE.B, ESE.nus_weights))
    return (2*Ju + 1)*line.B_ul*sum_qq


def RA(ESE, Li, J, M, Mp, rad, cdt):

    sum_u = (0+0j) / u.s
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
                                                          rad.Jqq_nu(cdt, line, q, qp, Mu, M, ESE.B, ESE.nus_weights))
                sum_u += (2*J+1)*line.B_lu*sum_qqMu

    return 0.5*sum_u


def RE(ESE, Li, J, M, Mp):

    # RE(\alpha J M M') = 1/2 * \delta_{M M'} * \sum_{\alpha_l J_l} A(\alpha J -> \alpha_l J_l)

    sum_l = (0+0j) / u.s
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


def RS(ESE, Li, J, M, Mp, rad, cdt):

    sum_l = (0+0j) / u.s
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
                                           rad.Jqq_nu(cdt, line, q, qp, M, Ml, ESE.B, ESE.nus_weights))

                sum_l += (2*J+1)*line.B_ul*sum_qqMl

    return 0.5*sum_l
