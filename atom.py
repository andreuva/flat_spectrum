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
        self.nu3 = self.nu*self.nu*self.nu

        self.A_lu = Alu
        self.A_ul = Alu
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

        self.solveESE = self.solveESE_new

        ####################################################
        ####################################################
        ####################################################
        # print('AD-HOC INITIALIZATION IN ESE')
        # for i, lev in enumerate(self.atom.dens_elmnt):

        #     Ll = lev[0]
        #     Ml = lev[-2]
        #     Mlp = lev[-1]

        #     if Mlp == Ml:   # If it's a population compute the Boltzman LTE ratio
        #         self.rho[i] = np.exp(-c.h.cgs*c.c.cgs*self.atom.levels[Ll].E/c.k_B.cgs/T) / \
        #                       len(self.atom.levels[Ll].M)
        #         self.populations += self.rho[i]

        # self.rho = self.rho/self.populations

        # for i, lev in enumerate(self.atom.dens_elmnt):
        #     Ll = lev[0]
        #     Ml = lev[-2]
        #     Mlp = lev[-1]
        #     self.rho[i] += np.absolute(Ml-Mlp)*0.2
        #     self.rho[i] += 1j*(Ml-Mlp)*0.33
        ####################################################
        ####################################################
        ####################################################

        self.N_rho = len(self.rho)
        self.coherences = self.N_rho - self.populations
        self.ESE_indep = np.zeros((self.N_rho, self.N_rho)) / u.s

    def solveESE_new(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """

        Bnorm = np.linalg.norm(cdt.B.value)

       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0],-1,-1,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0],-1, 0,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0],-1, 1,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 0,-1,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 0, 0,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 0, 1,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 1,-1,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 1, 0,0,0,0.,self.nus_weights)
       #Jqq = rad.Jqq_nu_debug(cdt, self.atom.lines[0], 1, 1,0,0,0.,self.nus_weights)

       #f = open('newstat','w')

        # Initialize to zero
        for i, p_lev in enumerate(self.atom.dens_elmnt_indep):
            self.ESE_indep[i] = np.zeros_like(self.ESE_indep[i])

        # Magnetic term
        if Bnorm > 0.:

            # Row
            for i, p_lev in enumerate(self.atom.dens_elmnt_indep):

                # Get level data
                Li = p_lev[0]
                Ji = p_lev[1]
                M = p_lev[2]
                Mp = p_lev[3]
                imag = p_lev[4]
                q = M - Mp
                nu_L = 1.3996e6*Bnorm     # Eq 3.10 LL04 Larmor freq
                gamma = 2.0*np.pi*nu_L*self.atom.levels[p_level[0]]

                # Column
                for j, q_lev in enumerate(self.atom.dens_elmnt_indep):

                    # Get level data
                    Lj = q_lev[0]
                    if Lj != Li:
                        continue
                    Jj = q_lev[1]
                    if Jj != Ji:
                        continue
                    cM = q_lev[2]
                    if cM != M:
                        continue
                    cMp = q_lev[3]
                    if cMp != Mp:
                        continue
                    cimag = q_lev[4]
                    if cimag == imag:
                        continue

                    if imag:
                        self.ESE_indep[i][j] -= dif*gamma
                    else:
                        self.ESE_indep[i][j] += dif*gamma

        # For each radiative transition
        for line in self.atom.lines:

            # Get levels
            ll = line.levels[0]
            lu = line.levels[1]

            # If no field, compute \bar{J}_qq here
            if Bnorm <= 0.:
                Jqq = {'-1-1': rad.Jqq_nu(cdt, line,-1,-1,0,0,0.,self.nus_weights), \
                       '-1 0': rad.Jqq_nu(cdt, line,-1, 0,0,0,0.,self.nus_weights), \
                       '-1 1': rad.Jqq_nu(cdt, line,-1, 1,0,0,0.,self.nus_weights), \
                       ' 0 0': rad.Jqq_nu(cdt, line, 0, 0,0,0,0.,self.nus_weights), \
                       ' 0 1': rad.Jqq_nu(cdt, line, 0, 1,0,0,0.,self.nus_weights), \
                       ' 1 1': rad.Jqq_nu(cdt, line, 1, 1,0,0,0.,self.nus_weights)}
                Jqq[' 0-1'] = Jqq['-1 0'].conjugate()
                Jqq[' 1-1'] = Jqq['-1 1'].conjugate()
                Jqq[' 1 0'] = Jqq[' 0 1'].conjugate()

            #
            # Get rates
            #

            lMu = -1000
            lMl = -1000

            # Get upper level
            for i, u_lev in enumerate(self.atom.dens_elmnt_indep):

                # Get level info and check with line
                li = u_lev[0]
                if li != lu:
                    continue
                Ju = u_lev[1]
                Mu = u_lev[2]
                Mup = u_lev[3]
                imagu = u_lev[4]
                if imagu and Mu == Mup:
                    continue

                # Get lower level
                for j, l_lev in enumerate(self.atom.dens_elmnt_indep):

                    # Get level info and check with line
                    lj = l_lev[0]
                    if lj != ll:
                        continue
                    Jl = l_lev[1]
                    Ml = l_lev[2]
                    Mlp = l_lev[3]
                    imagl = l_lev[4]
                    if imagl and Ml == Mlp:
                        continue

                    # Compute \bar{J}_qq if magnetic
                    if Bnorm > 0. and (Mu != lMu or Ml != lMl):
                        Jqq = {'-1-1': rad.Jqq_nu(cdt, line,-1,-1,Mu,Ml,Bnorm,self.nus_weights), \
                               '-1 0': rad.Jqq_nu(cdt, line,-1, 0,Mu,Ml,Bnorm,self.nus_weights), \
                               '-1 1': rad.Jqq_nu(cdt, line,-1, 1,Mu,Ml,Bnorm,self.nus_weights), \
                               ' 0 0': rad.Jqq_nu(cdt, line, 0, 0,Mu,Ml,Bnorm,self.nus_weights), \
                               ' 0 1': rad.Jqq_nu(cdt, line, 0, 1,Mu,Ml,Bnorm,self.nus_weights), \
                               ' 1 1': rad.Jqq_nu(cdt, line, 1, 1,Mu,Ml,Bnorm,self.nus_weights)}
                        Jqq[' 0-1'] = Jqq['-1 0'].conjugate()
                        Jqq[' 1-1'] = Jqq['-1 1'].conjugate()
                        Jqq[' 1 0'] = Jqq[' 0 1'].conjugate()
                        lMu = Mu
                        lMl = Ml

                    #
                    # Get transfer rates
                    #

                    # Absorption transfer
                    TA  = TA_n(self,Ju,Mu,Mup,Jl,Ml,Mlp,Jqq,line)
                    if Mlp != Ml:
                        TAp = TA_n(self,Ju,Mu,Mup,Jl,Mlp,Ml,Jqq,line)
                    # Stimulated emission transfer
                    TS  = TS_n(self,Jl,Ml,Mlp,Ju,Mu,Mup,Jqq,line)
                    if Mup != Mu:
                        TSp = TS_n(self,Jl,Ml,Mlp,Ju,Mup,Mu,Jqq,line)
                    # Espontaneous emission transfer
                    TE  = TE_n(self,Jl,Ml,Mlp,Ju,Mu,Mup,line)
                    if Mup != Mu:
                        TEp = TE_n(self,Jl,Ml,Mlp,Ju,Mup,Mu,line)

                    #
                    # Add to ESE
                    #

                    # Imaginary upper level (row TA | column TS, TE))
                    if imagu:

                        # Imaginary lower level (row TS, TE | column TA)
                        if imagl:

                            self.ESE_indep[i][j] += TA.real - TAp.real

                            self.ESE_indep[j][i] += TS.real - TSp.real

                            self.ESE_indep[j][i] += TE.real - TEp.real

                           #f.write('TA {0} {1}   {2}\n'.format(i,j,TA.real-TAp.real))
                           #f.write('TS {0} {1}   {2}\n'.format(j,i,TS.real-TSp.real))
                           #f.write('TE {0} {1}   {2}\n'.format(j,i,TE.real-TEp.real))

                        # Real lower level (row TS, TE | column TA)
                        else:

                            self.ESE_indep[i][j] += TA.imag
                            if Ml != Mlp:
                                self.ESE_indep[i][j] += TAp.imag
                           #    f.write('TA {0} {1}   {2}\n'.format(i,j,TA.imag+TAp.imag))
                           #else:
                           #    f.write('TA {0} {1}   {2}\n'.format(i,j,TA.imag))

                            self.ESE_indep[j][i] += TSp.imag - TS.imag
                           #f.write('TS {0} {1}   {2}\n'.format(j,i,TSp.imag-TS.imag))

                    # Real upper level (row TA | columns TS, TE)
                    else:

                        # Imaginary lower level (row TS, TE | column TA)
                        if imagl:

                            self.ESE_indep[i][j] += TAp.imag - TA.imag
                           #f.write('TA {0} {1}   {2}\n'.format(i,j,TAp.imag-TA.imag))

                            self.ESE_indep[j][i] += TS.imag
                            if Mu != Mup:
                                self.ESE_indep[j][i] += TSp.imag
                           #    f.write('TS {0} {1}   {2}\n'.format(j,i,TS.imag+Tsp.imag))
                           #else:
                           #    f.write('TS {0} {1}   {2}\n'.format(j,i,TS.imag))

                        # Real lower level (row TS, TE | column TA)
                        else:

                            self.ESE_indep[i][j] += TA.real
                            if Ml != Mlp:
                                self.ESE_indep[i][j] += TAp.real
                           #    f.write('TA {0} {1}   {2}\n'.format(i,j,TA.real+TAp.real))
                           #else:
                           #    f.write('TA {0} {1}   {2}\n'.format(i,j,TA.real))

                            self.ESE_indep[j][i] += TS.real
                            self.ESE_indep[j][i] += TE.real
                            if Mu != Mup:
                                self.ESE_indep[j][i] += TSp.real
                                self.ESE_indep[j][i] += TEp.real
                           #    f.write('TS {0} {1}   {2}\n'.format(j,i,TS.real+TSp.real))
                           #    f.write('TE {0} {1}   {2}\n'.format(j,i,TE.real+TEp.real))
                           #else:
                           #    f.write('TS {0} {1}   {2}\n'.format(j,i,TS.real))
                           #    f.write('TE {0} {1}   {2}\n'.format(j,i,TE.real))

                    #
                    # Get relaxation rates
                    #

                    # Absorption

                    # Only once per upper level magnetic component
                    if Mu == Mup and not imagu:

                        # Get second lower level
                        for k, k_lev in enumerate(self.atom.dens_elmnt_indep):

                            # Get level info and check with line
                            lk = k_lev[0]
                            if ll != lk:
                                continue
                            Jk = k_lev[1]
                            if Jl != Jk:
                                continue
                            Mk = k_lev[2]
                            Mkp = k_lev[3]
                            imagk = k_lev[4]
                            if imagk and Mk == Mkp:
                                continue

                            # If diagonal Ml or Mlp
                            if (Mk == Ml and Mkp == Ml) or \
                               (Mk == Mlp and Mkp == Mlp):

                                # Get rate
                                RA = RA_n(self,Jl,Ml,Mlp,Ju,Mu,Jqq,line)

                                # Diagonal in Ml
                                if (Mk == Ml and Mkp == Ml):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                                # Diagonal in Mlp
                                if (Mk == Mlp and Mkp == Mlp):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                            #
                            # Not diagonal
                            #

                            # M'' < M sum
                            if Mk < Ml and Mkp == Ml:

                                # Get rate
                                RA = RA_n(self,Jl,Mk,Mlp,Ju,Mu,Jqq,line)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.real))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                            # M'' > M sum
                            if Mkp > Ml and Mk == Ml:

                                # Get rate
                                RA = RA_n(self,Jl,Mkp,Mlp,Ju,Mu,Jqq,line)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.imag))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                            # M'' < M' sum
                            if Mk < Mlp and Mkp == Mlp:

                                # Get rate
                                RA = RA_n(self,Jl,Mk,Ml,Ju,Mu,Jqq,line)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.imag))

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.imag))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))

                            # M'' > M sum
                            if Mkp > Mlp and Mk == Mlp:

                                # Get rate
                                RA = RA_n(self,Jl,Mkp,Ml,Ju,Mu,Jqq,line)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,RA.real))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.imag))

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.imag))

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real
                           #            f.write('RA {0} {1}   {2}\n'.format(j,k,-RA.real))


                    # Emission

                    # Only once per lower level magnetic component
                    if Ml == Mlp and not imagl:

                        # Get second upper level
                        for k, k_lev in enumerate(self.atom.dens_elmnt_indep):

                            # Get level info and check with line
                            lk = k_lev[0]
                            if lu != lk:
                                continue
                            Jk = k_lev[1]
                            if Ju != Jk:
                                continue
                            Mk = k_lev[2]
                            Mkp = k_lev[3]
                            imagk = k_lev[4]
                            if imagk and Mk == Mkp:
                                continue

                            # If diagonal Mu or Mup
                            if (Mk == Mu and Mkp == Mu) or \
                               (Mk == Mup and Mkp == Mup):

                                # Get rate
                                RS = RS_n(self,Ju,Mu,Mup,Jl,Ml,Jqq,line)

                                # If diagonal Mu
                                if (Mk == Mu and Mkp == Mu):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.imag))

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_n(self,Ju,Mu,Mup,Jl,Ml,line)

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                                # If diagonal Mup
                                if (Mk == Mup and Mkp == Mup):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.imag))

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_n(self,Ju,Mu,Mup,Jl,Ml,line)

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                            #
                            # Not diagonal
                            #

                            # M'' < M sum
                            if Mk < Mu and Mkp == Mu:

                                # Get rate
                                RS = RS_n(self,Ju,Mk,Mup,Jl,Ml,Jqq,line)
                                RE = RE_n(self,Ju,Mk,Mup,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,RE.real))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.imag))

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.imag))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                            # M'' > M sum
                            if Mkp > Mu and Mk == Mu:

                                # Get rate
                                RS = RS_n(self,Ju,Mkp,Mup,Jl,Ml,Jqq,line)
                                RE = RE_n(self,Ju,Mkp,Mup,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.imag))

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.imag))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                            # M'' < M' sum
                            if Mk < Mup and Mkp == Mup:

                                # Get rate
                                RS = RS_n(self,Ju,Mk,Mu,Jl,Ml,Jqq,line)
                                RE = RE_n(self,Ju,Mk,Mu,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.imag))

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.imag))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))

                            # M'' > M sum
                            if Mkp > Mup and Mk == Mup:

                                # Get rate
                                RS = RS_n(self,Ju,Mkp,Mu,Jl,Ml,Jqq,line)
                                RE = RE_n(self,Ju,Mkp,Mu,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,RE.real))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.imag))

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,RS.imag))

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real
                           #            f.write('RS {0} {1}   {2}\n'.format(i,k,-RS.real))
                           #            f.write('RE {0} {1}   {2}\n'.format(i,k,-RE.real))


        indep = np.zeros(self.N_rho)/u.s
        indep[0] = 1/u.s

        for i, lev in enumerate(self.atom.dens_elmnt_indep):
            Ml = lev[2]
            Mlp = lev[3]
            if Mlp == Ml:
                self.ESE_indep[0, i] = 1/u.s
            else:
                self.ESE_indep[0, i] = 0/u.s

       #for i, lev in enumerate(self.atom.dens_elmnt_indep):
       #    msg = '{0} '.format(i)
       #    for j, val in enumerate(self.ESE_indep[i]):
       #        msg += ' {0:17.10e}'.format(val)
       #    msg += '\n'
       #    f.write(msg)
       #f.close()

        # Print the ESE matrix to a file
        # extension = 'txt'
        # ind = 0
        # dir = 'checks/'
        # if not os.path.exists(dir):
        #     os.makedirs(dir)

        # path = dir + 'ESE_matrix' + f'_{ind}.' + extension
        # while os.path.exists(path):
        #     ind += 1
        #     path = dir + 'ESE_matrix' + f'_{ind}.' + extension

        solve = np.real(self.ESE_indep.value)
        # rows = len(solve)
        # cols = len(solve[0])
        # with open(path, 'w') as f:
        #     for i in range(rows):
        #         print(f'Row {i+1}\t', end='', file=f)
        #         for j in range(cols):
        #             if solve[i][j] >= 0:
        #                 print(' ', end='', file=f)
        #             print(f'{solve[i][j]:.2E}', end='', file=f)
        #         print(f'= {indep[i].value}', file=f)

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

        # path = dir + 'rho_array' + f'_{ind}.' + extension
        # with open(path, 'w') as f:
        #     print('RHO\n----------\n', file=f)
        #     for i in range(rows):
        #         print(f'{self.rho[i]}', file=f)

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


    def solveESE_old(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """

        f = open('oldstat','w')

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
                                f.write('TS {0} {1}   {2}\n'.format(i,j,np.real(TS(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + TE(self, JJ, M, Mp, Jp, N, N, line.A_lu)
                                f.write('TE {0} {1}   {2}\n'.format(i,j,TE(self, JJ, M, Mp, Jp, N, N, line.A_lu)))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.imag(TS(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))

                        elif Np > M:
                            if imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TS {0} {1}   {2}\n'.format(i,j,(np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu)
                                                                               - TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))
                                f.write('TE {0} {1}   {2}\n'.format(i,j,(TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu)
                                                                               - TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TS {0} {1}   {2}\n'.format(i,j,(np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                            elif not imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))
                                f.write('TS {0} {1}   {2}\n'.format(i,j,(np.imag(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))))
                            elif not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TS {0} {1}   {2}\n'.format(i,j,(np.real(TS(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TS(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (TE(self, JJ, M, Mp, Jp, Np, N, line.A_lu)
                                                                               + TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))
                                f.write('TE {0} {1}   {2}\n'.format(i,j,(TE(self, JJ, M, Mp, Jp, Np, N, line.A_lu)
                                                                               + TE(self, JJ, M, Mp, Jp, N, Np, line.A_lu))))
                            else:
                                print('ERROR IN FILLING ESE MATRIX')
                                exit()

                    elif Lj < Li:
                        if Np == N:
                            if not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.real(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,np.real(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + np.imag(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,np.imag(TA(self, JJ, M, Mp, Jp, N, N, rad, line, cdt))))

                        elif Np > N:

                            if imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,(np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               - np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                            elif imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,(np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                            elif not imag_row and imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,(np.imag(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt))
                                                                               - np.imag(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt)))))
                            elif not imag_row and not imag_col:
                                self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))
                                f.write('TA {0} {1}   {2}\n'.format(i,j,(np.real(TA(self, JJ, M, Mp, Jp, N, Np, rad, line, cdt))
                                                                               + np.real(TA(self, JJ, M, Mp, Jp, Np, N, rad, line, cdt)))))
                            else:
                                print('ERROR IN FILLING ESE MATRIX')
                                exit()

                    elif Lj == Li:
                        # calculate the RA and RE
                        if M == N:
                            if Np == M:
                                if not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,-(np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,-(np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,-(np.real(RE(self, Li, JJ, M, Mp)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,(np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,(np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))))

                            elif Np > M:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, -(np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,- (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, Np, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, Np, Mp)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,(np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,(np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,- (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,- (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,- (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,- (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, Np, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, Np, Mp)))))

                            elif Np < M:

                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j,(np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,(np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RE(self, Li, JJ, Np, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,(np.real(RE(self, Li, JJ, Np, Mp)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, (np.imag(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, (np.imag(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.real(RA(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.real(RS(self, Li, JJ, Np, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, Np, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, Np, Mp)))))

                        if Mp == Np:
                            if N == Mp:
                                if not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.real(RA(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.real(RS(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, M, Mp)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, (np.imag(RA(self, Li, JJ, M, Mp, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, (np.imag(RS(self, Li, JJ, M, Mp, rad, cdt)))))

                            elif N > Mp:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, (np.real(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, (np.real(RS(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.real(RE(self, Li, JJ, N, M)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j, (np.real(RE(self, Li, JJ, N, M)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j,- (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, N, M)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, N, M)))))

                            elif N < Mp:
                                if imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, M, Mp)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j, - (np.real(RE(self, Li, JJ, N, M)))))
                                elif imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))))
                                elif not imag_row and imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, (np.imag(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] + (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, (np.imag(RS(self, Li, JJ, N, M, rad, cdt)))))
                                elif not imag_row and not imag_col:
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RA {0} {1}   {2}\n'.format(i,j, - (np.real(RA(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))
                                    f.write('RS {0} {1}   {2}\n'.format(i,j, - (np.real(RS(self, Li, JJ, N, M, rad, cdt)))))
                                    self.ESE_indep[i][j] = self.ESE_indep[i][j] - (np.real(RE(self, Li, JJ, N, M)))
                                    f.write('RE {0} {1}   {2}\n'.format(i,j,- (np.real(RE(self, Li, JJ, N, M)))))

                    else:
                        print("Error in the ESE matrix calculation")
                        exit()

            nu_L = 1.3996e6*np.linalg.norm(cdt.B.value)     # Eq 3.10 LL04 Larmor freq
            if M != Mp:
                if not imag_row:
                    self.ESE_indep[i][i+1] = self.ESE_indep[i][i+1] + 2*np.pi*(M - Mp)*nu_L*self.atom.levels[Li].g
                else:
                    self.ESE_indep[i][i-1] = self.ESE_indep[i][i-1] - 2*np.pi*(M - Mp)*nu_L*self.atom.levels[Li].g
            else:
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


        for i, lev in enumerate(self.atom.dens_elmnt_indep):
            msg = '{0} '.format(i)
            for j, val in enumerate(self.ESE_indep[i]):
                msg += ' {0:17.10e}'.format(val)
            msg += '\n'
            f.write(msg)
        f.close()

        # Print the ESE matrix to a file
        # extension = 'txt'
        # ind = 0
        # dir = 'checks/'
        # if not os.path.exists(dir):
        #     os.makedirs(dir)

        # path = dir + 'ESE_matrix' + f'_{ind}.' + extension
        # while os.path.exists(path):
        #     ind += 1
        #     path = dir + 'ESE_matrix' + f'_{ind}.' + extension

        solve = np.real(self.ESE_indep.value)
        # rows = len(solve)
        # cols = len(solve[0])
        # with open(path, 'w') as f:
        #     for i in range(rows):
        #         print(f'Row {i+1}\t', end='', file=f)
        #         for j in range(cols):
        #             if solve[i][j] >= 0:
        #                 print(' ', end='', file=f)
        #             print(f'{solve[i][j]:.2E}', end='', file=f)
        #         print(f'= {indep[i].value}', file=f)

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

        # path = dir + 'rho_array' + f'_{ind}.' + extension
        # with open(path, 'w') as f:
        #     print('RHO\n----------\n', file=f)
        #     for i in range(rows):
        #         print(f'{self.rho[i]}', file=f)

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
def TA_n(ESE, J, M, Mp, Jl, Ml, Mlp, Jqq, line):

    # Applied selection rules to remove sumation
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return (0+0j)/u.s
    qp = int(Mlp - Mp)
    if np.absolute(qp) > 1:
        return (0+0j)/u.s
    tag = f'{q:2d}{qp:2d}'

    sum_qq = 3*ESE.jsim.sign(Ml-Mlp) * \
               ESE.jsim.j3(J, Jl, 1, -M,  Ml, -q) * \
               ESE.jsim.j3(J, Jl, 1, -Mp, Mlp, -qp) * \
               Jqq[tag]

    return (2*Jl + 1)*line.B_lu*sum_qq


def TA(ESE, J, M, Mp, Jl, Ml, Mlp, rad, line, cdt):
    # Applied selection rules to remove sumation
    q = int(Ml - M)
    qp = int(Mlp - Mp)

    sum_qq = 3*(-1)**(Ml - Mlp)*(ESE.jsim.j3(J, Jl, 1, -M,  Ml, -q) *
                                 ESE.jsim.j3(J, Jl, 1, -Mp, Mlp, -qp) *
                                 rad.Jqq_nu(cdt, line, q, qp, M, Ml, ESE.B, ESE.nus_weights))
    return (2*Jl + 1)*line.B_lu*sum_qq

def TE_n(ESE, J, M, Mp, Ju, Mu, Mup, line):

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    qp = int(M - Mu)
    if q != qp:
        return (0+0j)/u.s

    sum_q = ESE.jsim.sign(Mu - Mup) * \
            ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) * \
            ESE.jsim.j3(Ju, J, 1, -Mu,  M, -q)

    return (2*Ju + 1)*line.A_ul*sum_q

def TE(ESE, J, M, Mp, Ju, Mu, Mup, Aul):
    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    sum_q = (-1)**(Mu - Mup)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                              ESE.jsim.j3(Ju, J, 1, -Mu,  M, -q))
    return (2*Ju + 1)*Aul*sum_q


def TS_n(ESE, J, M, Mp, Ju, Mu, Mup, Jqq, line):

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    if np.absolute(q) > 1:
        return (0+0j)/u.s
    qp = int(M - Mu)
    if np.absolute(qp) > 1:
        return (0+0j)/u.s
    tag = f'{q:2d}{qp:2d}'

    sum_qq = 3*ESE.jsim.sign(Mp - M) * \
               ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) * \
               ESE.jsim.j3(Ju, J, 1, -Mu,  M, -qp) * \
               Jqq[tag]

    return (2*Ju + 1)*line.B_ul*sum_qq

def TS(ESE, J, M, Mp, Ju, Mu, Mup, rad, line, cdt):
    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    qp = int(M - Mu)

    sum_qq = 3*(-1)**(Mp - M)*(ESE.jsim.j3(Ju, J, 1, -Mup, Mp, -q) *
                               ESE.jsim.j3(Ju, J, 1, -Mu,  M, -qp) *
                               rad.Jqq_nu(cdt, line, q, qp, Mu, M, ESE.B, ESE.nus_weights))
    return (2*Ju + 1)*line.B_ul*sum_qq


def RA_n(ESE, J, M, Mp, Ju, Mu, Jqq, line):

    # Selection rules
    q = int(M - Mu)
    if np.absolute(q) > 1:
        return (0+0j)/u.s
    qp = int(Mp - Mu)
    if np.absolute(qp) > 1:
        return (0+0j)/u.s
    tag = f'{q:2d}{qp:2d}'

    sum_u = 3*ESE.jsim.sign(M - Mp) * \
              ESE.jsim.j3(Ju, J, 1, -Mu,  M,  -q) * \
              ESE.jsim.j3(Ju, J, 1, -Mu, Mp, -qp) * \
              Jqq[tag]

    return 0.5*(2*J+1)*line.B_lu*sum_u

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


def RE_n(ESE, J, M, Mp, Jl, Ml, line):

    # Selection rules
    q = int(Ml - M)
    qp = int(Ml - Mp)
    if q != qp:
        return (0+0j)/u.s

    return 0.5*line.A_ul

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


def RS_n(ESE, J, M, Mp, Jl, Ml, Jqq, line):

    # Selection rules
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return (0+0j)/u.s
    qp = int(Ml - Mp)
    if np.absolute(qp) > 1:
        return (0+0j)/u.s
    tag = f'{q:2d}{qp:2d}'

    sum_l = 3*ESE.jsim.j3( J,Jl, 1, -M, Ml, -q) * \
              ESE.jsim.j3( J,Jl, 1,-Mp, Ml,-qp) * \
              Jqq[tag]

    if J == 1. and ((M == -1 and Mp == 0) or (M == 0 and Mp == -1)):
        print('\nNew calling RS')
        print('Arguments',J,M,Mp,Jl,Ml)
        print('Jqq',q,qp,tag,Jqq[tag])
        print('Sum',sum_l)
        print('Rest',line.B_ul*(2*J+1)*0.5)
        print('Return',0.5*(2*J+1)*line.B_ul*sum_l)

    return 0.5*(2*J+1)*line.B_ul*sum_l

def RS(ESE, Li, J, M, Mp, rad, cdt):

    if J == 1. and ((M == -1 and Mp == 0) or (M == 0 and Mp == -1)):
        print('\nTrue old calling RS',Li,J,M,Mp)

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

                            ss = 3*(ESE.jsim.j3(J, Jk, 1, -M, Ml, -q) *
                                    ESE.jsim.j3(J, Jk, 1, -Mp, Ml, -qp) *
                                    rad.Jqq_nu(cdt, line, q, qp, M, Ml, ESE.B, ESE.nus_weights))
                            if J == 1. and ((M == -1 and Mp == 0) or (M == 0 and Mp == -1)) and \
                               (np.absolute(ss.real) > 0. or np.absolute(ss.imag) > 0.):
                                print('\nOld calling RS',Lk)
                                print('Arguments',J,M,Mp,Jk,Ml)
                                print('Jqq',q,qp,rad.Jqq_nu(cdt,line,q,qp,M,Ml,ESE.B,ESE.nus_weights))
                                print('Sum',ss)
                                print('Rest',0.5*(2*J+1)*line.B_ul)
                                print('Return',0.5*(2*J+1)*line.B_ul*ss)

                sum_l += (2*J+1)*line.B_ul*sum_qqMl

    if J == 1. and ((M == -1 and Mp == 0) or (M == 0 and Mp == -1)):
        print('\nTrue return RS',0.5*sum_l)

    return 0.5*sum_l
