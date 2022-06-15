import pickle
import time
import numpy as np
import os,sys,copy
import scipy.linalg as linalg
import constants as c
from physical_functions import Tqq_all, jsymbols, rotate_ist

################################################################################
################################################################################
################################################################################

class term_class():
    """Class that defines the energy term of the atomic model
    """

    def __init__(self, LL, SS, JJ, Energy, JS, B=0, i0=0):
        """ Initialize the term class
            Input:
               LL: Orbital angular momentum
               SS: Spin angular momentum
               JJ: List of total angular momentum for the levels
                   within the term.
               EE: Energy for each J level [cm^-1]
               JS: Instance of Racah algebra class already initialized
                   (passed to take advantage of memoization)
                B: Magnetic field strength [G]
               i0: Global density matrix index corresponding to the
                   first index of this term within the atom
            Output: None
            Internal:
              self.L: Term orbital angular momentum
              self.S: Term spin angular momentum
              self.TE: Term energy
              self.LE: List of energies of the different levels
              self.nM: Number of possible M magnetic number values in
                       this term
              self.M: Possible M magnetic number values in this term
              self.nJ: Number of levels within the term
              self.g: Degeneracy of the term
              self.NN: Number of density matrix elements for the term
              self.Mblock: Index data for the blocks of M magnetic
                           number
              self.index_nuM: List of indexes of all jM combinations.
                              Each element has:
                               index, J level, J value, j level, M value
              self.eigval: List of the Hamiltonian eigenvalues, ordered
                           by Mblock indexes (self.index_muM[0]).
              self.eigvec: List of the Hamiltonian eigenvectors, ordered
                           by Mblock indexes (self.index_muM[0]), with
                           each element a vector with indexes given by
                           self.index_nuM[1].
              self.i0: Initial global atomic index for the density
                       matrix elements of this term
              self.index: Indexing of the density matrix elements.
                          Notation: rho(jM,j'M').
                          For each element of the list, a list with:
                            index: Proper index of the matrix element
                            jM index: Position in index_muM of the jM
                                      level
                            j'M' index: Position in index_muM of the
                                        j'M' level
                            M index: index of M in self.Mblock
                            M value: M magnetic number
                            mu index: index of j in self.index_muM
                            M' index: index of M' in self.Mblock
                            M' value: M' magnetic number
                            mu' index: index of j' in self.index_muM
                            Imag: Bool indicating if element is imaginary
                                  (True) or real (False)
              self.NN: Number of density matrix elements for this term
              self.NN_next: First index for the density matrix element
                            of the next term, if any

            Notes:
                JJ and Energy must have the same dimension and it
              needs to be consistent with the orbital and spin
              angular momentum.
                Only independent density matrix elements are indexed and
              used in the code
        """

        # Initialize term energy
        self.TE = 0.
        deg = 0.

        # Compute term energy
        for j,E in zip(JJ,Energy):
            f = (2.*j + 1.)
            self.TE += f*E
            deg += f
        self.TE /= deg

        # Save quantum numbers
        self.L = LL
        self.S = SS

        # Get number of levels
        minJ = np.absolute(LL-SS)
        maxJ = LL + SS
        self.nJ = int(round(maxJ - minJ + 1))

        # Multiplicity
        self.g = (2.*SS + 1.)*(2.*LL + 1.)

        # Initialize number of components rhojmj'm'
        self.NN = 0

        # Initialize block size for each M
        self.Mblock = []

        # Initialize eigenvalues and eigenvectors
        self.eigval = []
        self.eigvec = []

        # Save modified index
        self.i0 = i0

        # Sanity check
        if self.nJ != len(Energy) or self.nJ != len(JJ):
            print('Critical error, number of energy levels different ' + \
                  'than number of levels in term class initialization')
            sys.exit()

        # Save energies (fine structure)
        self.LE = Energy

        # J values
        self.J = np.array(JJ)
        if np.absolute(np.max(self.J) - maxJ) > 1e-6:
            print('Critical error, maximum total angular momentum in input ' + \
                  'list in term different to maximum possible angular momentum')
        if np.absolute(np.min(self.J) - minJ) > 1e-6:
            print('Critical error, minimum total angular momentum in input ' + \
                  'list in term different to minimum possible angular momentum')

        # M Values
        self.nM = int(round(2*maxJ + 1))
        self.M = np.linspace(-maxJ,maxJ,self.nM,endpoint=True,dtype=np.float32)

        #
        # Diagonalize
        #

        # Get magnetic field modulus
        Bnorm = B

        # Spin factor
        pS = np.sqrt(SS*(SS + 1.0)*(2.0*SS + 1.0))

        # Block size
        nM = self.nM

        # Max size
        smax = nM*nM

        # Initialize muM indexing
        self.index_muM = []
        ii = -1

        # For each M
        for M in self.M:

            # Initialize matrices
            ene = np.zeros((smax))
            diag = np.zeros((smax))
            odiag = np.zeros((smax))

            # Minimum J
            Jm = np.max([np.absolute(M), minJ])

            # Append a new block
            self.index_muM.append([])

            # Initialize column index
            i = -1

            # For each level J
            for k,J,E in zip(range(self.nJ),self.J,self.LE):

                # Check above minimum
                if J < Jm:
                    continue

                # J factor
                pJ = np.sqrt(2.0*J + 1.0)

                # Row index
                i1 = i

                # Advance column
                i += 1

                # Append to indexing
                ii += 1
                mu = i
                self.index_muM[-1].append([ii,k,J,mu,M])

                # For each J
                for J1 in self.J[i:]:

                    # Check above minimum
                    if J1 < Jm:
                        continue

                    # J difference
                    dJ = int(round(np.absolute(J1 - J)))

                    # Selection rules
                    if dJ > 1:
                        continue

                    # J1 factor
                    pJ1 = np.sqrt(2.0*J1 + 1.0)

                    # Atomic part
                    comm = (-1.0)**(int(round(maxJ+J+J1+M)))* \
                           pJ*pJ1*pS* \
                           JS.j3(J1,J,1.0,-M,M,0.0)* \
                           JS.j6(J1,J,1.0,SS,SS,LL)

                    # Advance row
                    i1 += 1

                    # If diagonal
                    if i == i1:
                        ene[i] = E
                        diag[i] = comm + M
                    else:
                        odiag[i1-1] = comm

            # Size of this block, append to list of sizes
            nblock = i + 1
            self.Mblock.append(nblock)

            # Reset local solutions
            ly = []

            # Compute larmor factor
            larmor = c.nuL*Bnorm

            # Multiply by larmor
            diag = diag[:nblock]*larmor
            odiag = odiag[:nblock-1]*larmor

            # Add energy
            for j in range(nblock):
                diag[j] += ene[j]

            # Diagonalize if there is dimensionality
            if nblock > 1:
                w, v = linalg.eigh_tridiagonal(diag, odiag, \
                                               check_finite=False)
            else:
                w = diag[0:1]
                v = np.array([[1.0]])

            # Store in term data
            for iv in range(nblock):

                # Append eigenvalue
                self.eigval.append(w[iv])

                # Ensure diagonal elements positive (arbitrary)
                if v[iv,iv] < 0:
                    self.eigvec.append(-v[:,iv])
                else:
                    self.eigvec.append(v[:,iv])

        #
        # Index the density matrix elements
        #

        # Initialize indexes matrix and running index
        self.index = []
        ii = -1

        # Initialize left index
        i1 = -1

        # For each "left" M
        for iM,M,Mblock in zip(range(self.nM),self.M,self.Mblock):

            # For each "left" mu
            for mu in range(Mblock):

                # Get mu, M index
                i1 += 1

                # Initialize right index
                i2 = -1

                # For each "right" M
                for iM1,M1,Mblock1 in zip(range(self.nM),self.M,self.Mblock):

                    # If dM > 1, skip
                    if int(round(np.absolute(M1 - M))) > int(round(2.*maxJ)):

                        # Advance the index anyways
                        i2 += Mblock1

                        # And skip
                        continue

                    # For each "right" mu
                    for mu1 in range(Mblock1):

                        # Get mu1, M1 index
                        i2 += 1

                        # Only if i2 >= i1
                        if i2 >= i1:

                            # If diagonal
                            if i1 == i2:
                                ii += 1
                                self.index.append([ii+self.i0,i1,i2,iM,M,mu,iM1,M1,mu1,False])
                            # Not diagonal
                            else:
                                ii += 1
                                self.index.append([ii+self.i0,i1,i2,iM,M,mu,iM1,M1,mu1,False])
                                ii += 1
                                self.index.append([ii+self.i0,i1,i2,iM,M,mu,iM1,M1,mu1,True])

        # Save number of independent density matrix elements
        self.NN = ii + 1
        self.NN_next = self.NN + self.i0

################################################################################
################################################################################
################################################################################

class level():
    """Class that defines the energy level of the atomic model
       (Not updated for MT version)
    """

    def __init__(self, energy, JJ, g):
        """ Initialize the level class
        """

        # Save atomic data: energy, degeneracy, total angular momentum
        self.E = energy
        self.g = g
        self.J = JJ

        # Index the MM' combinations
        self.MMp = []
        self.M = []
        for M in range(-self.J, self.J+1):
            self.M.append(M)
            for Mp in range(-self.J, self.J+1):
                self.MMp.append([self.E, self.J, M, Mp])

        # Index the density matrix elements
        self.MMp_indep = []
        for M in range(-self.J, self.J+1):
            for Mp in range(-self.J, self.J+1):
                if Mp == M:
                    self.MMp_indep.append([self.J, M, Mp, False])
                elif Mp > M:
                    self.MMp_indep.append([self.J, M, Mp, False])
                    self.MMp_indep.append([self.J, M, Mp, True])

################################################################################
################################################################################
################################################################################

class line_class():
    """Class that defines the lines of the atomic model
    """

    def __init__(self, multiterm, data, JS):
        """ Initialize the line class.
            Input:
               multiterm: bool that specifies that this is for a multi-term
                          atom
               data: List with data depending on the type of model
                 multiterm: a list with:
                    List of terms of the atom
                    List of indexes of the terms involved in the transition
                    List of indexes of the terms involved in the transition
                      (yes, it is here twice, not sure why)
                    Einstein coefficient for espontaneous emission [s^-1]
                    Number of frequencies to discretize the frequencies for
                      this line
                    Number of those frequencies to discretize just the core
                    Number of Doppler widths for half the range of this line
                    Number of those Doppler widths just for the core
                    JS: Instance of Racah algebra class already initialized
                        (passed to take advantage of memoization)
                 multilevel: a list with:
                    Upper level orbital angular momentum
                    Lower level orbital angular momentum
                    spin angular momentum
                    Upper level total angular momentum
                    Lower level total angular momentum
                    Upper level energy [cm^-1]
                    Lower level energy [cm^-1]
                    Einstein coefficient for espontaneous emission [s^-1]
                    Number of frequencies to discretize the frequencies for
                      this line
                    Number of those frequencies to discretize just the core
                    Number of Doppler widths for half the range of this line
                    Number of those Doppler widths just for the core
            Output: None
            Internal:
              self.multiterm: Identifies the line as one in a multi-term atom
              Depending on the type of model
                multiterm:
                  self.terms: Stores the terms of the atom (again?)
                  self.jlju: Indexes of the involved terms
                  self.dL: Jump in orbitan momentum
                multilevel:
                  self.dJ: Jump in total momentum
                common:
                  self.nw: Number of frequencies to discretize the
                           frequencies for this line
                  self.nwc: Number of those frequencies to discretize just
                            the core
                  self.dw: Number of Doppler widths for half the range of
                           this line
                  self.dwc: Number of those Doppler widths just for the core
                  self.gl: degeneracy lower term
                  self.gu: degeneracy upper term
                  self.wavelength: Resonance wavelength [cm]
                  self.energy: Resonance energy [erg]
                  self.nu: Resonance frequency [Hz]
                  self.nu3: Resonance frequency to the power of three [Hz^3]
                  self.A_ul: Einstein coefficient for spontaneous emission [s^-1]
                  self.B_ul: Einstein coefficient for stimulated emission
                             [erg^-1 cm^2 Hz sr]
                  self.B_lu: Einstein coefficient for absorption [erg^-1 cm^2 Hz sr]
                  self.especial: Bool that tells if this line is to be split in
                                 components
                  self.prof: Space to store later the profiles to integrate Jqq'
                  self.jqq: Jqq' integrated over the line profile
            Note: self.prof is an array if self.especial is False, but is a
                  dictionary with the arrays identified by a resonance if
                  self.especial is True
                  Likewise, self.jqq is a list of lists (both size 3) or a
                  dictionary with this list of lists identified by a resonance
                  is self.especial is True
        """

        # Distinguish
        self.multiterm = multiterm

        # If multi-term
        if self.multiterm:

            # Get data
            terms, line_terms, lllu, Aul, nw, nwc, dw, dwc = data

            # Get indexes of involved terms
            self.terms = line_terms
            self.jlju = lllu

            # Store line parameters
            self.nw = nw
            self.nwc = nwc
            self.dw = dw
            self.dwc = dwc

            # Get degeneracies
            self.gl = terms[line_terms[0]].g
            self.gu = terms[line_terms[1]].g

            # Compute resonance
            self.wavelength = 1/(terms[line_terms[1]].TE - terms[line_terms[0]].TE)
            self.energy = c.h * c.c / self.wavelength
            self.nu = self.energy/c.h
            self.nu3 = self.nu*self.nu*self.nu

            # Compute Einstein coefficients
            self.A_ul = Aul
            self.B_ul = Aul * (c.c*c.c/(2*c.h*self.nu3))
            self.B_lu = self.B_ul * (terms[line_terms[1]].g/terms[line_terms[0]].g)

            # Get L jump
            self.dL = terms[line_terms[1]].L - terms[line_terms[0]].L

            # Especial treatment
            self.especial = False

            # Define profile
            self.prof = None

            # Defining the Jqq as nested dictionaries'
            self.jqq = {}
            for qq in [-1, 0, 1]:
                self.jqq[qq] = {}
                for qp in [-1, 0, 1]:
                    self.jqq[qq][qp] = 0.0

        # If multi-level
        else:

            Lu, Ll, S, Ju, Jl, Eu, El, Alu, nw, nwc, dw, dwc = data

            self.nw = nw
            self.nwc = nwc
            self.dw = dw
            self.dwc = dwc

            self.gl = 2.*Jl + 1.
            self.gu = 2.*Ju + 1

            self.wavelength = 1/(Eu - El)
            self.energy = c.h * c.c / self.wavelength
            self.nu = self.energy/c.h
            self.nu3 = self.nu*self.nu*self.nu

            W6 = JS.j6(Lu,Ll,1,Jl,Ju,S)
            W6 *= W6

            self.A_lu = Alu*(2.*Lu + 1.)*(2.*Jl + 1.)*W6
            self.A_ul = Alu
            self.B_lu = Alu * (c.c*c.c/(2*c.h*self.nu3))
            self.B_ul = self.B_lu * ((2.*Ju + 1.)/(2.*Jl + 1.))

            self.dJ = Ju - Jl

            self.especial = False

            # Define profile
            self.prof = None

            # Defining the Jqq as nested dictionaries'
            self.jqq = {}
            for qq in [-1, 0, 1]:
                self.jqq[qq] = {}
                for qp in [-1, 0, 1]:
                    self.jqq[qq][qp] = 0.0

################################################################################

    def initialize_profiles_first(self, nus_N):
        """ Initialize to zero the line profiles the first time
            Input: number of wavelengths
            Output: None
            Internal: Makes self.prof equal to zero (or the components
                      of the dictionary is self.especial)
        """

        # If special Helium case
        if self.especial:
            # For each component
            for comp in self.prof:
                self.prof[comp] = np.zeros((nus_N))
        # Usual multi-term
        else:
            self.prof = np.zeros((nus_N))

################################################################################

    def initialize_profiles(self):
        """ Initialize to zero the line profiles
            Input: None
            Output: None
            Internal: Makes self.prof equal to zero (or the components
                      of the dictionary is self.especial)
        """

        # If special Helium case
        if self.especial:
            # For each component
            for comp in self.prof:
                self.prof[comp] *= 0.0
        # Usual multi-term
        else:
            self.prof *= 0.0

################################################################################

    def add_contribution_profiles(self, contr, nu0=None):
        """ Add Contritubion to the line profile
            Input:
              Contribution: Profile contribution already multiplied
                            by the frequency weights
              nu0: If self.especial, the contribution is added to the
                   closest component. This frequency is in [Hz]
        """

        # If special Helium case
        if self.especial:

            # Check nu0
            if nu0 is None:
                print('If line distinguishes components, it is necessary ' + \
                      'to include the nu0 argument to add_contribution_profiles()')
                sys.exit()

            # Get difference between components and resonance
            dd = np.absolute(self.resos - nu0 - self.nu)

            # Get component closest to this resonance
            comp = self.resos[np.argmin(dd)]

            # Add contribution
            self.prof[comp] += contr

        # Usual contribution
        else:

            self.prof += contr

################################################################################

    def normalize_profiles(self):
        """ Normalizes the profiles to its integral
            Input: None
            Output: None
            Internal: Normalizes self.prof. The contributions must have
                      been previously added.
        """

        # If special Helium case
        if self.especial:
            for comp in self.prof:
                self.prof[comp] /= self.prof[comp].sum()
        # Usual multi-term
        else:
            self.prof /= self.prof.sum()

################################################################################

    def reset_radiation(self):
        """ Initialize to zero the profile integrated jqq
            Input: None
            Output: None
            Internal: self.jqq is set to zero
        """

        # If special Helium case
        if self.especial:
            for comp in self.jqq:
                for qq in [-1, 0, 1]:
                    self.jqq[comp][qq] = {}
                    for qp in [-1, 0, 1]:
                        self.jqq[comp][qq][qp] = (0. + 0j)
        # Usual multi-term
        else:
            for qq in [-1, 0, 1]:
                self.jqq[qq] = {}
                for qp in [-1, 0, 1]:
                    self.jqq[qq][qp] = (0. + 0j)

################################################################################

    def sumStokes(self, ray, Tqq, stokes, nus_weights):
        """ Method to add a contribution to the Jqq 
            Input:
              ray: Object with the propagation direction information
              Tqq: Geometrical tensors in the pertinent reference frame
              stokes: Stokes spectrum for the point and propagation direction
              nus_weights: Frequency weights for the integral
            Output: None
            Internal: self.jqq gets updated with the contribution for this
                      direction
            Note: This routine manages the call of the actual routine
                  adding the contribution, so we can easily account for the
                  self.especial case
        """

        # If special Helium case
        if self.especial:

            # For each component
            for comp in self.jqq:

                # Call the actual sum
                self.jqq[comp] = self.actually_sumStokes(self.jqq[comp], \
                                                         Tqq, \
                                                         self.prof[comp], \
                                                         ray,stokes,nus_weights)

        # Usual multi-term
        else:

            # Call the actual sum
            self.jqq = self.actually_sumStokes(self.jqq, Tqq, self.prof, \
                                               ray,stokes,nus_weights)

################################################################################

    def actually_sumStokes(self, jqq, Tqq, prof, ray, stokes, nus_weights):
        """ Method to add Jqq contribution, but seriously now
            Input:
              jqq: List of lists for the line integrated Jqq'
              ray: Object with the propagation direction information
              Tqq: Geometrical tensors in the pertinent reference frame
              prof: Normalized profiles to integrate stokes over
              stokes: Stokes spectrum for the point and propagation direction
              nus_weights: Frequency weights for the integral
            Output: frequency integrated contribution to the Jqq' for this
                    propagation direction
            Note: We only compute Jqq' for q'>=q
        """

        # For each Stokes parameter
        for i in range(4):

            # Get profile
            contr = (stokes[i]*prof*ray.weight).sum()

            # If no contribution, skip
            if np.absolute(contr) <= 0.:
                continue

            # For each q and q'
            for qq in range(-1,2):
                for qp in range(qq,2):

                    jqq[qq][qp] += contr*Tqq[i][f'{qq}{qp}']

        return jqq

################################################################################

    def fill_Jqq(self):
        """ Get the q q' components of Jqq' that have not been computed yet
            Input: None
            Output: None
            Internal: Update the self.jqq dictionary/list
            Note: This routine manages the call of the actual routine
                  computing the missing Jqq', so we can easily account
                  for the self.especial case
        """

        # If special Helium case
        if self.especial:

            # For each component
            for comp in self.jqq:

                # Call the actual sum
                self.jqq[comp] = self.actually_fill_Jqq(self.jqq[comp])

        # Usual multi-term
        else:

            # Call the actual sum
            self.jqq = self.actually_fill_Jqq(self.jqq)

################################################################################

    def actually_fill_Jqq(self,jqq):
        """ Get the q q' components of Jqq' that have not been computed yet,
            but seriously now
            Input:
              jqq: List of lists for the line integrated Jqq' (only q'>=q)
            Output: frequency integrated Jqq' with all qq' combinations
                    propagation direction
        """

        # For each q and q'
        for qq in range(-1,2):
            for qp in range(-1,qq):
                jqq[qq][qp] = np.conjugate(jqq[qp][qq])
        return jqq

################################################################################

    def reduce_Jqq(self,f):
        """ Reduce the Jqq components multiplying by f
            Input:
              f: factor to multiply the Jqq with
            Output: None
            Internal: Update the self.jqq after multiplying by this factor
            Note: This routine manages the call of the actual routine
                  applying the factor, so we can easily account for the
                  self.especial case
                  This is implemented because Hazel has it, and I needed
                  to properly compare between the two codes
        """

        # If special Helium case
        if self.especial:

            # For each component
            for comp in self.jqq:

                # Call the actual sum
                self.jqq[comp] = self.actually_reduce_Jqq(self.jqq[comp],f)

        # Usual multi-term
        else:

            # Call the actual sum
            self.jqq = self.actually_reduce_Jqq(self.jqq,f)

################################################################################

    def actually_reduce_Jqq(self,jqq,f):
        """ Reduce the Jqq components multiplying by f, but seriously now
            Input:
              jqq: List of lists for the line integrated Jqq'
              f: factor to multiply the Jqq with
            Output: Input multiplied by the f factor
        """

        # For each q and q'
        for qq in range(-1,2):
            for qp in range(-1,2):

                jqq[qq][qp] *= f

        return jqq

################################################################################

    def rotate_Jqq(self, DKQQ, JS):
        """ Method to rotate the Jqq for the line
        """

        # If special Helium case
        if self.especial:

            # For each component
            for comp in self.jqq:

                # Call the actual sum
                self.jqq[comp] = self.actually_rotate_Jqq(self.jqq[comp], DKQQ, JS)

        # Usual rotation
        else:

            # Call the actual sum
            self.jqq = self.actually_rotate_Jqq(self.jqq, DKQQ, JS)

################################################################################

    def actually_rotate_Jqq(self, jqq, DKQQ, JS):
        """ Method to rotate the Jqq for the line, but seriously now
        """

        #
        # Build JKQ tensors
        #

        # Create JKQ dictionary
        JKQ = {}

        # For each multipole
        for K in range(3):

            # Initialize this multipole
            JKQ[K] = {}

            # Compute positive Q
            for Q in range(0,K+1):

                # Initialize
                JKQ[K][Q] = 0.

                # Factor for K
                f1 = np.sqrt(3.*(2.*K + 1.))

                # For each q and q'
                for qq in range(-1,2):

                    # Factor for q
                    f2 = f1*JS.sign(1+qq)

                    # Only contribution from q' from 3J symbol
                    qp = qq - Q

                    # Control qp
                    if np.absolute(qp) > 1:
                        continue

                    # Add contribution for qq'
                    JKQ[K][Q] += jqq[qq][qp]*f2*JS.j3(1.,1.,K,qq,-qp,-Q)

            # Compute negative Q
            if K > 0:
                JKQ[K][-1] = -1.0*np.conjugate(JKQ[K][1])
            if K > 1:
                JKQ[K][-2] =      np.conjugate(JKQ[K][2])

        #
        # Rotate JKQ
        #

        # Initialize rotated JKQ tensor
        JKQ_n = {}

        # K == 0 does not change
        JKQ_n[0] = JKQ[0]

        # For each multipole
        for K in range(1,3):

            # Initialize this multipole
            JKQ_n[K] = {}

            # Compute positive Q
            for Q in range(0,K+1):

                # Initialize component
                JKQ_n[K][Q] = 0.

                # For Q' in old JKQ
                for Qp in range(-K,K+1):

                    # Add contribution
                    JKQ_n[K][Q] += DKQQ[K][Q][Qp]*JKQ[K][Qp]

            # Compute negative Q
            if K > 0:
                JKQ_n[K][-1] = -1.0*np.conjugate(JKQ_n[K][1])
            if K > 1:
                JKQ_n[K][-2] =      np.conjugate(JKQ_n[K][2])

        #
        # Get the new q q' back
        #

        jqq = {}

        # For each q
        for qq in range(-1,2):

            # Factor for q
            f1 = JS.sign(1+qq)

            # Initialize jqq
            jqq[qq] = {}

            # For each q'
            for qp in range(-1,2):

                # Initialize
                jqq[qq][qp] = 0.

                # For each K
                for K in range(3):

                    # K factor
                    f2 = f1*np.sqrt((2.*K+1.)/3.)

                    # Get Q from 3J
                    Q = qq - qp

                    # Control Q
                    if np.absolute(Q) > K:
                        continue

                    # Add contribution
                    jqq[qq][qp] += f2*JS.j3(1,1,K,qq,-qp,-Q)*JKQ_n[K][Q]

        # And back
        return jqq

################################################################################

    def get_Jqq(self, nu0=None):
        """ Get the proper Jqq back
        """

        # If special Helium case
        if self.especial:

            # Get difference between components and resonance
            dd = np.absolute(self.resos - nu0)

            # Get component closest to this resonance
            comp = self.resos[np.argmin(dd)]

            # Add contribution
            return self.jqq[comp]

        # Usual contribution
        else:

            return self.jqq

################################################################################
################################################################################
################################################################################

class HeI_1083_2Ltest():
    """ Class to acces the atomic model
    """

    def __init__(self,JS):

        # Set as multiterm atom
        self.multiterm = False

        # MNIST data for HeI levels (2 level atom), energy cm^-1
        levels = [level(169_086.8428979, 1, 3),
                  level(159_855.9743297, 0, 1)]

        indx = np.argsort([lev.E.value for lev in levels])
        self.levels = []
        for i, ord in enumerate(indx):
            self.levels.append(levels[ord])

        # Line inverse lifetime s^-1
        self.lines = [line(self.multiterm,[self.levels, (0, 1), (0, 1), 1.0216e+07, \
                                           125,85,30.,5.],JS)]

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

################################################################################
################################################################################
################################################################################

class HeI_1083():
    """ Class to acces the atomic model. 2-term atom Helium 10830
    """
    def __init__(self,JS,B=0.0,especial=False):

        # Set as multiterm atom
        self.multiterm = True

        # Multi-term helium 10830 atom
        self.terms = []

        # Internal switch
        not_twoterm = False

        # Add the two terms. Energy cm^-1
        self.terms.append(term_class(0.0,1.0,      [1.], \
                                    [159855.9743297],JS,B,0))
        self.terms.append(term_class(1.0,1.0,[2.,1.,0.], \
                                    [169086.7664725, \
                                     169086.8428979, \
                                     169087.8308131],JS,B, \
                                     self.terms[-1].NN_next))
        if not_twoterm:
            self.terms.append(term_class(0.0,1.0,      [1.], \
                                     [183236.79170],JS,B, \
                                     self.terms[-1].NN_next))
            self.terms.append(term_class(1.0,1.0,[2.,1.,0.], \
                                     [185564.561920, \
                                      185564.583895, \
                                      185564.854540],JS,B, \
                                      self.terms[-1].NN_next))
            self.terms.append(term_class(2.0,1.0,[3.,2.,1.], \
                                     [186101.5461767, \
                                      186101.5486891, \
                                      186101.5928903],JS,B, \
                                      self.terms[-1].NN_next))
        # Get dimension of rho vector
        self.ndim = self.terms[-1].NN_next

        # Add 10830 line. Inverse lifetime in s^-1
        self.lines = []
        self.lines.append(line_class(self.multiterm,[self.terms, (0, 1), (0, 1), \
                                     1.0216e+07, \
                                     125, 55, 15., 2.5], JS))
#                                     75, 45, 15., 2.5], JS))
#                                    15,  5, 15., 2.5], JS))
        if not_twoterm:
            self.lines.append(line_class(self.multiterm,[self.terms, (0, 3), (0, 3), \
                                                     9.4746e+06, \
                                                     35, 15, 15., 2.5], JS))
#                                                    55, 35, 15., 2.5], JS))
            self.lines.append(line_class(self.multiterm,[self.terms, (1, 2), (1, 2), \
                                                     2.78532e7, \
                                                     35, 15, 15., 2.5], JS))
#                                                    55, 35, 15., 2.5], JS))
            self.lines.append(line_class(self.multiterm,[self.terms, (1, 4), (1, 4), \
                                                     7.0702687e7, \
                                                     35, 15, 15., 2.5], JS))
#                                                    55, 35, 15., 2.5], JS))
            self.reduction_f = [1.,0.2,1.,1.,1.]

        # Set mass
        self.mass = 4.002602 * c.amu

        # Initialize forbidden density matrix elements
        self.forbidden = []

        # If especial Helium case
        if especial:

            # Flag the line as especial
            self.lines[0].especial = True

            # Get energies for red and blue components
            red = 0.5*(self.terms[1].LE[0] + self.terms[1].LE[1]) - \
                  self.terms[0].LE[0]
            blue = self.terms[1].LE[2] - self.terms[0].LE[0]
            red *= c.c
            blue *= c.c
            self.lines[0].resos = [red,blue]

            # Define profiles
            self.lines[0].prof = {}
            for reso in self.lines[0].resos:
                self.lines[0].prof[reso] = None

            self.lines[0].resos = np.array(self.lines[0].resos)

            # Defining the Jqq as nested dictionaries'
            self.lines[0].jqq = {red: {}, blue: {}}
            for comp in self.lines[0].jqq:
                for qq in [-1, 0, 1]:
                    self.lines[0].jqq[comp][qq] = {}
                    for qp in [-1, 0, 1]:
                        self.lines[0].jqq[comp][qq][qp] = 0.0

            # For the upper term
            term = self.terms[1]

            # Jump in energy to cover mid distance between components
            GAM = 0.5*np.absolute(0.5*(self.terms[1].LE[0] + \
                                       self.terms[1].LE[1]) - \
                                  self.terms[1].LE[2])

            # Go by every density matrix element
            for index in term.index:

                # If not diagonal
                if index[1] != index[2]:

                    # Get energies
                    E1 = term.eigval[index[1]]
                    E2 = term.eigval[index[2]]

                    # If too far in energy, forbid it
                    if np.absolute(E1 - E2) > GAM:
                        self.forbidden.append(index[0])

################################################################################

    def initialize_profiles(self, nus_N):
        """ Method to initialize vectors in atom profiles
        """

        # For each line
        for line in self.lines:
            # Reset profile
            line.initialize_profiles_first(nus_N)

################################################################################

    def reset_jqq(self, nus_N):
        """ Method to reset Jqq and profiles for atom
        """

        # For each line
        for line in self.lines:
            # Reset Jqq
            line.reset_radiation()
            # Reset profile
            line.initialize_profiles()

################################################################################

    def sumStokes(self, ray, Tqq, stokes, nus_weights):
        """ Method to add Jqq contribution
        """

        # For each line
        for line in self.lines:
            line.sumStokes(ray, Tqq, stokes, nus_weights)

################################################################################

    def fill_Jqq(self):
        """ Method to fill the missing Jqq
        """

        # For each line
        for line in self.lines:
            line.fill_Jqq()

################################################################################

    def reduce_Jqq(self,redf):
        """ Call the method on the lines
        """

        # For each reduction factor and line
        for f,line in zip(redf,self.lines):
            line.reduce_Jqq(f)

################################################################################

    def rotate_Jqq(self, DKQQ, JS):
        """ Method to rotate the Jqq
        """

        # For each line
        for line in self.lines:
            line.rotate_Jqq(DKQQ, JS)

################################################################################
################################################################################
################################################################################

class ESE:
    """ A class that stores the atomic state and needs to be constantly updated
        during the Lambda iterations by providing the Stokes parameters.
        After every Lambda iteration, solveESE() needs to be called.
        It is assumed that only one spectral line is involved in the problem.
        This class needs to be instantiated at every grid point.
    """

    def __init__(self, v_dop, a_voigt, B, T, jsim, equilibrium=False, iz=0, especial=True):
        """ Initialize ESE class instance
            v_dop: Atom's Doppler width
            a_voigt: Damping parameter
            B: object of the magnetic field vector with in polar coordinates (gauss,rad,rad)
            T: Temperature
            equilibrium: Decices the initialization
            return value: None
        """

        # Flag for debugging prints in the SEE
        self.debug = False

        # Keep here the magnetic field vector [G]
        self.B = B[0]
        self.theta = B[1]
        self.phi = B[2]

        # Store height index for debugging
        self.iz = iz

        # If magnetic field > 0, get rotation matrix for JKQ
        if self.B > 0 and np.absolute(B[1]) > 1e-8:

            # Get rotation matrixes
            self.calc_DKQQ(B[1],B[2])
            self.rotate = True
            self.Tqq = {}

        else:

            # No need of rotation matrices
            self.DKQQ = None
            self.rotate = False

        # Initialize atom
        self.atom = HeI_1083(jsim, B=self.B, especial=especial)

        # If multi-term atom
        if self.atom.multiterm:

            # Allocate density matrix
            self.rho = np.zeros((self.atom.ndim))

            # Allocate SEE
            self.ESE_indep = np.zeros((self.atom.ndim, self.atom.ndim))

            # LTE
            if equilibrium:

                # Debug
                if self.debug:
                    print('Initializing Multi-term atom in equilibrium')

                # Initialize total population
                populations = 0

                # Run over terms
                for term in self.atom.terms:

                    # Run over elements
                    for index in term.index:

                        # Debug
                        if self.debug:
                            print('Index',index)

                        # If diagonal
                        if self.isdiag(index):

                            # Get energy
                            E = term.eigval[index[1]]

                            # Debug
                            if self.debug:
                                print('DIAGONAL',E)

                            # Add numerator LTE
                            self.rho[index[0]] = \
                                np.exp(-c.h*c.c*E/c.k_B/T) / term.g

                            # Add to normalize later
                            populations += self.rho[index[0]]

                # Normalize
                self.rho = self.rho/populations

                # Debug
                if self.debug:
                    print('rho')
                    print(self.rho)

            # Full ground term
            else:

                # Debug
                if self.debug:
                    print('Initializing Multi-term atom to ground term')

                # Run over indexes in first term
                for index in self.atom.terms[0].index:

                    # Debug
                    if self.debug:
                        print('Index',index)

                    # If diagonal
                    if self.isdiag(index):

                        # Debug
                        if self.debug:
                            print('DIAGONAL')

                        self.rho[index[0]] = 1.

                # Normalize
                self.rho /= self.rho.sum()

                # Debug
                if self.debug:
                    print('rho')
                    print(self.rho)

            # Point the solver function to multi-term
            self.solveESE = self.solveESE_MT

        # If multi-level atom
        else:

            # Allocate density matrix
            self.rho = np.zeros(len(self.atom.dens_elmnt)).astype('complex128')

            # Initialize populations
            populations = 0

            # Initialize in LTE
            if equilibrium:

                # For each level
                for i, lev in enumerate(self.atom.dens_elmnt):

                    # Get quantum numbers
                    Ll = lev[0]
                    Ml = lev[-2]
                    Mlp = lev[-1]

                    # If population (diagonal)
                    if Mlp == Ml:

                        # Get LTE numerator
                        self.rho[i] = np.exp(-c.h*c.c* \
                                             self.atom.levels[Ll].E/c.k_B/T) / \
                                      len(self.atom.levels[Ll].M)

                        # Add to populations to normalize later
                        populations += self.rho[i]

                # Normalize
                self.rho = self.rho/populations

            # Full ground term
            else:

                # Ad-hoc for just one magnetic component
                self.rho[0] = 1

            # Point solver to multi-level
            self.solveESE = self.solveESE_ML

            # Define other variables
            self.N_rho = len(self.rho)
            self.coherences = self.N_rho - self.populations
            self.ESE_indep = np.zeros((self.N_rho, self.N_rho))

################################################################################

    def initialize_profiles(self, nus_N):
        """ Just calls the same method for the atom
        """
        self.atom.initialize_profiles(nus_N)

################################################################################

    def reset_jqq(self, nus_N):
        """ Just calls the same method for the atom
        """
        self.atom.reset_jqq(nus_N)

################################################################################

    def sumStokes(self, ray, stokes, nus_weights, jsim):
        """ Just calls the same method for the atom
        """
        # If need to rotate
        if self.rotate:
            self.atom.sumStokes(ray, self.get_Tqq(ray,jsim), stokes, nus_weights)
        else:
            self.atom.sumStokes(ray, ray.Tqq, stokes, nus_weights)

################################################################################

    def fill_Jqq(self):
        """ Just calls the same method for the atom
        """

        # For each line
        self.atom.fill_Jqq()

################################################################################

    def reduce_Jqq(self,redf):
        """ Just calls the same method for the atom
        """

        # For each line
        self.atom.reduce_Jqq(redf)

################################################################################

    def rotate_Jqq(self, JS):
        """ Just calls the same method for the atom
        """

        # For each line
        self.atom.rotate_Jqq(self.DKQQ, JS)

################################################################################

    def isdiag(self, index):
        """ Returns true if the rho element is diagonal in mu and M
            in a multi-term atom
        """
        return (index[1] == index[2])

################################################################################

    def rho_call(self, lev, JJ, M, Mp):
        """ Return the rho element (ML) for a given combination of quantum
            numbers
        """
        index = self.atom.dens_elmnt.index([lev, self.atom.levels[lev].E, JJ, M, Mp])
        return self.rho[index]

################################################################################

    def calc_DKQQ(self, theta, chi, conj=False, back=False):
        """ Compute DKQQ rotation matrix from B vector
        """

        # Get instance of rotation class
        rota = rotate_ist()

        # Get rotation matrix
        self.DKQQ = rota.get_DKQQ(2,theta,chi,conj,back)

################################################################################

    def get_Tqq(self,ray,jsim):
        """ Compute Tqq in the magnetic reference frame from the vertical ones
        """

        # Try to get precomputed Tqq
        tag = f'{ray.inc}:{ray.az}'
        try:
            return self.Tqq[tag]
        except KeyError:
           #print('Vertical')
           #print(ray.Tqq)
            self.Tqq[tag] = self.rotate_Tqq(ray.Tqq,jsim)
           #print('Magn')
           #print(self.Tqq)
           #sys.exit()
            return self.Tqq[tag]
        except:
            raise

################################################################################

    def rotate_Tqq(self,Tqq,jsim):
        """ Rotates the Tqq tensors from vertical to magnetic
        """

        Tqqn = [{'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}]

        # Generate TKQ
        TKQ = [ [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]] ]
        nTKQ = [[[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]], \
                [[0.],[0.,0.,0.],[0.,0.,0.,0.,0.]] ]

        # For each K
        for K in range(3):

            # For each Q
            for iQ,Q in enumerate(range(-K,K+1)):

                # For each q
                for q in range(-1,2):

                    # Valid qp
                    qp = q - Q
                    if np.absolute(qp) > 1.:
                        continue

                    # Factor
                    ff = jsim.sign(1.+q)*np.sqrt(3.*(2.*K+1.))*jsim.j3(1.,1.,K,q,-qp,-Q)

                    # For each Stokes
                    for i in range(4):
                        TKQ[i][K][iQ] += ff*Tqq[i][f'{q}{qp}'] 

        # Rotate TKQ

        # For each Stokes
        for i in range(4):

            # K = 0 does not rotate
            nTKQ[i][0] = TKQ[i][0]

            # For each multipole
            for K in range(1,3):

                # Negative Q
                for Q in range(-K,1):

                    # Q index
                    iQ = Q + K

                    # For each Q'
                    for Qp in range(-K,K+1):

                        # T idex
                        iQp = Qp + K

                        nTKQ[i][K][iQ] += TKQ[i][K][iQp]*self.DKQQ[K][Q][Qp]

            # Positive Q
            nTKQ[i][1][2] = -1.*np.conjugate(nTKQ[i][1][0])
            nTKQ[i][2][3] = -1.*np.conjugate(nTKQ[i][2][1])
            nTKQ[i][2][4] =     np.conjugate(nTKQ[i][2][0])


        # For each q
        for q in range(-1,2):

            # For each qp
            for qp in range(q,2):

                # For each K
                for K in range(3):

                    # Valid Q
                    Q = q - qp
                    if np.absolute(Q) > K:
                        continue
                    iQ = Q + K

                    # Factor
                    ff = jsim.sign(1.+q)*np.sqrt((2.*K+1.)/3.)*jsim.j3(1.,1.,K,q,-qp,-Q)

                    # For each Stokes parameter
                    for i in range(4):
                        Tqqn[i][f'{q}{qp}'] += ff*nTKQ[i][K][iQ]

        # Do symmetrics
        for q in range(-1,2):
            for qp in range(-1,q):
                # For each stokes
                for i in range(4):
                    Tqqn[i][f'{q}{qp}'] = np.conjugate(Tqqn[i][f'{qp}{q}'])

        # And come back
        return Tqqn

################################################################################

    def rotate_Tqq_doubtful(self,Tqq):
        """ Rotates the Tqq tensors from vertical to magnetic
        """

        Tqqn = [{'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}, \
                {'-1-1':0j,'-10':0j,'-11':0j,'00':0j,'01':0j,'11':0j}]

        # For each q
        for q in range(-1,2):
            # For each qp
            for qp in range(q,2):
                tag = f'{q}{qp}'
                # For each p
                for p in range(-1,2):
                    # For each pp
                    for pp in range(-1,2):
                        DP = self.DKQQ[1][p][q]*np.conjugate(self.DKQQ[1][pp][qp])
                        # For each Stokes
                        for i in range(4):
                            Tqqn[i][tag] += DP*Tqq[i][f'{p}{pp}']
        # Do symmetrics
        for q in range(-1,2):
            for qp in range(-1,q):
                # For each stokes
                for i in range(4):
                    Tqqn[i][f'{q}{qp}'] = np.conjugate(Tqqn[i][f'{qp}{q}'])

        # And come back
        return Tqqn

################################################################################

    def solveESE_ML(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            Solves the SEE for a multi-level atom in standard representation
            Eqs. 7.5 in LL04
            return value: maximum relative change of the level population
        """

        # Shortcut to J symbols
        JS = cdt.JS

        # Fill the missing Jqq
        self.fill_Jqq()

        # Get magnetic field strength
        Bnorm = self.B

        # If there is magnetic field, need to rotate Jqq
        if self.rotate:
            self.rotate_Jqq(JS)

        # Initialize to zero the coefficients
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

                # Get magnetic contribution
                nu_L = c.nu_L*Bnorm
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

                    # Add contribution
                    if imag:
                        self.ESE_indep[i][j] -= q*gamma
                    else:
                        self.ESE_indep[i][j] += q*gamma

        # For each radiative transition
        for line in self.atom.lines:

            # Get levels
            ll = line.levels[0]
            lu = line.levels[1]

            # Get Jqq
            Jqq = line.get_Jqq()

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

                    #
                    # Get transfer rates
                    #

                    # Absorption transfer
                    TA  = TA_ML(self,Ju,Mu,Mup,Jl,Ml,Mlp,Jqq,line,JS)
                    if Mlp != Ml:
                        TAp = TA_ML(self,Ju,Mu,Mup,Jl,Mlp,Ml,Jqq,line,JS)
                    # Stimulated emission transfer
                    TS  = TS_ML(self,Jl,Ml,Mlp,Ju,Mu,Mup,Jqq,line,JS)
                    if Mup != Mu:
                        TSp = TS_ML(self,Jl,Ml,Mlp,Ju,Mup,Mu,Jqq,line,JS)
                    # Espontaneous emission transfer
                    TE  = TE_ML(self,Jl,Ml,Mlp,Ju,Mu,Mup,line,JS)
                    if Mup != Mu:
                        TEp = TE_ML(self,Jl,Ml,Mlp,Ju,Mup,Mu,line,JS)

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

                            # Debug
                            if self.debug:
                                f.write(f'TA {i} {j}  {TA.real-TAp.real}\n')
                                f.write(f'TS {j} {i}  {TS.real-TSp.real}\n')
                                f.write(f'TE {j} {i}  {TE.real-TEp.real}\n')

                        # Real lower level (row TS, TE | column TA)
                        else:

                            self.ESE_indep[i][j] += TA.imag
                            if Ml != Mlp:
                                self.ESE_indep[i][j] += TAp.imag

                            # Debug
                            if self.debug:
                                if Ml != Mlp:
                                    f.write(f'TA {i} {j}  {TA.imag+TAp.imag}\n')
                                else:
                                    f.write(f'TA {i} {j}  {TA.imag}\n')

                            self.ESE_indep[j][i] += TSp.imag - TS.imag

                            # Debug
                            if self.debug:
                                f.write(f'TS {j} {i}  {TSp.imag-TS.imag}\n')

                    # Real upper level (row TA | columns TS, TE)
                    else:

                        # Imaginary lower level (row TS, TE | column TA)
                        if imagl:

                            self.ESE_indep[i][j] += TAp.imag - TA.imag

                            # Debug
                            if debug:
                                f.write(f'TA {i} {j}  {TAp.imag-TA.imag}\n')

                            self.ESE_indep[j][i] += TS.imag
                            if Mu != Mup:
                                self.ESE_indep[j][i] += TSp.imag

                            # Debug
                            if self.debug:
                                if Mu != Mup:
                                    f.write(f'TS {j} {i}  {TS.imag+Tsp.imag}\n')
                                else:
                                    f.write(f'TS {j} {i}  {TS.imag}\n')

                        # Real lower level (row TS, TE | column TA)
                        else:

                            self.ESE_indep[i][j] += TA.real
                            if Ml != Mlp:
                                self.ESE_indep[i][j] += TAp.real

                            # Debug
                            if self.debug:
                                if Ml != Mlp:
                                    f.write(f'TA {i} {j}  {TA.real+TAp.real}\n')
                                else:
                                    f.write(f'TA {i} {j}  {TA.real}\n')

                            self.ESE_indep[j][i] += TS.real
                            self.ESE_indep[j][i] += TE.real
                            if Mu != Mup:
                                self.ESE_indep[j][i] += TSp.real
                                self.ESE_indep[j][i] += TEp.real

                            # Debug
                            if self.debug:
                                if Mu != Mup:
                                    f.write(f'TS {j} {i}  {TS.real+TSp.real}\n')
                                    f.write(f'TE {j} {i}  {TE.real+TEp.real}\n')
                                else:
                                    f.write(f'TS {j} {i}  {TS.real}\n')
                                    f.write(f'TE {j} {i}  {TE.real}\n')

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
                                RA = RA_ML(self,Jl,Ml,Mlp,Ju,Mu,Jqq,line,JS)

                                # Diagonal in Ml
                                if (Mk == Ml and Mkp == Ml):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                                # Diagonal in Mlp
                                if (Mk == Mlp and Mkp == Mlp):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                            #
                            # Not diagonal
                            #

                            # M'' < M sum
                            if Mk < Ml and Mkp == Ml:

                                # Get rate
                                RA = RA_ML(self,Jl,Mk,Mlp,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                            # M'' > M sum
                            if Mkp > Ml and Mk == Ml:

                                # Get rate
                                RA = RA_ML(self,Jl,Mkp,Mlp,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                            # M'' < M' sum
                            if Mk < Mlp and Mkp == Mlp:

                                # Get rate
                                RA = RA_ML(self,Jl,Mk,Ml,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')

                            # M'' > M sum
                            if Mkp > Mlp and Mk == Mlp:

                                # Get rate
                                RA = RA_ML(self,Jl,Mkp,Ml,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA {j} {k}  {-RA.real}\n')


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
                                RS = RS_ML(self,Ju,Mu,Mup,Jl,Ml,Jqq,line,JS)

                                # If diagonal Mu
                                if (Mk == Mu and Mkp == Mu):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.imag}\n')

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_ML(self,Ju,Mu,Mup,Jl,Ml,line)

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                                # If diagonal Mup
                                if (Mk == Mup and Mkp == Mup):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.imag}\n')

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_ML(self,Ju,Mu,Mup,Jl,Ml,line)

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                            #
                            # Not diagonal
                            #

                            # M'' < M sum
                            if Mk < Mu and Mkp == Mu:

                                # Get rate
                                RS = RS_ML(self,Ju,Mk,Mup,Jl,Ml,Jqq,line,JS)
                                RE = RE_ML(self,Ju,Mk,Mup,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.real}\n')
                                            f.write(f'RE {i} {k}  {RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                            # M'' > M sum
                            if Mkp > Mu and Mk == Mu:

                                # Get rate
                                RS = RS_ML(self,Ju,Mkp,Mup,Jl,Ml,Jqq,line,JS)
                                RE = RE_ML(self,Ju,Mkp,Mup,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                            # M'' < M' sum
                            if Mk < Mup and Mkp == Mup:

                                # Get rate
                                RS = RS_ML(self,Ju,Mk,Mu,Jl,Ml,Jqq,line,JS)
                                RE = RE_ML(self,Ju,Mk,Mu,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')

                            # M'' > M sum
                            if Mkp > Mup and Mk == Mup:

                                # Get rate
                                RS = RS_ML(self,Ju,Mkp,Mu,Jl,Ml,Jqq,line,JS)
                                RE = RE_ML(self,Ju,Mkp,Mu,Jl,Ml,line)

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.real}\n')
                                            f.write(f'RE {i} {k}  {RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS {i} {k}  {-RS.real}\n')
                                            f.write(f'RE {i} {k}  {-RE.real}\n')


        # Set independent term
        indep = np.zeros(self.N_rho)
        indep[0] = 1

        # Add mass conservation row
        for i, lev in enumerate(self.atom.dens_elmnt_indep):
            Ml = lev[2]
            Mlp = lev[3]
            if Mlp == Ml:
                self.ESE_indep[0, i] = 1
            else:
                self.ESE_indep[0, i] = 0

        # Solve the system of equations
        rho_n = linalg.solve(self.ESE_indep, indep)

        # Construct the full rho (complex)
        rho_comp = np.zeros_like(self.rho).astype('complex128')

        # Initialize indexes
        indexes = []

        # Initialize change populations and coherences
        change_p = -1e99
        change_c = -1e99

        # For each density matrix element (unrolled)
        for i, lev in enumerate(self.atom.dens_elmnt_indep):

            # Get quantum numbers data
            ll = lev[0]
            JJ = lev[1]
            M = lev[2]
            Mp = lev[3]
            imag = lev[4]

            # Get index in complex rho and append it in the list
            index = self.atom.dens_elmnt.index([ll, self.atom.levels[ll].E, JJ, M, Mp])
            indexes.append(index)

            # If real part
            if not imag:
                rho_comp[index] += rho_n[i]
                if np.absolute(M - Mp) < 0.25:
                    change_p = np.max([change_c,np.absolute(rho_comp[index] - self.rho[index])])
                else:
                    change_c = np.max([change_c,np.absolute(rho_comp[index] - self.rho[index])])
            # If imaginary part
            else:
                rho_comp[index] += 1j*rho_n[i]
                change_c = np.max([change_c,np.absolute(rho_comp[index] - self.rho[index])])


        # For the indexes stored from the last loop
        for index in indexes:

            # Get quantum numbers data
            lev = self.atom.dens_elmnt[index]
            ll = lev[0]
            JJ = lev[-3]
            M = lev[-2]
            Mp = lev[-1]

            # Find the "opposite" index
            op_index = self.atom.dens_elmnt.index([ll, self.atom.levels[ll].E, JJ, Mp, M])
            # And compute with the complex conjugate
            rho_comp[op_index] = np.conjugate(rho_comp[index])

        # Compute the change and store the new rho array
       #change = np.abs(rho_comp - self.rho)/np.abs((rho_comp + 1e-40))
        self.rho = rho_comp.copy()

        # Check for the populations to be > 0 and to be normaliced

        # Initialize sum
        suma = 0

        # For each density matrix element
        for i, lev in enumerate(self.atom.dens_elmnt):

            # Get element data
            Ll = lev[0]
            Ml = lev[-2]
            Mlp = lev[-1]
            JJ = lev[-3]

            # If diagonal
            if Mlp == Ml:

                # Add to sum
                suma += self.rho[i]

                # Check physical
                if self.rho[i] < 0:

                    print(f"Warning: Negative population of the level: " + \
                          f"L={Ll},J={JJ}, M={Ml},M'={Mlp}")
                    print(f"with population of rho={self.rho[i]}")
                    # input("Press Enter to continue...")

        if not 0.98 < suma < 1.02:
            print("Warning: Not normaliced populations in this itteration")
            print(f"With the sum of the populations rho = {suma}")
            # input("Press Enter to continue...")

        # Return MRC
        return change_p,change_c

################################################################################

    def solveESE_MT(self, rad, cdt):
        """
            Called at every grid point at the end of the Lambda iteration.
            Solves the SEE for a multi-term atom in standard representation
            Eqs. 7.29 in LL04
            return value: maximum relative change of the level population
        """

        # if self.iz == 0:
        #     print("saving_jqq")
        #     with open(f'{cdt.dir}jqq_solve_MT_{time.strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        #         pickle.dump(self.atom.lines[0].jqq, f)

        # Shortcut to J symbols
        JS = cdt.JS

        # Fill the missing Jqq
        self.fill_Jqq()

        # Reduce illumination ad-hoc as Hazel
        try:
            self.reduce_Jqq(self.atom.reduction_f)
        except AttributeError:
            pass
        except:
            raise

        # Debug
        if self.debug:

            # Open debug file
            f = open(f'debug_SEE_MT-{self.iz}','w')

            # Write Jqq pre-rotation
            f.write("J_qq'\n")
            for il,line in enumerate(self.atom.lines):
                f.write(f'Line {il} l0 {c.c*1e7/line.nu}\n')
                if line.especial:
                    for comp in line.resos:
                        f.write(f'Component l0 {c.c*1e7/comp}\n')
                        JKQ = {0: {0: 0.}, \
                               1: {-1:0.,0:0.,1:0.}, \
                               2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}}
                        for qq in range(-1,2):
                            for qp in range(-1,2):
                                f.write(f'    J_{qq:2d}{qp:2d} {line.jqq[comp][qq][qp]}\n')
                                for K in range(3):
                                    Q = qq - qp
                                    if np.absolute(Q) > K:
                                        continue
                                    JKQ[K][Q] += cdt.JS.sign(1 + qq)* \
                                                 np.sqrt(3.*(2.*K + 1.))* \
                                                 cdt.JS.j3(1.,1.,K,qq,-qp,-Q)* \
                                                 line.jqq[comp][qq][qp]
                        for K in range(0,3):
                            for Q in range(-K,K+1):
                                f.write(f'    J^{K:1d}{Q:2d} {JKQ[K][Q]}\n')
                else:
                    JKQ = {0: {0: 0.}, \
                           1: {-1:0.,0:0.,1:0.}, \
                           2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}}
                    for qq in range(-1,2):
                        for qp in range(-1,2):
                            f.write(f'    J_{qq:2d}{qp:2d} {line.jqq[qq][qp]}\n')

                            for K in range(3):
                                Q = qq - qp
                                if np.absolute(Q) > K:
                                    continue
                                JKQ[K][Q] += cdt.JS.sign(1 + qq)* \
                                             np.sqrt(3.*(2.*K + 1.))* \
                                             cdt.JS.j3(1.,1.,K,qq,-qp,-Q)* \
                                             line.jqq[qq][qp]
                    for K in range(0,3):
                        for Q in range(-K,K+1):
                            f.write(f'    J^{K:1d}{Q:2d} {JKQ[K][Q]}\n')

        # If there is magnetic field, need to rotate Jqq
        if self.rotate:

            # Rotate Jqq
            self.rotate_Jqq(JS)

            # Debug
            if self.debug:

                # Write Jqq post-rotation
                f.write("ROTATED J'\n")
                f.write("J_qq'\n")
                for il,line in enumerate(self.atom.lines):
                    f.write(f'Line {il} l0 {c.c*1e7/line.nu}\n')
                    if line.especial:
                        for comp in line.resos:
                            f.write(f'Component l0 {c.c*1e7/comp}\n')
                            JKQ = {0: {0: 0.}, \
                                   1: {-1:0.,0:0.,1:0.}, \
                                   2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}}
                            for qq in range(-1,2):
                                for qp in range(-1,2):
                                    f.write(f'    J_{qq:2d}{qp:2d} {line.jqq[comp][qq][qp]}\n')
                                    for K in range(3):
                                        Q = qq - qp
                                        if np.absolute(Q) > K:
                                            continue
                                        JKQ[K][Q] += cdt.JS.sign(1 + qq)* \
                                                     np.sqrt(3.*(2.*K + 1.))* \
                                                     cdt.JS.j3(1.,1.,K,qq,-qp,-Q)* \
                                                     line.jqq[comp][qq][qp]
                            for K in range(0,3):
                                for Q in range(-K,K+1):
                                    f.write(f'    J^{K:1d}{Q:2d} {JKQ[K][Q]}\n')
                    else:
                        JKQ = {0: {0: 0.}, \
                               1: {-1:0.,0:0.,1:0.}, \
                               2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}}
                        for qq in range(-1,2):
                            for qp in range(-1,2):
                                f.write(f'    J_{qq:2d}{qp:2d} {line.jqq[qq][qp]}\n')

                                for K in range(3):
                                    Q = qq - qp
                                    if np.absolute(Q) > K:
                                        continue
                                    JKQ[K][Q] += cdt.JS.sign(1 + qq)* \
                                                 np.sqrt(3.*(2.*K + 1.))* \
                                                 cdt.JS.j3(1.,1.,K,qq,-qp,-Q)* \
                                                 line.jqq[qq][qp]
                        for K in range(0,3):
                            for Q in range(-K,K+1):
                                f.write(f'    J^{K:1d}{Q:2d} {JKQ[K][Q]}\n')


        # Debug
        if self.debug:

            # If forbidden elements
            if len(self.atom.forbidden) > 0:

                f.write('\nForbidden elements:\n')

                # Write forbidden indexes
                for term in self.atom.terms:
                    for index in term.index:
                        if index[0] in self.atom.forbidden:
                            i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
                            f.write(f'[{i:2d}][{ii:2d},{ii1:2d}] ' + \
                                    f'rho({mu:2d},{M:3.1f};{mu1:2d},{M1:3.1f})\n')

                # Write Jqq pre-rotation
                f.write('\n')


        #
        # Initialize to zero and add trivial rows
        #

        # For each rho element
        for i in range(self.atom.ndim):

            self.ESE_indep[i] = np.zeros_like(self.ESE_indep[i])

            # If forbidden row, make it trivial solution
            if i in self.atom.forbidden:

                # rho(jM,j'M') = 0.
                self.ESE_indep[i][i] = 1.0

        #
        # Interference term (diagonal)
        #

        # For each atomic term
        for term in self.atom.terms:

            # Run over rho elements
            for index in term.index:

                # If forbidden, skip
                if index[0] in self.atom.forbidden:
                    continue

                # If diagonal, skip
                if self.isdiag(index):
                    continue

                # Get eigenvalues
                EE  = term.eigval[index[1]]
                EE1 = term.eigval[index[2]]

                # Get multiplicative term
                Gamma = 2.*np.pi*(EE - EE1)*c.c

                # If imaginary row
                if index[-1]:

                    self.ESE_indep[index[0]][index[0]-1] = -Gamma

                    # Debug
                    if self.debug:
                        f.write(f'MK {index[0]} {index[0]-1}  {-Gamma} ' + \
                                f'{index[1]} -> {EE} {index[2]} -> {EE1}\n')

                # If real row
                else:

                    self.ESE_indep[index[0]][index[0]+1] = Gamma

                    # Debug
                    if self.debug:
                        f.write(f'MK {index[0]} {index[0]+1}  {Gamma} ' + \
                                f'{index[1]} -> {EE} {index[2]} -> {EE1}\n')

        #
        # Radiative terms
        #

        # For each radiative transition
        for line in self.atom.lines:

            # Point to the upper and lower terms
            termu = self.atom.terms[line.terms[1]]
            terml = self.atom.terms[line.terms[0]]

            # If line is not especial, we can get the Jqq already
            if not line.especial:
                Jqq = line.get_Jqq()

            # Run over upper term indexes
            for indexu in termu.index:

                # Get variables in shorter form
                i,iu,iu1,iMu,Mu,muu,iMu1,Mu1,muu1,imagu = indexu

                # Get the energies of the juMu and ju'Mu' states
                Eu  = termu.eigval[iu]*c.c
                Eu1 = termu.eigval[iu1]*c.c

                # Run over lower term indexes
                for indexl in terml.index:

                    # Get variables in shorter form
                    j,il,il1,iMl,Ml,mul,iMl1,Ml1,mul1,imagl = indexl

                    # If the line is special
                    if line.especial:

                        # Get the energies of the jlMl and jl'Ml' states
                        El  = terml.eigval[il]*c.c
                        El1 = terml.eigval[il1]*c.c

                        # Get components frequencies
                        nu_ul   = Eu  - El
                        nu_u1l  = Eu1 - El
                        nu_ul1  = Eu  - El1
                        nu_u1l1 = Eu1 - El1

                        # Get transfer frequency
                        nu_t = 0.25*(nu_ul + nu_u1l + nu_ul1 + nu_u1l1)

                        # And request the Jqq
                        Jqq = line.get_Jqq(nu_t)

                    #
                    # Get transfer rates
                    #

                    # Ignoring this density matrix element
                    if i in self.atom.forbidden or j in self.atom.forbidden:

                        # No contribution
                        TA = 0.0
                        TAp = 0.0
                        TS = 0.0
                        TSp = 0.0
                        TE = 0.0
                        TEp = 0.0

                        # Debug
                        if self.debug:
                            f.write('Skip transfer rates for '+ \
                                    f'{i} {muu}{Mu} {muu1,Mu1} {imagu}, '+ \
                                    f'{j} {mul}{Ml} {mul1,Ml1} {imagl}\n')

                    # Including the density matrix element
                    else:

                        # Absorption transfer
                        TA  = TA_MT(self,termu,iu,muu,iMu,Mu,iu1,muu1,iMu1,Mu1, \
                                         terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1,Jqq,line,JS)
                        if il != il1:
                            TAp = TA_MT(self,termu,iu,muu,iMu,Mu,iu1,muu1,iMu1,Mu1, \
                                             terml,il1,mul1,iMl1,Ml1,il,mul,iMl,Ml,Jqq,line,JS)
                        # Stimulated emission transfer
                        TS  = TS_MT(self,terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1, \
                                         termu,iu,muu,iMu,Mu,iu1,muu1,iMu1,Mu1,Jqq,line,JS)
                        if iu != iu1:
                            TSp = TS_MT(self,terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1, \
                                             termu,iu1,muu1,iMu1,Mu1,iu,muu,iMu,Mu,Jqq,line,JS)
                        # Espontaneous emission transfer
                        TE  = TE_MT(self,terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1, \
                                         termu,iu,muu,iMu,Mu,iu1,muu1,iMu1,Mu1,line,JS)
                        if iu != iu1:
                            TEp = TE_MT(self,terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1, \
                                             termu,iu1,muu1,iMu1,Mu1,iu,muu,iMu,Mu,line,JS)

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

                                # Debug
                                if self.debug:
                                    f.write(f'TA1 {i} {j} --  {TA.real-TAp.real}\n')
                                    f.write(f'TS1 {j} {i} --  {TS.real-TSp.real}\n')
                                    f.write(f'TE1 {j} {i} --  {TE.real-TEp.real}\n')

                            # Real lower level (row TS, TE | column TA)
                            else:

                                self.ESE_indep[i][j] += TA.imag
                                if il != il1:
                                    self.ESE_indep[i][j] += TAp.imag

                                # Debug
                                if self.debug:
                                    if il != il1:
                                        f.write(f'TA2 {i} {j} --  {TA.imag+TAp.imag}\n')
                                    else:
                                        f.write(f'TA2 {i} {j} --  {TA.imag}\n')

                                self.ESE_indep[j][i] += TSp.imag - TS.imag

                                # Debug
                                if self.debug:
                                    f.write(f'TS2 {j} {i} --  {TSp.imag-TS.imag}\n')

                        # Real upper level (row TA | columns TS, TE)
                        else:

                            # Imaginary lower level (row TS, TE | column TA)
                            if imagl:

                                self.ESE_indep[i][j] += TAp.imag - TA.imag

                                # Debug
                                if self.debug:
                                    f.write(f'TA3 {i} {j} --  {TAp.imag-TA.imag}\n')

                                self.ESE_indep[j][i] += TS.imag
                                if iu != iu1:
                                    self.ESE_indep[j][i] += TSp.imag

                                # Debug
                                if self.debug:
                                    if iu != iu1:
                                        f.write(f'TS3 {j} {i} --  {TS.imag+TSp.imag}\n')
                                    else:
                                        f.write(f'TS3 {j} {i} --  {TS.imag}\n')

                            # Real lower level (row TS, TE | column TA)
                            else:

                                self.ESE_indep[i][j] += TA.real
                                if il != il1:
                                    self.ESE_indep[i][j] += TAp.real

                                # Debug
                                if self.debug:
                                    if il != il1:
                                        f.write(f'TA4 {i} {j} --  {TA.real+TAp.real}\n')
                                    else:
                                        f.write(f'TA4 {i} {j} --  {TA.real}\n')

                                self.ESE_indep[j][i] += TS.real
                                self.ESE_indep[j][i] += TE.real
                                if iu != iu1:
                                    self.ESE_indep[j][i] += TSp.real
                                    self.ESE_indep[j][i] += TEp.real

                                # Debug
                                if self.debug:
                                    if iu != iu1:
                                        f.write(f'TS4 {j} {i} --  {TS.real+TSp.real}\n')
                                        f.write(f'TE4 {j} {i} --  {TE.real+TEp.real}\n')
                                    else:
                                        f.write(f'TS4 {j} {i} --  {TS.real}\n')
                                        f.write(f'TE4 {j} {i} --  {TE.real}\n')

                    #
                    # Get relaxation rates
                    #

                    # Absorption

                    # Only once per upper level magnetic component
                    if iu == iu1 and not imagu and j not in self.atom.forbidden:

                        # Get Ju value
                        Ju = termu.index_muM[iMu][muu][2]

                        # Get second lower level
                        for indexk in terml.index:

                            # Get variables in shorter form
                            k,ik,ik1,iMk,Mk,muk,iMk1,Mk1,muk1,imagk = indexk

                            # If ignoring this density matrix element
                            if k in self.atom.forbidden:

                                # Debug
                                if self.debug:
                                    f.write('Skip absorption relaxation for '+ \
                                            f'{k} {muk}{Mk} {muk1,Mk1} {imagk}\n')
                                continue

                            # If especial
                            if line.especial:

                                # Get energies
                                Ek  = terml.eigval[ik]*c.c
                                Ek1 = terml.eigval[ik1]*c.c

                                # And component frequency
                                nu_uk   = Eu - Ek
                                nu_uk1  = Eu - Ek1

                            # If diagonal jM or jpMp
                            if (ik == il and ik1 == il) or \
                               (ik == il1 and ik1 == il1):

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_ra = (nu_ul + nu_ul1 + nu_uk)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_ra)

                                # Get rate
                                RA = RA_MT(self,terml,il,mul,iMl,Ml,il1,mul1,iMl1,Ml1, \
                                                termu,Ju,Mu,Jqq,line,JS)

                                # Diagonal in jM
                                if (ik == il and ik1 == il):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA1 {j} {k} {i}  {RA.imag}\n')

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA2 {j} {k} {i}  {-RA.real}\n')

                                # Diagonal in jpMp
                                if (ik == il1 and ik1 == il1):

                                    # Imaginary lower level (row)
                                    if imagl:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA3 {j} {k} {i}  {RA.imag}\n')

                                    # Real lower level (row)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA4 {j} {k} {i}  {-RA.real}\n')

                            #
                            # Not diagonal
                            #

                            # j''M'' < jM sum
                            if ik < il and ik1 == il:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_ra = (nu_ul + nu_ul1 + nu_uk)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_ra)

                                # Get rate
                                RA = RA_MT(self,terml,ik,muk,iMk,Mk,il1,mul1,iMl1,Ml1, \
                                                termu,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA5 {j} {k} {i}  {RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA6 {j} {k} {i}  {RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA7 {j} {k} {i}  {RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA8 {j} {k} {i}  {-RA.real}\n')

                            # j''M'' > jM sum
                            if ik1 > il and ik == il:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_ra = (nu_ul + nu_ul1 + nu_uk1)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_ra)

                                # Get rate
                                RA = RA_MT(self,terml,ik1,muk1,iMk1,Mk1,il1,mul1,iMl1,Ml1, \
                                                termu,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RA9 {j} {k} {i}  {-RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAA {j} {k} {i}  {RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAB {j} {k} {i}  {-RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAC {j} {k} {i}  {-RA.real}\n')

                            # j''M'' < jpMp sum
                            if ik < il1 and ik1 == il1:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_ra = (nu_ul + nu_ul1 + nu_uk)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_ra)

                                # Get rate
                                if j == 1 and k == 1 and i == 7:
                                    print(f'Just before: {RA}')
                                RA = RA_MT(self,terml,ik,muk,iMk,Mk,il,mul,iMl,Ml, \
                                                termu,Ju,Mu,Jqq,line,JS)
                                if j == 1 and k == 1 and i == 7:
                                    print(f'Just computed: {RA}')

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAD {j} {k} {i}  {-RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAE {j} {k} {i}  {-RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAF {j} {k} {i}  {RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        if j == 1 and k == 1 and i == 7:
                                            print(f'RA for 1,1, 7')
                                            print(f'Index {j}: {mul} {Ml},{mul1} {Ml1}')
                                            print(f'Index {k}: {muk} {Mk},{muk1} {Mk1}')
                                            print(f'Index {i}: {muu} {Mu},{muu1} {Mu1}')
                                            print(f'   {RA}')
                                            print(f'   {RA.real}')

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAG {j} {k} {i}  {-RA.real}\n')

                            # j''M'' > jM sum
                            if ik1 > il1 and ik == il1:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_ra = (nu_ul + nu_ul1 + nu_uk1)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_ra)

                                # Get rate
                                RA = RA_MT(self,terml,ik1,muk1,iMk1,Mk1,il,mul,iMl,Ml, \
                                                termu,Ju,Mu,Jqq,line,JS)

                                # Imaginary lower level (row)
                                if imagl:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] += RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAH {j} {k} {i}  {RA.real}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAI {j} {k} {i}  {-RA.imag}\n')

                                # Real lower level (row)
                                else:

                                    # Imaginary second lower level (column)
                                    if imagk:

                                        self.ESE_indep[j][k] -= RA.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAJ {j} {k} {i}  {-RA.imag}\n')

                                    # Real second lower level (column)
                                    else:

                                        self.ESE_indep[j][k] -= RA.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RAK {j} {k} {i}  {-RA.real}\n')


                    # Emission

                    # Only once per lower level magnetic component
                    if il == il1 and not imagl and i not in self.atom.forbidden:

                        # Get Jl value
                        Jl = terml.index_muM[iMl][mul][2]

                        # Get inverse of degeneracy
                        invg = 1./terml.g

                        # Get second upper level
                        for indexk in termu.index:

                            # Get variables in shorter form
                            k,ik,ik1,iMk,Mk,muk,iMk1,Mk1,muk1,imagk = indexk

                            # If ignoring this density matrix element
                            if k in self.atom.forbidden:

                                # Debug
                                if self.debug:
                                    f.write('Skip emission relaxation for '+ \
                                            f'{k} {muk}{Mk} {muk1,Mk1} {imagk}\n')
                                continue

                            # If especial
                            if line.especial:

                                # Get energies
                                Ek  = termu.eigval[ik]*c.c
                                Ek1 = termu.eigval[ik1]*c.c

                                # And component frequency
                                nu_kl   = Ek  - El
                                nu_k1l  = Ek1 - El

                            # If diagonal Mu or Mup
                            if (ik == iu and ik1 == iu) or \
                               (ik == iu1 and ik1 == iu1):

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_rs = (nu_ul + nu_u1l + nu_kl)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_rs)

                                # Get rate
                                RS = RS_MT(self,termu,iu,muu,iMu,Mu,iu1,muu1,iMu1,Mu1, \
                                                terml,Jl,Ml,Jqq,line,JS)

                                # If diagonal jM
                                if (ik == iu and ik1 == iu):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS1 {i} {k} {j}  {-RS.imag}\n')

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_MT(self,termu,iu,iu1,line)*invg

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS2 {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'RE2 {i} {k} {j}  {-RE.real}\n')

                                # If diagonal jupMup
                                if (ik == iu1 and ik1 == iu1):

                                    # Imaginary upper level (row)
                                    if imagu:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS3 {i} {k} {j}  {-RS.imag}\n')

                                    # Real upper level (row)
                                    else:

                                        # Get rate
                                        RE = RE_MT(self,termu,iu,iu1,line)*invg

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS4 {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'RE4 {i} {k} {j}  {-RE.real}\n')

                            #
                            # Not diagonal
                            #

                            # j''M'' < jM sum
                            if ik < iu and ik1 == iu:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_rs = (nu_ul + nu_u1l + nu_kl)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_rs)

                                # Get rate
                                RS = RS_MT(self,termu,ik,muk,iMk,Mk,iu1,muu1,iMu1,Mu1, \
                                                terml,Jl,Ml,Jqq,line,JS)
                                RE = RE_MT(self,termu,ik,iu1,line)*invg

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS5 {i} {k} {j}  {RS.real}\n')
                                            f.write(f'RE5 {i} {k} {j}  {RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS6 {i} {k} {j}  {-RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS7 {i} {k} {j}  {-RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS8 {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'RE8 {i} {k} {j}  {-RE.real}\n')

                            # j''M'' > jM sum
                            if ik1 > iu and ik == iu:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_rs = (nu_ul + nu_u1l + nu_k1l)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_rs)

                                # Get rate
                                RS = RS_MT(self,termu,ik1,muk1,iMk1,Mk1,iu1,muu1,iMu1,Mu1, \
                                                terml,Jl,Ml,Jqq,line,JS)
                                RE = RE_MT(self,termu,ik1,iu1,line)*invg

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RS9 {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'RE9 {i} {k} {j}  {-RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSA {i} {k} {j}  {-RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSB {i} {k} {j}  {RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSC {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'REC {i} {k} {j}  {-RE.real}\n')

                            # j''M'' < j'M' sum
                            if ik < iu1 and ik1 == iu1:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_rs = (nu_ul + nu_u1l + nu_kl)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_rs)

                                # Get rate
                                RS = RS_MT(self,termu,ik,muk,iMk,Mk,iu,muu,iMu,Mu, \
                                                terml,Jl,Ml,Jqq,line,JS)
                                RE = RE_MT(self,termu,ik,iu,line)*invg

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSD {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'RED {i} {k} {j}  {-RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSE {i} {k} {j}  {RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] -= RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSF {i} {k} {j}  {-RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSG {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'REG {i} {k} {j}  {-RE.real}\n')

                            # j''M'' > jM sum
                            if ik1 > iu1 and ik == iu1:

                                # Get Jqq if especial
                                if line.especial:

                                    # Get average frequency
                                    nu_rs = (nu_ul + nu_u1l + nu_k1l)/3.

                                    # Get Jqq
                                    Jqq = line.get_Jqq(nu_rs)

                                # Get rate
                                RS = RS_MT(self,termu,ik1,muk1,iMk1,Mk1,iu,muu,iMu,Mu, \
                                                terml,Jl,Ml,Jqq,line,JS)
                                RE = RE_MT(self,termu,ik1,iu,line)*invg

                                # Imaginary upper level (row)
                                if imagu:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.real
                                        self.ESE_indep[i][k] += RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSH {i} {k} {j}  {RS.real}\n')
                                            f.write(f'REH {i} {k} {j}  {RE.real}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSI {i} {k} {j}  {RS.imag}\n')

                                # Real upper level (row)
                                else:

                                    # Imaginary second upper level (column)
                                    if imagk:

                                        self.ESE_indep[i][k] += RS.imag

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSJ {i} {k} {j}  {RS.imag}\n')

                                    # Real second upper level (column)
                                    else:

                                        self.ESE_indep[i][k] -= RS.real
                                        self.ESE_indep[i][k] -= RE.real

                                        # Debug
                                        if self.debug:
                                            f.write(f'RSK {i} {k} {j}  {-RS.real}\n')
                                            f.write(f'REK {i} {k} {j}  {-RE.real}\n')


        # Mass conservation

        indep = np.zeros(self.atom.ndim)
        irow = 0
        indep[irow] = 1

        # For every term
        for term in self.atom.terms:

            # For every density matrix element
            for index in term.index:

                # Get variables in shorter form
                i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index

                # If diagonal
                if ii == ii1:
                    self.ESE_indep[irow,i] = 1
                else:
                    self.ESE_indep[irow,i] = 0

        '''
        print('AD-HOC SYSTEM LOWER TERM ISOTROPIC')
        for index in self.atom.terms[0].index:

                # Get variables in shorter form
                i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index

                self.ESE_indep[i,:] = 0
                self.ESE_indep[i,i] = 1

                # If diagonal
                if ii == ii1:
                    indep[i] = 1
                else:
                    indep[i] = 0
        print('AD-HOC SYSTEM LOWER TERM ISOTROPIC')

        print('AD-HOC SYSTEM UPPER TERM ISOTROPIC')
        for index in self.atom.terms[1].index:

                # Get variables in shorter form
                i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index

                self.ESE_indep[i,:] = 0
                self.ESE_indep[i,i] = 1

                # If diagonal
                if ii == ii1:
                    indep[i] = 1
                else:
                    indep[i] = 0
        print('AD-HOC SYSTEM UPPER TERM ISOTROPIC')
        '''

        # Debug
        if self.debug:
            def formated(val):
                if np.absolute(val) > 0.:
                    return f'{val:13.6e}'
                else:
                    return '    ---    '
            f.write('SEE:\n')
            # For each line
            for i in range(self.atom.ndim):
                row = f'{i}: '
                # For each column
                for j in range(self.atom.ndim):
                    row += f' {j}:{formated(self.ESE_indep[i,j])}'
                row += f'  ; {formated(indep[i])}\n'
                f.write(row+'\n')
               #if i == 27 or i == 28 or i == 82 or i == 83:
               #    print(self.ESE_indep[i,:])

        # Copy old rho
        rho_old = self.rho.copy()

        # Solve SEE and store new density matrix
        try:
            self.rho = linalg.solve(self.ESE_indep, indep)
        except np.linalg.LinAlgError:
            print('Sinfular matrix in SEE')
            # For each line
            for i in range(self.atom.ndim):
                row = f'{i}: '
                # For each column
                for j in range(self.atom.ndim):
                    row += f' {j}:{self.ESE_indep[i,j]:13.6e}'
                row += f'  ; {indep[i]:13.6e}\n'
                print(row)
            raise
        except:
            raise

        # Debug
        if self.debug:
            f.write("rho(jM,j'M'):\n")
            for term in self.atom.terms:
                for index in term.index:
                    i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
                    f.write(f'[{i:2d}][{ii:2d},{ii1:2d}] ' + \
                            f'rho({mu:2d},{M:3.1f};{mu1:2d},{M1:3.1f}) = '+ \
                            f'{rho_old[i]:13.6e}  {self.rho[i]:13.6e}\n')

            f.write('\n')

        # Debug
        if self.debug:
            print('')
            f.write("rho^K_Q(J,J'):\n")
            for term in self.atom.terms:

                indexes = copy.deepcopy(term.index)
                rhos0 = copy.deepcopy(rho_old).tolist()[:term.NN_next]
                rhos1 = copy.deepcopy(self.rho).tolist()[:term.NN_next]
                LL = term.L
                SS = term.S
                maxJ = LL + SS
                f.write(f"Term L {LL} S {SS}\n")
                print(f"Term L {LL} S {SS}")

                # Initialize next index
                ii = term.NN_next-1

                # Initialize left index
                i1 = -1

                # Add missing indexes

                # For each "left" M
                for iM,M,Mblock in zip(range(term.nM),term.M,term.Mblock):
                    # For each "left" mu
                    for mu in range(Mblock):

                        # Get mu, M index
                        i1 += 1

                        # Initialize right index
                        i2 = -1

                        # For each "right" M
                        for iM1,M1,Mblock1 in zip(range(term.nM),term.M,term.Mblock):

                            # If dM > 1, skip
                            if int(round(np.absolute(M1 - M))) > int(round(2.*maxJ)):

                                # Advance the index anyways
                                i2 += Mblock1

                                # And skip
                                continue

                            # For each "right" mu
                            for mu1 in range(Mblock1):

                                # Get mu1, M1 index
                                i2 += 1

                                # Only if i2 < i1
                                if i2 < i1:

                                    for index in term.index:
                                        ii2 = index[1]
                                        ii1 = index[2]
                                        if i1 == ii1 and i2 == ii2:
                                            crrho0 = rho_old[index[0]]
                                            cirho0 = rho_old[index[0]+1]
                                            crrho1 = self.rho[index[0]]
                                            cirho1 = self.rho[index[0]+1]
                                            break

                                    ii += 1
                                    indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,False])
                                    rhos0.append(crrho0)
                                    rhos1.append(crrho0)
                                    ii += 1
                                    indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,True])
                                    rhos0.append(-1.*cirho0)
                                    rhos1.append(-1.*cirho1)
                rhos0 = np.array(rhos0)
                rhos1 = np.array(rhos1)

                # Output
                for J in term.J:

                    # Possible M
                    Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

                    for Jp in term.J:
                        minK = int(round(np.absolute(J-Jp)))
                        maxK = int(round(J + Jp))

                        for K in range(minK,maxK+1):
                            for Q in range(-K,K+1):

                                # Initialize rhoKQ
                                rho0 = 1j*0.
                                rho1 = 1j*0.

                                # For each M
                                for M in Ms:

                                    # Known contribution

                                    # Decide M'
                                    Mp = M - Q

                                    # Skip invalid M'
                                    if np.absolute(Mp) > Jp:
                                        continue

                                    # Sign and weight
                                    SW = JS.sign(J-M)* \
                                         np.sqrt(2.*K + 1.)

                                    # Go by the whole term
                                    for index in indexes:

                                        # Extract data
                                        ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index

                                        # If M,M'
                                        if (M1 != M or M2 != Mp):
                                            continue

                                        # Run over J
                                        for Jindex1 in term.index_muM[iM1]:
                                            J1 = Jindex1[2]
                                            iJ1 = Jindex1[1]

                                            # Relevant J
                                            if np.absolute(J1-J) > 0.25:
                                                continue

                                            # Run over J'
                                            for Jindex2 in term.index_muM[iM2]:
                                                J2 = Jindex2[2]
                                                iJ2 = Jindex2[1]

                                                # Relevant J'
                                                if np.absolute(J2-Jp) > 0.25:
                                                    continue

                                                # Contribution
                                                CC = term.eigvec[i1][iJ1]* \
                                                     term.eigvec[i2][iJ2]* \
                                                     SW*JS.j3(J,Jp,K,M,-Mp,-Q)

                                                # If imag
                                                if imag:
                                                    rho0 += CC*rhos0[ii]*1j
                                                    rho1 += CC*rhos1[ii]*1j
                                                else:
                                                    rho0 += CC*rhos0[ii]
                                                    rho1 += CC*rhos1[ii]

                                # Print rhoKQ
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                        f'{rho0.real:13.6e}  {rho1.real:13.6e}\n')
                                if Q != 0:
                                    f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                            f'{rho0.imag:13.6e}  {rho1.imag:13.6e}\n')

        # Get maximum relative change and check population sum
        MRC_p = 0.0
        MRC_c = 0.0
        suma = 0.

        # For each term
        for term in self.atom.terms:

            # Get scale
            scale = -1.

            # For each element
            for index in term.index:

                # If not diagonal, skip
                if not self.isdiag(index):
                    continue

                # Update scale
                scale = np.max([scale, \
                                np.max([np.max(rho_old[index[0]]), \
                                        np.max(self.rho[index[0]])])])

            # For each element
            for index in term.index:

                # Easier pointer
                old = rho_old[index[0]]
                new = self.rho[index[0]]

                # If diagonal
                if self.isdiag(index):

                    # Get MRC
                    if old < 1e-15 and new < 1e-15:
                        continue
                    if old > 1e-15:
                        MRC = np.absolute(old-new)/np.absolute(old)
                    elif new > 1e-15:
                        MRC = np.absolute(old-new)/np.absolute(new)
                    else:
                        MRC = np.absolute(old-new)

                    # Update maximum relative change
                    MRC_p = np.max([MRC_p,MRC])

                    # Add to population
                    suma += self.rho[index[0]]

                    # Check sign
                    if self.rho[index[0]] < 0:
                        print(f"Warning: Negative population of the level: " + \
                               "L={term.L},j={index[5]}, M={index[4]}," + \
                               "j'={index[8]},M'={index[7]},real={index[9]}")
                        print(f"with population of rho={self.rho[index[0]]}")

                # If not diagonal
                else:

                    # Get MRC
                    if old*scale < 1e-15 and new*scale < 1e-15:
                        continue
                    if old*scale > 1e-15:
                        MRC = np.absolute(old-new)/np.absolute(old)
                    elif new*scale > 1e-15:
                        MRC = np.absolute(old-new)/np.absolute(new)
                    else:
                        MRC = np.absolute(old-new)

                    # Update maximum relative change
                    MRC_c = np.max([MRC_c,MRC])

        # If bad normalization
        if not 0.98 < suma < 1.02:
            print("Warning: Not normaliced populations in this itteration")
            print(f"With the sum of the populations rho = {suma}")

        # Debug
        if self.debug:
            f.close()

        return MRC_p,MRC_c

################################################################################
################################################################################
################################################################################

# Eq 7.34a from LL04 for the SEE coeficients
def TA_MT(ESE,term,i,mu,iM,M,ip,mup,iMp,Mp,terml,il,mul,iMl,Ml,ilp,mulp,iMlp,Mlp,Jqq,line,jsim):

    # Initialize sum
    sum_qq = (0+0j)

    # Applied selection rules to remove sumation
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return sum_qq
    qp = int(Mlp - Mp)
    if np.absolute(qp) > 1:
        return sum_qq

    # For each J
    for Jindex in term.index_muM[iM]:

        J = Jindex[2]
        iJ = Jindex[1]
        pJ = np.sqrt(2.*J + 1.)*term.eigvec[i][iJ]

        # For each Jl
        for Jlindex in terml.index_muM[iMl]:

            Jl = Jlindex[2]
            iJl = Jlindex[1]

            # Dipole
            if np.absolute(J-Jl) > 1.25 or (J + Jl) < 0.25:
                continue

            WW = pJ*np.sqrt(2.*Jl + 1.)*terml.eigvec[il][iJl]* \
                 jsim.j6(term.L,terml.L,1.,Jl,J,term.S)* \
                 jsim.j3(J,Jl,1.,-M,Ml,-q)

            # For each J'
            for Jpindex in term.index_muM[iMp]:

                Jp = Jpindex[2]
                iJp = Jpindex[1]

                pJp = np.sqrt(2.*Jp + 1.)*term.eigvec[ip][iJp]

                # For each Jl'
                for Jlpindex in terml.index_muM[iMlp]:

                    Jlp = Jlpindex[2]
                    iJlp = Jlpindex[1]

                    # Dipole
                    if np.absolute(Jp-Jlp) > 1.25 or (Jp + Jlp) < 0.25:
                        continue

                    WWp = WW*pJp*np.sqrt(2.*Jlp+1.)*terml.eigvec[ilp][iJlp]* \
                          jsim.j6(term.L,terml.L,1.,Jlp,Jp,term.S)* \
                          jsim.j3(Jp,Jlp,1.,-Mp,Mlp,-qp)

                    sum_qq += WWp*Jqq[q][qp]

    return (2.*terml.L + 1.)*3.*jsim.sign(Ml-Mlp)*line.B_lu*sum_qq

# Eq 7.9 from LL04 for the SEE coeficients
def TA_ML(ESE, J, M, Mp, Jl, Ml, Mlp, Jqq, line,jsim):

    # Applied selection rules to remove sumation
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return (0+0j)
    qp = int(Mlp - Mp)
    if np.absolute(qp) > 1:
        return (0+0j)

    sum_qq = 3.*jsim.sign(Ml-Mlp) * \
                jsim.j3(J, Jl, 1, -M,  Ml, -q) * \
                jsim.j3(J, Jl, 1, -Mp, Mlp, -qp) * \
                Jqq[q][qp]

    return (2.*Jl + 1.)*line.B_lu*sum_qq


# Eq 7.34b from LL04 for the SEE coeficients
def TE_MT(ESE,term,i,mu,iM,M,ip,mup,iMp,Mp,termu,iu,muu,iMu,Mu,iup,muup,iMup,Mup,line,jsim):

    # Initialize sum
    sum_qq = (0+0j)

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    qp = int(M - Mu)
    if q != qp:
        return sum_qq

    # For each J
    for Jindex in term.index_muM[iM]:

        J = Jindex[2]
        iJ = Jindex[1]
        pJ = np.sqrt(2.*J + 1.)*term.eigvec[i][iJ]

        # For each Ju
        for Juindex in termu.index_muM[iMu]:

            Ju = Juindex[2]
            iJu = Juindex[1]

            # Dipole
            if np.absolute(J-Ju) > 1.25 or (J + Ju) < 0.25:
                continue

            WW = pJ*np.sqrt(2.*Ju + 1.)*termu.eigvec[iu][iJu]* \
                 jsim.j6(termu.L,term.L,1.,J,Ju,term.S)* \
                 jsim.j3(Ju,J,1.,-Mu,M,-q)

            # For each J'
            for Jpindex in term.index_muM[iMp]:

                Jp = Jpindex[2]
                iJp = Jpindex[1]

                pJp = np.sqrt(2.*Jp + 1.)*term.eigvec[ip][iJp]

                # For each Ju'
                for Jupindex in termu.index_muM[iMup]:

                    Jup = Jupindex[2]
                    iJup = Jupindex[1]

                    # Dipole
                    if np.absolute(Jp-Jup) > 1.25 or (Jp + Jup) < 0.25:
                        continue

                    WWp = WW*pJp*np.sqrt(2.*Jup+1.)*termu.eigvec[iup][iJup]* \
                          jsim.j6(termu.L,term.L,1.,Jp,Jup,term.S)* \
                          jsim.j3(Jup,Jp,1.,-Mup,Mp,-q)

                    if np.absolute(WWp) <= 0.:
                        continue

                    sum_qq += WWp

    return (2.*termu.L + 1)*jsim.sign(Mu-Mup)*line.A_ul*sum_qq

def TE_ML(ESE, J, M, Mp, Ju, Mu, Mup, line, jsim):

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    qp = int(M - Mu)
    if q != qp:
        return (0+0j)

    sum_q = jsim.sign(Mu - Mup) * \
            jsim.j3(Ju, J, 1, -Mup, Mp, -q) * \
            jsim.j3(Ju, J, 1, -Mu,  M, -q)

    return (2.*Ju + 1.)*line.A_ul*sum_q

# Eq 7.34c from LL04 for the SEE coeficients
def TS_MT(ESE,term,i,mu,iM,M,ip,mup,iMp,Mp,termu,iu,muu,iMu,Mu,iup,muup,iMup,Mup,Jqq,line,jsim):

    # Initialize sum
    sum_qq = (0+0j)

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    if np.absolute(q) > 1:
        return sum_qq
    qp = int(M - Mu)
    if np.absolute(qp) > 1:
        return sum_qq

    # For each J
    for Jindex in term.index_muM[iM]:

        J = Jindex[2]
        iJ = Jindex[1]
        pJ = np.sqrt(2.*J + 1.)*term.eigvec[i][iJ]

        # For each Ju
        for Juindex in termu.index_muM[iMu]:

            Ju = Juindex[2]
            iJu = Juindex[1]

            # Dipole
            if np.absolute(J-Ju) > 1.25 or (J + Ju) < 0.25:
                continue

            WW = pJ*np.sqrt(2.*Ju + 1.)*termu.eigvec[iu][iJu]* \
                 jsim.j6(termu.L,term.L,1.,J,Ju,term.S)* \
                 jsim.j3(Ju,J,1.,-Mu,M,-qp)

            # For each J'
            for Jpindex in term.index_muM[iMp]:

                Jp = Jpindex[2]
                iJp = Jpindex[1]

                pJp = np.sqrt(2.*Jp + 1.)*term.eigvec[ip][iJp]

                # For each Ju'
                for Jupindex in termu.index_muM[iMup]:

                    Jup = Jupindex[2]
                    iJup = Jupindex[1]

                    # Dipole
                    if np.absolute(Jp-Jup) > 1.25 or (Jp + Jup) < 0.25:
                        continue

                    WWp = WW*pJp*np.sqrt(2.*Jup+1.)*termu.eigvec[iup][iJup]* \
                          jsim.j6(termu.L,term.L,1.,Jp,Jup,term.S)* \
                          jsim.j3(Jup,Jp,1.,-Mup,Mp,-q)

                    sum_qq += WWp*Jqq[q][qp]

    return (2.*termu.L + 1)*3.*jsim.sign(M-Mp)*line.B_ul*sum_qq


def TS_ML(ESE, J, M, Mp, Ju, Mu, Mup, Jqq, line, jsim):

    # Applied selection rules to remove sumation
    q = int(Mp - Mup)
    if np.absolute(q) > 1:
        return (0+0j)
    qp = int(M - Mu)
    if np.absolute(qp) > 1:
        return (0+0j)

    sum_qq = 3.*jsim.sign(Mp - M) * \
                jsim.j3(Ju, J, 1, -Mup, Mp, -q) * \
                jsim.j3(Ju, J, 1, -Mu,  M, -qp) * \
                Jqq[q][qp]

    return (2.*Ju + 1.)*line.B_ul*sum_qq

def RA_MT(ESE,term,i,mu,iM,M,ip,mup,iMp,Mp,termu,Ju,Mu,Jqq,line,jsim):

    # In itialize sum
    sum_u = (0.+0.j)

    # Selection rules
    q = int(M - Mu)
    if np.absolute(q) > 1:
        return sum_u
    qp = int(Mp - Mu)
    if np.absolute(qp) > 1:
        return sum_u

    # For each J
    for Jindex in term.index_muM[iM]:

        J = Jindex[2]
        iJ = Jindex[1]

        # Dipole
        if np.absolute(J-Ju) > 1.25 or (J + Ju) < 0.25:
            continue

        WW = np.sqrt(2.*J + 1.)*term.eigvec[i][iJ]* \
             jsim.j6(termu.L,term.L,1.,J,Ju,term.S)* \
             jsim.j3(Ju,J,1.,-Mu,M,-q)

        # For each J'
        for Jpindex in term.index_muM[iMp]:

            Jp = Jpindex[2]
            iJp = Jpindex[1]

            # Dipole
            if np.absolute(Jp-Ju) > 1.25 or (Jp + Ju) < 0.25:
                continue

            WWp = WW*np.sqrt(2.*Jp + 1.)*term.eigvec[ip][iJp]* \
                  jsim.j6(termu.L,term.L,1.,Jp,Ju,term.S)* \
                  jsim.j3(Ju,Jp,1.,-Mu,Mp,-qp)

            sum_u += WWp*Jqq[q][qp]

    return 1.5*(2.*Ju + 1.)*(2.*term.L + 1.)*line.B_lu*sum_u*jsim.sign(q+qp)

def RA_ML(ESE, J, M, Mp, Ju, Mu, Jqq, line, jsim):

    # Selection rules
    q = int(M - Mu)
    if np.absolute(q) > 1:
        return (0+0j)
    qp = int(Mp - Mu)
    if np.absolute(qp) > 1:
        return (0+0j)

    sum_u = 3*jsim.sign(M - Mp) * \
              jsim.j3(Ju, J, 1, -Mu,  M,  -q) * \
              jsim.j3(Ju, J, 1, -Mu, Mp, -qp) * \
              Jqq[q][qp]

    return 0.5*(2*J+1)*line.B_lu*sum_u

def RE_MT(ESE,term,i,ip,line):

    # Diagonal
    if i == ip:
        return 0.5*line.A_ul
    else:
        return (0+0j)

def RE_ML(ESE, J, M, Mp, Jl, Ml, line):

    # Selection rules
    q = int(Ml - M)
    qp = int(Ml - Mp)
    if q != qp:
        return (0+0j)

    return 0.5*line.A_ul


def RS_MT(ESE,term,i,mu,iM,M,ip,mup,iMp,Mp,terml,Jl,Ml,Jqq,line,jsim):

    # In itialize sum
    sum_l = (0+0j)

    # Selection rules
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return sum_l
    qp = int(Ml - Mp)
    if np.absolute(qp) > 1:
        return sum_l

    # For each J
    for Jindex in term.index_muM[iM]:

        J = Jindex[2]
        iJ = Jindex[1]

        # Dipole
        if np.absolute(Jl-J) > 1.25 or (Jl + J) < 0.25:
            continue

        WW = np.sqrt(2.*J + 1.)*term.eigvec[i][iJ]* \
             jsim.j6(term.L,terml.L,1.,Jl,J,term.S)* \
             jsim.j3(J,Jl,1.,-M,Ml,-q)

        # For each J'
        for Jpindex in term.index_muM[iMp]:

            Jp = Jpindex[2]
            iJp = Jpindex[1]

            # Dipole
            if np.absolute(Jl-Jp) > 1.25 or (Jl + Jp) < 0.25:
                continue

            WWp = WW*np.sqrt(2.*Jp + 1.)*term.eigvec[ip][iJp]* \
                  jsim.j6(term.L,terml.L,1.,Jl,Jp,term.S)* \
                  jsim.j3(Jp,Jl,1.,-Mp,Ml,-qp)

            sum_l += WWp*Jqq[q][qp]

    return 1.5*(2.*Jl + 1.)*(2.*term.L + 1.)*line.B_ul*sum_l


def RS_ML(ESE, J, M, Mp, Jl, Ml, Jqq, line, jsim):

    # Selection rules
    q = int(Ml - M)
    if np.absolute(q) > 1:
        return (0+0j)
    qp = int(Ml - Mp)
    if np.absolute(qp) > 1:
        return (0+0j)

    sum_l = 3*jsim.j3( J,Jl, 1, -M, Ml, -q) * \
              jsim.j3( J,Jl, 1,-Mp, Ml,-qp) * \
              Jqq[q][qp]

    return 0.5*(2*J+1)*line.B_ul*sum_l
