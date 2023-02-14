import copy
import numpy as np
import matplotlib.pyplot as plt
import constants as cts


class RTcoefs:
    """ Radiative transfer coefficients.
        Just one instance of the class needs to be created in the program.
    """

    def __init__(self, nus, nus_weights, mode):
        """
            nus: array of the line frequencies
            nus_weights: array of the line frequency weights
            return value: None
        """

        # Defining units to avoid quotients and some products later
        self.rtc4prototype = np.zeros((4, nus.size))
        self.rtc3prototype = np.zeros((3, nus.size))
        self.hnu = 0.25*cts.h/(np.pi)
        self.hcm2 = 2.0*cts.h/(cts.c*cts.c)

        # Multi-level atom
        if mode == 0:
            self.getRTcoefs = self.getRTcoefs_ML
        elif mode == 1:
            self.getRTcoefs = self.getRTcoefs_MT

        # No warning sent yet
        self.no_warning = False


    def getRTcoefs_ML(self, ese, ray, cdts):
        """ Provides the 4-vector of epsilon and the 4x4 K-matrix for the point
            with given ESE state and ray direction.
            Multi-level atom case in standard representation.
            Eqs. 7.10 of LL04.
              ese: the local instance of the ESE class
              ray: object with .theta and .chi variables defining the ray
                   of propagation direction
              return value: [S (source function vector in frequencies), K (4x4 list of
                             vectors in frequencies)]
        """

        # Point to J symbols
        jsim = cdts.JS

        # Get geomtrical tensors
        # If there is need to rotate
        if ese.rotate:
            TQQ = ese.get_Tqq(ray,jsim)
        else:
            TQQ = ray.Tqq

        # Add density to global constant
        hnuN = self.hnu*cdts.n_dens

        # Get factor for Zeeman splitting later
        dB = 1.3996*ese.B

        # Initialize RT coefficients
        eta_a = copy.copy(self.rtc4prototype)
        eta_s = copy.copy(eta_a)
        rho_a = copy.copy(self.rtc3prototype)
        rho_s = copy.copy(rho_a)
        eps = copy.copy(eta_a)

        # For each transition
        for line in ese.atom.lines:

            # Initialize profiles
            line.initialize_profiles()

            # Get levels quantum numbers
            Ll = line.levels[0]
            Jl = line.jlju[0]
            Lu = line.levels[1]
            Ju = line.jlju[1]

            # Initialize line contributions
            sum_etaa0 = 0
            sum_etaa1 = 0
            sum_etaa2 = 0
            sum_etaa3 = 0
            sum_etas0 = 0
            sum_etas1 = 0
            sum_etas2 = 0
            sum_etas3 = 0
            sum_rhoa1 = 0
            sum_rhoa2 = 0
            sum_rhoa3 = 0
            sum_rhos1 = 0
            sum_rhos2 = 0
            sum_rhos3 = 0
            sum_eps0  = 0
            sum_eps1  = 0
            sum_eps2  = 0
            sum_eps3  = 0

            # Get global factors which depend on the line
            lfactor = 3*(2*Jl + 1)*line.B_lu
            ufactor = 3*(2*Ju + 1)*line.B_ul

            # For each pair of Ml, Mu
            for Ml in ese.atom.levels[Ll].M:
              for Mu in ese.atom.levels[Lu].M:

                # Get q value and check valid
                q = int(round(Ml - Mu))
                if np.absolute(q) > 1:
                  continue

                # Get voigt profile with Zeeman splitting
                voigt = cdts.voigt_profile(line, \
                                            dB*(line.gu*Mu - \
                                                line.gl*Ml)*\
                                            line.nu.unit)

                # Get first 3J
                j3 = jsim.j3(Ju, Jl, 1, -Mu, Ml, -q)

                # Absorption, run over Ml'
                for Mlp in ese.atom.levels[Ll].M:

                  # Get q' value and check valid
                  qp = int(round(Mlp - Mu))
                  if np.absolute(qp) > 1:
                    continue

                  # Get 3J factors
                  factor = lfactor*jsim.sign(Ml-Mlp) * \
                           j3 * jsim.j3(Ju, Jl, 1, -Mu, Mlp, -qp)

                  # Product of Voigt profile and rhomm'
                  esevoigt = voigt * ese.rho_call(Ll, Jl, Ml, Mlp)

                  # Stokes parameters
                  # This ugly unrolling is to avoid astropy errors
                  # It may work if sum_* are lists? Maybe, but did
                  # not even try, it will look more elegant for sure
                  # if it did work
                  tag = f'{q}{qp}'
                  Cfactor = TQQ[0][tag]*esevoigt
                  sum_etaa0 += factor*np.real(Cfactor)

                  Cfactor = TQQ[1][tag]*esevoigt
                  sum_etaa1 += factor*np.real(Cfactor)
                  sum_rhoa1 += factor*np.imag(Cfactor)

                  Cfactor = TQQ[2][tag]*esevoigt
                  sum_etaa2 += factor*np.real(Cfactor)
                  sum_rhoa2 += factor*np.imag(Cfactor)

                  Cfactor = TQQ[3][tag]*esevoigt
                  sum_etaa3 += factor*np.real(Cfactor)
                  sum_rhoa3 += factor*np.imag(Cfactor)

                # Emission, run over Mu'
                for Mup in ese.atom.levels[Lu].M:

                  # Get q' value and check valid
                  qp = int(round(Ml - Mup))
                  if np.absolute(qp) > 1:
                    continue

                  # Get 3J factors
                  factor = ufactor * j3 * \
                           jsim.j3(Ju, Jl, 1, -Mup, Ml, -qp)

                  # Product of Voigt profile and rhomm'
                  esevoigt = voigt * ese.rho_call(Lu, Ju, Mup, Mu)

                  # Stokes parameters
                  tag = f'{q}{qp}'
                  Cfactor = TQQ[0][tag]*esevoigt
                  sum_etas0 += factor*np.real(Cfactor)

                  Cfactor = TQQ[1][tag]*esevoigt
                  sum_etas1 += factor*np.real(Cfactor)
                  sum_rhos1 += factor*np.imag(Cfactor)

                  Cfactor = TQQ[2][tag]*esevoigt
                  sum_etas2 += factor*np.real(Cfactor)
                  sum_rhos2 += factor*np.imag(Cfactor)

                  Cfactor = TQQ[3][tag]*esevoigt
                  sum_etas3 += factor*np.real(Cfactor)
                  sum_rhos3 += factor*np.imag(Cfactor)

            # Add to the profile in the line and normalize it
            line.add_contribution_profiles(sum_etaa0.real*cdts.nus_weights)
            line.normalize_profiles()

            # Get final constants for the RT coefficients
            hnu = hnuN*line.nu
            hnu3 = hnu * self.hcm2*line.nu3

            # Add to line contribution
            # rho U is negative because is the upper part of the triangular
            # propagation matrix
            try:
                eta_a[0, :] += hnu * sum_etaa0
                eta_a[1, :] += hnu * sum_etaa1
                eta_a[2, :] += hnu * sum_etaa2
                eta_a[3, :] += hnu * sum_etaa3
                eta_s[0, :] += hnu * sum_etas0
                eta_s[1, :] += hnu * sum_etas1
                eta_s[2, :] += hnu * sum_etas2
                eta_s[3, :] += hnu * sum_etas3
                rho_a[0, :] += hnu * sum_rhoa1
                rho_a[1, :] -= hnu * sum_rhoa2
                rho_a[2, :] += hnu * sum_rhoa3
                rho_s[0, :] += hnu * sum_rhos1
                rho_s[1, :] -= hnu * sum_rhos2
                rho_s[2, :] += hnu * sum_rhos3
                eps[0, :]   += hnu3 * sum_etas0
                eps[1, :]   += hnu3 * sum_etas1
                eps[2, :]   += hnu3 * sum_etas2
                eps[3, :]   += hnu3 * sum_etas3
            except NameError:
                pass
            except:
                pass

        # Correct for stimulated emission
        eta = eta_a - eta_s
        rho = rho_a - rho_s

        # Check physical absorption
        if np.any(eta[0] < 0):
            print("Warning: eta_I < 0")

        # Scale eta and rho
        for ii in range(1,4):
            eta[ii] /= (eta[0]+cts.vacuum)
            rho[ii] /= (eta[0]+cts.vacuum)

        # Build propagation matrix
        KK = [eta,rho]
        #KK = np.array([[eta[0],  eta[1],  eta[2],  eta[3]],
        #               [eta[1],  eta[0],  rho[2], -rho[1]],
        #               [eta[2], -rho[2],  eta[0],  rho[0]],
        #               [eta[3],  rho[1], -rho[0],  eta[0]]])

        # Build source function
        SS = eps/(eta[0]+cts.vacuum)

        return SS, KK


    def getRTcoefs_MT(self, ese, ray, cdts):
        """ Provides the 4-vector of epsilon and the 4x4 K-matrix for the point
            with given ESE state and ray direction.
            Multi-term atom case in standard representation.
            Eqs. 7.35 of LL04.
              ese: the local instance of the ESE class
              ray: object with .theta and .chi variables defining the ray
                   of propagation direction
              return value: [S (source function vector in frequencies), K (4x4 list of
                             vectors in frequencies)]
        """

        # Point to J symbols
        jsim = cdts.JS

        # Get geomtrical tensors
        # If there is need to rotate
        if ese.rotate:
            TQQ = ese.get_Tqq(ray,jsim)
        else:
            TQQ = ray.Tqq


        # Add density to global constant
        hnuN = self.hnu*cdts.n_dens

        # For each transition
        for line in ese.atom.lines:

          # Initialize profiles
          line.initialize_profiles()

          # Initialize line contributions for both components
          sum_etaa0 = {res: 0 for res in line.resos}
          sum_etaa1 = {res: 0 for res in line.resos}
          sum_etaa2 = {res: 0 for res in line.resos}
          sum_etaa3 = {res: 0 for res in line.resos}
          sum_etas0 = {res: 0 for res in line.resos}
          sum_etas1 = {res: 0 for res in line.resos}
          sum_etas2 = {res: 0 for res in line.resos}
          sum_etas3 = {res: 0 for res in line.resos}
          sum_rhoa1 = {res: 0 for res in line.resos}
          sum_rhoa2 = {res: 0 for res in line.resos}
          sum_rhoa3 = {res: 0 for res in line.resos}
          sum_rhos1 = {res: 0 for res in line.resos}
          sum_rhos2 = {res: 0 for res in line.resos}
          sum_rhos3 = {res: 0 for res in line.resos}

          # Point to terms involved
          termu = ese.atom.terms[line.terms[1]]
          terml = ese.atom.terms[line.terms[0]]

          # Get quantum numbers
          Ll = terml.L
          Lu = termu.L
          SS = termu.S

          # Frequency constant
          hnu = hnuN*line.nu

          # Proportionality factors that are line dependent
          lfactor = 3*(2*Ll + 1)*line.B_lu*hnu
          ufactor = 3*(2*Lu + 1)*line.B_ul*hnu

          # For each Ml
          for iMl,Mlblock in enumerate(terml.index_muM):

            # Get Ml
            Ml = Mlblock[0][-1]

            # For each Mu
            for iMu,Mublock in enumerate(termu.index_muM):

              # Get Mu
              Mu = Mublock[0][-1]

              # Get q value and check valid
              q = int(round(Ml - Mu))
              if np.absolute(q) > 1:
                continue

              # For each jl
              for mul_index in Mlblock:

                # Get index
                imul = mul_index[-2]

                # Get energy and index for rho
                el = terml.eigval[mul_index[0]]
                il0 = mul_index[0]

                # For each ju
                for muu_index in Mublock:

                  # Get index
                  imuu = muu_index[-2]

                  # Get energy and index for rho
                  eu = termu.eigval[muu_index[0]]
                  iu1 = muu_index[0]

                  # Resonance for Voigt profile (need to subtract line.nu
                  # to get just displacement)
                  nu0 = (eu - el)*cts.c - line.nu

                  # Get Voigt profile
                  voigt = cdts.voigt_profile(line, nu0)

                  # select wich component to include the contribution
                  # as the closest in frequency to the center
                  dd = np.absolute(line.resos - nu0 - line.nu)
                  comp = line.resos[np.argmin(dd)]

                  '''
                  try:
                      import matplotlib.pyplot as plt
                  except:
                      pass
                  lamb = cts.c.cgs*(1e7*unt.nm/unt.cm)/cdts.nus
                  plt.plot(lamb,voigt.real)
                  plt.plot(lamb,voigt.real,marker='*')
                  plt.plot(lamb,voigt.imag)
                  plt.plot(lamb,voigt.imag,marker='*')
                  print('Plot Voigt')
                  plt.show()
                  '''

                  # For each Jl
                  for Jl_index in Mlblock:

                    # Get quantum number, index, and eigenvector
                    Jl = Jl_index[2]
                    iJl = Jl_index[1]
                    CjlJl = terml.eigvec[mul_index[0]][iJl]

                    # For each Ju
                    for Ju_index in Mublock:

                      # Get quantum number, index, and eigenvector
                      Ju = Ju_index[2]
                      iJu = Ju_index[1]
                      Clu = CjlJl*termu.eigvec[muu_index[0]][iJu]

                      # Common 3J and 6J with eigenvectors
                      j36 = jsim.j3(Ju,Jl,1.,-Mu,Ml,-q)* \
                            jsim.j6(Lu,Ll,1.,Jl,Ju,SS)* \
                            Clu

                      # Check non-zero
                      if np.absolute(j36) <= 0.:
                        continue

                      # Factors
                      j36 *= np.sqrt((2.*Jl + 1.)*(2.*Ju + 1.))

                      #
                      # Absorption
                      #

                      # For each Ju'
                      for Ju1_index in Mublock:

                        # Get quantum number, index, and eigenvector
                        Ju1 = Ju1_index[2]
                        iJu1 = Ju1_index[1]
                        CC = j36*termu.eigvec[muu_index[0]][iJu1]*lfactor

                        # Eigenvec value
                        if np.absolute(CC) <= 0.:
                            continue

                        # Common 3J and 6J with eigenvectors
                        CC *= np.sqrt(2.*Ju1 + 1.)

                        # For Ml'
                        for iMl1,Ml1block in enumerate(terml.index_muM):

                          # Get Ml'
                          Ml1 = Ml1block[0][-1]

                          # Get q1 value and check valid
                          q1 = int(round(Ml1 - Mu))
                          if np.absolute(q1) > 1:
                            continue

                          # Multiply Tqq and voigt
                          tag = f'{q}{q1}'
                          Tqq0voigt = TQQ[0][tag]*voigt
                          Tqq1voigt = TQQ[1][tag]*voigt
                          Tqq2voigt = TQQ[2][tag]*voigt
                          Tqq3voigt = TQQ[3][tag]*voigt

                          # Get jl'
                          for mul1_index in Ml1block:

                            # Get index
                            imul1 = mul1_index[-2]
                            il1 = mul1_index[0]

                            # Get Jl'
                            for Jl1_index in Ml1block:

                              # Get quantum number, index, and eigenvector
                              Jl1 = Jl1_index[2]
                              iJl1 = Jl1_index[1]
                              CCb = CC*terml.eigvec[mul1_index[0]][iJl1]* \
                                    jsim.sign(q+q1)* \
                                    jsim.j3(Ju1,Jl1,1.,-Mu,Ml1,-q1)* \
                                    jsim.j6(Lu,Ll,1.,Jl1,Ju1,SS)

                              # Check magnitude
                              if np.absolute(CCb) <= 0.:
                                  continue

                              # Factor
                              CCb *= np.sqrt(2.*Jl1 + 1.)

                              # If il1 < il0, conjugate
                              if il1 < il0:
                                jl0 = il1
                                jl1 = il0
                                conj = True
                              else:
                                jl0 = il0
                                jl1 = il1
                                conj = False

                              # Initialize rho
                              rho = 0j

                              # Find rho index
                              for index in terml.index:

                                # ju'Mu',juMu
                                if index[1] == jl0 and index[2] == jl1:

                                    # Add real
                                    rho += ese.rho[index[0]]

                                    # Not diagonal
                                    if jl0 != jl1:
                                        if conj:
                                            rho -= 1j*ese.rho[index[0]+1]
                                        else:
                                            rho += 1j*ese.rho[index[0]+1]
                                    # And exit
                                    break

                              # Scale
                              rho *= CCb

                              # Contribution
                              contr = np.real(rho*Tqq0voigt)
                              c1 = rho*Tqq1voigt
                              c2 = rho*Tqq2voigt
                              c3 = rho*Tqq3voigt

                              sum_etaa0[comp] += contr
                              sum_etaa1[comp] += np.real(c1)
                              sum_etaa2[comp] += np.real(c2)
                              sum_etaa3[comp] += np.real(c3)
                              sum_rhoa1[comp] += np.imag(c1)
                              sum_rhoa2[comp] += np.imag(c2)
                              sum_rhoa3[comp] += np.imag(c3)

                              contr = contr*cdts.nus_weights
                              line.add_contribution_profiles(contr, nu0)

                      # Emission

                      # For each Jl'
                      for Jl1_index in Mlblock:

                        # Get quantum number, index, and eigenvector
                        Jl1 = Jl1_index[2]
                        iJl1 = Jl1_index[1]
                        CC = j36*terml.eigvec[mul_index[0]][iJl1]*ufactor

                        # Eigenvec value
                        if np.absolute(CC) <= 0.:
                            continue

                        # Common 3J and 6J with eigenvectors
                        CC *= np.sqrt(2.*Jl1 + 1.)

                        # For Mu'
                        for iMu1,Mu1block in enumerate(termu.index_muM):

                          # Get Ml'
                          Mu1 = Mu1block[0][-1]

                          # Get q1 value and check valid
                          q1 = int(round(Ml - Mu1))
                          if np.absolute(q1) > 1:
                            continue

                          # Multiply Tqq and voigt
                          tag = f'{q}{q1}'
                          Tqq0voigt = TQQ[0][tag]*voigt
                          Tqq1voigt = TQQ[1][tag]*voigt
                          Tqq2voigt = TQQ[2][tag]*voigt
                          Tqq3voigt = TQQ[3][tag]*voigt

                          # Get ju'
                          for muu1_index in Mu1block:

                            # Get index
                            imuu1 = muu1_index[-2]
                            iu0 = muu1_index[0]

                            # Get Ju'
                            for Ju1_index in Mu1block:

                              # Get quantum number, index, and eigenvector
                              Ju1 = Ju1_index[2]
                              iJu1 = Ju1_index[1]
                              CCb = CC*termu.eigvec[muu1_index[0]][iJu1]* \
                                    jsim.j3(Ju1,Jl1,1.,-Mu1,Ml,-q1)* \
                                    jsim.j6(Lu,Ll,1.,Jl1,Ju1,SS)

                              # Check magnitude
                              if np.absolute(CCb) <= 0.:
                                  continue

                              # Factor
                              CCb *= np.sqrt(2.*Ju1 + 1.)

                              # If iu1 < iu0, conjugate
                              if iu1 < iu0:
                                ju0 = iu1
                                ju1 = iu0
                                conj = True
                              else:
                                ju0 = iu0
                                ju1 = iu1
                                conj = False

                              # Initialize rho
                              rho = 0j

                              # Find rho index
                              for index in termu.index:

                                # ju'Mu',juMu
                                if index[1] == ju0 and index[2] == ju1:

                                    # Add real
                                    rho += ese.rho[index[0]]

                                    # Not diagonal
                                    if ju0 != ju1:
                                        if conj:
                                            rho -= 1j*ese.rho[index[0]+1]
                                        else:
                                            rho += 1j*ese.rho[index[0]+1]
                                    # And exit
                                    break

                              # Scale
                              rho *= CCb

                              # Contribution
                              c0 = np.real(rho*Tqq0voigt)
                              c1 = rho*Tqq1voigt
                              c2 = rho*Tqq2voigt
                              c3 = rho*Tqq3voigt

                              sum_etas0[comp] += c0
                              sum_etas1[comp] += np.real(c1)
                              sum_etas2[comp] += np.real(c2)
                              sum_etas3[comp] += np.real(c3)
                              sum_rhos1[comp] += np.imag(c1)
                              sum_rhos2[comp] += np.imag(c2)
                              sum_rhos3[comp] += np.imag(c3)

          # Get final constants for the RT coefficients
         #hnu3 = hnu * self.hcm2*line.nu3
          hnutohnu3 = self.hcm2*line.nu3

          # Normalize line profile
          line.normalize_profiles()

          # Add to line contribution
          # rho U is negative because is the upper part of the triangular
          # propagation matrix
          try:
              for res in line.resos:
                eta_a0[res] += sum_etaa0[res]
                eta_a1[res] += sum_etaa1[res]
                eta_a2[res] += sum_etaa2[res]
                eta_a3[res] += sum_etaa3[res]
                eta_s0[res] += sum_etas0[res]
                eta_s1[res] += sum_etas1[res]
                eta_s2[res] += sum_etas2[res]
                eta_s3[res] += sum_etas3[res]
                rho_a0[res] += sum_rhoa1[res]
                rho_a1[res] -= sum_rhoa2[res]
                rho_a2[res] += sum_rhoa3[res]
                rho_s0[res] += sum_rhos1[res]
                rho_s1[res] -= sum_rhos2[res]
                rho_s2[res] += sum_rhos3[res]
                eps0[res]   += hnutohnu3 * sum_etas0[res]
                eps1[res]   += hnutohnu3 * sum_etas1[res]
                eps2[res]   += hnutohnu3 * sum_etas2[res]
                eps3[res]   += hnutohnu3 * sum_etas3[res]
          except NameError:
              eta_a0 =  {res: sum_etaa0[res] for res in line.resos}
              eta_a1 =  {res: sum_etaa1[res] for res in line.resos}
              eta_a2 =  {res: sum_etaa2[res] for res in line.resos}
              eta_a3 =  {res: sum_etaa3[res] for res in line.resos}
              eta_s0 =  {res: sum_etas0[res] for res in line.resos}
              eta_s1 =  {res: sum_etas1[res] for res in line.resos}
              eta_s2 =  {res: sum_etas2[res] for res in line.resos}
              eta_s3 =  {res: sum_etas3[res] for res in line.resos}
              rho_a0 =  {res: sum_rhoa1[res] for res in line.resos}
              rho_a1 =  {res: -sum_rhoa2[res] for res in line.resos}
              rho_a2 =  {res: sum_rhoa3[res] for res in line.resos}
              rho_s0 =  {res: sum_rhos1[res] for res in line.resos}
              rho_s1 =  {res: -sum_rhos2[res] for res in line.resos}
              rho_s2 =  {res: sum_rhos3[res] for res in line.resos}
              eps0  =  {res: hnutohnu3*sum_etas0[res] for res in line.resos}
              eps1  =  {res: hnutohnu3*sum_etas1[res] for res in line.resos}
              eps2  =  {res: hnutohnu3*sum_etas2[res] for res in line.resos}
              eps3  =  {res: hnutohnu3*sum_etas3[res] for res in line.resos}

          except:
              raise

        # Correct for stimulated emission
        eta0 = {res:eta_a0[res] - eta_s0[res] for res in line.resos}
        eta1 = {res:eta_a1[res] - eta_s1[res] for res in line.resos}
        eta2 = {res:eta_a2[res] - eta_s2[res] for res in line.resos}
        eta3 = {res:eta_a3[res] - eta_s3[res] for res in line.resos}
        rho0 = {res:rho_a0[res] - rho_s0[res] for res in line.resos}
        rho1 = {res:rho_a1[res] - rho_s1[res] for res in line.resos}
        rho2 = {res:rho_a2[res] - rho_s2[res] for res in line.resos}

        # Scale eta and rho
        for res in line.resos:
          eta1[res] /= (eta0[res] + cts.vacuum)
          eta2[res] /= (eta0[res] + cts.vacuum)
          eta3[res] /= (eta0[res] + cts.vacuum)
          rho0[res] /= (eta0[res] + cts.vacuum)
          rho1[res] /= (eta0[res] + cts.vacuum)
          rho2[res] /= (eta0[res] + cts.vacuum)
          eps0[res] /= (eta0[res] + cts.vacuum)
          eps1[res] /= (eta0[res] + cts.vacuum)
          eps2[res] /= (eta0[res] + cts.vacuum)
          eps3[res] /= (eta0[res] + cts.vacuum)

          # Check physical absorption
          if np.any(eta0[res] < 0) and self.no_warning:
              print(f"Warning: eta_I < 0 at iz = {ese.iz} dir = {ray.rinc}x{ray.raz}")
              for ifreq,e0 in enumerate(eta0[res]):
                  if e0 < 0.:
                      print(f"  Lambda {cts.c*1e7/cdts.nus[ifreq]:12.6f}: " + \
                            f"{eta_a0[ifreq]:13.6e} - {eta_s0[ifreq]:13.6e} = " + \
                            f"{e0:13.6e}    MAX: {np.max(eta0[res]):13.6e}")
              self.no_warning = False
              print(f"Will not bother you with more instances of this warning")

        # Build propagation matrix
        KK = [[eta0,eta1,eta2,eta3],[rho0,rho1,rho2]]

        # Build source function
        # SS = np.concatenate((eps0,eps1,eps2,eps3)).reshape((4,cdts.nus_N))
        SS = [eps0,eps1,eps2,eps3]


        return SS, KK
