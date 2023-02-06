import numpy as np
from numpy import sin, cos, exp, sqrt, conj
import math,copy


#@jit(nopython=True)
def voigt(v, a):
    """ Compute the Voigt profile for a list of frequencies (normalized)
    """

    ss = np.abs(v)+a
    dd = (.195e0*np.abs(v)-.176e0)
    zz = a - 1.j*v
    res = v*0.j

    # Run over frequencies
    for i,s,z,d in zip(range(len(ss)),ss,zz,dd):

        if s >= .15e2:
            t = .5641896e0*z/(.5+z*z)
        else:

            if s >= .55e1:

                u = z*z
                t = z*(.1410474e1 + .5641896e0*u)/(.75e0 + u*(.3e1 + u))

            else:

                if a >= d:
                    nt = .164955e2 + z*(.2020933e2 + z*(.1196482e2 +
                                            z*(.3778987e1 + .5642236e0*z)))
                    dt = .164955e2 + z*(.3882363e2 + z*(.3927121e2 +
                                            z*(.2169274e2 + z*(.6699398e1 + z))))
                    t = nt / dt
                else:
                    u = z*z
                    x = z*(.3618331e5 - u*(.33219905e4 - u*(.1540787e4 -
                               u*(.2190313e3 - u*(.3576683e2 - u*(.1320522e1 -
                                  .56419e0*u))))))
                    y = .320666e5 - u*(.2432284e5 - u*(.9022228e4 -
                                       u*(.2186181e4 - u*(.3642191e3 - u*(.6157037e2 -
                                          u*(.1841439e1 - u))))))
                    t = np.exp(u) - x/y
        res[i] = t

    return res

def voigt_custom(x, sigma, gamma, x0=0):
    """
    Return the Voigt line shape at x with Lorentzian component gamma
    and Gaussian component sigma.
    """
    return np.real(special.wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) \
        / sigma / np.sqrt(2*np.pi)

def Tqq_all(theta,chi):
    """ Return all the Tqq
    """

    # Initialize output
    out = [{},{},{},{}]

    ct = cos(theta)
    st = sin(theta)
    st2 = st*st
    ct2 = ct*ct
    cst = ct*st
    cc = cos(chi)
    sc = sin(chi)
    c2c = cos(2.*chi)
    s2c = sin(2.*chi)
    s2i = 1./np.sqrt(2)
    ec = (cc - 1j*sc)
    e2c = (c2c - 1j*s2c)

    # I
    out[0]['-1-1'] = 0.25*(1+ct2)
    out[0]['00'] = 0.5*st2
    out[0]['11'] = out[0]['-1-1']
    out[0]['-10'] = -0.5*s2i*cst*ec
    out[0]['0-1'] = np.conjugate(out[0]['-10'])
    out[0]['-11'] = 0.25*st2*e2c
    out[0]['1-1'] = np.conjugate(out[0]['-11'])
    out[0]['01'] = -1.*out[0]['-10']
    out[0]['10'] = np.conjugate(out[0]['01'])
    # Q
    out[1]['-1-1'] = -0.25*st2
    out[1]['00'] = 0.5*st2
    out[1]['11'] = out[1]['-1-1']
    out[1]['-10'] = -0.5*s2i*cst*ec
    out[1]['0-1'] = np.conjugate(out[1]['-10'])
    out[1]['-11'] = -0.25*(1+ct2)*e2c
    out[1]['1-1'] = np.conjugate(out[1]['-11'])
    out[1]['01'] = -1.*out[1]['-10']
    out[1]['10'] = np.conjugate(out[1]['01'])
    # U
    out[2]['-1-1'] = 0.
    out[2]['00'] = 0.
    out[2]['11'] = 0.
    out[2]['-10'] = 0.5*s2i*1j*st*ec
    out[2]['0-1'] = np.conjugate(out[2]['-10'])
    out[2]['-11'] = 0.5*ct*e2c*1j
    out[2]['1-1'] = np.conjugate(out[2]['-11'])
    out[2]['01'] = -1.*out[2]['-10']
    out[2]['10'] = np.conjugate(out[2]['01'])
    # V
    out[3]['-1-1'] = -0.5*ct
    out[3]['00'] = 0.
    out[3]['11'] = 0.5*ct
    out[3]['-10'] = 0.5*s2i*st*ec
    out[3]['0-1'] = np.conjugate(out[3]['-10'])
    out[3]['-11'] = 0.
    out[3]['1-1'] = 0.
    out[3]['01'] = out[3]['-10']
    out[3]['10'] = np.conjugate(out[3]['01'])

    # And return full Tqq
    return out

#@jit(nopython=True)
def Tqq(q1, q2, i, theta, chi):
    t = 0.+0j
    c = False
    if q2 < q1:
        c = True
        tmp = q1
        q1 = q2
        q2 = tmp

    if i == 0:
        if q1 == q2 == -1 or q1 == q2 == 1:
            t = 1/4*(1+cos(theta)**2) + 0j
        elif q1 == q2 == 0:
            t = 1/2*sin(theta)**2 + 0j
        elif q1 == -1 and q2 == 0:
            t = -1/2/sqrt(2)*sin(theta)*cos(theta)*exp(-chi*1j)
        elif q1 == -1 and q2 == 1:
            t = 1/4*sin(theta)**2*exp(-chi*2j)
        elif q1 == 0 and q2 == 1:
            t = 1/2/sqrt(2)*sin(theta)*cos(theta)*exp(-chi*1j)
    elif i == 1:
        if q1 == q2 == -1 or q1 == q2 == 1:
            t = -1/4*sin(theta)**2 + 0j
        elif q1 == q2 == 0:
            t = 1/2*sin(theta)**2 + 0j
        elif q1 == -1 and q2 == 0:
            t = -1/2/sqrt(2)*cos(theta)*sin(theta)*exp(-chi*1j)
        elif q1 == -1 and q2 == 1:
            t = -1/4*(1+cos(theta)**2)*exp(-chi*2j)
        elif q1 == 0 and q2 == 1:
            t = 1/2/sqrt(2)*cos(theta)*sin(theta)*exp(-chi*1j)
    elif i == 2:
        if q1 == -1 and q2 == 0:
            t = 1/2/sqrt(2)*1j*sin(theta)*exp(-chi*1j)
        elif q1 == -1 and q2 == 1:
            t = 1/4*2j*cos(theta)*exp(-chi*2j)
        elif q1 == 0 and q2 == 1:
            t = -1/2/sqrt(2)*1j*sin(theta)*exp(-chi*1j)
    elif i == 3:
        if q1 == q2 == -1:
            t = -1/2*cos(theta) + 0j
        elif q1 == q2 == 1:
            t = 1/2*cos(theta) + 0j
        elif q1 == -1 and q2 == 0:
            t = 1/2/sqrt(2)*sin(theta)*exp(-chi*1j)
        elif q1 == 0 and q2 == 1:
            t = 1/2/sqrt(2)*sin(theta)*exp(-chi*1j)

    if not c:
        return t
    else:
        return conj(t)


######################################################################
# Tanaus\'u del Pino Alem\'an                                        #
#   Instituto de Astrof\'isica de Canarias                           #
#   Class to compute 3, 6, and 9 j-symbols                           #
#                                                                    #
#  26/10/2018 - V1.0.0 - First version. (TdPA)                       #
######################################################################

class jsymbols():
    ''' Class to compute 3, 6, and 9 j-symbols
    '''

    def __init__(self, initialize=None, memoization=None):
        ''' Class initializer
        '''

        # Initialize factorial and sign list
        self.__fact = [0.]
        self.__sign = [1.]

        # Initialize factorial list up to the specified value
        if initialize is not None:
            if isinstance(initialize, int) or \
               isinstance(initialize, float):
                a = self.logfct(initialize)
                a = self.sign(initialize)

        # Check memoization
        self.memoization = False
        if isinstance(memoization, bool):
            if memoization:
                self.memoization = True
                self.memo = {}

######################################################################

    def logfct(self, val):
        ''' Returns the logarithm of the factorial of val
            val is expected to be an integer
        '''

        ival = abs(val)

        # Compute and store up to val
        while len(self.__fact) <= ival:

            self.__fact.append(self.__fact[-1] +
                               math.log(float(len(self.__fact))))

        return self.__fact[ival]

######################################################################

    def sign(self, val):
        ''' Returns the sign of val
        '''

        # Real val
        ival = int(math.fabs(val))

        # Compute and store up to val
        while len(self.__sign) <= ival:

            self.__sign.append(self.__sign[-1] * -1.)

        return self.__sign[ival]

######################################################################

    def __fn1(self, j1, j2, j3, m1, m2, m3):
        ''' Auxiliar used in 3j calculations
        '''

        l1 = int(round(j1 + j2 - j3))
        l2 = int(round(j2 + j3 - j1))
        l3 = int(round(j3 + j1 - j2))
        l4 = int(round(j1 + j2 + j3) + 1)
        l5 = int(round(j1 + m1))
        l6 = int(round(j1 - m1))
        l7 = int(round(j2 + m2))
        l8 = int(round(j2 - m2))
        l9 = int(round(j3 + m3))
        l10 = int(round(j3 - m3))

        fn1 = 0.5*(self.logfct(l1) + self.logfct(l2) +
                   self.logfct(l3) - self.logfct(l4) +
                   self.logfct(l5) + self.logfct(l6) +
                   self.logfct(l7) + self.logfct(l8) +
                   self.logfct(l9) + self.logfct(l10))

        return fn1

######################################################################

    def __fn2(self, ij1, ij2, ij3):
        ''' Auxiliar used in 3j calculations
        '''

        l1 = int(round(ij1+ij2-ij3))//2
        l2 = int(round(ij2+ij3-ij1))//2
        l3 = int(round(ij3+ij1-ij2))//2
        l4 = int(round(ij1+ij2+ij3))//2 + 1

        fn2 = 0.5*(self.logfct(l1) + self.logfct(l2) +
                   self.logfct(l3) - self.logfct(l4))

        return fn2

######################################################################

    def j3(self, j1, j2, j3, m1, m2, m3):
        ''' Compute 3j symbol
        '''

        # If memoization
        if self.memoization:

            # Get tag
           #tag = f'{j1}{j2}{j3}{m1}{m2}{m3}'
            tag = (j1,j2,j3,m1,m2,m3)

            try:
                return self.memo[tag]
            except:
                pass

        # Initialize value
        js3 = 0.0

        # Conver to integer combinations
        ij1 = int(round(j1 + j1))
        ij2 = int(round(j2 + j2))
        ij3 = int(round(j3 + j3))

        # Selection rules
        if ij1 + ij2 - ij3 < 0:
            return js3
        if ij2 + ij3 - ij1 < 0:
            return js3
        if ij3 + ij1 - ij2 < 0:
            return js3

        # Conver to integer combinations
        im1 = int(round(m1 + m1))
        im2 = int(round(m2 + m2))
        im3 = int(round(m3 + m3))

        # Selection rules
        if im1 + im2 + im3 != 0:
            return js3
        if math.fabs(im1) > ij1:
            return js3
        if math.fabs(im2) > ij2:
            return js3
        if math.fabs(im3) > ij3:
            return js3

        # Get minimum index to run from
        kmin = (ij3 - ij1 - im2)//2
        kmin1 = int(kmin)
        kmin2 = int((ij3 - ij2 + im1)//2)
        kmin = max(-1*min(kmin1, kmin2), 0)

        # Get maximum index to run to
        kmax = int(round(j1 + j2 - j3))
        kmax1 = int(kmax)
        kmax2 = int(round(j1 - m1))
        kmax3 = int(round(j2 + m2))
        kmax = min([kmax, kmax2, kmax3])

        if kmin <= kmax:

            term1 = self.__fn1(j1, j2, j3, m1, m2, m3)

            sgn = self.sign((ij1 - ij2 - im3)//2)

            for i in range(kmin, kmax+1):

                term2 = self.logfct(i) + self.logfct(kmin1+i) + \
                    self.logfct(kmin2+i) + self.logfct(kmax1-i) + \
                    self.logfct(kmax2-i) + self.logfct(kmax3-i)
                js3 = self.sign(i)*math.exp(term1-term2) + js3

            js3 = sgn*js3

        # If memoization
        if self.memoization:
            self.memo[tag] = js3

        return js3

######################################################################

    def j6(self, j11, j12, j13, j21, j22, j23):
        ''' Compute 6j symbol
        '''

        # If memoization
        if self.memoization:

            # Get tag
           #tag = f'{j11}{j12}{j13}{j21}{j22}{j23}'
            tag = (j11,j12,j13,j21,j22,j23)

            try:
                return self.memo[tag]
            except:
                pass

        # Initialize value
        js6 = 0.0

        # Conver to integer combinations
        ij1 = int(round(j11 + j11))
        ij2 = int(round(j12 + j12))
        ij3 = int(round(j13 + j13))
        ij4 = int(round(j21 + j21))
        ij5 = int(round(j22 + j22))
        ij6 = int(round(j23 + j23))

        ijm1 = (ij1 + ij2 + ij3)//2
        ijm2 = (ij1 + ij5 + ij6)//2
        ijm3 = (ij4 + ij2 + ij6)//2
        ijm4 = (ij4 + ij5 + ij3)//2

        ijm = ijm1

        ijm = max([ijm, ijm2, ijm3, ijm4])

        ijx1 = (ij1 + ij2 + ij4 + ij5)//2
        ijx2 = (ij2 + ij3 + ij5 + ij6)//2
        ijx3 = (ij3 + ij1 + ij6 + ij4)//2

        ijx = ijx1

        ijx = min([ijx, ijx2, ijx3])

        if ijm <= ijx:

            term1 = self.__fn2(ij1, ij2, ij3) + \
                    self.__fn2(ij1, ij5, ij6) + \
                    self.__fn2(ij4, ij2, ij6) + \
                    self.__fn2(ij4, ij5, ij3)

            for i in range(int(ijm), int(ijx)+1):

                term2 = self.logfct(i+1) - self.logfct(i-ijm1) - \
                    self.logfct(i-ijm2) - self.logfct(i-ijm3) - \
                    self.logfct(i-ijm4) - self.logfct(ijx1-i) - \
                    self.logfct(ijx2-i) - self.logfct(ijx3-i)
                js6 = self.sign(i)*math.exp(term1+term2) + js6

        # If memoization
        if self.memoization:
            self.memo[tag] = js6

        return js6

######################################################################

    def j9(self, j11, j12, j13, j21, j22, j23, j31, j32, j33):
        ''' Compute 9j symbol
        '''

        # If memoization
        if self.memoization:

            # Get tag
           #tag = f'{j11}{j12}{j13}{j21}{j22}{j23}{j31}{j32}{j33}'
            tag = (j11,j12,j13,j21,j22,j23,j31,j32,j33)

            try:
                return self.memo[tag]
            except:
                pass

        # Initialize value
        js9 = 0.0

        # Conver to integer combinations
        ij11 = int(round(j11 + j11))
        ij12 = int(round(j12 + j12))
        ij13 = int(round(j13 + j13))
        ij21 = int(round(j21 + j21))
        ij22 = int(round(j22 + j22))
        ij23 = int(round(j23 + j23))
        ij31 = int(round(j31 + j31))
        ij32 = int(round(j32 + j32))
        ij33 = int(round(j33 + j33))

        kmin1 = int(round(math.fabs(ij11 - ij33)))
        kmin2 = int(round(math.fabs(ij32 - ij21)))
        kmin3 = int(round(math.fabs(ij23 - ij12)))

        kmin1 = max([kmin1, kmin2, kmin3])

        kmax1 = ij11 + ij33
        kmax2 = ij32 + ij21
        kmax3 = ij23 + ij12

        kmax1 = min(kmax1, kmax2, kmax3)

        if kmin1 <= kmax1:

            for k in range(kmin1, kmax1+1, 2):

                hk = 0.5*float(k)

                js9 = self.sign(k)*float(k+1) * \
                      self.j6(j11, j21, j31, j32, j33, hk) * \
                      self.j6(j12, j22, j32, j21, hk, j23) * \
                      self.j6(j13, j23, j33, hk, j11, j12) + js9

        # If memoization
        if self.memoization:
            self.memo[tag] = js9

        return js9

######################################################################
# Tanaus\'u del Pino Alem\'an                                        #
#   Instituto de Astrof\'isica de Canarias                           #
#   Class to compute rotation matrices                               #
#   Modified to return complex number instead of list                #
#                                                                    #
#  15/01/2021 - V1.0.0 - First version. (TdPA)                       #
######################################################################

######################################################################
######################################################################
######################################################################

class rotate_ist():
    ''' Class to rotate irreducible spherical tensors
    '''

######################################################################
######################################################################
######################################################################

    def __init__(self, initialize=None):
        ''' Class initializer
        '''

        # Initialize factorial and sign list
        self.__fact = [0.]
        self.__sign = [1.]

        # Initialize factorial list up to the specified value
        if initialize is not None:
            if isinstance(initialize, int) or \
               isinstance(initialize, float):
                a = self.logfct(initialize)
                a = self.sign(initialize)

######################################################################
######################################################################
######################################################################

    def logfct(self, val):
        ''' Returns the logarithm of the factorial of val
        '''

        # Real val
        ival = int(math.fabs(val))

        # Compute and store up to val
        while len(self.__fact) <= ival:

            self.__fact.append(self.__fact[-1] + \
                               math.log(float(len(self.__fact))))

        return self.__fact[ival]

######################################################################
######################################################################
######################################################################

    def sign(self, val):
        ''' Returns the sign of val
        '''

        # Real val
        ival = int(math.fabs(val))

        # Compute and store up to val
        while len(self.__sign) <= ival:

            self.__sign.append(self.__sign[-1] * -1.)

        return self.__sign[ival]

######################################################################
######################################################################
######################################################################

    def __nint(self,num):
        ''' Fortran's nint
        '''

        return int(round(num))

######################################################################
######################################################################
######################################################################

    def __rdmat(self,rJ,rM1,rM,theta):
        ''' Compute rotation matrix d[rJ,rM1,rM](theta)
            -180 < theta < 180
        '''

        thalf = 0.5*theta

        ss = math.sin(thalf)
        cc = math.cos(thalf)

        ifs = self.__nint(rJ+rM)
        ifd = self.__nint(rJ-rM)
        ifs1 = self.__nint(rJ+rM1)
        ifd1 = self.__nint(rJ-rM1)
        imd = self.__nint(rM1-rM)
        imdf2 = self.__nint(rJ+rJ) - imd

        imax = min(ifs,ifd1)
        imin = max(0,-imd)

        tmp = 0.0

        for i in range(imin,imax+1):

            i2 = i + i

            k1 = imdf2 - i2
            k2 = imd + i2

            ess = 0.0
            ecc = 0.0

            if math.fabs(theta-math.pi) > 0.0:
                ecc = cc**k1
            elif k1 == 0:
                ecc = 1.0

            if math.fabs(theta) > 0.0:
                ess = ss**k2
            elif K2 == 0:
                ess = 1.0

            k = imd + i

            tmp = self.sign(k)* \
                  math.exp(-self.logfct(k) - \
                            self.logfct(ifs-i) - \
                            self.logfct(ifd1-i) - \
                            self.logfct(i))*ecc*ess + tmp

        rdmat = tmp*math.exp(0.5* \
                (self.logfct(ifs) + self.logfct(ifd) + \
                 self.logfct(ifs1) + self.logfct(ifd1)))

        return rdmat

######################################################################
######################################################################
######################################################################

    def get_DKQQ(self,kmax,theta,chi,conjugate=None,backwards=None):
        ''' Return the DKQQ tensor for the given maximum rank,
            angles, and rotation variables.
            Rotated an angle theta (polar) and chi (azimuth)
        '''

        # Deal with inputs
        if isinstance(conjugate, bool):
            conj = conjugate
        else:
            conj = False

        # Deal with inputs
        if isinstance(backwards, bool):
            back = backwards
        else:
            back = False

        # Compute complex exponential part

        # Initialize arrays
        ceR = []
        ceI = []
        ceIm = []

        Kmax = kmax + 1

        # For each K to include
        for K in range(1,Kmax):
            ang = float(K)*chi
            ceR.append(math.cos(ang))
            ceI.append(math.sin(ang))
            ceIm.append(-ceI[-1])

        # Rearrange the numbers
        ceR = ceR[::-1] + [1.0] + ceR
        ceI = ceIm[::-1] + [0.0] + ceI

        # Compute rotation tensor D[K][Q][Q'] and rotate
        DKQQ = {}

        # For each K multipole
        for K in range(1,Kmax):

            # Float and dimension in Q space
            rK = float(K)
            NQ = 2*K + 1

            # Initialize Q space
            DKQQ[K] = {}

            # Initialize rolling Q
            aQ = -K - 1

            # For each Q
            for iQ in range(NQ):

                # Rolling Q and its float
                aQ += 1
                rQ = float(aQ)

                # Initialize Q' space
                DKQQ[K][aQ] = {}

                # Initialize rolling Q
                aQ1 = -K - 1

                # For each Q'
                for iQ1 in range(NQ):

                    # Rolling Q' and its float
                    aQ1 += 1
                    rQ1 = float(aQ1)

                    # Get dKQQ
                    dkqq = self.__rdmat(rK,rQ1,rQ,theta)

                    #
                    # Get exponential parts depending on the
                    # configuration
                    #

                    # If rotating with complex conjugated
                    if conj:

                        # If rotating backwards
                        if back:
                            cer = ceR[kmax+aQ]
                            cei = ceI[kmax+aQ]
                        # If rotating forwards
                        else:
                            cer = ceR[kmax+aQ1]
                            cei = ceI[kmax+aQ1]

                    # If rotating with normal tensor
                    else:

                        # If rotating backwards
                        if back:
                            cer = ceR[kmax-aQ]
                            cei = ceI[kmax-aQ]
                        # If rotating forwards
                        else:
                            cer = ceR[kmax-aQ1]
                            cei = ceI[kmax-aQ1]

                    # Add the DKQQ value
                    DKQQ[K][aQ][aQ1] = cer*dkqq + 1j*cei*dkqq

        # Return the new tensor
        return DKQQ
