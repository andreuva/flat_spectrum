import numpy as np
import scipy.interpolate as inter
import constants as c


class Allen_class():
    """ Compute specific intensity from Allen
    """

    def __init__(self):
        """ Create interpolation functions to be used later.
            Inputs: None
            Outputs: None
            Internal: Defines interpolation functions for CLV
                      coefficients and continuum intensity (in
                      wavelength)
        """

        # Size
        N = 22

        # Lambda [micron]
        x = np.array([0.2e0,0.22e0,0.245e0,0.265e0,0.28e0, \
                      0.3e0,0.32e0,0.35e0,0.37e0,0.38e0, \
                      0.4e0,0.45e0,0.5e0,0.55e0,0.6e0,0.8e0, \
                      1e0,1.5e0,2e0,3e0,5e0,10e0])
        # u1 (CLV coefficient order 1)
        y1 = np.array([0.12e0,-1.30e0,-0.10e0,-0.10e0,0.38e0, \
                       0.74e0,0.88e0,0.98e0,1.03e0,0.92e0, \
                       0.91e0,0.99e0,0.97e0,0.93e0,0.88e0, \
                       0.73e0,0.64e0,0.57e0,0.48e0,0.35e0, \
                       0.22e0,0.15e0])
        # u2 (CLV coefficient order 2)
        y2 = np.array([0.33e0,1.6e0,0.85e0,0.90e0,0.57e0,0.20e0, \
                       0.03e0,-0.10e0,-0.16e0,-0.05e0,-0.05e0, \
                       -0.17e0,-0.22e0,-0.23e0,-0.23e0,-0.22e0, \
                       -0.20e0,-0.21e0,-0.18e0,-0.12e0,-0.07e0, \
                       -0.07e0])

        # Lambda [micron]
        xI = np.array([0.2e0,0.22e0,0.24e0,0.26e0,0.28e0,0.3e0, \
                       0.32e0,0.34e0,0.36e0,0.37e0,0.38e0,0.39e0, \
                       0.4e0,0.41e0,0.42e0,0.43e0,0.44e0,0.45e0, \
                       0.46e0,0.48e0,0.5e0,0.55e0,0.6e0,0.65e0, \
                       0.7e0,0.75e0,0.8e0,0.9e0,1e0,1.1e0,1.2e0, \
                       1.4e0,1.6e0,1.8e0,2e0,2.5e0,3e0,4e0,5e0, \
                       6e0,8e0,10e0,12e0])

        # Intensity [10^10 erg s^-1 cm^-2 sr^-1 micron^-1] TODO -> {CHECK}
        yI = np.array([0.06e0,0.21e0,0.29e0,0.6e0,1.3e0,2.45e0, \
                       3.25e0,3.77e0,4.13e0,4.23e0,4.63e0,4.95e0, \
                       5.15e0,5.26e0,5.28e0,5.24e0,5.19e0,5.1e0, \
                       5e0,4.79e0,4.55e0,4.02e0,3.52e0,3.06e0, \
                       2.69e0,2.28e0,2.03e0,1.57e0,1.26e0,1.01e0, \
                       0.81e0,0.53e0,0.36e0,0.238e0,0.16e0,0.078e0, \
                       0.041e0,0.0142e0,0.0062e0,0.0032e0,0.00095e0, \
                       0.00035e0,0.00018e0])

        # Convert units to cgs in Hz bandwidth
        yI *= 1e10*xI*xI/(c.c*1e4)

        # Interpolation functions
        self.fu1 = inter.interp1d(x,y1,kind='linear', \
                         bounds_error=False, \
                         fill_value=(y1[0],y1[-1]),assume_sorted=True)
        self.fu2 = inter.interp1d(x,y2,kind='linear', \
                         bounds_error=False, \
                         fill_value=(y2[0],y2[-1]),assume_sorted=True)
        self.fI = inter.interp1d(xI,yI,kind='linear', \
                        bounds_error=False, \
                        fill_value=(yI[0],yI[-1]),assume_sorted=True)

        # Initialize gamma angle
        self.sinG = 0.999999
        self.cosG = np.sqrt(1. - self.sinG*self.sinG)

    def get_gamma(self,h):
        """ Get gamma angle functions for a given height.
            Input: height [cm]
            Output: None
            Internal: Defines the cosine and the sine of the
                      gamma angle, the angle of the cone subtending
                      the Sun from a point at the height given in the
                      input
        """
        self.h = h
        self.sinG = c.R_sun / (c.R_sun + h)
        self.cosG = np.sqrt(1. - self.sinG*self.sinG)

    def get_ab_coeffs(self, h):
        """ Get the CLV coefficients for a given height.
            Input: height [cm]
            Output: None
            Internal: Defines the CLV coefficients a and b
        """
        self.get_gamma(h)
        cg = self.cosG
        cg2 = cg*cg
        cg3 = cg2*cg

        sg = self.sinG

        self.a0 = (1-cg)
        self.a1 = cg - 0.5 - 0.5*cg2/sg*np.log((1+sg)/cg)
        self.a2 = (cg+2)*(cg-1)/(3*(cg+1))

        self.b0 = 1/3 * (1-cg3)
        self.b1 = 1/24* (8*cg3 - 3*cg2 -2) - 1/8* cg3*cg/sg*np.log((1+sg)/cg)
        self.b2 = (cg-1)*(3*cg3 + 6*cg2 + 4*cg + 2)/(15*(cg+1))

    def get_anisotropy(self, nu, h):
        """ Get the anisotropy function for a given height.
            Input: None
            Output: None
            Internal: Defines the anisotropy function
        """
        self.get_ab_coeffs(h)
        I_0 = self.get_radiation(nu)
        # Convert input to lambda [micron]
        lamb = c.c*1e4/nu
        # u1 and u2 from pre-defined interpolation functions
        u1 = self.fu1(lamb)
        u2 = self.fu2(lamb)

        J_nu = 0.5*I_0*(self.a0 + self.a1*u1 + self.a2*u2)
        K_nu = 0.5*I_0*(self.b0 + self.b1*u1 + self.b2*u2)

        self.wnu_JK = (3*K_nu-J_nu)/(2*J_nu)

        J00 = J_nu
        J20 = J00*self.wnu_JK/np.sqrt(2)

        return self.wnu_JK, J00, J20

    def get_radiation(self,nus):
        """ Get radiation intensity from Allen
            Input: numpy array of frequencies [Hz]
            Output: continuum intensity at the requested frequencies
                    [erg s^-1 cm^-2 sr^-1 Hz^-1]
            Internal: None
        """

        # Convert to lambda [micron]
        lamb = c.c*1e4/nus

        # Intensity
        return self.fI(lamb)

    def get_clv(self,ray,nus):
        """ Get CLV from Allen
            Input: numpy array of frequencies [Hz]
            Output: CLV factor for the known height.
            Note: The height is pre-determined because
                  self.get_gamma() needs to be called first for
                  the desired height.
        """

        # Convert input to lambda [micron]
        lamb = c.c*1e4/nus

        # u1 and u2 from pre-defined interpolation functions
        u1 = self.fu1(lamb)
        u2 = self.fu2(lamb)

        # Get cosine of alpha angle (heliocentric angle)
        mu = np.cos(ray.rinc)
        arg = mu*mu - self.cosG*self.cosG
        if arg >= 0.:
            cosa = np.sqrt(mu*mu - self.cosG*self.cosG)/self.sinG
        else:
            return 0.*u1

        # Return geometric factor
        return 1. - u1*(1. - cosa) - u2*(1. - cosa*cosa)
