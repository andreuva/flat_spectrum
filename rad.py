import numpy as np
from tqq import Tqq
from astropy import units as u
from parameters import I_units as Iu
from astropy.modeling.models import BlackBody as bb

class RTE:
    """RTE class containing the stokes parameters and the Jqq as well as other
    Radiation information"""

    def __init__(self, nus_N, v_dop):
        # Initialicing the I to 0
        I = np.zeros(nus_N)
        Q = np.zeros(nus_N)
        U = np.zeros(nus_N)
        V = np.zeros(nus_N)
        self.stokes = np.array([I,Q,U,V])*Iu

        # Defining the Jqq as nested dictionaries'
        self.jqq = {}
        for qq in [-1,0,1]:
            self.jqq[qq] = {}
            for qp in [-1,0,1]:
                self.jqq[qq][qp] = np.zeros(nus_N)*Iu * u.sr

    
    def make_IC(self, nus):
        # If a point is defined as IC put Q=U=V=0 and I to BB
        self.stokes = self.stokes*0
        self.stokes[0] = bb(temperature = 5772 * u.K )(nus)


    def sumStokes(self, ray):
        """
            Called per every Lambda iteration, grid point, and ray direction.
            stokes: list of 4 arrays of Stokes parameters [I,Q,U,V] in given
            point and direction (all frequencies)
            ray: object with .theta, .chi and weight variables defining the
            ray of propagation direction
            return value: None
        """
        for i in range(4):
            for qq in [-1,0,1]:
                for qp in [-1,0,1]:
                    self.jqq[qq][qp] = self.jqq[qq][qp] + ray.weight * Tqq(qq, qp, i, ray.inc.value, ray.az.value) * self.stokes[i] * u.sr

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state."""
        for qq in [-1,0,1]:
                for qp in [-1,0,1]:
                    self.jqq[qq][qp] = self.jqq[qq][qp] * 0
    
    
    def voigt(v, a):

        s = abs(v)+a
        d = .195e0*abs(v)-.176e0
        z = a - 1j*v

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
                    x = z*(.3618331e5 - u*(.33219905e4 - u*(.1540787e4 - \
                        u*(.2190313e3 - u*(.3576683e2 - u*(.1320522e1 - \
                        .56419e0*u))))))
                    y = .320666e5 - u*(.2432284e5 - u*(.9022228e4 - \
                        u*(.2186181e4 - u*(.3642191e3 - u*(.6157037e2 - \
                        u*(.1841439e1 - u))))))
                    t = np.exp(u) - x/y
        return t


    def voigt_custom(x, sigma, gamma, x0=0):
        """
        Return the Voigt line shape at x with Lorentzian component gamma
        and Gaussian component sigma.
        """
        return np.real(special.wofz(((x-x0) + 1j*gamma)/sigma/np.sqrt(2))) \
            / sigma / np.sqrt(2*np.pi)
