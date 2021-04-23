import numpy as np
from tqq import Tqq
from astropy import units as u
from parameters import I_units as Iu
from astropy.modeling.models import BlackBody as bb

class RTE:
    """RTE class containing the stokes parameters and the Jqq as well as other
    Radiation information"""

    def __init__(self, nus_N):
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
