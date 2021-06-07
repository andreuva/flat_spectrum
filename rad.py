from physical_functions import Tqq
from parameters import I_units as Iu

from astropy.modeling.models import BlackBody as bb
from astropy import units as u
import numpy as np


class RTE:
    """RTE class containing the stokes parameters and the Jqq as well as other
    Radiation information"""

    def __init__(self, nus, v_dop):
        # Initialicing the I to 0
        II = np.zeros(len(nus))
        QQ = np.zeros(len(nus))
        UU = np.zeros(len(nus))
        VV = np.zeros(len(nus))
        self.stokes = np.array([II, QQ, UU, VV])*Iu

        self.nus = nus
        # Defining the Jqq as nested dictionaries'
        self.jqq = {}
        for qq in [-1, 0, 1]:
            self.jqq[qq] = {}
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = np.zeros(len(nus))*Iu * u.sr

    def make_IC(self, nus):
        # If a point is defined as IC put Q=U=V=0 and I to BB
        self.stokes = self.stokes*0
        self.stokes[0] = bb(temperature=5772*u.K)(nus)

    def check_I(self):
        if np.any(self.stokes[0] < 0):
            print(f"Warning: Negative intensity {self.stokes[0][self.stokes[0] < 0]}")
            input("Press Enter to continue...")

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
            for qq in [-1, 0, 1]:
                for qp in [-1, 0, 1]:
                    self.jqq[qq][qp] = self.jqq[qq][qp] + \
                                       ray.weight*Tqq(qq, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)\
                                       * self.stokes[i] * u.sr

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state."""
        for qq in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = self.jqq[qq][qp] * 0

    def Jqq_nu(self, q, qp, nu):
        jqqp = self.jqq[q][qp]
        return np.interp(nu, self.nus, jqqp).value
