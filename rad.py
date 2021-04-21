import numpy as np
from tqq import Tqq
from astropy import units as u
from parameters import I_units as Iu
from astropy.modeling.models import BlackBody as bb

class RTE:

    def __init__(self, nus_N):
        I = np.zeros(nus_N)*Iu
        Q = np.zeros(nus_N)*Iu
        U = np.zeros(nus_N)*Iu
        V = np.zeros(nus_N)*Iu
        self.stokes = [I,Q,U,V]

        # Defining the Jqq' (z = 0, m = -1, p = +1)
        self.jzz = np.zeros(nus_N)*Iu
        self.jpp = np.zeros(nus_N)*Iu
        self.jmm = np.zeros(nus_N)*Iu
        self.jmz = np.zeros(nus_N)*Iu
        self.jmp = np.zeros(nus_N)*Iu
        self.jzp = np.zeros(nus_N)*Iu

    
    def make_IC(self, nus):
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
            self.jzz = self.jzz + ray.weight * Tqq(0, 0, i, ray.inc.value, ray.az.value) * self.stokes[i]
            self.jpp = self.jpp + ray.weight * Tqq(1, 1, i, ray.inc.value, ray.az.value) * self.stokes[i]
            self.jmm = self.jmm + ray.weight * Tqq(-1, -1, i, ray.inc.value, ray.az.value) * self.stokes[i]
            self.jmz = self.jmz + ray.weight * Tqq(-1, 0, i, ray.inc.value, ray.az.value) * self.stokes[i]
            self.jmp = self.jmp + ray.weight * Tqq(-1, 1, i, ray.inc.value, ray.az.value) * self.stokes[i]
            self.jzp = self.jzp + ray.weight * Tqq(0, 1, i, ray.inc.value, ray.az.value) * self.stokes[i]

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state."""
        self.jzz = self.jzz * 0
        self.jpp = self.jpp * 0
        self.jmm = self.jmm * 0
        self.jmz = self.jmz * 0
        self.jmp = self.jmp * 0
        self.jzp = self.jzp * 0
