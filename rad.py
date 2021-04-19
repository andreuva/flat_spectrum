import numpy as np
from tqq import Tqq

class RTE:

    def __init__(self, nus_N):
        I = np.zeros(nus_N)
        Q = np.zeros(nus_N)
        U = np.zeros(nus_N)
        V = np.zeros(nus_N)
        self.stokes = [I,Q,U,V]

        # Defining the Jqq' (z = 0, m = -1, p = +1)
        self.jzz = np.zeros(nus_N)
        self.jpp = np.zeros(nus_N)
        self.jmm = np.zeros(nus_N)
        self.jmz = np.zeros(nus_N)
        self.jmp = np.zeros(nus_N)
        self.jzp = np.zeros(nus_N)

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
