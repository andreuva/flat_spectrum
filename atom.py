import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.modeling.models import BlackBody as bb


class level():
    """Class that defines the energy level of the atomic model"""
    def __init__(self, energy, JJ, g):
        self.E = energy
        self.g = g
        self.J = JJ
        self.M = [ml for ml in range(-self.J, self.J+1)]


class transition():
    def __init__(self, level_u, level_d, Alu):
        self.upper = level_u
        self.lower = level_d

        self.wavelength = 1/(level_u.E - level_d.E)
        self.energy = c.h.cgs * c.c.cgs / self.wavelength
        self.nu = self.energy/c.h.cgs

        self.A_lu = Alu
        self.B_lu = Alu * c.c.cgs**3/(2*c.h.cgs*self.nu**3)
        self.B_ul = self.B_lu * self.upper.g/self.lower.g

        self.dJ = level_u.J - level_d.J


class HeI_1083():
    """
    Class to acces the atomic model
    """
    def __init__(self):
        self.levels = [level(169_086.8428979/u.cm, 1, 3),      # MNIST data for HeI levels (2 level atom)
                       level(159_855.9743297/u.cm, 0, 3)]

        self.transitions = [transition(*self.levels, 1.0216e+07)]


class ESE:
    """ A class that stores the atomic state and needs to be constantly updated
        during the Lambda iterations by providing the Stokes parameters.
        After every Lambda iteration, solveESE() needs to be called.
        It is assumed that only one spectral line is involved in the problem.
        This class needs to be instantiated at every grid point.
    """

    def __init__(self, v_dop, a_voigt, nus, nus_weights, B):
        """
            nus_weights: array of the frequency quadrature weights
            B: object of the magnetic field vector with xyz components (gauss)
            return value: None
        """
        self.rho_l = 1
        self.rho_u = 1

    def solveESE(self, rad):
        """ Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """
        return 1
