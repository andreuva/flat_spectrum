import numpy as np
from astropy import units as u
from astropy.modeling.models import BlackBody as bb


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
