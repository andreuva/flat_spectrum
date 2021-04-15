class ESE:
    """ A class that stores the atomic state and needs to be constantly updated
        during the Lambda iterations by providing the Stokes parameters.
        After every Lambda iteration, solveESE() needs to be called.
        It is assumed that only one spectral line is involved in the problem.
        This class needs to be instantiated at every grid point.
    """

    def __init__(self, nus_weights, B):
        """
            nus_weights: array of the frequency quadrature weights
            B: object of the magnetic field vector with xyz components (gauss)
            return value: None
        """
        pass

    def sumStokes(self, stokes, ray, ray_w):
        """
            Called per every Lambda iteration, grid point, and ray direction.
            stokes: list of 4 arrays of Stokes parameters [I,Q,U,V] in given
            point and direction (all frequencies)
            ray: object with .theta and .chi variables defining the ray of
            propagation direction
            ray_w: direction quadrature weight of the ray
            return value: None
        """
        pass

    def solveESE(self):
        """ Called at every grid point at the end of the Lambda iteration.
            return value: maximum relative change of the level population
        """
        pass

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state.
        """
        pass
