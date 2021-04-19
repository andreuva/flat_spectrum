import numpy as np


class RTcoefs:
    """ Radiative transfer coefficients.
        Just one instance of the class needs to be created in the program.
    """

    def __init__(self, nus):
        """
            nus: array of the line frequencies
            return value: None
        """
        pass

    def getRTcoefs(self, ese, ray):
        """ Provides the 4-vector of epsilon and the 4x4 K-matrix for the point
            with given ESE state and ray direction.
            ese: the local instance of the ESE class
            ray: object with .theta and .chi variables defining the ray ç
            of propagation direction
            return value: [epsilon (vector in frequencies), K (4x4 list of
            vectors in frequencies)]
        """
        z = np.zeros(64)
        eps = [z, z, z, z]
        KK = [[z, z, z, z] for i in range(4)]
        return eps, KK
