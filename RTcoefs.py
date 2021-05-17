import parameters as pm
from physical_functions import Tqq, jsymbols

import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as cts
from astropy import units as unt
# from py3nj import wigner3j


class RTcoefs:
    """ Radiative transfer coefficients.
        Just one instance of the class needs to be created in the program.
    """
    jsim = jsymbols()

    def __init__(self, nus):
        """
            nus: array of the line frequencies
            return value: None
        """
        pass

    def getRTcoefs(self, ese, ray, cdts):
        """ Provides the 4-vector of epsilon and the 4x4 K-matrix for the point
            with given ESE state and ray direction.
            ese: the local instance of the ESE class
            ray: object with .theta and .chi variables defining the ray
            of propagation direction
            return value: [S (source function vector in frequencies), K (4x4 list of
            vectors in frequencies)]
        """

        # Eq 7.10 of LL04

        KK = np.zeros((4, 4, pm.wn))
        SS = np.zeros((4, pm.wn))*pm.I_units

        return SS, KK
