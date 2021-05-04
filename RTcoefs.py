import parameters as pm
from physical_functions import Tqq, jsymbols

import numpy as np
from astropy import constants as cts
from astropy import units as unt


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

        Blu = 1
        sum_mq = 0
        for q in [-1, 0, 1]:
            for q_p in [-1, 0, 1]:
                for M_u in [-1, 0, 1]:
                    sum_mq = sum_mq + 3*self.jsim.j6(cdts.ju, cdts.jl, 1, -M_u, 0, -q) * \
                                        self.jsim.j6(cdts.ju, cdts.jl, 1, -M_u, 0, -q_p)
                    sum_mq = sum_mq + np.real(Tqq(q, q_p, 0, ray.inc.value, ray.az.value) * ese.rho_l * cdts.voigt_profile(cdts.a_voigt))

        eta = cts.h.cgs*cdts.nus/(4*np.pi) * \
            (2*cdts.jl + 1) * Blu * sum_mq

        S = np.zeros((4, pm.wn))*pm.I_units
        KK = np.zeros((4, 4, pm.wn))
        return S, KK
