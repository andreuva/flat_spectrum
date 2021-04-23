import numpy as np
from astropy import units, constants
from atom import ESE
from rad import RTE


class ray:
    """ray class representing each ray in the angular quadrature"""

    def __init__(self, weight, inclination, azimut):
        self.weight = weight
        self.inc = inclination
        self.az = azimut

    def is_downward(self):
        """ Checking if the ray is downward (inclination > 90º or mu < 0)"""
        if self.inc > 90 * units.deg:
            return True
        else: 
            return False


class conditions:
    """Class cointaining the conditions that are not going to change during the program
    such as grids, quadratures, auxiliare matrices and some parameters"""

    def __init__(self, parameters):
        """To initialice the conditions a parameters.py file is needed with the parameters"""
        # grid in heights
        self.z0 = parameters.z0.cgs
        self.zf = parameters.zf.cgs
        self.z_N = parameters.zn
        self.zz = np.linspace(self.z0, self.zf, self.z_N)

        # points and the weights for frequency quadrature (equispaced for now)
        self.wf = constants.c.cgs / parameters.lamb0.cgs
        self.w0 = constants.c.cgs / parameters.lambf.cgs
        self.nus_N = parameters.wn
        self.nus = np.linspace(self.w0, self.wf, self.nus_N)
        self.nus_weights = np.ones(self.nus_N)
        self.nus_weights[0] = 0.5
        self.nus_weights[-1] = 0.5

        # weights and directions of the angular quadrature
        self.rays = []
        for data_ray in np.loadtxt(parameters.ray_quad):
            self.rays.append(ray(data_ray[0],
                                 data_ray[1] * units.deg,
                                 data_ray[2] * units.deg))
        self.rays_N = len(self.rays)

        # Maximum lambda itterations
        self.max_iter = int(parameters.max_iter)

        # Auxiliar Identity tensor and matrix to not reallocate them later computations
        self.Id_tens = np.repeat(np.identity(4)[ :, :, np.newaxis], self.nus_N, axis=2)
        self.identity = np.identity(4)

class state:
    """state class containing the current state of the solution, this include the 
    radiation and atomic state of each point as well as the MRC, mag field, and optical depth"""
    def __init__(self, cdts):

        # Initializing the maximum relative change
        self.mrc = np.ones(cdts.z_N)

        # Initialice the array of the magnetic field vector
        self.B = np.zeros((cdts.z_N, 3)) * units.G

        # Initialicing the atomic state instanciating ESE class for each point
        self.atomic = [ESE(cdts.nus, cdts.nus_weights, vector) for vector in self.B]

        # Initialicing the radiation state instanciating RTE class for each point
        self.radiation = [RTE(cdts.nus_N) for z in cdts.zz]

        # Make the first point the IC with I=BB(T=5772 K) and Q=U=V=0
        self.radiation[0].make_IC(cdts.nus)

        # Setting the optical depth of each point
        self.tau = [val for val in np.linspace(100, 0, cdts.z_N)]


    def update_mrc(self):
        """Update the mrc of the current state by finding the
        maximum mrc over all points in z (computed in ESE method)"""
        for i, point in enumerate(self.atomic):
            self.mrc[i] = point.solveESE(self.radiation[i])


    def new_itter(self):
        """Update the source funtions of all the points with the new radiation field
        computed in the previous itteration and reseting the internal state of the rad class"""
        for rad, at in zip(self.radiation, self.atomic):
            at.getSourceFunc(rad)
            rad.resetRadiation()
