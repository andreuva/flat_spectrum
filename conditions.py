import numpy as np
from astropy import units, constants
from atom import ESE
from rad import RTE


class ray:
    def __init__(self, weight, inclination, azimut):
        self.weight = weight
        self.inc = inclination
        self.az = azimut

    def is_downward(self):
        if self.inc > 90 * units.deg:
            return True
        else: 
            return False


class conditions:
    def __init__(self, parameters):

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


        self.Id_tens = np.repeat(np.identity(4)[ :, :, np.newaxis], self.nus_N, axis=2)


class state:

    def __init__(self, cdts):

        # Initializing the maximum relative change
        self.mrc = np.ones(cdts.z_N)

        # Initialice the array of the magnetic field vector
        self.B = np.zeros((cdts.z_N, 3)) * units.G

        # Initialicing the atomic state instanciating ESE class for each point
        self.atomic = [ESE(cdts.nus, cdts.nus_weights, vector) for vector in self.B]

        # Initialicing the radiation state instanciating RTE class for each point
        self.radiation = [RTE(cdts.nus_N) for z in cdts.zz]

        self.radiation[0].make_IC(cdts.nus)

        self.tau = [val for val in np.linspace(100, 0, cdts.z_N)]

    def update_mrc(self):
        """Update the mrc of the current state by finding the
        maximum mrc over all points in z (computed in ESE method)"""
        for i, point in enumerate(self.atomic):
            self.mrc[i] = point.solveESE(self.radiation[i])


    def new_itter(self):
        for rad, at in zip(self.radiation, self.atomic):
            at.getSourceFunc(rad)
            rad.resetRadiation()
