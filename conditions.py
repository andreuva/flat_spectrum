from physical_functions import voigt
from atom import ESE, HeI_1083
from rad import RTE

import numpy as np
from numpy.linalg import norm
from numpy import real
import matplotlib.pyplot as plt
from astropy import units, constants


class ray:
    """ray class representing each ray in the angular quadrature"""
    def __init__(self, weight, inclination, azimut, alpha, theta_crit, z0):

        # basic properties of the ray (inclination, azimut and weight in the slab RF
        self.weight = weight
        self.inc = inclination
        self.az = azimut

        # Rotate the RF to the global one and store the result as a attribute
        xyz_slab = np.array([np.sin(self.inc)*np.cos(self.az),
                             np.sin(self.inc)*np.sin(self.az),
                             np.cos(self.inc)])*units.cm
        rotation_matrix = np.array([[np.cos(alpha), 0, np.sin(-alpha)],
                                    [0,             1,              0],
                                    [np.sin(alpha), 0,  np.cos(alpha)]])
        xyz_global = rotation_matrix @ xyz_slab

        self.inc_glob = np.arccos(xyz_global[2]/units.cm).to('deg')
        global_az = np.arctan2(xyz_global[1], xyz_global[0]).to('deg')
        global_az[global_az < 0] += 360*units.deg
        self.az_glob = global_az

        # Compute the CLV for the intersecting rays (Astrophysical quantities)
        self.clv = 0

        if theta_crit < self.inc_glob:
            theta_clv = 180*units.deg - np.arcsin((constants.R_sun.cgs + z0)/constants.R_sun.cgs * np.sin(180*units.deg-self.inc_glob))
            self.clv = 1 - 0.64 + 0.2 + 0.64*np.cos(theta_clv) - 0.2*np.cos(theta_clv)**2

        # Check if the ray is upwards or not in the global RF
        if self.inc_glob > 90 * units.deg:
            self.is_downward = True
        else:
            self.is_downward = False


class point:
    '''class to store the atomic and radiation info in each iteration'''

    def __init__(self, atomic, radiation, height):
        self.radiation = radiation
        self.atomic = atomic
        self.z = height


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
        self.dz = self.zz[1] - self.zz[0]

        # points and the weights for frequency quadrature (equispaced for now)
        self.wf = constants.c.cgs / parameters.lamb0.cgs
        self.w0 = constants.c.cgs / parameters.lambf.cgs
        self.nus_N = parameters.wn
        self.nus = np.linspace(self.w0, self.wf, self.nus_N)
        self.nus_weights = np.ones(self.nus_N)
        self.nus_weights[0] = 0.5
        self.nus_weights[-1] = 0.5

        # Parameters of the rotation of the slab and the global ref frame
        self.alpha = parameters.alpha

        self.theta_crit = 180*units.deg-np.arcsin(constants.R_sun.cgs/(constants.R_sun.cgs + self.z0))

        # weights and directions of the angular quadrature
        self.rays = []
        for data_ray in np.loadtxt(parameters.ray_quad):
            self.rays.append(ray(data_ray[0],
                                 data_ray[1] * units.deg,
                                 data_ray[2] * units.deg,
                                 self.alpha, self.theta_crit, self.z0))
        self.rays_N = len(self.rays)

        # Dopler velocity
        self.v_dop = parameters.v_dop
        self.a_voigt = parameters.a_voigt
        self.n_dens = parameters.n_dens

        # Initialice the array of the magnetic field vector
        self.B = np.zeros((self.z_N, 3)) * units.G

        # Maximum lambda itterations
        self.max_iter = int(parameters.max_iter)

        # Auxiliar Identity tensor and matrix to not reallocate them later computations
        self.Id_tens = np.repeat(np.identity(4)[:, :, np.newaxis], self.nus_N, axis=2)
        self.identity = np.identity(4)

    def voigt_profile(self, line, Mu, Ml, B):
        vs = self.nus.value
        v0 = line.nu.value + (norm(B)/constants.h.cgs.value *
                              (line.gu*Mu - line.gl*Ml))
        return voigt(vs-v0, self.a_voigt)


class state:
    """state class containing the current state of the solution, this include the
    radiation and atomic state of each point as well as the MRC, mag field, and optical depth"""
    def __init__(self, cdts):

        # Initializing the maximum relative change
        self.mrc = np.ones(cdts.z_N)

        # Initialice the array of the magnetic field vector (defined in conditions)
        # self.B = cdts.B

        # Initialicing the atomic state instanciating ESE class for each point
        self.atomic = [ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, np.linalg.norm(vec)) for vec in cdts.B]

        # Define the IC for the downward and upward rays as an atomic class
        self.space_atom = ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, np.zeros(3)*units.G)
        self.sun_atom = ESE(cdts.v_dop, cdts.a_voigt, cdts.nus, cdts.nus_weights, np.ones(3)*100*units.G)

        # Initialicing the radiation state instanciating RTE class for each point
        self.radiation = [RTE(cdts.nus, cdts.v_dop) for z in cdts.zz]

        # Define the IC for the downward and upward rays as a radiation class
        self.space_rad = RTE(cdts.nus, cdts.v_dop)
        self.sun_rad = RTE(cdts.nus, cdts.v_dop)
        self.sun_rad.make_IC(cdts.nus)

        # Make the first point the IC with I=BB(T=5772 K) and Q=U=V=0
        self.radiation[0].make_IC(cdts.nus)

    def update_mrc(self, cdts):
        """Update the mrc of the current state by finding the
        maximum mrc over all points in z (computed in ESE method)"""
        for i, point in enumerate(self.atomic):
            self.mrc[i] = point.solveESE(self.radiation[i], cdts)

    def new_itter(self):
        """Update the source funtions of all the points with the new radiation field
        computed in the previous itteration and reseting the internal state of the rad class"""
        for rad, at in zip(self.radiation, self.atomic):
            rad.resetRadiation()


def plot_quadrature(cdt):
    plt.figure()
    plt.subplot(projection="aitoff")

    inclinations_loc = np.array([ray.inc.value for ray in cdt.rays])
    azimuts_loc = np.array([ray.az.value for ray in cdt.rays])

    inclinations_glob = np.array([ray.inc_glob.value for ray in cdt.rays])
    azimuts_glob = np.array([ray.az_glob.value for ray in cdt.rays])

    plt.plot(inclinations_loc, azimuts_loc, 'o', label='local RF')
    plt.plot(inclinations_glob, azimuts_glob, 'o', alpha=0.5, label='global RF')
    plt.grid(True)
    plt.title('Quadrature in the two reference frames')
    plt.legend()
    plt.show()
