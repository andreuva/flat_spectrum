import sys,copy
from physical_functions import voigt, jsymbols, Tqq_all
from atom import ESE
from rad import RTE

import numpy as np
import constants as constants
from get_nus import get_nus
from allen import Allen_class

################################################################################
################################################################################
################################################################################

class ray:
    """ ray class representing each ray in the angular quadrature
    """

    def __init__(self, weight, inclination, azimut, theta_crit, z0):

        # basic properties of the ray (inclination, azimut and weight in the slab RF
        self.weight = weight
        self.inc = inclination
        self.az = azimut

        self.rinc = self.inc*constants.degtorad
        self.raz = self.az*constants.degtorad

        # Compute the xyz vector for the intersecting rays (Slab quantities)
        self.xyz_slab = np.array([np.sin(self.rinc)*np.cos(self.raz),
                             np.sin(self.rinc)*np.sin(self.raz),
                             np.cos(self.rinc)])

        # Compute the CLV for the intersecting rays (Astrophysical quantities)
        self.clv = 0

        # If below critical inclination
        if theta_crit > self.inc:

            # Get inclination for CLV
            arg = (constants.R_sun + z0)/constants.R_sun * \
                   np.sin(180. - self.inc)
            if abs(arg) <= 1.:
                theta_clv = 180.0 - \
                            np.arcsin((constants.R_sun + z0)/constants.R_sun * \
                                      np.sin(180. - self.inc))
            else:
                theta_clv = 180.

            # Get LV factor
            self.clv = 1 - 0.64 + 0.2 + 0.64*np.abs(np.cos(theta_clv)) - \
                       0.2*np.cos(theta_clv)**2

        # If factor is negative
        if self.clv < 0:
            print(f"WARNING: CLV < 0 in ray {inclination}-{azimut}: {self.clv}")

        # Check if the ray is upwards or not
        if self.inc > 90.0:
            self.is_downward = True
        else:
            self.is_downward = False

        # Calculate Tqq here once
        self.Tqq = Tqq_all(self.rinc,self.raz)

################################################################################
################################################################################
################################################################################

class point:
    '''class to store the atomic and radiation info in each iteration'''

    def __init__(self, atomic, radiation, height):
        self.radiation = radiation
        self.atomic = atomic
        self.z = height

    def sumStokes(self, ray, nus_weights, JS):
        """ Add contribution to Jqq
        """

        self.radiation.sumStokes(ray)
        self.atomic.sumStokes(ray, self.radiation.stokes, nus_weights, JS)

    def setradiationas(self,asthis):
        """ Sets the stokes parameters as in asthis
        """
        self.radiation.stokes = copy.deepcopy(asthis.stokes)

################################################################################
################################################################################
################################################################################

class conditions:
    """Class cointaining the conditions that are not going to change
       during the program such as grids, quadratures, auxiliare
       matrices and some parameters
    """

    def __init__(self, parameters):
        """ To initialice the conditions a parameters.py file is needed
            with the parameters
        """

        # grid in heights
        self.z0 = parameters.z0
        self.zf = parameters.zf
        self.z_N = parameters.zn
        self.zz = np.linspace(self.z0, self.zf, self.z_N, endpoint=True)
        self.dz = self.zz[1] - self.zz[0]

        self.especial = parameters.especial
        self.verbose = parameters.verbose
        self.extra_plots = parameters.extra_plots
        self.extra_save = parameters.extra_save

        # Initialize J symbols
        self.JS = jsymbols(memoization=True)

        self.theta_crit = 180. - \
                          np.arcsin(constants.R_sun/(constants.R_sun + self.z0))

        # weights and directions of the angular quadrature
        self.rays = []
        for data_ray in np.loadtxt(parameters.ray_quad):
            self.rays.append(ray(data_ray[0], \
                                 data_ray[1], \
                                 data_ray[2], \
                                 self.theta_crit, self.z0))
        self.rays_N = len(self.rays)

        # Output rays
        self.orays = []
        for LOS in parameters.ray_out:
            self.orays.append(ray(1.0, \
                                  (np.arccos(LOS[0]) * 180.0 / np.pi), \
                                  LOS[1], \
                                  self.theta_crit, self.z0))

        # Get copy of the atom
        atom = ESE(0.,0.,np.zeros((3)),0.,self.JS, especial=self.especial)
        atom = atom.atom

        # Dopler velocity
        self.v_dop_0 = parameters.v_dop
        self.a_voigt = parameters.a_voigt
        self.n_dens = parameters.n_dens
        self.temp = parameters.temp
        self.v_dop = np.sqrt(2.*constants.k_B*self.temp/atom.mass)/constants.c

        self.velocity = np.array(parameters.velocity)

        # Get frequency vectors and size
        self.nus, self.nus_weights = get_nus(atom,self.v_dop_0)
        self.nus_N = self.nus.size

        # Initialice the array of the magnetic field vector
        self.B = np.zeros((self.z_N, 3))

        # Constant field
        # print('Ad-hoc constant field in conditions.__init__()')
        for iz in range(self.z_N):
            self.B[iz,0] = parameters.B
            self.B[iz,1] = parameters.B_inc*np.pi/180.
            self.B[iz,2] = parameters.B_az*np.pi/180.

        # If starting from equilibrium
        self.equi = parameters.initial_equilibrium

        # Maximum lambda itterations
        self.max_iter = int(parameters.max_iter)
        self.tolerance_p = parameters.tolerance_p
        self.tolerance_c = parameters.tolerance_c

        self.dir = parameters.dir

        # Auxiliar Identity tensor and matrix to not reallocate them later computations
        self.Id_tens = np.repeat(np.identity(4)[:, :, np.newaxis], self.nus_N, axis=2)
        self.identity = np.identity(4)

        # Atomic model
        if atom.multiterm:
            self.mode = 1
        else:
            self.mode = 0

        # Initialize cache for Voigt profiles
        self.cache = {}

    def voigt_profile_calc(self, line, dE=0.):
        """ Computes the Voigt function given the line variable with the nu field, and the
            energy displacement
        """

        # Get line center and Doppler width
        v0 = line.nu + dE
        delt_v = line.nu*self.v_dop

        # Call profile calculation and apply normalization (physical)
        profile = voigt((v0-self.nus)/delt_v, self.a_voigt)
        profile = profile / (np.sqrt(np.pi) * delt_v)

        # Apply normalization (numerical)
        normalization = np.sum(profile.real*self.nus_weights)
        profile.real = profile.real/normalization
        # profile = np.real(profile)/normalization + 1j*np.imag(profile)

        return profile

    def voigt_profile(self, line, dE=0.):
        """ Check if the Voigt profile is stored and, if not, call for its calculation
        """

        try:
            return self.cache[line][dE]
        except:

            if line not in self.cache:
                self.cache[line] = {dE: self.voigt_profile_calc(line,dE)}
            else:
                cache = self.cache[line]
                if dE not in cache:
                    cache[dE] = self.voigt_profile_calc(line,dE)
            return self.cache[line][dE]

################################################################################
################################################################################
################################################################################

class state:
    """state class containing the current state of the solution, this include the
       radiation and atomic state of each point as well as the MRC, mag field, and
       optical depth
    """

    def __init__(self, cdts):
        """ Initialize from the conditions
        """

        # Initializing the maximum relative change
        self.mrc_p = -1.
        self.mrc_c = -1.

        # Initialicing the atomic state instanciating ESE class for each point
        self.atomic = [ESE(cdts.v_dop, cdts.a_voigt, vec, cdts.temp,\
                           cdts.JS, cdts.equi, iz, cdts.especial)
                       for iz,vec in enumerate(cdts.B)]
        for atomic in self.atomic:
            atomic.initialize_profiles(cdts.nus_N)

        # Initialicing the radiation state instanciating RTE class for each point
        self.radiation = [RTE(cdts.nus, cdts.v_dop) for z in cdts.zz]

        # Get Allen class instance and gamma angles
        Allen = Allen_class()
        Allen.get_gamma(np.min(cdts.zz))

        # Define the IC for the downward and upward rays as a radiation class
        self.space_rad = RTE(cdts.nus, cdts.v_dop)
        self.sun_rad = []
        self.osun_rad = []
        for ray in cdts.rays:
            self.sun_rad.append(RTE(cdts.nus, cdts.v_dop))
            if ray.rinc < 0.5*np.pi:
                if cdts.velocity.sum() == 0:
                    self.sun_rad[-1].make_IC(cdts.nus, ray, Allen)
                else:
                    self.sun_rad[-1].make_IC_velocity(cdts.nus, ray, Allen, cdts.velocity)
        for ray in cdts.orays:
            self.osun_rad.append(RTE(cdts.nus, cdts.v_dop))
            if ray.rinc < 0.5*np.pi:
                if cdts.velocity.sum() == 0:
                    self.osun_rad[-1].make_IC(cdts.nus, ray, Allen)
                else:
                    self.osun_rad[-1].make_IC_velocity(cdts.nus, ray, Allen, cdts.velocity)

    def update_mrc(self, cdts, itter):
        """Update the mrc of the current state by finding the
           maximum mrc over all points in z (computed in ESE method)
        """

        # For each point
        for i, point in enumerate(self.atomic):

            # Get MRC
            mrc_p, mrc_c = point.solveESE(self.radiation[i], cdts)
            self.mrc_p = np.max([self.mrc_p,mrc_p])
            self.mrc_c = np.max([self.mrc_c,mrc_c])

    def new_itter(self):
        """Update the source funtions of all the points with the new radiation field
           computed in the previous itteration and reseting the internal state of the
           rad class
        """

        # Reset MRC
        self.mrc_p = -1.
        self.mrc_c = -1.

        # Reset radiation
        for rad, at in zip(self.radiation, self.atomic):
            rad.resetRadiation()
            at.atom.reset_jqq(rad.nus.size)
