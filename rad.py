from physical_functions import Tqq
import constants as c
import numpy as np


def gaussian(x, mu, std):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(std, 2.)))


class RTE:
    """RTE class containing the stokes parameters and the Jqq as well as other
       Radiation information
    """

    def __init__(self, nus, v_dop):

        # Initialicing the I to 0
        II = np.zeros(len(nus))
        QQ = np.zeros(len(nus))
        UU = np.zeros(len(nus))
        VV = np.zeros(len(nus))
        self.stokes = np.array([II, QQ, UU, VV])

        # Frequency axis
        self.nus = nus

        # Defining the Jqq as nested dictionaries'
        self.jqq = {}
        for qq in [-1, 0, 1]:
            self.jqq[qq] = {}
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = np.zeros(len(nus),dtype=np.complex128)

    def bb(self, T, nus):
        """ Black body radiation
        """

        numc = nus/c.c
        arg = c.h/c.k_B/T
        arg *= nus

        return (2.*c.h*numc*numc*nus)/(np.exp(arg) - 1.)

    def make_IC(self, nus, ray, Allen):
        """ Get continuum radiation
        """

        # If a point is defined as IC put Q=U=V=0 and I to BB
        self.stokes = self.stokes*0
       #self.stokes[0] = self.bb(T,nus)
       #self.stokes[0] = self.bb(T,nus) * Allen.get_clv(ray,nus)
        self.stokes[0] = Allen.get_radiation(nus) * Allen.get_clv(ray,nus)

    def make_IC_velocity(self, nus, ray, Allen, velocity):
        """ Get continuum radiation
        """

        # If a point is defined as IC put Q=U=V=0 and I to BB
        self.stokes = self.stokes*0
        self.stokes[0] = Allen.get_radiation(nus) * Allen.get_clv(ray,nus)

        nu_1 = 2.76733e14
        nu_2 = 2.76764e14

        gaussian_width = 7e9
        gaussian_1_height = self.stokes[0].max()*0.5 #(0.6 + np.sin(ray.raz)*0.3)
        gaussian_2_height = gaussian_1_height/7
        gauss_1 = gaussian(nus, nu_1, gaussian_width)*gaussian_1_height
        gauss_2 = gaussian(nus, nu_2, gaussian_width)*gaussian_2_height
        self.stokes[0] = self.stokes[0] - (gauss_1 + gauss_2)

        ray_cart = np.array([np.sin(ray.inc)*np.cos(ray.az),
                             np.sin(ray.inc)*np.sin(ray.az),
                             np.cos(ray.inc)])
        vlos = np.dot(ray_cart, velocity)
        nus_p = nus*(1+vlos/299792458)

        if ray.inc > np.arctan2(10,5)*180/np.pi:
            self.stokes[0] = 0*self.stokes[0]
        else:
            self.stokes[0] = np.interp(nus, nus_p, self.stokes[0])


    def check_I(self):
        """ Check physical intensity
        """

        if np.any(self.stokes[0] < 0):
            print(f"Warning: Negative intensity {self.stokes[0][self.stokes[0] < 0]}")
            input("Press Enter to continue...")

    def sumStokes(self, ray):
        """
            Called per every Lambda iteration, grid point, and ray direction.
            stokes: list of 4 arrays of Stokes parameters [I,Q,U,V] in given
            point and direction (all frequencies)
            ray: object with .theta, .chi and weight variables defining the
            ray of propagation direction
            return value: None
        """

        # For each Stokes parameter
        for i in range(4):
            # For each qq'
            for qq in range(-1,2):
                for qp in range(qq,2):
                    self.jqq[qq][qp] = self.jqq[qq][qp] + \
                                       ray.weight* \
                                       Tqq(qq,qp,i,ray.rinc,ray.raz)* \
                                       self.stokes[i]
                for qp in range(-1,qq):
                    self.jqq[qq][qp] = self.jqq[qp][qq].conj()

    def resetStokes(self):
        """ Set Stokes paramaters to zero
        """

        for i in range(4):
            self.stokes[i] = 0

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state."""

        for qq in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = (0.+0j)
