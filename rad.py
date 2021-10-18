from physical_functions import Tqq
from parameters import I_units as Iu

from astropy.modeling.models import BlackBody as bb
from astropy import units as u
import numpy as np


class RTE:
    """RTE class containing the stokes parameters and the Jqq as well as other
    Radiation information"""

    def __init__(self, nus, v_dop):
        # Initialicing the I to 0
        II = np.zeros(len(nus))
        QQ = np.zeros(len(nus))
        UU = np.zeros(len(nus))
        VV = np.zeros(len(nus))
        self.stokes = np.array([II, QQ, UU, VV])*Iu

        self.nus = nus
        # Defining the Jqq as nested dictionaries'
        self.jqq = {}
        for qq in [-1, 0, 1]:
            self.jqq[qq] = {}
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = np.zeros(len(nus))*Iu * u.sr

    def make_IC(self, nus, T):
        # If a point is defined as IC put Q=U=V=0 and I to BB
        self.stokes = self.stokes*0
        self.stokes[0] = bb(temperature=T)(nus)

    def check_I(self):
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
       #f = open('debugJ','a')
       #il = len(self.stokes[0])//2
       #f.write('    Weight {0}\n'.format(ray.weight))
        for i in range(4):
           #f.write('    Stokes {0}: {1}\n'.format(i,self.stokes[i][il]))
            for qq in [-1, 0, 1]:
                for qp in [-1, 0, 1]:
                   #f.write('      Contribution {0:2d}{1:2d}\n'.format(qq,qp))
                   #f.write('        tqqp {0}\n'.format(Tqq(qq, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)))
                   #f.write('        all {0}\n'.format(ray.weight*Tqq(qq, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)*self.stokes[i][il]))
                    self.jqq[qq][qp] = self.jqq[qq][qp] + \
                                       ray.weight*Tqq(qq, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)\
                                       * self.stokes[i]
                   #f.write('        cumulative {0}\n'.format(self.jqq[qq][qp][il]))
       #f.close()


    def sumStokes_debug(self, ray, fd):
        """
            Called per every Lambda iteration, grid point, and ray direction.
            stokes: list of 4 arrays of Stokes parameters [I,Q,U,V] in given
            point and direction (all frequencies)
            ray: object with .theta, .chi and weight variables defining the
            ray of propagation direction
            return value: None
        """
        fd.write(f'Direction {ray.inc} {ray.az}\n')
        for i in range(4):
            if np.max(np.absolute(self.stokes[i])) <= 0.:
                continue
            fd.write(f'  Stokes {i}\n')
            for qq in [-1, 0, 1]:
                for qp in [-1, 0, 1]:
                    fd.write(f'    q {qq} qp {qp} jqqp {self.jqq[qq][qp]}')
                    fd.write(f' W {ray.weight} Tqq {Tqq(qq,qp,i,ray.inc.to("rad").value,ray.az.to("rad").value)}\n')
                    if isinstance(self.jqq[qq][qp], float):
                        for j,s in enumerate(self.stokes[i]):
                            fd.write(f'   {i} {0.0} {self.stokes[i][j]} ')
                            fd.write(f'{ray.weight*Tqq(qq, qp, i, ray.inc.to("rad").value, ray.az.to("rad").value) * self.stokes[i][j]}\n')
                    else:
                        for j,s in enumerate(self.stokes[i]):
                            fd.write(f'   {i} {self.jqq[qq][qp][j]} {self.stokes[i][j]} ')
                            fd.write(f'{self.jqq[qq][qp][j] + ray.weight*Tqq(qq, qp, i, ray.inc.to("rad").value, ray.az.to("rad").value) * self.stokes[i][j]}\n')
                    self.jqq[qq][qp] = self.jqq[qq][qp] + \
                                       ray.weight*Tqq(qq, qp, i, ray.inc.to('rad').value, ray.az.to('rad').value)\
                                       * self.stokes[i]

    def resetStokes(self):
        for i in range(4):
            self.stokes[i] = 0

    def resetRadiation(self):
        """ Called at the beginning of every Lambda iteration.
            It initializes the internal state."""
        for qq in [-1, 0, 1]:
            for qp in [-1, 0, 1]:
                self.jqq[qq][qp] = 0.

    def Jqq_nu(self, cdt, line, q, qp, Mu, Ml, B, nus_weights):
        jqqp = self.jqq[q][qp]
        # profile = cdt.voigt_profile(line, Mu, Ml, B)
        profile = (cdt.voigt_profile(line).real + 0j)
        return np.sum(jqqp*profile*nus_weights)

    def Jqq_nu_debug(self, cdt, line, q, qp, Mu, Ml, B, nus_weights):
        jqqp = self.jqq[q][qp]
        # profile = cdt.voigt_profile(line, Mu, Ml, B)
        profile = (cdt.voigt_profile(line).real + 0j)
        f = open(f'jqq_integral_debug_{q}-{qp}','w')
        for i in range(nus_weights.size):
            f.write('{0} {1} {2} {3}\n'.format(i,nus_weights[i],profile[i],jqqp[i]))
        f.write('{0}\n'.format(np.sum(jqqp*profile*nus_weights)))
        f.close()
        return np.sum(jqqp*profile*nus_weights)
