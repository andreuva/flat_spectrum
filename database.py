from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters_rtcoefs as pm
import numpy as np

if __name__ == '__main__':

    cdt = conditions(pm)
    st = state(cdt)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)

    # select the ray direction as the otuput ray
    ray = cdt.orays[0]
    point_O = point(st.atomic[0], st.radiation[0], cdt.zz[0])

    # Set Stokes
    # point_O.setradiationas(st.sun_rad[0])
    # point_O.sumStokes(ray, cdt.nus_weights, cdt.JS)
    point_O.radiation.jqq[0][0] = np.ones_like(point_O.radiation.jqq[0][0])

    # Solve the ESE
    point_O.atomic.solveESE(point_O.radiation, cdt)

    sf, kk = RT_coeficients.getRTcoefs(point_O.atomic, ray, cdt)