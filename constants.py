import numpy as np

# Constants
pi = np.pi                   # pi
R_sun = 6.957e10             # solar radius [cm]
c = 2.99792458e10            # speed of light [cm/s]
k_B = 1.38064852e-16         # Boltzmann constant [erg K^-1]
h = 6.62607004e-27           # Planck constant [erg s]
nuL = 4.6686437e-5           # Larmor frequency divided by field [cm^-1 G^-1]
nu_L = 1.3996e6              # Lamor frequency Eq. 3.10 LL04
amu = 1.6605402e-24          # Atomic mass unit [g]
vacuum = 1e-200              # Vacuum absorptivity [cm^-1]

# Transformations
degtorad = pi/180.           # Transformation from degree to radian (angles)
radtodeg = 180./pi           # Transformation from radian to degree (angles)

