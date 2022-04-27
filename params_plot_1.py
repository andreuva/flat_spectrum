import time

z0 = 30010.5*1e5   # height of the slav over the surfave [cm] (around 3 arcsec)
zf = 30000.5*1e5   # geometrical height of the slab [cm] (around 3 arcsec)
zn = 10           # Points in z in the slab
alpha = 0.        # angle between the vertical of the slav and the radial line [deg]

ray_quad = "gaussian_quadrature_32x8.dat"# file where the angular quadrature is saved
ray_out = [[0.7, 0.]]             # List of mu,phi for emergent radiation directions

v_dop = 5*1e5              # Dopler velocity (not the actual one) [cm/s]
a_voigt = 1e-1             # voigt damping parameter of the line profile
n_dens = 5.8e4             # Density wich defines the optical thickness [cm^-3]
temp = 1e4                 # Temperature of the slab [K] (6 km/s)
Trad = 6e3                 # Black body star temperature for radiation [K]

B = 1.0e0                  # Magnetic field strength [T]
B_inc = 90.                # Magnetic field inclination [deg]
B_az = 0.0                 # Magnetic field azimuth [deg]

max_iter = 200       # Maximum itterations for the forward method
tolerance_p = 5e-6   # tolerated relative change in populations
tolerance_c = 5e-4   # tolerated relative change in coherences

initial_equilibrium = True   # Start from equilibrium

dir = f'output_{time.strftime("%Y%m%d-%H%M%S")}/'
del time