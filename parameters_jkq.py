import time

z0 = 6000.0*1e5    # height of the slav over the surfave [cm]
zf = 6190.0*1e5    # geometrical height of the slab [cm]
zn = 10            # Points in z in the slab

ray_quad = "quadrature_gaussian_16x4_inv.dat" # file where the angular quadrature is read from
ray_out = [[1.0, 0.]]             # List of mu,phi for emergent radiation directions

v_dop = 5.0*1e5            # Dopler velocity (not the actual one) [cm/s]
a_voigt = 1e-5             # voigt damping parameter of the line profile
n_dens = 5.8e4             # Density wich defines the optical thickness [cm^-3]
temp = 1e4                 # Temperature of the slab [K] (6 km/s)

B = 1.0e1                  # Magnetic field strength [T]
B_inc = 90.0               # Magnetic field inclination [deg]
B_az = 0.0                 # Magnetic field azimuth [deg]

velocity = [0., 0., 0.]    # Velocity of the slab [cm/s]

max_iter = 200       # Maximum itterations for the forward method
tolerance_p = 5e-6   # tolerated relative change in populations
tolerance_c = 5e-4   # tolerated relative change in coherences

initial_equilibrium = True   # Start from equilibrium
verbose = False
extra_plots = False
extra_save = True
especial = True

dir = f'cpp/jkq/'
del time