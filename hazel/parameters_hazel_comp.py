import time

z0 = 6000.0*1e5   # height of the slav over the surfave [cm]
zf = 6045.0*1e5   # geometrical height of the slab [cm]
zn = 10            # Points in z in the slab

ray_quad = "quadrature_gaussian_32x8.dat"# file where the angular quadrature is saved
ray_out = [[-0.13 , 0.]]             # List of mu,phi for emergent radiation directions

v_dop = 5.0*1e5            # Dopler velocity (not the actual one) [cm/s]
a_voigt = 0.19             # voigt damping parameter of the line profile
n_dens = 5.8e4             # Density wich defines the optical thickness [cm^-3]
temp = 7.75e3              # Temperature of the slab [K] (6.4 km/s)

B = 22.8                   # Magnetic field strength [T]
B_inc = 149.0              # Magnetic field inclination [deg]
B_az = -0.1                # Magnetic field azimuth [deg]

velocity = [0., 0., 0.]   # Velocity of the slab [cm/s]

max_iter = 200       # Maximum itterations for the forward method
tolerance_p = 5e-6   # tolerated relative change in populations
tolerance_c = 5e-4   # tolerated relative change in coherences

initial_equilibrium = True   # Start from equilibrium
verbose = False
extra_plots = True
extra_save = True
especial = True

dir = f'output_hazel_comp_{time.strftime("%Y%m%d-%H%M%S")}/'
del time