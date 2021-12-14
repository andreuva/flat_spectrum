import time

z0 = 100*1e5                 # height of the slav over the surfave [cm]
#zf = 110*1e5                 # geometrical height of the slab [cm]
zf = 1100*1e5                 # geometrical height of the slab [cm]
#zf = 1000*1e5                # geometrical height of the slab [cm]
#zn = 5                       # Points in z in the slab
#zn = 35                      # Points in z in the slab
zn = 100                      # Points in z in the slab
#zn = 10                      # Points in z in the slab
alpha = 0.                   # angle between the vertical of the slav and the radial line [deg]

ray_quad = "gaussian_quadrature_8x8.dat"   # file where the angular quadrature is saved
#ray_quad = 'gaussian_quadrature_2x1.dat'
ray_out = [[0.1,0.],[1.0,0.]]                       # List of mu,phi for emergent radiation directions

v_dop = 5.0*1e5              # Dopler velocity (not the actual one) [cm/s]
a_voigt = 1e-4               # voigt damping parameter of the line profile
n_dens = 1e4                 # Density wich defines the optical thickness [cm^-3]
temp = 1e4                   # Temperature of the slab [K]

max_iter = 200               # Maximum itterations for the forward method
tolerance_p = 1e-6           # tolerated relative change in populations
tolerance_c = 1e-6           # tolerated relative change in coherences

initial_equilibrium = True   # Start from equilibrium

#dir = f'output_{time.strftime("%Y%m%d-%H%M%S")}/'
dir = f'output_testing_2/'
