import time

z0 = 2174.5*1e5   # height of the slav over the surfave [cm] (around 3 arcsec)
#z0 = 100*1e5     # height of the slav over the surfave [cm]
#zf = 110*1e5     # geometrical height of the slab [cm]
#zf = 1100*1e5    # geometrical height of the slab [cm]
zf = 2175.5*1e5   # geometrical height of the slab [cm] (around 3 arcsec)
#zf = 1000*1e5    # geometrical height of the slab [cm]
#zn = 5           # Points in z in the slab
#zn = 35          # Points in z in the slab
#zn = 3            # Points in z in the slab
#zn = 100         # Points in z in the slab
zn = 10          # Points in z in the slab
alpha = 0.        # angle between the vertical of the slav and the radial line [deg]

#ray_quad = "gaussian_quadrature_8x8.dat" # file where the angular quadrature is saved
#ray_quad = "gaussian_quadrature_8x4.dat" # file where the angular quadrature is saved
#ray_quad = "gaussian_quadrature_16x4.dat"# file where the angular quadrature is saved
#ray_quad = "gaussian_quadrature_32x4.dat"# file where the angular quadrature is saved
ray_quad = "gaussian_quadrature_32x8.dat" # file where the angular quadrature is saved
#ray_quad = 'gaussian_quadrature_2x1.dat'
ray_out = [[0.1,0.],[1.0,0.]]             # List of mu,phi for emergent radiation directions
#ray_out = [[0.112,11.]]             # List of mu,phi for emergent radiation directions

v_dop = 5.0*1e5            # Dopler velocity (not the actual one) [cm/s]
a_voigt = 1e-99            # voigt damping parameter of the line profile
#n_dens = 1e5              # Density wich defines the optical thickness [cm^-3]
#n_dens = 8e4               # Density wich defines the optical thickness [cm^-3]
n_dens = 5.8e4               # Density wich defines the optical thickness [cm^-3]
#n_dens = 1e10               # Density wich defines the optical thickness [cm^-3]
#n_dens = 8e3              # Density wich defines the optical thickness [cm^-3]
#temp = 1e4                # Temperature of the slab [K]
temp = 8.665251563142749e3 # Temperature of the slab [K] (6 km/s)
Trad = 6e3                 # Black body star temperature for radiation [K]

max_iter =   1       # Maximum itterations for the forward method
tolerance_p = 5e-6   # tolerated relative change in populations
#tolerance_c = 5e-5  # tolerated relative change in coherences
tolerance_c = 5e-4   # tolerated relative change in coherences

initial_equilibrium = True   # Start from equilibrium
#initial_equilibrium = False # Start from equilibrium

#dir = f'output_{time.strftime("%Y%m%d-%H%M%S")}/'
#dir = f'output_testing_2/'
#dir = f'output_hazel_like_B0/'
#dir = f'output_hazel_like_B1-30-120/'
dir = f'output_hazel_like_B1-0-0/'

