from astropy import units as u
import time

z0 = 100 * u.km                 # height of the slav over the surfave
#zf = 110 * u.km                # geometrical height of the slav
zf = 1000 * u.km                # geometrical height of the slav
zn = 32                         # Points in z in the slav
alpha = 0 * u.deg              # angle between the vertical of the slav and the radial line

#lamb0 = 1081.3 * u.nm             # Initial frequency (nm)
#lambf = 1085.3 * u.nm             # Final frequency (nm)
lamb0 = 1082.5216751075432 * u.nm             # Initial frequency (nm)
lambf = 1084.121675107543 * u.nm             # Final frequency (nm)
wn = 64                        # Points in the frequency quadrature (grid)

ray_quad = "gaussian_quadrature_8x8.dat"       # file where the angular quadrature is saved

v_dop = 5.0 * u.km / u.s         # Dopler velocity
a_voigt = 1                      # voigt width of the line profile
n_dens = 1e6 * 1/u.cm**3         # Density wich defines the optical thickness
temp = 1e4 * u.K                 # Temperature of the slab

I_units = u.erg / (u.cm**2 * u.Hz * u.s * u.sr)

max_iter = 100                    # Maximum itterations for the forward method
tolerance = 1e-6               # tolerated relative change in populations

dir = f'plots_{time.strftime("%Y%m%d-%H%M%S")}/'
