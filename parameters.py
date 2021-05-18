from astropy import units as u

z0 = 500 * u.km                 # height of the slav over the surfave
zf = 2500 * u.km                # geometrical height of the slav
zn = 32                         # Points in z in the slav
alpha = 10 * u.deg              # angle between the vertical of the slav and the radial line

lamb0 = 1082 * u.nm             # Initial frequency (nm)
lambf = 1085 * u.nm             # Final frequency (nm)
wn = 32                         # Points in the frequency quadrature (grid)

ray_quad = "pl13n100.dat"       # file where the angular quadrature is saved

v_dop = 5.0 * u.km / u.s         # Dopler velocity
a_voigt = 7e10                   # voigt width of the line profile
n_dens = 1e10 * 1/u.cm**3        # Density wich defines the optical thickness

I_units = u.erg / (u.cm**2 * u.Hz * u.s * u.sr)

max_iter = 5                   # Maximum itterations for the forward method
tolerance = 1e-10               # tolerated relative change in populations
