from astropy import units as u

z0 = 0 * u.km                   # Initial height
zf = 2500 * u.km                # final height
zn = 64                           # Points in z

lamb0 = 1082 * u.nm             # Initial frequency (nm)
lambf = 1085 * u.nm             # Final frequency (nm)
wn = 64                         # Points in the frequency quadrature (grid)

ray_quad = "pl13n100.dat"       # file where the angular quadrature is saved

I_units = u.erg / (u.cm**2 * u.Hz * u.s * u.sr)

max_iter = 20                  # Maximum itterations for the forward method
tolerance = 1e-10               # tolerated relative change in populations
