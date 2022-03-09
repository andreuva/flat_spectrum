import sys
import numpy as np
from atom import ESE
import constants as c

def get_nus(atom, vdop):
    ''' Returns the frequency and frequency weight vectors
    '''

    # Minimum resolution [cm]
    resol = 1e-6 * 1e-7

    # Jump to change integration interval [cm]
    jump = 20.0 * 1e-7

    # Get Doppler width
    DwT = vdop/c.c

    #
    # Count components
    #

    # Initialize axis
    nus = None

    # For each transition
    for line in atom.lines:

        # Get terms
        terml = atom.terms[line.terms[0]]
        termu = atom.terms[line.terms[1]]

        # Get number of frequencies
        lnfreq = line.nw
        lnfreqc = line.nwc

        # Proper Doppler width
        DW = DwT*line.nu

        # Correct if even or non-sense
        if lnfreq % 2 == 0:
            lnfreq += 1
        if lnfreqc % 2 == 0:
            lnfreqc += 1
        if lnfreqc > lnfreq:
            lnfreq = lnfreqc

        # J combinations
        for Ju,Eu in zip(termu.J,termu.LE):
            for Jl,El in zip(terml.J,terml.LE):

                # Append resonance
                nu = Eu - El
                nu *= c.c

                # Initialize auxiliar axis
                omaux = [0.]
                nega = []

                # Core part
                for ii in range(2,lnfreqc//2+2):
                    omaux.append(omaux[-1] + line.dwc/float(lnfreqc//2))
                    nega.append(-omaux[-1])

                # Logarithmic part
                v = np.log10(line.dwc)
                dv = np.log10(line.dw)
                for ii in range(lnfreqc//2+2,lnfreq//2+2):
                    v += dv/float(lnfreq//2 - lnfreqc//2)
                    omaux.append(10.0**v)
                    nega.append(-omaux[-1])

                # Mount two halves
                omaux = nega[::-1] + omaux

                # Get actual axis
                omaux = np.array(omaux)*DW + nu

                # Add to total axis
                if nus is None:
                    nus = omaux.copy()
                else:
                    nus = np.concatenate((nus,omaux))

    #
    # Check for duplicates
    #

    # Create flag
    N = nus.size
    flag = np.ones((N))*1.1

    # For each frequency
    for ii in range(N):

        # If flagged, skip
        if flag[ii] < 1:
            continue

        # Get lambda
        l0 = c.c/nus[ii]

        # For each other frequency
        for jj in range(ii+1,N):

            # If flagged skip
            if flag[jj] < 1:
                continue

            # Get lambda
            l1 = c.c/nus[jj]

            # If repeated, flag to remove
            if np.absolute(l1-l0) < resol:
                flag[jj] = 0

    # Get only non-repeated
    final_nu = []

    # Go by the original
    for nu,fl in zip(nus,flag):

        # If not flagged, add
        if flag[jj] > 1:
            final_nu.append(nu)

    # Get array and sort
    nus = np.array(final_nu)
    nus = np.sort(nus, axis=None)

    # Save space for frequency weights
    wnus = np.empty((nus.size))

    #
    # Define integration weights
    #

    # Initialize weight and integral
    wnus[0] = 0.5*(nus[1] - nus[0])
    norm1 = wnus[0]

    # Save initial lower limit
    O0 = nus[0]

    # Pointer to first element current interval
    cfreq = 0

    # Flag saying point 1 is not the initial of interval
    init = False

    # current wavelength
    cw = c.c/nus[1]

    # For the rest of frequencies but the last one
    for ii in range(1,nus.size-1):

        # Next wavelength
        nw = c.c/nus[ii+1]

        # If initial in an interval
        if init:

            # Trapezoidal only one side
            wnus[ii] = 0.5*(nus[ii+1] - nus[ii])

            # Next point cannot be first
            init = False

            # Initialize integral to normalize weights
            norm1 = wnus[ii]

            # Pointer is now here
            cfreq = ii

            # And beginning current interval
            O0 = nus[ii]

        # Not starting an interval
        else:

            # If big difference with next one, last point of an interval
            if np.absolute(cw - nw) > jump:

                # Weight
                wnus[ii] = 0.5*(nus[ii] - nus[ii-1])

                # End of current interval
                O1 = nus[ii]

                # Add to integral
                norm1 += wnus[ii]

                # We know that the integral must be
                norm = O1 - O0

                # Normalization factor
                norm = norm/norm1

                # Renormalize weights
                wnus[cfreq:ii+1] /= norm

                # Next point is the first in a new interval
                init = True

            # Keep the interval
            else:

                # Weight
                wnus[ii] = 0.5*(nus[ii+1] - nus[ii-1])

                # Add to integral
                norm1 += wnus[ii]

        # Current wavelength
        cw = nw

    # Deal with last point
    wnus[-1] = 0.5*(nus[-1] - nus[-2])

    # Must be the end of the interval
    O1 = nus[-1]

    # Add to norm
    norm1 += wnus[-1]

    # We know that the integral must be
    norm = O1 - O0

    # Normalization factor
    norm = norm/norm1

    # Renormalize weights
    wnus[cfreq:] /= norm

    # Return frequency and weights
    return nus,wnus
