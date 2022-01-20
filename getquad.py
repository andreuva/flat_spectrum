import sys
import numpy as np

def gaussianQ(n1):
    ''' Returns gaussian nodes and weights for a given order
    '''

    x0 = 0.0
    x1 = 1.0

    n2 = (n1+1)//2

    xp = 0.5*(x1+x0)
    xm = 0.5*(x1-x0)

    xx = np.zeros(n1)
    ww = np.zeros(n1)

    for i1 in range(1,n2+1):

        mu0 = np.cos(np.pi*(float(i1) - 0.25)/(float(n1) + 0.5))

        while True:

            d1 = 1.0
            d2 = 0.0

            for i2 in range(1,n1+1):

                d3 = d2
                d2 = d1
                d1 = ((2.0*float(i2) - 1.0)*mu0*d2 - \
                      (float(i2) - 1.0)*d3)/i2

            d4 = float(n1)*(mu0*d1 - d2)/(mu0*mu0 - 1.0)
            mu1 = mu0
            mu0 = mu1 - d1/d4

            if np.absolute(mu1 - mu0) <= 1e-15:
                break

        xx[i1-1] = xp - xm*mu0
        ww[i1-1] = 2.0*xm/((1.0 - mu0*mu0)*d4*d4)

        xx[n1-i1] = xp + xm*mu0
        ww[n1-i1] = ww[i1-1]

        # Normalize
        ww = ww/ww.sum()

    return {'n': n1, 'x': xx, 'w': ww}

def trapezoidalQ(n1, circ=None):
    ''' Returns trapezoidal nodes and weights for a given octant
    '''

    # Argument control
    if circ is None:
        icirc = False
    else:
        icirc = circ
        if not isinstance(icirc, bool):
            icirc = False

    xx = np.zeros((n1))
    ww = np.zeros((n1))

    # If circular
    if icirc:

        for i in range(1,n1+1):

            xx[i-1] = 2.0*np.pi*(float(i) - 0.5)/float(n1)
            ww[i-1] = 1.0/float(n1)

    else:

        xx = np.linspace(0.0,1.0,n1,endpoint=True)
        ww[:] = 1.0/float(n1)

    ww = ww/ww.sum()

    return {'n': n1, 'x': xx, 'w': ww}

def main():
    ''' Build quadrature as tensor product ng gaussian nodes (-1,-1)
        and nt trapezoidal "circular" nodes (0,2pi)
    '''

    # Params
    ng = 32
    nt = 8
    filename = 'gaussian_quadrature_32x8.dat'

    # Get each quadrature
    GQuad = gaussianQ(ng)
    TQuad = trapezoidalQ(nt, True)

    # Create file
    f = open(filename, 'w')
    for i in range(ng):
        for j in range(nt):
            f.write(f"{GQuad['w'][i]*TQuad['w'][j]} {180.0*np.arccos(GQuad['x'][i])/np.pi} {180.0*TQuad['x'][j]/np.pi}\n")
    f.close()

if __name__ == '__main__':
    main()
