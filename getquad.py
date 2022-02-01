import sys
import numpy as np

def gaussianQ(n1):
    ''' Returns gaussian nodes and weights for a given order
    '''

    nang = (n1 + 1)//2
    da = (n1 % 2) - 1
    x0 = -1.
    x1 = 1.
    n = n1

    n2 = (n1+1)//2

    xp = 0.5*(x1+x0)
    xm = 0.5*(x1-x0)

    x = np.zeros(n1)
    w = np.zeros(n1)

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

        x[i1-1] = xp - xm*mu0
        w[i1-1] = 2.0*xm/((1.0 - mu0*mu0)*d4*d4)

        x[n1-i1] = xp + xm*mu0
        w[n1-i1] = w[i1-1]

    xx = np.zeros(n)
    ww = np.zeros(n)

    # Reorder the nodes and weights second half
    for ii in range(n,nang-da-1,-1):

      xx[n - ii] = x[ii-1]
      ww[n - ii] = w[ii-1]

    for ii in range(1,nang-da):

      xx[nang + ii - 1] = x[nang - da - ii - 1]
      ww[nang + ii - 1] = w[nang - da - ii - 1]

    X = np.zeros(2*n)
    W = np.zeros(2*n)

    for ii in range(1,n+1):
        X[ii-1] = .5*(xx[n - ii] - 1.)
        W[ii-1] = .5*ww[n - ii]
        X[2*n-ii] = .5*(xx[ii-1] + 1.)
        W[2*n-ii] = .5*ww[ii-1]

    # Normalize
    W = W/W.sum()

    return {'n': n1, 'x': X, 'w': W}

def trapezoidalQ(n0, circ=None):
    ''' Returns trapezoidal nodes and weights for a given octant
    '''

    # Argument control
    if circ is None:
        icirc = False
    else:
        icirc = circ
        if not isinstance(icirc, bool):
            icirc = False

    n1 = n0*4

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
    ng = 4
    nt = 1
    filename = f'gaussian_quadrature_{ng*2}x{nt*4}.dat'

    # Get each quadrature
    GQuad = gaussianQ(ng)
    TQuad = trapezoidalQ(nt, True)

    # Create file
    f = open(filename, 'w')
    for i in range(ng*2):
        for j in range(nt*4):
            f.write(f"{GQuad['w'][i]*TQuad['w'][j]} {180.0*np.arccos(GQuad['x'][i])/np.pi} {180.0*TQuad['x'][j]/np.pi}\n")
    f.close()

if __name__ == '__main__':
    main()
