import copy
import numpy as np
import constants as c


def BESSER(point_M, point_O, point_P, \
           sf_m, sf_o, sf_p, \
           kk_m, kk_o, kk_p, \
           ray, cdt, tau_tot, quad, \
           tau):
    """ Solve SC step with BESSER
    """

    # Compute optical depth step
    tauMO = 0.5*(kk_m[0][0] + kk_o[0][0])*np.absolute((point_O.z - point_M.z)/np.cos(ray.rinc)) + c.vacuum
    tau += tauMO

    # Add to total tau
    tau_tot = np.append(tau_tot, tau_tot[-1] + tauMO[cdt.nus_N//2])

    # Compute exponentials
    exp_tauMO = np.empty(tauMO.shape)
    # Small linear
    small = (tauMO < 1e-7)
   #small = (tauMO < 1e-299)
    exp_tauMO[small] = 1. - tauMO[small] + 0.5*tauMO[small]*tauMO[small]
    # Normal
    normal = (tauMO >= 1e-7) & (tauMO < 300.)
   #normal = (tauMO < 300.)
    exp_tauMO[normal] = np.exp(-tauMO[normal])

    # Compute linear coeff
    psi_m, psi_o = psi_lin(exp_tauMO,tauMO)

   #print('tN',tauMO[9:12])
   #print('eN',exp_tauMO[9:12])
   #print('pmN',psi_m[9:12])
   #print('poN',psi_o[9:12])
   #print('wmN',wm)
   #print('woN',wo)
   #print('wcN',wc)
   #print('cmN',cm)

    # Build kappa and matrix
    kappa = []
    matri = []

    # First row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[0][1])
    kappa[-1].append( psi_o*kk_o[0][2])
    kappa[-1].append( psi_o*kk_o[0][3])
    matri[-1].append( exp_tauMO.copy())
    matri[-1].append(-psi_m*kk_m[0][1])
    matri[-1].append(-psi_m*kk_m[0][2])
    matri[-1].append(-psi_m*kk_m[0][3])
    # Second row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][1].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[1][2])
    kappa[-1].append( psi_o*kk_o[1][1])
   #matri[-1].append( matri[0][1].copy())
    matri[-1].append( 0.)
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)
    matri[-1].append(-psi_m*kk_m[1][2])
    matri[-1].append(-psi_m*kk_m[1][1])
    # Third row
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][2].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[1][2].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
    kappa[-1].append( psi_o*kk_o[1][0])
   #matri[-1].append( matri[0][2].copy())
    matri[-1].append( 0.)
    matri[-1].append(-matri[1][2].copy())
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)
    matri[-1].append(-psi_m*kk_m[1][0])
    # Fourth
    kappa.append([])
    matri.append([])
   #kappa[-1].append( kappa[0][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[1][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append(-kappa[2][3].copy())
    kappa[-1].append( 0.)
   #kappa[-1].append( np.ones(exp_tauMO.shape))
    kappa[-1].append( 1.)
   #matri[-1].append( matri[0][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append(-matri[1][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append(-matri[2][3].copy())
    matri[-1].append( 0.)
   #matri[-1].append( exp_tauMO.copy())
    matri[-1].append( 1.)

    # Invert matrix
    kappa = matinv(kappa)

    # Matrix time vectors
    v1 = matrivec(matri,point_M.radiation.stokes)
    v2 = matvec(kappa,v1)

    # Quadratic
    if quad:

        # Compute optical depth step
        tauOP = 0.5*(kk_o[0][0] + kk_p[0][0])* \
                np.absolute((point_O.z - point_M.z)/np.cos(ray.rinc)) + c.vacuum

        # Compute BESSER coefficients
        wm,wo,wc = rt_omega(exp_tauMO,tauMO)

        # BESSER coefficient
        cm = BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p)

        ss = [wm*sf_m[0] + wo*sf_o[0] +  wc*cm[0], \
              wm*sf_m[1] + wo*sf_o[1] +  wc*cm[1],
              wm*sf_m[2] + wo*sf_o[2] +  wc*cm[2],
              wm*sf_m[3] + wo*sf_o[3] +  wc*cm[3]]
        ss = matvec(kappa,ss)
        point_O.radiation.stokes = sumstkl(v2, ss)

    # Lineal
    else:

        ss = [psi_m*sf_m[0] + psi_o*sf_o[0], \
              psi_m*sf_m[1] + psi_o*sf_o[1], \
              psi_m*sf_m[2] + psi_o*sf_o[2], \
              psi_m*sf_m[3] + psi_o*sf_o[3]]
        ss = matvec(kappa, ss)
        point_O.radiation.stokes = sumstkl(v2, ss)

    return tau_tot


def psi_lin(ex,t):
    """ Compute linear contributions
    """

    big = t > 0.11
    small = t <= 0.11

    psi_m = np.empty(t.shape)
    psi_o = np.empty(t.shape)

    psi_m[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                    ((63e0 - 8e0*t[small])*t[small] - 432e0) + 2520e0) - \
                    12096e0) + 45360e0) - 120960e0) + 181440e0))/362880e0)
    psi_o[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                   ((9e0 - t[small])*t[small] - 72e0) + 504e0) - \
                    3024e0) + 15120e0) - 60480e0) + 181440e0))/362880e0)

    psi_m[big] = (1.-ex[big]*(1.+t[big]))/t[big]
    psi_o[big] = (ex[big]+t[big]-1.)/t[big]
   #psi_m = (1.-ex*(1.+t))/t
   #psi_o = (ex+t-1.)/t

    return psi_m,psi_o

def rt_omega(ex,t):
    """ Compute BESSER contributions
    """

    big = t > 0.14
    small = t <= 0.14

    omega_m = np.empty(t.shape)
    omega_o = np.empty(t.shape)
    omega_c = np.empty(t.shape)

    omega_m[big] = (2e0 - ex[big]*(t[big]*t[big] + 2e0*t[big] + 2e0))/(t[big]*t[big])
    omega_m[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                      ((140e0 - 18e0*t[small])*t[small] - \
                      945e0) + 5400e0) - 25200e0) + 90720e0) - \
                      226800e0) + 302400e0))/907200e0)

    big = t > 0.18
    small = t <= 0.18

    omega_o[big] = 1e0 - 2e0*(t[big] + ex[big] - 1e0)/(t[big]*t[big])
    omega_o[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                       ((10e0 - t[small])*t[small] - 90e0) + \
                       720e0) - 5040e0) + 30240e0) - 151200e0) + \
                       604800e0))/1814400e0)
    omega_c[big] = 2.0*(t[big] - 2.0 + ex[big]*(t[big] + 2.0))/(t[big]*t[big])
    omega_c[small] = ((t[small]*(t[small]*(t[small]*(t[small]*(t[small]*(t[small]* \
                       ((35e0 - 4e0*t[small])*t[small] - \
                       270e0) + 1800e0) - 10080e0) + 45360e0) - \
                       151200e0) + 302400e0))/907200e0)

    return omega_m,omega_o,omega_c


def matinv(M):
    """ Inverts a matrix 4x4 with the absorption matrix symmetry properties\n
    """

    a = M[0][1]
    b = M[0][2]
    c = M[0][3]
    d = M[1][2]
    e = M[1][3]
    f = M[2][3]
    o = np.ones(a.shape)
    aa = a*a
    ab = a*b
    ac = a*c
    ad = a*d
    ae = a*e
    af = a*f
    bb = b*b
    bc = b*c
    bd = b*d
    be = b*e
    bf = b*f
    cc = c*c
    cd = c*d
    ce = c*e
    cf = c*f
    dd = d*d
    de = d*e
    df = d*f
    ee = e*e
    ef = e*f
    ff = f*f
    T0 = (af - be + cd)
    P1 = T0*a
    P2 = T0*b
    P3 = T0*c
    P4 = T0*d
    P5 = T0*e
    P6 = T0*f
    S1 = bc - de
    S2 = ac + df
    S3 = ab - ef
    S4 = ae + bf
    S5 = ad - cf
    S6 = bd + ce

    M[0][0] =   o + dd + ee + ff
    M[1][0] = - a + S6 - P6
    M[2][0] = - b - S5 + P5
    M[3][0] = - c - S4 - P4
    M[0][1] = - a - S6 - P6
    M[1][1] =   o - bb - cc + ff
    M[2][1] =   d + S3 - P3
    M[3][1] =   e + S2 + P2
    M[0][2] = - b + S5 + P5
    M[1][2] = - d + S3 + P3
    M[2][2] =   o - aa - cc + ee
    M[3][2] =   f + S1 - P1
    M[0][3] = - c + S4 - P4
    M[1][3] = - e + S2 - P2
    M[2][3] = - f + S1 + P1
    M[3][3] =   o - aa - bb + dd

    idet = o/(M[0][0] + a*M[1][0] + b*M[2][0] + c*M[3][0])
   
    for i in range(4):
        for j in range(4):
            M[i][j] *= idet

    return M

def matvec(A,b):
    """ Product matrix and vector with vector coefficients, for RT
    """

    c = [[],[],[],[]]

    c[0] = A[0][0]*b[0] + A[0][1]*b[1] + A[0][2]*b[2] + A[0][3]*b[3]
    c[1] = A[1][0]*b[0] + A[1][1]*b[1] + A[1][2]*b[2] + A[1][3]*b[3]
    c[2] = A[2][0]*b[0] + A[2][1]*b[1] + A[2][2]*b[2] + A[2][3]*b[3]
    c[3] = A[3][0]*b[0] + A[3][1]*b[1] + A[3][2]*b[2] + A[3][3]*b[3]

    return c

def matrivec(A,b):
    """ Product matrix and vector with vector coefficients, but it takes
        into account the symmetry properties of matri in BESSER, for RT
    """

    c = [[],[],[],[]]

    c[0] = A[0][0]*b[0] + A[0][1]*b[1] + A[0][2]*b[2] + A[0][3]*b[3]
    c[1] = A[0][1]*b[0] + A[0][0]*b[1] + A[1][2]*b[2] + A[1][3]*b[3]
    c[2] = A[0][2]*b[0] - A[1][2]*b[1] + A[0][0]*b[2] + A[2][3]*b[3]
    c[3] = A[0][3]*b[0] - A[1][3]*b[1] - A[2][3]*b[2] + A[0][0]*b[3]

    return c

def matmat(A,B):
    """ Product two 4x4 matrices
    """

    C = []

    # Row
    for ii in range(4):
        C.append([])
        for jj in range(4):
            C[-1].append(A[ii][jj]*B[jj][ii])

    return C

def sumstkl(a,b):
    """ Sum list of Stokes parameter contributions
    """

    c = []

    for i in range(4):
        c.append(a[i] + b[i])

    return c

def BESSER_old(point_M, point_O, point_P, \
           sf_m, sf_o, sf_p, \
           kk_m, kk_o, kk_p, \
           ray, cdt, tau_tot, clv=1):

    kp_m = (kk_m/kk_m[0, 0] - cdt.Id_tens)
    kp_o = (kk_o/kk_o[0, 0] - cdt.Id_tens)
    # kp_p = (kk_p/kk_p[0, 0] - 1)

    # BESSER coeficients to solve RTE (Jiri Stepan and Trujillo Bueno A&A 557 2013)
    tauMO = ((kk_m[0, 0] + kk_o[0, 0])/2) * \
            np.abs((point_O.z - point_M.z)/np.cos(ray.rinc)) + 1e-30
    tau_tot = np.append(tau_tot, tau_tot[-1] + tauMO[int(cdt.nus_N/2)])
    tauOP = ((kk_o[0, 0] + kk_p[0, 0])/2) * \
            np.abs((point_P.z - point_O.z)/np.cos(ray.rinc)) + 1e-30

    exp_tauMO = np.exp(-np.where(tauMO < 300, tauMO, 300))

    psi_m = (1 - exp_tauMO*(1 + tauMO))/(tauMO)
    psi_o = (exp_tauMO + tauMO - 1)/(tauMO)

    # Compute the wm, wo and wc for the BESSER and taylor aprox if needed
    wm = (2 - exp_tauMO*(tauMO**2 + 2*tauMO + 2))/(tauMO**2)
    wo = 1 - 2*(exp_tauMO + tauMO - 1)/(tauMO**2)
    wc = 2*(tauMO - 2 + exp_tauMO*(tauMO + 2))/(tauMO**2)

    to_taylor_m = tauMO < 0.14
    to_taylor_oc = tauMO < 0.18

    wm[to_taylor_m] = (tauMO[to_taylor_m] *
                       (tauMO[to_taylor_m] *
                        (tauMO[to_taylor_m] *
                         (tauMO[to_taylor_m] *
                          (tauMO[to_taylor_m] *
                           (tauMO[to_taylor_m] *
                            ((140-18*tauMO[to_taylor_m])*tauMO[to_taylor_m] - 945) + 5400)
                              - 25_200) + 90_720) - 226_800) + 302_400))/907_200

    wo[to_taylor_oc] = (tauMO[to_taylor_oc] *
                        (tauMO[to_taylor_oc] *
                         (tauMO[to_taylor_oc] *
                          (tauMO[to_taylor_oc] *
                           (tauMO[to_taylor_oc] *
                            (tauMO[to_taylor_oc] *
                             ((10-tauMO[to_taylor_oc])*tauMO[to_taylor_oc] - 90) + 720)
                               - 5040) + 30_240) - 151_200) + 604_800))/1_814_400

    wc[to_taylor_oc] = (tauMO[to_taylor_oc] *
                        (tauMO[to_taylor_oc] *
                         (tauMO[to_taylor_oc] *
                         (tauMO[to_taylor_oc] *
                          (tauMO[to_taylor_oc] *
                           (tauMO[to_taylor_oc] *
                            ((35-4*tauMO[to_taylor_oc])*tauMO[to_taylor_oc] - 270) + 1800)
                              - 10_800) + 45_360) - 151_200) + 302_400))/907_200

    # BESSER INTERPOLATION Jiri Stepan A&A 2013
    # Step 1: calculate dm(p) = (y0(p) - ym(0))/hm(p)
    cm = BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p)

   #print('tO',tauMO[9:12])
   #print('eO',exp_tauMO[9:12])
   #print('pmO',psi_m[9:12])
   #print('poO',psi_o[9:12])
   #print('wmO',wm)
   #print('woO',wo)
   #print('wcO',wc)
   #print('cmO',cm)

    k_1_inv = (cdt.Id_tens + psi_o*kp_o)

    # Inverting the matrices K^-1 for all the wavelenghts
    k_1 = np.zeros_like(k_1_inv)
    for k in range(cdt.nus_N):
        k_1[:, :, k] = np.linalg.inv(k_1_inv[:, :, k])
    k_2 = (exp_tauMO*cdt.Id_tens - psi_m*kp_m)
    # Multipling matrices of all wavelengths with at once (eq 7 and 8)
    k_3 = np.einsum("ijb, jkb -> ikb", k_1, k_2)
    kt = np.einsum("ijk, jk -> ik", k_3, point_M.radiation.stokes)
    # Bring all together to compute the new stokes parameters
    point_O.radiation.stokes = kt*clv + wm*sf_m + wo*sf_o + wc*cm

    '''
    i0 = -4
    i1 = exp_tauMO.size
    i1 = -3
    print('O')
    print(f'{tauMO[i0]:23.16e} {exp_tauMO[i0]:23.16e} {(exp_tauMO[i0] + tauMO[i0] - 1e0)/tauMO[i0]:23.16e} {psi_o[i0]:23.16e}')
    print(tauMO[i0:i1],exp_tauMO[i0:i1])
    print(kk_o[0,0,i0:i1])
    print(kk_o[0,1,i0:i1]/kk_o[0,0,i0:i1])
    print('w')
    print(wm[i0:i1],wo[i0:i1],wc[i0:i1])
    print(cm[0][i0:i1],cm[1][i0:i1],cm[2][i0:i1],cm[3][i0:i1])
    print('psi')
    print(psi_m[i0:i1],kp_m[0][1][i0:i1],psi_m[i0:i1]*kp_m[0][1][i0:i1])
    print(psi_o[i0:i1],kp_o[0][1][i0:i1],psi_o[i0:i1]*kp_o[0][1][i0:i1])
    print('k')
    print(k_1_inv[0][0][i0:i1])
    print(k_1_inv[0][1][i0:i1])
    print(k_1_inv[0][2][i0:i1])
    print(k_1_inv[0][3][i0:i1])
    print(k_1_inv[1][0][i0:i1])
    print(k_1_inv[1][1][i0:i1])
    print(k_1_inv[1][2][i0:i1])
    print(k_1_inv[1][3][i0:i1])
    print(k_1_inv[2][0][i0:i1])
    print(k_1_inv[2][1][i0:i1])
    print(k_1_inv[2][2][i0:i1])
    print(k_1_inv[2][3][i0:i1])
    print(k_1_inv[3][0][i0:i1])
    print(k_1_inv[3][1][i0:i1])
    print(k_1_inv[3][2][i0:i1])
    print(k_1_inv[3][3][i0:i1])
    print('m')
    print(k_2[0][0][i0:i1])
    print(k_2[0][1][i0:i1])
    print(k_2[0][2][i0:i1])
    print(k_2[0][3][i0:i1])
    print(k_2[1][0][i0:i1])
    print(k_2[1][1][i0:i1])
    print(k_2[1][2][i0:i1])
    print(k_2[1][3][i0:i1])
    print(k_2[2][0][i0:i1])
    print(k_2[2][1][i0:i1])
    print(k_2[2][2][i0:i1])
    print(k_2[2][3][i0:i1])
    print(k_2[3][0][i0:i1])
    print(k_2[3][1][i0:i1])
    print(k_2[3][2][i0:i1])
    print(k_2[3][3][i0:i1])
    '''

    return tau_tot


def LinSC_old(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt):

    # Obtain the optical thicknes between the points in this ray and compute
    tauMO = ((kk_m[0, 0] + kk_o[0, 0])/2) * np.abs((point_O.z - point_M.z)/np.cos(ray.inc))

    # Compute the psi_m and psi_o
    to_taylor_psi = tauMO < 1e-3

    exp_tauMO = np.exp(-np.where(tauMO < 700, tauMO, 700))
    exp_tauMO = np.where(exp_tauMO > 1e-50, exp_tauMO, 0)

    u_0 = 1 - exp_tauMO
    u_0[to_taylor_psi] = tauMO[to_taylor_psi] - tauMO[to_taylor_psi]**2/2 + tauMO[to_taylor_psi]**3/6
    u_1 = tauMO - u_0

    psi_m = u_0 - u_1/tauMO
    psi_o = u_1/tauMO

    point_O.radiation.stokes = point_M.radiation.stokes*exp_tauMO + sf_m*psi_m + sf_o*psi_o

def ybetwab(y,a,b):
    return (a <= b and y >= a and y <= b) or \
           (a >= b and y <= a and y >= b)

def correctyab(y,a,b):
    if b > a:
        mini = a
        maxi = b
    else:
        mini = b
        maxi = a

    if y <= mini and y <= maxi:
        return y
    elif y < mini:
        return mini
    else:
        return maxi

#@jit(nopython=True)
def BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p):

    Cm = []

    # For Stokes
    for j in range(4):
        Cm.append(copy.copy(sf_o[j]))
        # For frequency
        for m, hm, hp, ym, yo, yp in zip(range(tauMO.size), tauMO, tauOP, sf_m[j],
                                         sf_o[j], sf_p[j]):

            # If both greater than 0
            if hm > 0. and hm > 0.:

                dm = (yo - ym)/hm
                dp = (yp - yo)/hp

            else:

                continue

            # If steps opposite sign
            if dm*dp <= 0.:
                continue

            # If same sign

            yder = (hm*dp + hp*dm)/(hm + hp)
            cm = yo - 0.5*hm*yder
            cp = yo + 0.5*hm*yder

            condm = ybetwab(cm, ym, yo)
            condp = ybetwab(cp, yo, yp)

            if condm and condp:
                Cm[j][m] = cm
            elif not condm:
                Cm[j][m] = correctyab(cm,ym,yo)
            elif not condp:
                cpp = correctyab(cp,yo,yp)
                yder = 2.0*(cpp - yo)/hp
                cm = yo - 0.5*hm*yder
                condpp = ybetwab(cm,ym,yo)

                if condpp:
                    Cm[j][m] = cm
                else:
                    Cm[j][m] = correctyab(cm,ym,yo)
    return Cm

#@jit(nopython=True)
def BESSER_interp_deactivated(tauMO, tauOP, sf_m, sf_o, sf_p):

    Cm = copy.copy(sf_o)

    # For Stokes
    for j in range(4):
        # For frequency
        for m, hm, hp, ym, yo, yp in zip(range(tauMO.size), tauMO, tauOP, sf_m[j, :],
                                         sf_o[j, :], sf_p[j, :]):

            # If both greater than 0
            if hm > 0. and hm > 0.:

                dm = (yo - ym)/hm
                dp = (yp - yo)/hp

            else:

                continue

            # If steps opposite sign
            if dm*dp <= 0.:
                continue

            # If same sign

            yder = (hm*dp + hp*dm)/(hm + hp)
            cm = yo - 0.5*hm*yder
            cp = yo + 0.5*hm*yder

            condm = ybetwab(cm, ym, yo)
            condp = ybetwab(cp, yo, yp)

            if condm and condp:
                Cm[j,m] = cm
            elif not condm:
                Cm[j,m] = correctyab(cm,ym,yo)
            elif not condp:
                cpp = correctyab(cp,yo,yp)
                yder = 2.0*(cpp - yo)/hp
                cm = yo - 0.5*hm*yder
                condpp = ybetwab(cm,ym,yo)

                if condpp:
                    Cm[j,m] = cm
                else:
                    Cm[j,m] = correctyab(cm,ym,yo)
    return Cm

#@jit(nopython=True)
def BESSER_interp_old(tauMO, tauOP, sf_m, sf_o, sf_p):

    hm = tauMO
    hp = tauOP
    ## No need to create arrays for these temporal quantities (except for cm, the output, one can also just copy sf_o)
    dm = (sf_o - sf_m)/(hm)
    dp = (sf_p - sf_o)/(hp)
    cm = np.ones_like(sf_o)
    cp = cm.copy()

    for i, cm_i in enumerate(cm):
        for w, cm_i_w in enumerate(cm_i):

            # Step 2: If it's monotonic dm*dp <= 0 cm=cp=y0 and exit
            if dm[i, w]*dp[i, w] <= 0:
                cm[i, w] = sf_o[i, w]
                ## No need to asign cp (I know this is a residual from when you returned both)
                cp[i, w] = sf_o[i, w]

            # Step 3: estimate the derivative at o (y0' = (hm*dp + hp*dm)/(hm + hp))
            else:
                dy = (hm[w]*dp[i, w] + hp[w]*dm[i, w])/(hm[w] + hp[w])

                # Step 4: calculate the initial positions of the control points
                ## product by 0.5 is always faster than division by 2 (unless python does something special with exact easy fractions, I do not know). It is not a relevant difference when doing it once, but really relevant when it happens millions of times
                cm[i, w] = sf_o[i, w] - dy*hm[w]/2
                cp[i, w] = sf_o[i, w] + dy*hp[w]/2

                # Step 5: check for the condition if satisfied -> 7 otherwise -> 6
                if not min([sf_m[i, w], sf_o[i, w]]) <= cm[i, w] <= max([sf_m[i, w], sf_o[i, w]]):
                    # Step 6: set cm = ym and exit the algorithm
                    cm[i, w] = sf_m[i, w]
                else:
                    # Step 7: check for the condition and if is not satisfied cp=yp and correct overshoot
                    if not min([sf_o[i, w], sf_p[i, w]]) <= cp[i, w] <= max([sf_o[i, w], sf_p[i, w]]):
                        cp[i, w] = sf_p[i, w]

                    # Step 8: Calculate the new derivative using the corrected cp
                    dy = (cp[i, w] - sf_o[i, w])/(hp[w]/2)
                    # Step 9: Calculate the new cm to keep the derivative smooth
                    cm[i, w] = sf_o[i, w] - dy*hm[w]/2

    return cm
