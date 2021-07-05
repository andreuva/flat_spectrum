import numpy as np
from numba import jit
from parameters import I_units


def BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, ray, cdt, tau_tot, clv=1):

    # BESSER coeficients to solve RTE (Jiri Stepan and Trujillo Bueno A&A 557 2013)
    k_p = np.moveaxis(np.diagonal(kk_p, 0, 0, 1), 0, -1).copy()
    k_m = np.moveaxis(np.diagonal(kk_m, 0, 0, 1), 0, -1).copy()
    k_o = np.moveaxis(np.diagonal(kk_o, 0, 0, 1), 0, -1).copy()

    tauMO = ((k_m + k_o)/2) * np.abs((point_O.z - point_M.z)/np.cos(ray.inc)) + 1e-30
    tau_tot = np.append(tau_tot, tau_tot[-1] + tauMO[0][2])
    tauOP = ((k_o + k_p)/2) * np.abs((point_P.z - point_O.z)/np.cos(ray.inc)) + 1e-30

    exp_tauMO = np.exp(-np.where(tauMO < 700, tauMO, 700))

    # Compute the psi_m and psi_o
    # to_taylor_psi = tauMO < 1e-3

    # u_0 = 1 - exp_tauMO
    # for i, disc in enumerate(to_taylor_psi):
    #     u_0[i][disc] = tauMO[i][disc] - tauMO[i][disc]**2/2 + tauMO[i][disc]**3/6
    # u_1 = tauMO - u_0

    # psi_m = u_0 - u_1/tauMO
    # psi_o = u_1/tauMO

    psi_m = (1 - exp_tauMO*(1 + tauMO))/(tauMO)
    psi_o = (exp_tauMO + tauMO - 1)/(tauMO)

    # Compute the wm, wo and wc for the BESSER and taylor aprox if needed
    wm = (2 - exp_tauMO*(tauMO**2 + 2*tauMO + 2))/(tauMO**2)
    wo = 1 - 2*(exp_tauMO + tauMO - 1)/(tauMO**2)
    wc = 2*(tauMO - 2 + exp_tauMO*(tauMO + 2))/(tauMO**2)

    to_taylor_m = tauMO < 0.14
    to_taylor_oc = tauMO < 0.18
    for i, disc in enumerate(to_taylor_m):
        wm[i][disc] = (tauMO[i][disc] *
                       (tauMO[i][disc] *
                        (tauMO[i][disc] *
                         (tauMO[i][disc] *
                          (tauMO[i][disc] *
                           (tauMO[i][disc] *
                            ((140-18*tauMO[i][disc])*tauMO[i][disc] - 945) + 5400)
                              - 25_200) + 90_720) - 226_800) + 302_400))/907_200

    for i, disc in enumerate(to_taylor_oc):
        wo[i][disc] = (tauMO[i][disc] *
                       (tauMO[i][disc] *
                        (tauMO[i][disc] *
                         (tauMO[i][disc] *
                          (tauMO[i][disc] *
                           (tauMO[i][disc] *
                            ((10-tauMO[i][disc])*tauMO[i][disc] - 90) + 720)
                             - 5040) + 30_240) - 151_200) + 604_800))/1_814_400

        wc[i][disc] = (tauMO[i][disc] *
                       (tauMO[i][disc] *
                        (tauMO[i][disc] *
                         (tauMO[i][disc] *
                          (tauMO[i][disc] *
                           (tauMO[i][disc] *
                            ((35-4*tauMO[i][disc])*tauMO[i][disc] - 270) + 1800)
                             - 10_800) + 45_360) - 151_200) + 302_400))/907_200

    # BESSER INTERPOLATION Jiri Stepan A&A 2013
    # Step 1: calculate dm(p) = (y0(p) - ym(0))/hm(p)
    cm = BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p)*I_units

    k_1_inv = (cdt.Id_tens*kk_o.unit + psi_o*kk_o)

    # Inverting the matrices K^-1 for all the wavelenghts
    k_1 = np.zeros_like(k_1_inv)/k_1_inv.unit**2
    for k in range(cdt.nus_N):
        k_1[:, :, k] = np.linalg.inv(k_1_inv[:, :, k])
    k_2 = (exp_tauMO*kk_m.unit - psi_m*kk_m)
    # Multipling matrices of all wavelengths with at once (eq 7 and 8)
    k_2 = np.einsum("ijb, jkb -> ikb", k_1, k_2)
    kt = np.einsum("ijk, jk -> ik", k_2, point_M.radiation.stokes)
    # Bring all together to compute the new stokes parameters
    point_O.radiation.stokes = kt*clv + wm*sf_m + wo*sf_o + wc*cm

    return tau_tot


def LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, ray, cdt):

    # Obtain the optical thicknes between the points in this ray and compute
    k_m = np.moveaxis(np.diagonal(kk_m, 0, 0, 1), 0, -1).copy()
    k_o = np.moveaxis(np.diagonal(kk_o, 0, 0, 1), 0, -1).copy()
    tauMO = ((k_m + k_o)/2) * np.abs((point_O.z - point_M.z)/np.cos(ray.inc))

    # Compute the psi_m and psi_o
    to_taylor_psi = tauMO < 1e-3

    exp_tauMO = np.exp(-np.where(tauMO < 700, tauMO, 700))
    exp_tauMO = np.where(exp_tauMO > 1e-50, exp_tauMO, 0)

    u_0 = 1 - exp_tauMO
    for i, disc in enumerate(to_taylor_psi):
        u_0[i][disc] = tauMO[i][disc] - tauMO[i][disc]**2/2 + tauMO[i][disc]**3/6
    u_1 = tauMO - u_0

    psi_m = u_0 - u_1/tauMO
    psi_o = u_1/tauMO

    point_O.radiation.stokes = point_M.radiation.stokes*exp_tauMO + sf_m*psi_m + sf_o*psi_o


@jit(nopython=True)
def BESSER_interp(tauMO, tauOP, sf_m, sf_o, sf_p):

    hm = tauMO
    hp = tauOP
    dm = (sf_o - sf_m)/(hm)
    dp = (sf_p - sf_o)/(hp)
    cm = np.ones_like(sf_o)
    cp = cm.copy()

    for i, cm_i in enumerate(cm):
        for w, cm_i_w in enumerate(cm_i):

            # Step 2: If it's monotonic dm*dp <= 0 cm=cp=y0 and exit
            if dm[i, w]*dp[i, w] <= 0:
                cm[i, w] = sf_o[i, w]
                cp[i, w] = sf_o[i, w]

            # Step 3: estimate the derivative at o (y0' = (hm*dp + hp*dm)/(hm + hp))
            else:
                dy = (hm[i, w]*dp[i, w] + hp[i, w]*dm[i, w])/(hm[i, w] + hp[i, w])

                # Step 4: calculate the initial positions of the control points
                cm[i, w] = sf_o[i, w] - dy*hm[i, w]/2
                cp[i, w] = sf_o[i, w] + dy*hp[i, w]/2

                # Step 5: check for the condition if satisfied -> 7 otherwise -> 6
                if not min([sf_m[i, w], sf_o[i, w]]) <= cm[i, w] <= max([sf_m[i, w], sf_o[i, w]]):
                    # Step 6: set cm = ym and exit the algorithm
                    cm[i, w] = sf_m[i, w]
                else:
                    # Step 7: check for the condition and if is not satisfied cp=yp and correct overshoot
                    if not min([sf_o[i, w], sf_p[i, w]]) <= cp[i, w] <= max([sf_o[i, w], sf_p[i, w]]):
                        cp[i, w] = sf_p[i, w]

                    # Step 8: Calculate the new derivative using the corrected cp
                    dy = (cp[i, w] - sf_o[i, w])/(hp[i, w]/2)
                    # Step 9: Calculate the new cm to keep the derivative smooth
                    cm[i, w] = sf_o[i, w] - dy*hm[i, w]/2

    return cm
