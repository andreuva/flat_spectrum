import numpy as np


def BESSER(point_M, point_O, point_P, sf_m, sf_o, sf_p, kk_m, kk_o, kk_p, tau_m, tau_o, tau_p, ray, cdt, clv=1):
    # BESSER coeficients to solve RTE (Jiri Stepan and Trujillo Bueno A&A 557 2013)
    tauMO = np.abs((tau_o - tau_m)/np.cos(ray.inc))
    tauOP = np.abs((tau_p - tau_o)/np.cos(ray.inc))

    # Compute the psi_m and psi_o
    psi_m = (1 - np.exp(-tauMO)*(1 + tauMO))/(tauMO)
    psi_o = (np.exp(-tauMO) + tauMO - 1)/(tauMO)

    # Compute the wm, wo and wc for the BESSER and taylor aprox if needed
    wm = (2 - np.exp(-tauMO)*(tauMO**2 + 2*tauMO + 2))/(tauMO**2)
    wo = 1 - 2*(np.exp(-tauMO) + tauMO - 1)/(tauMO**2)
    wc = 2*(tauMO - 2 + np.exp(-tauMO)*(tauMO + 2))/(tauMO**2)

    to_taylor = tauMO < 0.14
    wm[to_taylor] = (tauMO*(tauMO*(tauMO*(tauMO*(tauMO*(tauMO*((140-18*tauMO)*tauMO - 945) + 5400)
                            - 25_200) + 90_720) - 226_800) + 302_400))/907_200
    to_taylor = tauMO < 0.18
    wo[to_taylor] = (tauMO*(tauMO*(tauMO*(tauMO*(tauMO*(tauMO*((10-tauMO)*tauMO - 90) + 720)
                            - 5040) + 30_240) - 151_200) + 604_800))/1_814_400
    wc[to_taylor] = (tauMO*(tauMO*(tauMO*(tauMO*(tauMO*(tauMO*((35-4*tauMO)*tauMO - 270) + 1800)
                            - 10_800) + 45_360) - 151_200) + 302_400))/907_200

    # BESSER INTERPOLATION Jiri Stepan A&A 2013
    # Step 1: calculate dm(p) = (y0(p) - ym(0))/hm(p)
    hm = tau_o - tau_m
    hp = tau_p - tau_o
    dm = (sf_o - sf_m)/hm
    dp = (sf_p - sf_o)/hp
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

    k_1_inv = (cdt.Id_tens + psi_o*kk_m)

    # Inverting the matrices K^-1 for all the wavelenghts
    k_1 = np.zeros_like(k_1_inv)
    for k in range(cdt.nus_N):
        k_1[:, :, k] = np.linalg.solve(k_1_inv[:, :, k], cdt.identity)
    k_2 = (np.exp(-tauMO) - psi_m * kk_o)
    # Multipling matrices of all wavelengths with at once (eq 7 and 8)
    k_2 = np.einsum("ijb, jkb -> ikb", k_1, k_2)
    kt = np.einsum("ijk, jk -> ik", k_2, point_M.radiation.stokes)
    # Bring all together to compute the new stokes parameters
    point_O.radiation.stokes = kt*clv + wm*sf_m + wo*sf_o + wc*cm


def LinSC(point_M, point_O, sf_m, sf_o, kk_m, kk_o, tau_m, tau_o, ray, cdt):
    tauMO = np.abs((tau_o - tau_m)/np.cos(ray.inc))

    u_0 = 1 - np.exp(-tauMO)
    to_taylor = tauMO < 1e-3
    u_0[to_taylor] = tauMO[to_taylor] - tauMO[to_taylor]**2/2 + tauMO[to_taylor]**3/6
    u_1 = tauMO - u_0

    psi_m = u_0 - u_1/tauMO
    psi_o = u_1/tauMO

    point_O.radiation.stokes = point_M.radiation.stokes*np.exp(-tauMO) + sf_m*psi_m + sf_o*psi_o
