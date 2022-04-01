import numpy as np


def construct_JKQ_0():
    JKQ = {}
    for K in range(3):
        JKQ[K] = {}
        for Q in range(0,K+1):
            JKQ[K][Q] = 0.
        # Compute negative Q
        if K > 0:
            JKQ[K][-1] = 0.
        if K > 1:
            JKQ[K][-2] = 0.

    return JKQ


def Jqq_to_JKQ(Jqq,JS):

    # Create JKQ dictionary
    JKQ = {}

    # For each multipole
    for K in range(3):

        # Initialize this multipole
        JKQ[K] = {}

        # Compute positive Q
        for Q in range(0,K+1):

            # Initialize
            JKQ[K][Q] = 0.

            # Factor for K
            f1 = np.sqrt(3.*(2.*K + 1.))

            # For each q and q'
            for qq in range(-1,2):

                # Factor for q
                f2 = f1*JS.sign(1+qq)

                # Only contribution from q' from 3J symbol
                qp = qq - Q

                # Control qp
                if np.absolute(qp) > 1:
                    continue

                # Add contribution for qq'
                JKQ[K][Q] += Jqq[qq][qp]*f2*JS.j3(1.,1.,K,qq,-qp,-Q)

        # Compute negative Q
        if K > 0:
            JKQ[K][-1] = -1.0*np.conjugate(JKQ[K][1])
        if K > 1:
            JKQ[K][-2] =      np.conjugate(JKQ[K][2])
    return JKQ

def JKQ_to_Jqq(JKQ, JS):

    jqq = {}

    # For each q
    for qq in range(-1,2):

        # Factor for q
        f1 = JS.sign(1+qq)

        # Initialize jqq
        jqq[qq] = {}

        # For each q'
        for qp in range(-1,2):

            # Initialize
            jqq[qq][qp] = 0.

            # For each K
            for K in range(3):

                # K factor
                f2 = f1*np.sqrt((2.*K+1.)/3.)

                # Get Q from 3J
                Q = qq - qp

                # Control Q
                if np.absolute(Q) > K:
                    continue

                # Add contribution
                jqq[qq][qp] += f2*JS.j3(1,1,K,qq,-qp,-Q)*JKQ[K][Q]
    
    return jqq


def TKQ(i, K, Q, theta, chi, gamma=0.):
    conj = False
    t = 0.+0j
    if Q<0:
        conj = True
        Q = -Q
    if i==0:
        if K==0:
            t = 1.+0j
        elif K==2 and Q==0:
            t = 1./(2.*np.sqrt(2.)) * (3.*np.cos(theta)*np.cos(theta) - 1.) + 0j
        elif K==2 and Q==1:
            t = -np.sqrt(3.)/2. * np.sin(theta) * np.cos(theta) * np.exp(chi*1j)
        elif K==2 and Q==2:
            t = np.sqrt(3.)/4. * np.sin(theta) * np.sin(theta) * np.exp(chi*2j)
    elif i==1:
        if K==2 and Q==0:
            t = -3./(2.*np.sqrt(2.)) * np.cos(2.*gamma) * np.sin(theta) * np.sin(theta) + 0j
        elif K==2 and Q==1:
            t = -np.sqrt(3.)/2.*(np.cos(2.*gamma)*np.cos(theta) + np.sin(2.*gamma)*1j) * np.sin(theta) * np.exp(chi*1j)
        elif K==2 and Q==2:
            t = -np.sqrt(3.)/4.*(np.cos(2.*gamma)*(1. + np.cos(theta)*np.cos(theta)) + np.sin(2.*gamma)*np.cos(theta)*2j) * np.exp(chi*2j)
    elif i==2:
        if K==2 and Q==0:
            t = 3./(2.*np.sqrt(2.)) * np.sin(2.*gamma) * np.sin(theta) * np.sin(theta)
        elif K==2 and Q==1:
            t = np.sqrt(3.)/2. * (np.sin(2.*gamma)*np.cos(theta) - np.cos(2.*gamma) * 1j) * np.sin(theta) * np.exp(chi*1j)
        elif K==2 and Q==2:
            t = np.sqrt(3.)/4. * (np.sin(2.*gamma)*(1. + np.cos(theta)*np.cos(theta)) - np.cos(2.*gamma)*np.cos(theta)*2j) * np.exp(chi*2j)
    elif i==3:
        if K==1 and Q==0:
            t = np.sqrt(3./2.) * np.cos(theta)
        elif K==1 and Q==1:
            t = -np.sqrt(3.)/2. * np.sin(theta) * np.exp(chi*1j)
    if conj:
        t = t.conjugate() * (-1)**Q
    return t