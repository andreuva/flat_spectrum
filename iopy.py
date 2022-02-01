from physical_functions import jsymbols,rotate_ist
import constants as c
import numpy as np

def io_saverho(datadir,atomic):
    ''' Save density matrix everywhere
    '''

    # J symbols
    jsim = jsymbols()
    rota = rotate_ist()

    # Open file rhojMjpMp
    f = open(datadir+'rhos', 'w')

    # Atom0
    atom0 = atomic[0].atom

    # Go through terms
    for iterm in range(len(atom0.terms)):
        for index in atom0.terms[iterm].index:

            i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
            f.write(f'[{i:2d}][{ii:2d},{ii1:2d}] ' + \
                    f'rho({mu:2d}{M:4.1f},{mu1:2d}{M1:4.1f}): ')

            for atoml in atomic:
                rrho = atoml.rho
                atom = atoml.atom

                f.write(f'  {rrho[i]:13.6e}')

            f.write('\n')

    # Close file
    f.close()

    # Open file rhoKQ
    f = open(datadir+'rhokq', 'w')

    # For each height
    for atoml in atomic:

        # Atom
        atom = atoml.atom

        for term in atom.terms:
            f.write(f"Term L {term.L} S {term.S}\n")
            for J in term.J:

                # Possible M
                Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

                for Jp in term.J:
                    minK = int(round(np.absolute(J-Jp)))
                    maxK = int(round(J + Jp))
                    if atoml.rotate:
                        DKQQ = rota.get_DKQQ(2,-atoml.theta,-atoml.phi,conjugate=True,backwards=True)
                        mrho = {}
                    for K in range(minK,maxK+1):
                        if atoml.rotate:
                            mrho[K] = {}
                        for Q in range(-K,K+1):
                            if atoml.rotate:
                                mrho[K][Q] = 0j

                            # Initialize rhoKQ
                            rho = 0.

                            # For each M
                            for M in Ms:

                                # Decide M'
                                Mp = M - Q

                                # Sign and weight
                                SS = jsim.sign(J-M)* \
                                     np.sqrt(2.*K + 1.)

                                # Skip invalid M'
                                if np.absolute(Mp) > Jp:
                                    continue

                                # Go by the whole term
                                for index in term.index:

                                    # Extract data
                                    ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index
                                    # If M,M'
                                    if (M1 != M or M2 != Mp) and \
                                       (M1 != Mp or M2 != M):

                                        continue

                                    # If crossed, conjugate
                                    if M1 == Mp and M2 == M and i1 != i2:
                                        iiM1 = iM2
                                        ii1 = i2
                                        iiM2 = iM1
                                        ii2 = i1
                                        conj = -1.0
                                    else:
                                        iiM1 = iM1
                                        ii1 = i1
                                        iiM2 = iM2
                                        ii2 = i2
                                        conj = 1.0

                                    # Run over J
                                    for Jindex1 in term.index_muM[iiM1]:
                                        J1 = Jindex1[2]
                                        iJ1 = Jindex1[1]

                                        # Relevant J
                                        if np.absolute(J1-J) > 0.25:
                                            continue

                                        # Run over J'
                                        for Jindex2 in term.index_muM[iiM2]:
                                            J2 = Jindex2[2]
                                            iJ2 = Jindex2[1]

                                            # Relevant J'
                                            if np.absolute(J2-Jp) > 0.25:
                                                continue

                                            # Contribution
                                            CC = term.eigvec[ii1][iJ1]* \
                                                 term.eigvec[ii2][iJ2]* \
                                                 SS*jsim.j3(J,Jp,K,M,-Mp,-Q)

                                            # If imag
                                            if imag:
                                                rho += conj*CC*rrho[ii]*1j
                                            else:
                                                rho += CC*rrho[ii]
                            if atoml.rotate:
                                mrho[K][Q] = rho
                            # Print rhoKQ
                            f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_m = ' + \
                                    f'{rho.real:13.6e}\n')
                            if Q != 0:
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_m = ' + \
                                        f'{rho.imag:13.6e}\n')
                    if atoml.rotate:
                        for K in range(minK,maxK+1):
                            if K == 0:
                                continue
                            if K > 2:
                                continue
                            for Q in range(-K,K+1):
                                val = 0j
                                for Qp in range(-K,K+1):
                                    val += DKQQ[K][Q][Qp]*mrho[K][Qp]
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_v = ' + \
                                        f'{val.real:13.6e}\n')
                                if Q != 0:
                                    f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_v = ' + \
                                            f'{val.imag:13.6e}\n')
                    else:
                        for K in range(minK,maxK+1):
                            for Q in range(-K,K+1):
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_v = ' + \
                                        f'{mrho[K][Q].real:13.6e}\n')
                                if Q != 0:
                                    f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f})_v = ' + \
                                            f'{mrho[K][Q].imag:13.6e}\n')

    # Close file
    f.close()

    return

def io_saverho_alt(datadir,atomic):
    ''' Save density matrix everywhere
    '''

    # J symbols
    jsim = jsymbols()

    # Open file rhojMjpMp
    f = open(datadir+'rhos', 'w')

    # For each height
    for atoml in atomic:

        # Atom
        rrho = atoml.rho
        atom = atoml.atom

        # Write to file
        for term in atom.terms:
            for index in term.index:
                i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
                f.write(f'[{i:2d}][{ii:2d},{ii1:2d}] ' + \
                        f'rho({mu:2d}{M:2f},{mu1:2d}{M1:2f}) = '+ \
                        f'{rrho[i]:13.6e}\n')

    # Close file
    f.close()

    # Open file rhoKQ
    f = open(datadir+'rhokq', 'w')

    # For each height
    for atoml in atomic:

        # Atom
        atom = atoml.atom

        for term in atom.terms:
            f.write(f"Term L {term.L} S {term.S}\n")
            for J in term.J:

                # Possible M
                Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

                for Jp in term.J:
                    minK = int(round(np.absolute(J-Jp)))
                    maxK = int(round(J + Jp))
                    for K in range(minK,maxK+1):
                        for Q in range(-K,K+1):

                            # Initialize rhoKQ
                            rho = 0.

                            # For each M
                            for M in Ms:

                                # Decide M'
                                Mp = M - Q

                                # Sign and weight
                                SS = jsim.sign(J-M)* \
                                     np.sqrt(2.*K + 1.)

                                # Skip invalid M'
                                if np.absolute(Mp) > Jp:
                                    continue

                                # Go by the whole term
                                for index in term.index:

                                    # Extract data
                                    ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index
                                    # If M,M'
                                    if (M1 != M or M2 != Mp) and \
                                       (M1 != Mp or M2 != M):

                                        continue

                                    # If crossed, conjugate
                                    if M1 == Mp and M2 == M and i1 != i2:
                                        iiM1 = iM2
                                        ii1 = i2
                                        iiM2 = iM1
                                        ii2 = i1
                                        conj = -1.0
                                    else:
                                        iiM1 = iM1
                                        ii1 = i1
                                        iiM2 = iM2
                                        ii2 = i2
                                        conj = 1.0

                                    # Run over J
                                    for Jindex1 in term.index_muM[iiM1]:
                                        J1 = Jindex1[2]
                                        iJ1 = Jindex1[1]

                                        # Relevant J
                                        if np.absolute(J1-J) > 0.25:
                                            continue

                                        # Run over J'
                                        for Jindex2 in term.index_muM[iiM2]:
                                            J2 = Jindex2[2]
                                            iJ2 = Jindex2[1]

                                            # Relevant J'
                                            if np.absolute(J2-Jp) > 0.25:
                                                continue

                                            # Contribution
                                            CC = term.eigvec[ii1][iJ1]* \
                                                 term.eigvec[ii2][iJ2]* \
                                                 SS*jsim.j3(J,Jp,K,M,-Mp,-Q)

                                            # If imag
                                            if imag:
                                                rho += conj*CC*rrho[ii]*1j
                                            else:
                                                rho += CC*rrho[ii]
                            # Print rhoKQ
                            f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                    f'{rho.real:13.6e}\n')
                            if Q != 0:
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                        f'{rho.imag:13.6e}\n')

    # Close file
    f.close()



def io_saverad(datadir,atomic):
    ''' Save radiation tensors everywhere
    '''

    # J symbols
    jsim = jsymbols()
    rota = rotate_ist()

    # Open file rhojMjpMp
    f = open(datadir+'js', 'w')
    f2 = open(datadir+'jkq', 'w')

    # Atom0
    atom0 = atomic[0].atom

    # Write Jqq pre-rotation
    f.write("J_qq'\n")
    for il,line in enumerate(atom0.lines):
        f.write(f'Line {il} l0 {c.c*1e7/line.nu}\n')
        f2.write(f'Line {il} l0 {c.c*1e7/line.nu}\n')

        if line.especial:
            for comp in line.resos:
                f.write(f'Component l0 {c.c*1e7/comp}\n')
                f2.write(f'Component l0 {c.c*1e7/comp}\n')
                JKQ = []
                vJKQ = []
                for atoml in atomic:
                    JKQ.append({0: {0: 0.}, \
                                1: {-1:0.,0:0.,1:0.}, \
                                2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}})
                    vJKQ.append({0: {0: 0.}, \
                                1: {-1:0.,0:0.,1:0.}, \
                                2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}})
                for qq in range(-1,2):
                    for qp in range(-1,2):
                        f.write(f'J_{qq:2d}{qp:2d}')
                        for atoml in atomic:
                            f.write(f' {atoml.atom.lines[il].jqq[comp][qq][qp]}')
                        f.write('\n')
                        for K in range(3):
                            Q = qq - qp
                            if np.absolute(Q) > K:
                                continue
                            for i,atoml in enumerate(atomic):
                                JKQ[i][K][Q] += jsim.sign(1 + qq)* \
                                                 np.sqrt(3.*(2.*K + 1.))* \
                                                 jsim.j3(1.,1.,K,qq,-qp,-Q)* \
                                                 atoml.atom.lines[il].jqq[comp][qq][qp]
                for K in range(0,3):
                    for Q in range(-K,K+1):
                        f2.write(f'J^{K:1d}{Q:2d}_m')
                        for i in range(len(JKQ)):
                            f2.write(f' {JKQ[i][K][Q]}')
                        f2.write('\n')
                for K in range(1,3):
                    for Q in range(-K,K+1):
                        f2.write(f'J^{K:1d}{Q:2d}_v')
                        for i,atoml in enumerate(atomic):
                            if atoml.rotate:
                                DKQQ = rota.get_DKQQ(2,-atoml.theta,-atoml.phi,backwards=True)
                                val = 0j
                                for Qp in range(-K,K+1):
                                    val += DKQQ[K][Q][Qp]*JKQ[i][K][Qp]
                            else:
                                val = JKQ[i][K][Q]
                            f2.write(f' {JKQ[i][K][Q]}')
                        f2.write('\n')
        else:
            JKQ = []
            for atoml in atomic:
                JKQ.append({0: {0: 0.}, \
                            1: {-1:0.,0:0.,1:0.}, \
                            2: {-2:0.,-1:0.,0:0.,1:0.,2:0.}})
            for qq in range(-1,2):
                for qp in range(-1,2):
                    f.write(f'J_{qq:2d}{qp:2d}')
                    for atoml in atomic:
                        f.write(f' {atoml.atom.lines[il].jqq[qq][qp]}')
                    f.write('\n')
                    for K in range(3):
                        Q = qq - qp
                        if np.absolute(Q) > K:
                            continue
                        for i, atoml in enumerate(atomic):
                            JKQ[i][K][Q] += jsim.sign(1 + qq)* \
                                            np.sqrt(3.*(2.*K + 1.))* \
                                            jsim.j3(1.,1.,K,qq,-qp,-Q)* \
                                            atoml.atom.lines[il].jqq[qq][qp]
            for K in range(0,3):
                for Q in range(-K,K+1):
                    f2.write(f'J^{K:1d}{Q:2d}')
                    for i in range(len(JKQ)):
                        f2.write(f' {JKQ[i][K][Q]}')
                    f2.write('\n')
            for K in range(1,3):
                for Q in range(-K,K+1):
                    f.write(f'J^{K:1d}{Q:2d}_v')
                    for i,atoml in enumerate(atomic):
                        if atoml.rotate:
                            DKQQ = rota.get_DKQQ(2,-atoml.theta,-atoml.phi,backwards=True)
                            val = 0j
                            for Qp in range(-K,K+1):
                                val += DKQQ[K][Q][Qp]*JKQ[i][K][Qp]
                        else:
                            val = JKQ[i][K][Q]
                        f2.write(f' {JKQ[i][K][Q]}')
                    f2.write('\n')
    f.close()
    f2.close()
    return




def io_saverho_alt(datadir,atomic):
    ''' Save density matrix everywhere
    '''

    # J symbols
    jsim = jsymbols()

    # Open file rhojMjpMp
    f = open(datadir+'rhos', 'w')

    # For each height
    for atoml in atomic:

        # Atom
        rrho = atoml.rho
        atom = atoml.atom

        # Write to file
        for term in atom.terms:
            for index in term.index:
                i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
                f.write(f'[{i:2d}][{ii:2d},{ii1:2d}] ' + \
                        f'rho({mu:2d}{M:2f},{mu1:2d}{M1:2f}) = '+ \
                        f'{rrho[i]:13.6e}\n')

    # Close file
    f.close()

    # Open file rhoKQ
    f = open(datadir+'rhokq', 'w')

    # For each height
    for atoml in atomic:

        # Atom
        atom = atoml.atom

        for term in atom.terms:
            f.write(f"Term L {term.L} S {term.S}\n")
            for J in term.J:

                # Possible M
                Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

                for Jp in term.J:
                    minK = int(round(np.absolute(J-Jp)))
                    maxK = int(round(J + Jp))
                    for K in range(minK,maxK+1):
                        for Q in range(-K,K+1):

                            # Initialize rhoKQ
                            rho = 0.

                            # For each M
                            for M in Ms:

                                # Decide M'
                                Mp = M - Q

                                # Sign and weight
                                SS = jsim.sign(J-M)* \
                                     np.sqrt(2.*K + 1.)

                                # Skip invalid M'
                                if np.absolute(Mp) > Jp:
                                    continue

                                # Go by the whole term
                                for index in term.index:

                                    # Extract data
                                    ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index
                                    # If M,M'
                                    if (M1 != M or M2 != Mp) and \
                                       (M1 != Mp or M2 != M):

                                        continue

                                    # If crossed, conjugate
                                    if M1 == Mp and M2 == M and i1 != i2:
                                        iiM1 = iM2
                                        ii1 = i2
                                        iiM2 = iM1
                                        ii2 = i1
                                        conj = -1.0
                                    else:
                                        iiM1 = iM1
                                        ii1 = i1
                                        iiM2 = iM2
                                        ii2 = i2
                                        conj = 1.0

                                    # Run over J
                                    for Jindex1 in term.index_muM[iiM1]:
                                        J1 = Jindex1[2]
                                        iJ1 = Jindex1[1]

                                        # Relevant J
                                        if np.absolute(J1-J) > 0.25:
                                            continue

                                        # Run over J'
                                        for Jindex2 in term.index_muM[iiM2]:
                                            J2 = Jindex2[2]
                                            iJ2 = Jindex2[1]

                                            # Relevant J'
                                            if np.absolute(J2-Jp) > 0.25:
                                                continue

                                            # Contribution
                                            CC = term.eigvec[ii1][iJ1]* \
                                                 term.eigvec[ii2][iJ2]* \
                                                 SS*jsim.j3(J,Jp,K,M,-Mp,-Q)

                                            # If imag
                                            if imag:
                                                rho += conj*CC*rrho[ii]*1j
                                            else:
                                                rho += CC*rrho[ii]
                            # Print rhoKQ
                            f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                    f'{rho.real:13.6e}\n')
                            if Q != 0:
                                f.write(f'rho^{K:1d}_{Q:2d}({J:3f},{Jp:3f}) = ' + \
                                        f'{rho.imag:13.6e}\n')

    # Close file
    f.close()

    return
