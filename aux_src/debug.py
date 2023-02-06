# Import classes and parameters
from RTcoefs import RTcoefs
from conditions import conditions, state, point
import parameters as pm
from solver import BESSER, LinSC_old, BESSER_old
from plot_utils import *
import constants as c
from iopy import io_saverho,io_saverad
from atom import ESE
from rad import RTE

# Import needed libraries
import numpy as np
import pickle,struct,sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy,struct

def main():
    """ Main code
    """

    # Initializating the conditions, state and RT coefficients
    cdt = conditions(pm)
    RT_coeficients = RTcoefs(cdt.nus,cdt.nus_weights,cdt.mode)
    st = state(cdt)

    #############
    # Debug SEE #
    #############

    JS = cdt.JS
    nus = cdt.nus
    nus_w = cdt.nus_weights
    v_dop = cdt.v_dop

    # Write frequency file in helium debug folder
    print('SIZE',nus.size)
    f = open('/home/tanausu/Documents/Codes/MyCodes/HANLERT/Outputs/debugheliumcode/wave','wb')
    f.write(struct.pack('i',nus.size))
    f.write(struct.pack('d'*nus.size,*(nus*1e-5/c.c)))
    f.close()

    # Vertical field
   #B = np.array([1.0,0.0,0.0])
    B = np.array([1.0,63.0*np.pi/180.,111.0*np.pi/180.])
   #B = np.array([0.0,0.0,0.0])

    '''
    def fffu(Jl,Ml,Ju,Mu):
        ww = JS.j3(Ju,Jl,1,-Mu,Ml,Mu-Ml)
        return (2.*Ju+1.)*ww*ww
    def fffl(Jl,Ml,Ju,Mu):
        ww = JS.j3(Ju,Jl,1,-Mu,Ml,Mu-Ml)
        return (2.*Jl+1.)*ww*ww

    jls = [1]
    jus = [2,1,0]
    for jl in jls:
        mls = np.linspace(-jl,jl,2*jl+1,endpoint=True)
        for mll in mls:
            ml = int(round(mll))
            for ju in jus:
                mus = np.linspace(-ju,ju,2*ju+1,endpoint=True)
                for muu in mus:
                    mu = int(round(muu))
                    if np.absolute(mu-ml) > 1:
                        continue
                    print(f'({jl:1d}{ml:2d},{ju:1d}{mu:2d}) = ' + \
                          f'{fffu(jl,ml,ju,mu)} {fffl(jl,ml,ju,mu)}')
    sys.exit()
    '''

    # Define a point
    atomic = ESE(cdt.v_dop,cdt.a_voigt,B,cdt.temp,cdt.JS,cdt.equi,0)
    radiation = RTE(cdt.nus, cdt.v_dop)
    pointC = point(atomic, radiation, cdt.zz[1])

    # Diagon output
    f = open('pyth.300','w')
    for it,term in enumerate(atomic.atom.terms):
        f.write(f'    {it}\n')
        k = -1
        for iM,M,Mblock in zip(range(term.nM),term.M,term.Mblock):
            for mu in range(Mblock):
                k += 1
                f.write(f' {term.eigval[k]-term.TE:24.16e}')
               #f.write(f' {term.eigval[k]:24.16e}')
                for j1 in range(Mblock):
                    f.write(f' {term.eigvec[k][j1]:24.16e}')
                f.write('\n')


#           k = -1
#           for j in range(nM):
#               k += 1
#               f.write(f' {term.eigval[k]}')
#               for j1 in range(nM):
#                   print(k,nM,j,j1,term.eigvec[k])
#                   f.write(f' {term.eigvec[k][j1]}')
#               f.write('\n')
    f.close()

    # Ad-hoc radiation field
    JKQ = {0: {0: 100. + 0j}, \
           1: {0:   5. + 0.j, \
               1:  -3. + 7.j}, \
           2: {0:  21. + 0.j, \
               1: -15. + 17.j, \
               2:  11. - 13.j}}
#              1: 0. + 0.j, \

    # Get negative Q
    for K in range(3):
        for Q in range(1,K+1):
            if Q == 1:
                ss = -1.0
            elif Q == 2:
                ss = 1.0
            JKQ[K][-Q] = ss*np.conjugate(JKQ[K][Q])

    f = open('pyth.200','w')

    f.write('JKQ input\n')
    for K in range(3):
        for Q in range(-K,K+1):
            f.write(f'{K:2d} {Q:2d} {JKQ[K][Q].real:10.3e} {JKQ[K][Q].imag:10.3e}\n')

    # Get Jqq (assuming Helium)
    line = pointC.atomic.atom.lines[0]
    line.jqq = {}

    # For each q
    for qq in range(-1,2):

        # Factor for q
        f1 = JS.sign(1+qq)

        # Initialize jqq
        line.jqq[qq] = {}

        # For each q'
        for qp in range(-1,2):

            # Initialize
            line.jqq[qq][qp] = (0. + 0.j)

            # For each K
            for K in range(3):

                # K factor
                f2 = f1*np.sqrt((2.*K+1.)/3.)

                # Get Q from 3J
                Q = qq - qp

                # Control Q
                if np.absolute(Q) > K:
                    continue

                # Contribution
                contr = f2*JS.j3(1,1,K,qq,-qp,-Q)*JKQ[K][Q]

                # Add contribution
                line.jqq[qq][qp] += contr

    factor = 1e3/299792458e5
    f.write('\n')
    f.write('J00 Blu\n')
    f.write(f'{JKQ[0][0].real*line.B_lu*factor}\n')

    f.write('\n')
    f.write('\n')
    f.write('Jqq input\n')
    for q in range(-1,2):
        for qp in range(-1,2):
            f.write(f'{q:2d} {qp:2d} {line.jqq[q][qp].real:10.3e} {line.jqq[q][qp].imag:10.3e}\n')

    # Convert units
    for qq in line.jqq:
        for qp in line.jqq[qq]:
            line.jqq[qq][qp] *= factor

    # Summon SEE
    pointC.atomic.solveESE(None,cdt)

    atomic = pointC.atomic
    atom = pointC.atomic.atom
    f.write('\n')
    f.write('\n')
    f.write("rho(jM,j'M') output\n")
    for term in atom.terms:
        for index in term.index:
            i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
            if imag:
                continue
            f.write(f'[{i:3d}][{ii:2d},{ii1:2d}] ' + \
                    f'rho({mu:2d},{int(M):2d};{mu1:2d},{int(M1):2d}) = '+ \
                    f'{atomic.rho[i]:13.6e}')
                   #f'rho({mu:2d},{M:4.1f};{mu1:2d},{M1:4.1f}) = '+ \
            if ii == ii1:
                f.write(f'\n')
            else:
                f.write(f' {atomic.rho[i+1]:13.6e}\n')


    f.write('\n')
    f.write('\n')
    f.write("rhoKQ output\n")
    for it,term in enumerate(atom.terms):
       #f.write(f"Term L {term.L} S {term.S}\n")
       #print(f"Term L {term.L} S {term.S}")

        indexes = copy.deepcopy(term.index)
        rhos = copy.deepcopy(atomic.rho).tolist()[:term.NN_next]

        LL = term.L
        SS = term.S
        maxJ = LL + SS

        # Initialize next index
        ii = term.NN_next-1

        # Initialize left index
        i1 = -1

        # For each "left" M
        for iM,M,Mblock in zip(range(term.nM),term.M,term.Mblock):
            # For each "left" mu
            for mu in range(Mblock):

                # Get mu, M index
                i1 += 1

                # Initialize right index
                i2 = -1

                # For each "right" M
                for iM1,M1,Mblock1 in zip(range(term.nM),term.M,term.Mblock):

                    # If dM > 1, skip
                    if int(round(np.absolute(M1 - M))) > int(round(2.*maxJ)):

                        # Advance the index anyways
                        i2 += Mblock1

                        # And skip
                        continue

                    # For each "right" mu
                    for mu1 in range(Mblock1):

                        # Get mu1, M1 index
                        i2 += 1

                        # Only if i2 < i1
                        if i2 < i1:

                            for index in term.index:
                                ii2 = index[1]
                                ii1 = index[2]
                                if i1 == ii1 and i2 == ii2:
                                    crrho = atomic.rho[index[0]]
                                    cirho = atomic.rho[index[0]+1]
                                    break

                            ii += 1
                            indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,False])
                            rhos.append(crrho)
                            ii += 1
                            indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,True])
                            rhos.append(-1.*cirho)
        rhos = np.array(rhos)

        # Add missing indexes

        for J in term.J:

            # Possible M
            Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

            for Jp in term.J:
                minK = int(round(np.absolute(J-Jp)))
                maxK = int(round(J + Jp))

                for K in range(minK,maxK+1):
                    for Q in range(-K,K+1):

                        # Initialize rhoKQ
                        rho = 1j*0.

                        # For each M
                        for M in Ms:

                            #
                            # Known contribution
                            #

                            # Decide M'
                            Mp = M - Q

                            # Skip invalid M'
                            if np.absolute(Mp) > Jp:
                                continue

                            # Sign and weight
                            SS = JS.sign(J-M)* \
                                 np.sqrt(2.*K + 1.)

                            # Go by the whole term
                            for index in indexes:

                                # Extract data
                                ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index

                                # If M,M'
                                if (M1 != M or M2 != Mp):
                                    continue

                                # Run over J
                                for Jindex1 in term.index_muM[iM1]:
                                    J1 = Jindex1[2]
                                    iJ1 = Jindex1[1]

                                    # Relevant J
                                    if np.absolute(J1-J) > 0.25:
                                        continue

                                    # Run over J'
                                    for Jindex2 in term.index_muM[iM2]:
                                        J2 = Jindex2[2]
                                        iJ2 = Jindex2[1]

                                        # Relevant J'
                                        if np.absolute(J2-Jp) > 0.25:
                                            continue

                                        # Contribution
                                        CC = term.eigvec[i1][iJ1]* \
                                             term.eigvec[i2][iJ2]* \
                                             SS*JS.j3(J,Jp,K,M,-Mp,-Q)

                                        # If imag
                                        if imag:
                                            rho += CC*rhos[ii]*1j
                                        else:
                                            rho += CC*rhos[ii]
                                            print(f'{J} {Jp} {K} {Q} {i1}{M1} {i2}{M2} {ii} {CC*rhos[ii]}  {rho}')


                        # Print rhoKQ
                        f.write(f'{it:1d}')
                        f.write(f' {J:3.1f}')
                        f.write(f' {Jp:3.1f}')
                        f.write(f' {K:2d}')
                        f.write(f' {Q:2d}')
                        f.write(f' {rho.real:13.6e}')
                        if Q != 0 or J != Jp:
                            f.write(f' {rho.imag:13.6e}\n')
                        else:
                            f.write(f'\n')

    f.close()

    # Exit debug
   #sys.exit('Debugging SEE')
    print('Debugged SEE')

    # Open fortran rhoKQ file
    f = open('/home/tanausu/Documents/Codes/MyCodes/HANLERT/Outputs/debugheliumcode/fort.400','r')
    lines = list(f)
    f.close()
    # Empty first line
    lines.pop(0)
    # Initialize rhoKQ(J,J')
    rhoKQ = {}
    for line in lines:
        try:
            iterm,rJ,rJ1,K,Q,Re,Im = line.split()
        except:
            try:
                iterm,rJ,rJ1,K,Q,Re = line.split()
                Im = 0.
            except:
                break
        iterm = int(iterm) - 1
        rJ = float(rJ)
        rJ1 = float(rJ1)
        K = int(K)
        Q = int(Q)
        Re = float(Re)
        Im = float(Im)

        if iterm not in rhoKQ:
            rhoKQ[iterm] = {}
        if rJ not in rhoKQ[iterm]:
            rhoKQ[iterm][rJ] = {}
        if rJ1 not in rhoKQ[iterm][rJ]:
            rhoKQ[iterm][rJ][rJ1] = {}
        if K not in rhoKQ[iterm][rJ][rJ1]:
            rhoKQ[iterm][rJ][rJ1][K] = {}
        rhoKQ[iterm][rJ][rJ1][K][Q] = Re + 1j*Im

    # Build rhonn'
    atomic.rho *= 0.

    # For each term
    for it,term in enumerate(atomic.atom.terms):
        iJdic = {}
        for iJ,J in enumerate(term.J):
            iJdic[J] = iJ
        # For each index in the term
        for index in term.index:
            # Extract data
            ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index
            for rJ in term.J:
                if rJ < abs(M1):
                    continue
                f1 = JS.sign(rJ-M1)* \
                     term.eigvec[i1][iJdic[rJ]]
                for rJ1 in term.J:
                    if rJ1 < abs(M2):
                        continue
                    f2 = f1*term.eigvec[i2][iJdic[rJ1]]
                    for K in range(int(round(abs(rJ-rJ1))), \
                                   int(round(abs(rJ+rJ1)))+1):
                        f3 = f2*np.sqrt(2.*K + 1.)
                        for Q in range(-K,K+1):
                            if abs(M1 - M2 -Q) > 0.:
                                continue
                            f4 = f3*JS.j3(rJ,rJ1,K,M1,-M2,-Q)

                            if imag:
                                atomic.rho[ii] += f4*np.imag(rhoKQ[it][rJ][rJ1][K][Q])
                            else:
                                atomic.rho[ii] += f4*np.real(rhoKQ[it][rJ][rJ1][K][Q])

    f = open('pyth.400','w')
    f.write("rho(jM,j'M') intput\n")
    for term in atom.terms:
        for index in term.index:
            i,ii,ii1,iM,M,mu,iM1,M1,mu1,imag = index
            if imag:
                continue
            f.write(f'[{i:3d}][{ii:2d},{ii1:2d}] ' + \
                    f'rho({mu:2d},{int(M):2d};{mu1:2d},{int(M1):2d}) = '+ \
                    f'{atomic.rho[i]:13.6e}')
                   #f'rho({mu:2d},{M:4.1f};{mu1:2d},{M1:4.1f}) = '+ \
            if ii == ii1:
                f.write(f'\n')
            else:
                f.write(f' {atomic.rho[i+1]:13.6e}\n')


    f.write('\n')
    f.write('\n')
    f.write("rhoKQ input\n")
    for it,term in enumerate(atom.terms):
       #f.write(f"Term L {term.L} S {term.S}\n")
       #print(f"Term L {term.L} S {term.S}")

        indexes = copy.deepcopy(term.index)
        rhos = copy.deepcopy(atomic.rho).tolist()[:term.NN_next]

        LL = term.L
        SS = term.S
        maxJ = LL + SS

        # Initialize next index
        ii = term.NN_next-1

        # Initialize left index
        i1 = -1

        # For each "left" M
        for iM,M,Mblock in zip(range(term.nM),term.M,term.Mblock):
            # For each "left" mu
            for mu in range(Mblock):

                # Get mu, M index
                i1 += 1

                # Initialize right index
                i2 = -1

                # For each "right" M
                for iM1,M1,Mblock1 in zip(range(term.nM),term.M,term.Mblock):

                    # If dM > 1, skip
                    if int(round(np.absolute(M1 - M))) > int(round(2.*maxJ)):

                        # Advance the index anyways
                        i2 += Mblock1

                        # And skip
                        continue

                    # For each "right" mu
                    for mu1 in range(Mblock1):

                        # Get mu1, M1 index
                        i2 += 1

                        # Only if i2 < i1
                        if i2 < i1:

                            for index in term.index:
                                ii2 = index[1]
                                ii1 = index[2]
                                if i1 == ii1 and i2 == ii2:
                                    crrho = atomic.rho[index[0]]
                                    cirho = atomic.rho[index[0]+1]
                                    break

                            ii += 1
                            indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,False])
                            rhos.append(crrho)
                            ii += 1
                            indexes.append([ii,i1,i2,iM,M,mu,iM1,M1,mu1,True])
                            rhos.append(-1.*cirho)
        rhos = np.array(rhos)

        # Add missing indexes

        for J in term.J:

            # Possible M
            Ms = np.linspace(-J,J,int(round(2*J+1)),endpoint=True)

            for Jp in term.J:
                minK = int(round(np.absolute(J-Jp)))
                maxK = int(round(J + Jp))

                for K in range(minK,maxK+1):
                    for Q in range(-K,K+1):

                        # Initialize rhoKQ
                        rho = 1j*0.

                        # For each M
                        for M in Ms:

                            #
                            # Known contribution
                            #

                            # Decide M'
                            Mp = M - Q

                            # Skip invalid M'
                            if np.absolute(Mp) > Jp:
                                continue

                            # Sign and weight
                            SS = JS.sign(J-M)* \
                                 np.sqrt(2.*K + 1.)

                            # Go by the whole term
                            for index in indexes:

                                # Extract data
                                ii,i1,i2,iM1,M1,mu1,iM2,M2,mu2,imag = index

                                # If M,M'
                                if (M1 != M or M2 != Mp):
                                    continue

                                # Run over J
                                for Jindex1 in term.index_muM[iM1]:
                                    J1 = Jindex1[2]
                                    iJ1 = Jindex1[1]

                                    # Relevant J
                                    if np.absolute(J1-J) > 0.25:
                                        continue

                                    # Run over J'
                                    for Jindex2 in term.index_muM[iM2]:
                                        J2 = Jindex2[2]
                                        iJ2 = Jindex2[1]

                                        # Relevant J'
                                        if np.absolute(J2-Jp) > 0.25:
                                            continue

                                        # Contribution
                                        CC = term.eigvec[i1][iJ1]* \
                                             term.eigvec[i2][iJ2]* \
                                             SS*JS.j3(J,Jp,K,M,-Mp,-Q)

                                        # If imag
                                        if imag:
                                            rho += CC*rhos[ii]*1j
                                        else:
                                            rho += CC*rhos[ii]
                                           #print(f'{J} {Jp} {K} {Q} {i1}{M1} {i2}{M2} {ii} {CC*rhos[ii]}  {rho}')


                        # Print rhoKQ
                        f.write(f'{it:1d}')
                        f.write(f' {J:3.1f}')
                        f.write(f' {Jp:3.1f}')
                        f.write(f' {K:2d}')
                        f.write(f' {Q:2d}')
                        f.write(f' {rho.real:13.6e}')
                        if Q != 0 or J != Jp:
                            f.write(f' {rho.imag:13.6e}\n')
                        else:
                            f.write(f'\n')

    f.close()

    # Get RT_coeffs
    sf, kk = RT_coeficients.getRTcoefs(pointC.atomic, cdt.orays[0], cdt)

    f = open('pyth.500','w')
    for inu,nu in enumerate(nus):
        f.write(f'{c.c*1e7/nu:23.15e} {kk[0][0][inu]:23.15e} {kk[0][1][inu]:23.15e} ' + \
                f'{kk[0][2][inu]:23.15e} {kk[0][3][inu]:23.15e} ' + \
                f'{kk[1][0][inu]:23.15e} {kk[1][1][inu]:23.15e} ' + \
                f'{kk[1][2][inu]:23.15e} ' + \
                f'{sf[0,inu]:23.15e} {sf[1][inu]:23.15e} ' + \
                f'{sf[2,inu]:23.15e} {sf[3][inu]:23.15e}\n')
    f.close()


    sys.exit('Debugging RTc')
    #############
    # Debug SEE #
    #############


if __name__ == '__main__':
    main()
#
# Block of prints to check Tqq and TKQ in RTcoeff_MT
'''
TKQ = [{},{},{},{}]
#fill
for i in range(4):
    for K in range(0,3):
        TKQ[i][K] = {}
        for Q in range(-K,K+1):
            TKQ[i][K][Q] = 0. + 1j*0.
# Compute
for i in range(4):
    for K in range(0,3):
        for Q in range(-K,K+1):
            for q in range(-1,2):
                qp = q - Q
                if np.absolute(qp) > 1:
                    continue
                ff = jsim.sign(1+q)*np.sqrt(3.*(2.*K+1))* \
                     jsim.j3(1,1,K,q,-qp,-Q)
                tag = f'{q}{qp}'
                TKQ[i][K][Q] += ff*TQQ[i][tag]

f = open('pyth.400','a')
f.write('\n')
f.write('\n')
f.write('Tqq\n')
for q in range(-1,2):
    for qp in range(-1,2):
        tag = f'{q}{qp}'
        f.write(f'{q} {qp} ' + \
                f'{np.real(TQQ[0][tag]):23.15e} ' + \
                f'{np.imag(TQQ[0][tag]):23.15e} ' + \
                f'{np.real(TQQ[1][tag]):23.15e} ' + \
                f'{np.imag(TQQ[1][tag]):23.15e} ' + \
                f'{np.real(TQQ[2][tag]):23.15e} ' + \
                f'{np.imag(TQQ[2][tag]):23.15e} ' + \
                f'{np.real(TQQ[3][tag]):23.15e} ' + \
                f'{np.imag(TQQ[3][tag]):23.15e}\n')
f.write('\n')
f.write('\n')
f.write('TKQ\n')
for K in range(0,3):
    for Q in range(-K,K+1):
        f.write(f'{K:2d} {Q:2d} ' + \
                f'{np.real(TKQ[0][K][Q]):23.15e} ' + \
                f'{np.imag(TKQ[0][K][Q]):23.15e} ' + \
                f'{np.real(TKQ[1][K][Q]):23.15e} ' + \
                f'{np.imag(TKQ[1][K][Q]):23.15e} ' + \
                f'{np.real(TKQ[2][K][Q]):23.15e} ' + \
                f'{np.imag(TKQ[2][K][Q]):23.15e} ' + \
                f'{np.real(TKQ[3][K][Q]):23.15e} ' + \
                f'{np.imag(TKQ[3][K][Q]):23.15e}\n')
f.close()
'''

#
# Debug prints from term_class()
'''
        # Debug mode
        self.debug = False

        if self.debug:
            print('\n\n======================')
            print(f'\nTerm S={SS} L={LL} E={Energy}')
            if maxJ != minJ:
                print(f'J c [{minJ},{maxJ}]')
            else:
                print(f'J = {minJ}')

        if self.debug:
            print(f'M c [{-maxJ},{maxJ}] ({nM})')

            if self.debug:
                print(f'\n\nM={M}\n')

            if self.debug:
                print(f'Minimum J={Jm}')

                if self.debug:
                    print(f'Added index: [{ii},{k},{J},{mu},{M}]')

                    if self.debug:
                        print(f'Level J ={J}')
                        print(f"Level J'={J1}")

                    if self.debug:
                        print(f'comm [{i1},{i}]={comm}')

                        if self.debug:
                            print(f'diag {i,i1} {diag[i]}')
                            print(comm,M,c.nuL*Bnorm,E,self.TE)

                        if self.debug:
                            print(f'odiag {i,i1} {odiag[i1-1]}')
                            print(comm,c.nuL*Bnorm)

            if self.debug:
                print(f'Block size={nblock}')

            if self.debug:
                print(f'\nB={Bnorm} L={larmor}')
               #print('Matrix:')
               #print(rmatr)
                print('Diagon')
                print(diag)
                print('Odiagon')
                print(odiag)
                print('Diagonal:')
                print(w)
                if v is not None:
                    for i in range(len(w)):
                        print(w[i],v[:,i])

        # Debug
        if self.debug:
            for iM,Mindex in enumerate(self.index_muM):
                print(f'Mblock {iM}:')
                for index in Mindex:
                    vec = f'({self.eigvec[index[0]][0]})'
                    for i in range(1,self.eigvec[index[0]].size):
                        vec += f',{self.eigvec[index[0]][i]}'
                    vec += f')'
                    print(f'  i {index[0]} mu iJ {index[1]} J {index[2]} ' + \
                          f'mu {index[3]} M {index[4]} -> E ' + \
                          f'{self.eigval[index[0]]} Ev ' + vec)
            for index in self.index:
                print(f'Index {index[0]:2d}: ' + \
                      f'({index[5]:1d},{index[4]:3.1f};{index[8]:1d},{index[7]:3.1f}) ' + \
                      f'{index[9]}  ;  {index[1]},{index[2]}  ; {index[3]},{index[6]}')
'''

# Class for FS components that was in atom.py
'''
class line_FS():
    """Class that defines the lines of the atomic model for a multi-term atom
    """

    def __init__(self, levels, line_levels, jlju, Alu):

        self.levels = line_levels
        self.jlju = jlju

        self.gl = levels[line_levels[0]].g
        self.gu = levels[line_levels[1]].g

        self.wavelength = 1/(levels[line_levels[1]].E - levels[line_levels[0]].E)
        self.energy = c.h * c.c / self.wavelength
        self.nu = self.energy/c.h
        self.nu3 = self.nu*self.nu*self.nu

        self.A_lu = Alu
        self.A_ul = Alu
        self.B_lu = Alu * (c.c**2/(2*c.h*self.nu**3))
        self.B_ul = self.B_lu * (levels[line_levels[1]].g/levels[line_levels[0]].g)

        self.dJ = levels[line_levels[1]].J - levels[line_levels[0]].J
'''

# Code to define FS components that was in line_class
'''

            # Initialize fine structure components
            self.lines = []

            # Fill fine structure components
           #Lu = terms[line_terms[1]].L
           #Ll = terms[line_terms[0]].L
           #S = terms[line_terms[1]].S
           #for Ju,Eu in zip(terms[line_terms[1]].J,terms[line_terms[1]].LE):
           #    for Jl,El in zip(terms[line_terms[0]].J,terms[line_terms[0]].LE):

           #        # Check dipole
           #        if np.absolute(Ju - Jl) > 1.25 or np.absolute(Jl + Ju) < 0.25:
           #            continue

           #        self.lines.append(line_FS(Lu,Ll,S,Ju,Jl,Eu,El,Alu))
'''

# Debugging in line_class()
'''

        # Debugging
        self.debug = False

'''

# Debugging in line_class()'s add_contribution_profiles
'''
   #def add_contribution_profiles(self, contr, nus, nu0=None):

            try:
                import matplotlib.pyplot as plt
            except:
                pass
            lamb = c.c.cgs*(1e7*u.nm/u.cm)/nus
            plt.plot(lamb,contr)
            plt.plot(lamb,contr,marker='*')
            print('Plot contribution')
            plt.show()
'''

# Debugging in line_class()'s normalize_profiles()
'''
            print(f'Profile normalized to: {self.prof.sum()}')
            try:
                import matplotlib.pyplot as plt
            except:
                pass
            lamb = c.c.cgs*(1e7*u.nm/u.cm)/nus
            plt.plot(lamb,self.prof)
            plt.plot(lamb,self.prof,marker='*')
            print('Plot normalized')
            plt.show()
'''

# Debugging in line_class()'s sumStokes()
'''
            # Debug
            if self.debug:
                print('Summing Stokes in line, no especial')
'''

# Debugging in line_class()'s actuallysumStokes()
'''
            # Debug
            if self.debug:
                print(f'  Adding Stokes {i}, angular weight {ray.weight}')

            # Debug
            if self.debug:
                print(f'  Computed contribution to add {contr}')

                    # Debug
                    if self.debug:
                        print(f'    T{qq}{qp} {Tqq[i][f"{qq}{qp}"]}')
                        print(f'    Previous value J{qq}{qp} {jqq[qq][qp]}')

                    # Debug
                    if self.debug:
                        print(f'    New value J{qq}{qp} {jqq[qq][qp]}')
'''





