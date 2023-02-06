import matplotlib.pyplot as plt
import numpy as np

def rdif(x,y):
    return 1.0e2*np.absolute(x - y)/np.absolute(x)
def adif(x,y):
    return np.absolute(x - y)
def minimax(x,y):
    return f'[{np.min(np.absolute(x))},{np.min(np.absolute(y))};' + \
           f' {np.max(np.absolute(x))},{np.max(np.absolute(y))}]'

def main():

    file1 = 'pyth.500'
    file2 = '/home/tanausu/Documents/Codes/MyCodes/HANLERT/Outputs/debugheliumcode/fort.500'

    l1 = []
    l2 = []
    eta01 = []
    eta11 = []
    eta21 = []
    eta31 = []
    rho11 = []
    rho21 = []
    rho31 = []
    eps01 = []
    eps11 = []
    eps21 = []
    eps31 = []
    eta02 = []
    eta12 = []
    eta22 = []
    eta32 = []
    rho12 = []
    rho22 = []
    rho32 = []
    eps02 = []
    eps12 = []
    eps22 = []
    eps32 = []


    f1 = open(file1,'r')
    f2 = open(file2,'r')
    lines1 = list(f1)
    lines2 = list(f2)
    f1.close()
    f2.close()

    for lin1,lin2 in zip(lines1,lines2):
        col1 = lin1.split()
        col2 = lin2.split()

        l1.append(float(col1[0]))
        l2.append(float(col2[0]))
        eta01.append(float(col1[1]))
        eta02.append(float(col2[1]))
        eta11.append(float(col1[2]))
        eta12.append(float(col2[2]))
        eta21.append(float(col1[3]))
        eta22.append(float(col2[3]))
        eta31.append(float(col1[4]))
        eta32.append(float(col2[4]))
        rho11.append(float(col1[5]))
        rho12.append(float(col2[5]))
        rho21.append(float(col1[6]))
        rho22.append(float(col2[6]))
        rho31.append(float(col1[7]))
        rho32.append(float(col2[7]))
        eps01.append(float(col1[8]))
        eps02.append(float(col2[8]))
        eps11.append(float(col1[9]))
        eps12.append(float(col2[9]))
        eps21.append(float(col1[10]))
        eps22.append(float(col2[10]))
        eps31.append(float(col1[11]))
        eps32.append(float(col2[11]))
    l1 = np.array(l1)
    l2 = np.array(l2)
    eta01 = np.array(eta01)
    eta02 = np.array(eta02)
    eta11 = np.array(eta11)
    eta12 = np.array(eta12)
    eta21 = np.array(eta21)
    eta22 = np.array(eta22)
    eta31 = np.array(eta31)
    eta32 = np.array(eta32)
    rho11 = np.array(rho11)
    rho12 = np.array(rho12)
    rho21 = np.array(rho21)
    rho22 = np.array(rho22)
    rho31 = np.array(rho31)
    rho32 = np.array(rho32)
    eps01 = np.array(eps01)
    eps02 = np.array(eps02)
    eps11 = np.array(eps11)
    eps12 = np.array(eps12)
    eps21 = np.array(eps21)
    eps22 = np.array(eps22)
    eps31 = np.array(eps31)
    eps32 = np.array(eps32)

    scale = False
    vacuum = 1e-200
    if scale:
        eta11 *= (eta01 + vacuum)
        eta12 *= (eta01 + vacuum)
        eta21 *= (eta01 + vacuum)
        eta22 *= (eta01 + vacuum)
        eta31 *= (eta01 + vacuum)
        eta32 *= (eta01 + vacuum)
        rho11 *= (eta01 + vacuum)
        rho12 *= (eta01 + vacuum)
        rho21 *= (eta01 + vacuum)
        rho22 *= (eta01 + vacuum)
        rho31 *= (eta01 + vacuum)
        rho32 *= (eta01 + vacuum)
        eps01 *= (eta01 + vacuum)
        eps02 *= (eta01 + vacuum)
        eps11 *= (eta01 + vacuum)
        eps12 *= (eta01 + vacuum)
        eps21 *= (eta01 + vacuum)
        eps22 *= (eta01 + vacuum)
        eps31 *= (eta01 + vacuum)
        eps32 *= (eta01 + vacuum)

    # Transform HanleRT
    factor = 1e3/299792458e5
    eps02 *= factor
    eps12 *= factor
    eps22 *= factor
    eps32 *= factor

    # Wave
    d = adif(l1,l2)
    print(f'Wavelegth adif: {np.min(d)}--{np.max(d)} ' + \
          f'[{np.min(l1)},{np.min(l2)};{np.max(l1)},{np.max(l2)}]')
    d = rdif(l1,l2)
    print(f'Wavelegth rdif: {np.min(d)}--{np.max(d)}')

    # eta0
    d = adif(eta01,eta02)
    print(f'eta0 adif: {np.min(d)}--{np.max(d)} ' + minimax(eta01,eta02))
    d = rdif(eta01,eta02)
    print(f'eta0 rdif: {np.min(d)}--{np.max(d)}')
    fig = plt.figure(num=0)
    ax = plt.subplot()
    ax.plot(l1,eta01,'b')
    ax.plot(l2,eta02,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eta0')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eta1
    d = adif(eta11,eta12)
    print(f'eta1 adif: {np.min(d)}--{np.max(d)} ' +  minimax(eta11,eta12))
    d = rdif(eta11,eta12)
    print(f'eta1 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eta11,'b')
    ax.plot(l2,eta12,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eta1')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    ax2.plot(l1,d,'*')
    plt.show()
    ax2.cla()
    ax.cla()

    # eta2
    d = adif(eta21,eta22)
    print(f'eta2 adif: {np.min(d)}--{np.max(d)} ' +  minimax(eta21,eta22))
    d = rdif(eta21,eta22)
    print(f'eta2 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eta21,'b')
    ax.plot(l2,eta22,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eta2')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eta3
    d = adif(eta31,eta32)
    print(f'eta3 adif: {np.min(d)}--{np.max(d)} ' +  minimax(eta31,eta32))
    d = rdif(eta31,eta32)
    print(f'eta3 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eta31,'b')
    ax.plot(l2,eta32,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eta3')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # rho1
    d = adif(rho11,rho12)
    print(f'rho1 adif: {np.min(d)}--{np.max(d)} ' +  minimax(rho11,rho12))
    d = rdif(rho11,rho12)
    print(f'rho1 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,rho11,'b')
    ax.plot(l2,rho12,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'rho1')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
   #ax2.cla()
    ax.cla()

    # rho2
    d = adif(rho21,rho22)
    print(f'rho2 adif: {np.min(d)}--{np.max(d)} ' +   minimax(rho21,rho22))
    d = rdif(rho21,rho22)
    print(f'rho2 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,rho21,'b')
    ax.plot(l2,rho22,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'rho2')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # rho3
    d = adif(rho31,rho32)
    print(f'rho3 adif: {np.min(d)}--{np.max(d)} ' +   minimax(rho31,rho32))
    d = rdif(rho31,rho32)
    print(f'rho3 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,rho31,'b')
    ax.plot(l2,rho32,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'rho3')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eps0
    d = adif(eps01,eps02)
    print(f'eps0 adif: {np.min(d)}--{np.max(d)} ' +   minimax(eps01,eps02))
    d = rdif(eps01,eps02)
    print(f'eps0 rdif: {np.min(d)}--{np.max(d)}')
    fig = plt.figure(num=0)
    ax = plt.subplot()
    ax.plot(l1,eps01,'b')
    ax.plot(l2,eps02,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eps0')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eps1
    d = adif(eps11,eps12)
    print(f'eps1 adif: {np.min(d)}--{np.max(d)} ' + minimax(eps11,eps12))
    d = rdif(eps11,eps12)
    print(f'eps1 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eps11,'b')
    ax.plot(l2,eps12,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eps1')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eps2
    d = adif(eps21,eps22)
    print(f'eps2 adif: {np.min(d)}--{np.max(d)} ' + minimax(eps21,eps22))
    d = rdif(eps21,eps22)
    print(f'eps2 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eps21,'b')
    ax.plot(l2,eps22,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eps2')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()

    # eps3
    d = adif(eps31,eps32)
    print(f'eps3 adif: {np.min(d)}--{np.max(d)} ' + minimax(eps31,eps32))
    d = rdif(eps31,eps32)
    print(f'eps3 rdif: {np.min(d)}--{np.max(d)}')
    ax = plt.subplot()
    ax.plot(l1,eps31,'b')
    ax.plot(l2,eps32,'g')
    ax.set_xlabel(r'$\lambda$ [nm]')
    ax.set_ylabel(r'eps3')
    ax2 = ax.twinx()
    ax2.plot(l1,d,'b:')
    plt.show()
    ax2.cla()
    ax.cla()



def main_SEE():


    file1 = 'pyth.200'
    file2 = '/home/tanausu/Documents/Codes/MyCodes/HANLERT/Outputs/debugheliumcode/fort.200'

    f = open(file1,'r')
    lines1 = list(f)
    f.close()
    f = open(file2,'r')
    lines2 = list(f)
    f.close()

    while lines1[0].strip().lower() != 'rhokq output':
        lines1.pop(0)
    lines1.pop(0)
    while lines2[0].strip().lower() != 'rhokq output':
        lines2.pop(0)
    lines2.pop(0)

    # Compare line by line
    for l1,l2 in zip(lines1,lines2):
        d1 = l1.split()
        d2 = l2.split()

        try:
            it,J1,J2,K,Q,Re1,Im1 = d1
            it,J1,J2,K,Q,Re2,Im2 = d2
            Re1 = float(Re1)
            Re2 = float(Re2)
            Im1 = float(Im1)
            Im2 = float(Im2)
            dRa = adif(Re1,Re2)
            dIa = adif(Im1,Im2)
            dRr = rdif(Re1,Re2)
            dIr = rdif(Im1,Im2)
        except:
            it,J1,J2,K,Q,Re1 = d1
            it,J1,J2,K,Q,Re2 = d2
            Re1 = float(Re1)
            Re2 = float(Re2)
            Im1 = None
            Im2 = None
            dRa = adif(Re1,Re2)
            dRr = rdif(Re1,Re2)
        it = int(it)
        J1 = float(J1)
        J2 = float(J1)
        K = int(K)
        Q = int(Q)

        print(f"Term {it:1d} J {J1:1.0f} J' {J2:1.0f} K {K:1d} Q {Q:2d}:") 
       #print(f"    Real adif: {dRa}   {Re1:23.15e}  {Re2:23.15e}")
       #print(f"    Real rdif: {dRr}")
        if dRr > 1e-1:
            print(f"    Real rdif: {dRr:8.2e}   {Re1:23.15e}  {Re2:23.15e}    <===")
        elif dRr > 1e-4:
            print(f"    Real rdif: {dRr:8.2e}   {Re1:23.15e}  {Re2:23.15e}    *")
        else:
            print(f"    Real rdif: {dRr:8.2e}   {Re1:23.15e}  {Re2:23.15e}")
        if Im1 is not None:
           #print(f"    Imag adif: {dIa}   {Im1:23.15e}  {Im2:23.15e}")
           #print(f"    Imag rdif: {dIr}")
            if dIr > 1e-1:
                print(f"    Imag rdif: {dIr:8.2e}   {Im1:23.15e}  {Im2:23.15e}    <===")
            elif dIr > 1e-4:
                print(f"    Imag rdif: {dIr:8.2e}   {Im1:23.15e}  {Im2:23.15e}    *")
            else:
                print(f"    Imag rdif: {dIr:8.2e}   {Im1:23.15e}  {Im2:23.15e}")

if __name__ == '__main__':
   #main()
    main_SEE()
