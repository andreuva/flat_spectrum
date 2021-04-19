from numpy import sin, cos, exp, sqrt, conj

def Tqq(q1,q2, i, theta, chi):
    t = 0.
    c = False
    if q2<q1:
        c = True
        tmp = q1
        q1 = q2
        q2 = tmp
    
    if i==0:
        if q1==q2==-1 or q1==q2==1: t = 1/4*(1+cos(theta)**2)
        elif q1==q2==0: t = 1/2*sin(theta)**2
        elif q1==-1 and q2==0: t = -1/2/sqrt(2)*sin(theta)*cos(theta)*exp(-chi*1j)
        elif q1==-1 and q2==1: t = 1/4*sin(theta)**2*exp(-chi*2j)
        elif q1==0 and q2==1: t = 1/2/sqrt(2)*sin(theta)*cos(theta)*exp(-chi*1j)
    elif i==1:
        if q1==q2==-1 or q1==q2==1: t = -1/4*sin(theta)**2
        elif q1==q2==0: t = 1/2*sin(theta)**2
        elif q1==-1 and q2==0: t = -1/2/sqrt(2)*cos(theta)*sin(theta)*exp(-chi*1j)
        elif q1==-1 and q2==1: t = -1/4*(1+cos(theta)**2)*exp(-chi*2j)
        elif q1==0 and q2==1: t = 1/2/sqrt(2)*cos(theta)*sin(theta)*exp(-chi*1j)
    elif i==2:
        if q1==-1 and q2==0: t = 1/2/sqrt(2)*1j*sin(theta)*exp(-chi*1j)
        elif q1==-1 and q2==1: t = 1/4*2j*cos(theta)*exp(-chi*2j)
        elif q1==0 and q2==1: t = -1/2/sqrt(2)*1j*sin(theta)*exp(-chi*1j)
    elif i==3:
        if q1==q2==-1: t = -1/2*cos(theta)
        elif q1==q2==1: t = 1/2*cos(theta)
        elif q1==-1 and q2==0: t = 1/2/sqrt(2)*sin(theta)*exp(-chi*1j)
        elif q1==0 and q2==1: t = 1/2/sqrt(2)*sin(theta)*exp(-chi*1j)

    if not c:
        return t
    else:
        return conj(t)