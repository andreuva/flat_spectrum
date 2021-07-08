import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
import numpy as np
import os


def save_or_show(mode, file, directory):
    if mode == 'save':
        dir = './' + directory + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        ext = '.png'
        filename = dir+file+ext

        i = 0
        while os.path.exists(filename):
            filename = dir + file + '_' + str(i) + ext
            i += 1

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quadrature(cdt, mode='save', directory='plots'):
    plt.figure()
    plt.subplot(projection="aitoff")

    inclinations_loc = np.array([ray.inc.value for ray in cdt.rays])
    azimuts_loc = np.array([ray.az.value for ray in cdt.rays])

    inclinations_glob = np.array([ray.inc_glob.value for ray in cdt.rays])
    azimuts_glob = np.array([ray.az_glob.value for ray in cdt.rays])

    plt.plot(inclinations_loc, azimuts_loc, 'o', label='local RF')
    plt.plot(inclinations_glob, azimuts_glob, 'o', alpha=0.5, label='global RF')
    plt.grid(True)
    plt.title('Quadrature in the two reference frames')
    plt.legend()
    save_or_show(mode, 'quadrature', directory)


def plot_z_profile(cdt, st, nu='mean', norm=True, name='stokes_profile', mode='save', directory='plots'):

    normalization = st.sun_rad.stokes[0][int(len(st.sun_rad.stokes[0])/2)].value
    if nu == 'mean':
        name = name + '_mean'
        II = np.array([st.radiation[i].stokes[0].mean().value for i in range(cdt.z_N)])
        QQ = np.array([st.radiation[i].stokes[1].mean().value for i in range(cdt.z_N)])
        UU = np.array([st.radiation[i].stokes[2].mean().value for i in range(cdt.z_N)])
        VV = np.array([st.radiation[i].stokes[3].mean().value for i in range(cdt.z_N)])
    elif -cdt.nus_N < nu < cdt.nus_N and type(nu) == int:
        II = np.array([st.radiation[i].stokes[0][nu].value for i in range(cdt.z_N)])
        QQ = np.array([st.radiation[i].stokes[1][nu].value for i in range(cdt.z_N)])
        UU = np.array([st.radiation[i].stokes[2][nu].value for i in range(cdt.z_N)])
        VV = np.array([st.radiation[i].stokes[3][nu].value for i in range(cdt.z_N)])
    else:
        print(f'NOT PROFILE SELECTED, ERROR IN NU = {nu}')
        II = cdt.zz*0
        QQ = cdt.zz*0
        UU = cdt.zz*0
        VV = cdt.zz*0

    if norm:
        name = name + '_norm'
        QQ = QQ/(II + 1e-30)
        UU = UU/(II + 1e-30)
        VV = VV/(II + 1e-30)
        II = II/normalization
        lab = ['I(norm)', 'Q/I', 'U/I', 'V/I']
    else:
        lab = ['I', 'Q', 'U', 'V']

    stokes = [II, QQ, UU, VV]

    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 4+1):
        fig.add_subplot(2, 2, i)
        plt.plot(cdt.zz, stokes[i-1])
        plt.ticklabel_format(useOffset=False)
        plt.xlabel('Z (CGS)')
        plt.ylabel(lab[i-1])
        plt.title(lab[i-1])
    plt.tight_layout(pad=2.0)
    save_or_show(mode, name, directory)


def plot_stokes_im(cdt, st, norm=True, name='stokes_image', mode='save', directory='plots'):

    II = np.array([st.radiation[i].stokes[0].value for i in range(cdt.z_N)])
    QQ = np.array([st.radiation[i].stokes[1].value for i in range(cdt.z_N)])
    UU = np.array([st.radiation[i].stokes[2].value for i in range(cdt.z_N)])
    VV = np.array([st.radiation[i].stokes[3].value for i in range(cdt.z_N)])

    if norm:
        name = name + '_norm'
        normalization = st.sun_rad.stokes[0][int(len(st.sun_rad.stokes[0])/2)].value
        QQ = QQ/(II + 1e-30)
        UU = UU/(II + 1e-30)
        VV = VV/(II + 1e-30)
        II = II/normalization

    stokes = [II, QQ, UU, VV]
    lab = ['I(norm)', 'Q/I', 'U/I', 'V/I']
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 4+1):
        fig.add_subplot(2, 2, i)
        plt.imshow(stokes[i-1], aspect='auto', extent=[cdt.w0.value, cdt.wf.value, cdt.z0.value, cdt.zf.value])
        plt.xlabel('frequency')
        plt.ylabel('z')
        plt.colorbar()
        plt.title(lab[i-1])

    plt.tight_layout(pad=2.0)
    save_or_show(mode, name, directory)


def plot_quantity(cdt, xx, quantity, names=['x', 'quantity'], mode='save', directory='plots'):

    plt.plot(xx, quantity)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(names[1] + ' vs ' + names[0])
    save_or_show(mode, names[1], directory)
