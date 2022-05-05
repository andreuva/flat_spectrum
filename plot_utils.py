import constants
import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os


def unique_filename(directory, name, extension):
    i = 0
    path = directory + '/' + name + f'_{i}.' + extension
    while os.path.exists(path):
        i += 1
        path = directory + '/' + name + f'_{i}.' + extension

    return path


def save_or_show(mode, file, directory):
    if mode == 'save':

        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(unique_filename(directory, file, 'png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_quadrature(cdt, mode='save', directory='plots'):
    plt.figure()
    # plt.subplot(projection="aitoff")

    inclinations_loc = np.array([ray.rinc for ray in cdt.rays])
    azimuts_loc = np.array([ray.raz for ray in cdt.rays])

    inclinations_glob = np.array([ray.inc_glob*constants.degtorad for ray in cdt.rays])
    azimuts_glob = np.array([ray.az_glob*constants.degtorad for ray in cdt.rays])

    plt.plot(azimuts_loc, np.cos(inclinations_loc), 'o', label='local RF')
    plt.plot(azimuts_glob, np.cos(inclinations_glob), 'o', alpha=0.5, label='global RF')
    plt.title('Quadrature in the two reference frames')
    plt.legend()
    save_or_show(mode, 'quadrature', directory)

    # Create a sphere
    r = 0.95
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    # transform data
    theta, phi, r = inclinations_loc, azimuts_loc, np.ones_like(azimuts_loc)
    xx_loc = np.cos(phi)*np.sin(theta)
    yy_loc = np.sin(phi)*np.sin(theta)
    zz_loc = np.cos(theta)

    theta, phi, r = inclinations_glob, azimuts_glob, np.ones_like(azimuts_loc)
    xx_glob = np.cos(phi)*np.sin(theta)
    yy_glob = np.sin(phi)*np.sin(theta)
    zz_glob = np.cos(theta)

    # Set colours and render
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.5, linewidth=0)
    ax.scatter(xx_glob, yy_glob, zz_glob, color="r", s=50, alpha=1)
    ax.scatter(xx_loc, yy_loc, zz_loc, color="g", s=50, alpha=1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")

    plt.tight_layout()

    ax.view_init(elev=0., azim=0.)
    plt.savefig(f"{directory}filament_view.png")

    ax.view_init(elev=90., azim=0.)
    plt.savefig(f"{directory}prominence_view.png")


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
    lab = ['I', 'Q', 'U', 'V']

    if norm:
        name = name + '_norm'
        normalization = st.sun_rad.stokes[0][int(len(st.sun_rad.stokes[0])/2)].value
        QQ = QQ/(II + 1e-30)
        UU = UU/(II + 1e-30)
        VV = VV/(II + 1e-30)
        II = II/normalization
        lab = ['I(norm)', 'Q/I', 'U/I', 'V/I']

    stokes = [II, QQ, UU, VV]
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, 4+1):
        fig.add_subplot(2, 2, i)
        plt.imshow(stokes[i-1], aspect='auto', origin='lower', extent=[cdt.w0.value, cdt.wf.value, cdt.z0.value, cdt.zf.value], cmap='inferno')
        plt.xlabel('frequency')
        plt.ylabel('z')
        plt.colorbar()
        plt.title(lab[i-1])

    plt.tight_layout(pad=2.0)
    save_or_show(mode, name, directory)


def plot_quantity(xx, quantity, names=['x', 'quantity'], mode='save', directory='plots'):

    plt.plot(xx, quantity)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(names[1] + ' vs ' + names[0])
    save_or_show(mode, names[1], directory)


def plot_4_profiles(wave, eta_I, eta_Q, eta_U, eta_V, title=False, n=0, eps=False,
                    show=True, save=False, directory='plots', name='4_profiles'):
    
    # if there is no directory, create one
    if (not os.path.exists(directory)) and save:
        os.makedirs(directory)
    
    if eps:
        pre = r'$\epsilon $'
    else:
        pre = ''

    # plot the profiles of emission
    if n==0:
        plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(wave, eta_I, f'C{n}')
    plt.title(fr'{pre}$I$')
    plt.xlabel(r'$\nu$')
    plt.subplot(2,2,2)
    plt.plot(wave, eta_Q, f'C{n}')
    plt.title(fr'{pre}$Q$')
    plt.xlabel(r'$\nu$')
    plt.subplot(2,2,3)
    plt.plot(wave, eta_U, f'C{n}')
    plt.title(fr'{pre}$U$')
    plt.xlabel(r'$\nu$')
    plt.subplot(2,2,4)
    plt.plot(wave, eta_V, f'C{n}')
    plt.title(fr'{pre}$V$')
    plt.xlabel(r'$\nu$')
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(f'{directory}/{name}.png')
    if show:
        plt.show()
    if save or show:
        plt.close()
