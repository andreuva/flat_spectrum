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


def plot_z_profile(cdt, st, nu='mean', norm=True, mode='save', directory='plots'):

    normalization = st.sun_rad.stokes[0][int(len(st.sun_rad.stokes[0])/2)].value

    if nu == 'mean':
        profile = np.array([st.radiation[i].stokes[0].mean().value for i in range(cdt.z_N)])
    elif -cdt.nus_N < nu < cdt.nus_N and type(nu) == int:
        profile = np.array([st.radiation[i].stokes[0][nu].value for i in range(cdt.z_N)])
    else:
        print(f'NOT PROFILE SELECTED, ERROR IN NU = {nu}')
        profile = cdt.zz*0

    if norm:
        profile = profile/normalization

    plt.plot(cdt.zz, profile)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel('vertical height (CGS)')
    plt.ylabel('Intensity/B_w (normaliced)')
    plt.title('Vertical profile of radiation')
    save_or_show(mode, 'I_z_profile', directory)


def plot_stokes_im(cdt, st, norm=True, mode='save', directory='plots'):

    im = np.array([st.radiation[i].stokes[0].value for i in range(cdt.z_N)])
    if norm:
        normalization = st.sun_rad.stokes[0][int(len(st.sun_rad.stokes[0])/2)].value
        im = im/normalization

    plt.imshow(im, aspect='auto')
    plt.colorbar()
    plt.xlabel('frequency')
    plt.ylabel('z')
    save_or_show(mode, 'stokes_profile', directory)


def plot_quantity(cdt, xx, quantity, names=['x', 'quantity'], mode='save', directory='plots'):

    plt.plot(xx, quantity)
    plt.ticklabel_format(useOffset=False)
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title(names[1] + ' vs ' + names[0])
    save_or_show(mode, names[1], directory)
