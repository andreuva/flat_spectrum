import matplotlib.pyplot as plt
import numpy as np
import os


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
    if mode == 'save':
        dir = './' + directory + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        file = 'quadrature'
        ext = '.png'
        filename = dir+file+ext

        i = 0
        while os.path.exists(filename):
            filename = dir + file + '_' + str(i) + ext
            i += 1

        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_z_profile(cdt, st, mode='save', directory='./plots'):
    profile = np.array([st.radiation[i].stokes[0][5].value for i in range(cdt.z_N)])
    plt.plot(cdt.zz, profile)
    plt.xlabel('vertical height (Km)')
    plt.ylabel('Intensity (CGS)')
    plt.title('Vertical profile of radiation')
    if mode == 'save':
        dir = './' + directory + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)

        file = 'I_z_profile'
        ext = '.png'
        filename = dir+file+ext

        i = 0
        while os.path.exists(filename):
            filename = dir + file + '_' + str(i) + ext
            i += 1

        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
