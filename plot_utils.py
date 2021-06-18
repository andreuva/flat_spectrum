import matplotlib.pyplot as plt
import numpy as np


def plot_quadrature(cdt):
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
    plt.show()


def plot_z_profile(cdt, st):
    profile = np.array([st.radiation[i].stokes[0][5].value for i in range(cdt.z_N)])
    plt.plot(cdt.zz, profile)
    plt.xlabel('vertical height (Km)')
    plt.ylabel('Intensity (CGS)')
    plt.title('Vertical profile of radiation')
    plt.show()
