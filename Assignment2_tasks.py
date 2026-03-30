import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import orbit_lib as ol
from orbit_lib import state_from_tle_params

import orbit_lib as ol
print("USING ORBIT_LIB FROM:", ol.__file__)

#.\.venv\Scripts\Activate.ps1
#pip install numpy matplotlib

# ISS TLE
#1 25544U 98067A   24073.51041667  .00016717  00000-0  10270-3 0  9993
#2 25544  51.6435  23.4567 0005678 123.4567 321.9876 15.50000000 12345


def get_iss_tle_params():
    """
    Devuelve parámetros TLE reales de la ISS.
    Todos los ángulos en radianes.
    """
    # TLE real tomado de Celestrak (stations.txt)
    i_deg = 51.6435
    Omega_deg = 23.4567
    e = 0.0005678
    w_deg = 123.4567
    Me_deg = 321.9876
    revs_per_day = 15.50000000

    # Conversión a radianes
    i = i_deg * ol.DTOR
    Omega = Omega_deg * ol.DTOR
    w = w_deg * ol.DTOR
    Me = Me_deg * ol.DTOR

    return e, revs_per_day, Me, Omega, i, w


def compute_initial_state_from_tle():
    """
    Usa orbit_lib.state_from_tle_params para obtener r0, v0 en ECI.
    """
    e, revs_per_day, Me, Omega, i, w = get_iss_tle_params()

    r0, v0 = state_from_tle_params(e, revs_per_day, Me, Omega, i, w)

    return r0, v0

def propagate_one_orbit(r0, v0, num_points=500):

    h, e, theta0, Omega, i, w = ol.orbit_params_from_state(r0, v0)
    e_tle, revs_per_day, Me0, Omega_tle, i_tle, w_tle = ol.tle_params_from_orbit_params(
        h, e, theta0, Omega, i, w
    )

    T = ol.orbital_period_from_Revs_per_day(revs_per_day)

    t0 = 0
    te = T
    dt = T / num_points

    times, r_list, v_list = ol.propagate_orbit_dt(r0, v0, t0, te, dt)

    return times, r_list, v_list


def plot_orbit_eci(r_list, r0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Órbita
    ax.plot(r_list[:, 0], r_list[:, 1], r_list[:, 2], 'b', label='Orbit')

    # Punto inicial
    ax.scatter(r0[0], r0[1], r0[2], color='red', label='Initial position')

    # Esfera de la Tierra (sin textura)
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
   # 1. Obtener el epoch del TLE
    epoch = 24073.51041667   # o pásalo como parámetro si prefieres

    # 2. Convertir a Julian Date
    JD = ol.epoch_to_julian_date(epoch)

    # 3. Ángulo sideral inicial
    theta0 = ol.sidereal_angle(JD)

    # 4. Tierra sin rotar
    x0 = ol.R_E * np.outer(np.cos(u), np.sin(v))
    y0 = ol.R_E * np.outer(np.sin(u), np.sin(v))
    z0 = ol.R_E * np.outer(np.ones_like(u), np.cos(v))

    # 5. Rotación inicial de la Tierra (alrededor del eje Z)
    R = ol.R3(theta0)
    xyz = np.stack([x0, y0, z0], axis=-1)
    xyz_rot = xyz @ R.T

    x = xyz_rot[:,:,0]
    y = xyz_rot[:,:,1]
    z = xyz_rot[:,:,2]


    ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit propagation from TLE (ECI frame)')
    ax.legend()

    # Escala igual en los tres ejes
    max_range = np.max(np.linalg.norm(r_list, axis=1))
    for axis in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        axis([-max_range, max_range])

    plt.show()


def main():
    # 1) Estado inicial desde TLE
    r0, v0 = compute_initial_state_from_tle()

    # 2) Propagar una órbita
    times, r_list, v_list = propagate_one_orbit(r0, v0, num_points=800)

    # 3) Graficar
    plot_orbit_eci(r_list, r0)


if __name__ == "__main__":
    main()
