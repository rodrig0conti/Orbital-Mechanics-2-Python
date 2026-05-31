import numpy as np
import simutils_9 as su

mu = 3.986004418e5   # km^3/s^2
MU= mu
R_E = 6378           # km
w_E = 7.2921e-5      # rad/s
f = 1/298.257223563

AU = 149597870.0      # km, 
m_earth = 7.767e6     # T·km^3,
# Geodetic latitude
phi_geo = np.deg2rad(9.21)
phi_m = np.arctan((1 - f)**2 * np.tan(phi_geo))   # geocentric
lam_m = 0.0  

DTOR = np.pi / 180        
RTOD = 180 / np.pi       

def mean_anomaly_from_eccentric_anomaly(E, e):
    return E - e * np.sin(E)

def orbital_period_from_semi_major_axis(a):
    return 2 * np.pi * np.sqrt(a**3 / MU)

def orbital_period_from_Revs_per_day(revs_per_day):
    return 86400 / revs_per_day

def eccentric_anomaly_from_true_anomaly(theta, e):
    factor = np.sqrt((1 - e) / (1 + e))
    return 2 * np.arctan(factor * np.tan(theta / 2))

def true_anomaly_from_eccentric_anomaly(E, e):
    factor = np.sqrt((1 + e) / (1 - e))
    return 2 * np.arctan(factor * np.tan(E / 2))

def orbit_params_from_tle_params(e, revs_per_day, Me, Omega, i, w):
    #Orbital period from revs/day
    T = orbital_period_from_Revs_per_day(revs_per_day)
    a = (MU * (T / (2 * np.pi))**2) ** (1/3)
    h = np.sqrt(MU * a * (1 - e**2))

    E = eccentric_anomaly_from_mean_anomaly(Me, e)
    theta = true_anomaly_from_eccentric_anomaly(E, e)
    return h, e, theta, Omega, i, w

def tle_params_from_orbit_params(h, e, theta, Omega, i, w):
    a = h**2 / (MU * (1 - e**2))
    T = orbital_period_from_semi_major_axis(a)
    revs_per_day = 86400 / T
    E = eccentric_anomaly_from_true_anomaly(theta, e)
    Me = mean_anomaly_from_eccentric_anomaly(E, e)
    return e, revs_per_day, Me, Omega, i, w


def rotation_matrix_from_classical_euler_sequence(Omega, i, w):
    #Rotation matrix from perifocal frame to ECI using  euler sequence:
    #R = R3(Omega) * R1(i) * R3(w)
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)

    R3_O = np.array([[ cO, -sO, 0],
                     [ sO,  cO, 0],
                     [  0,   0, 1]])

    R1_i = np.array([[1,  0,   0],
                     [0, ci, -si],
                     [0, si,  ci]])

    R3_w = np.array([[ cw, -sw, 0],
                     [ sw,  cw, 0],
                     [  0,   0, 1]])

    return R3_O @ R1_i @ R3_w

def rotation_matrix_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    #R = R3(yaw) * R2(pitch) * R1(roll)
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    R1 = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    R2 = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])

    R3 = np.array([[ cy, -sy, 0],
                   [ sy,  cy, 0],
                   [  0,   0, 1]])

    return R3 @ R2 @ R1

def R3(theta):
    #Rotation matrix around the z axis by angle theta in radians
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def angle_wrap_radians(angle):
    #[0, 2π).
    return angle % (2 * np.pi)

def angle_wrap_degrees(angle):
    #[0, 360).
    return angle % 360.0

def quat_from_axis_angle(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])

def quaternion_from_classical_euler_sequence(Omega, i, w):
    #quaternion equivalent of R = R3(Omega) * R1(i) * R3(w)
    qO = quat_from_axis_angle([0, 0, 1], Omega)
    qi = quat_from_axis_angle([1, 0, 0], i)
    qw = quat_from_axis_angle([0, 0, 1], w)

    #q_total = qO * qi * qw
    return quat_multiply(quat_multiply(qO, qi), qw)

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    q_roll  = quat_from_axis_angle([1, 0, 0], roll)
    q_pitch = quat_from_axis_angle([0, 1, 0], pitch)
    q_yaw   = quat_from_axis_angle([0, 0, 1], yaw)
    return quat_multiply(quat_multiply(q_yaw, q_pitch), q_roll)


def eccentric_anomaly_from_mean_anomaly(Me, e, tol=1e-10, N=50):
    #Solve Me=E-e*sin(E) using Newton's method. radians
    # Initial guess
    if Me < np.pi:
        E = Me + e
    else:
        E = Me - e
    #Newton iterations
    for _ in range(N):
        f  = E - e*np.sin(E) - Me
        fp = 1 - e*np.cos(E)
        E_new = E - f/fp
        # convergence
        if abs(E_new - E) < tol:
            return E_new

        E = E_new
    # If doesnt converge return last value
    return E

def sidereal_angle(JD):
    #It computes the earths sidereal angle theta_G in radians from the Julian date JD
    JD_int = np.floor(JD)
    #Julian centuries since J2000.0
    T0 = (JD_int - 2451545.0) / 36525.0
    #theta_G0 at 00:00 UTC (in degrees)
    theta_G0 = (100.4606184
                + 36000.77005361 * T0
                + 0.00038793 * T0**2
                - 2.6e-8 * T0**3)

    #Add earth rotation since midnight 
    frac = (JD + 0.5) - np.floor(JD + 0.5)
    seconds_since_midnight = frac * 86400.0

    theta_G = theta_G0 + (w_E * seconds_since_midnight) * RTOD  # convert rad todeg
    theta_G = theta_G % 360.0 #wrap
    #give in radians
    return theta_G * DTOR

def state_from_orbit_params(h, e, theta, Omega, i, w):
    #give vector state r and v in ECI
    r_mag = (h**2 / MU) / (1 + e * np.cos(theta))
    r_p = np.array([
        r_mag * np.cos(theta),
        r_mag * np.sin(theta),
        0.0
    ])
    v_p = (MU / h) * np.array([
        -np.sin(theta),
        e + np.cos(theta),
        0.0
    ])
    R = rotation_matrix_from_classical_euler_sequence(Omega, i, w) #perifocal to ECI
    r_eci = R @ r_p
    v_eci = R @ v_p
    return r_eci, v_eci

def state_from_tle_params(e, revs_per_day, Me, Omega, i, w):
    # give r and v in eci from TLE
    h, e, theta, Omega, i, w = orbit_params_from_tle_params(
        e, revs_per_day, Me, Omega, i, w
    )
    r_eci, v_eci = state_from_orbit_params(h, e, theta, Omega, i, w)
    return r_eci, v_eci


def orbit_params_from_state(r, v):
    #Give classical orbital parameters from r and v in ECI
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    vr = np.dot(r, v) / r_mag

    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    i = np.arccos(h_vec[2] / h)

    k = np.array([0, 0, 1])
    N = np.cross(k, h_vec)
    N_mag = np.linalg.norm(N)

    # RAAN
    if N_mag != 0:
        Omega = np.arccos(N[0] / N_mag)
        if N[1] < 0:
            Omega = 2*np.pi - Omega
    else:
        Omega = 0.0  # equatorial orbit

    e_vec = (1/MU) * ((v_mag**2 - MU/r_mag)*r - vr*v)
    e = np.linalg.norm(e_vec)

    # Argument of perigee
    if N_mag != 0 and e > 1e-12:
        w = np.arccos(np.dot(N, e_vec) / (N_mag * e))
        if e_vec[2] < 0:
            w = 2*np.pi - w
    else:
        w = 0.0  # circular  orbit

    #True anomaly
    if e > 1e-12:
        theta = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if vr < 0:
            theta = 2*np.pi - theta
    else:
        # circular orbit: true anomaly undefined then use angle from N
        theta = np.arccos(np.dot(N, r) / (N_mag * r_mag))
        if r[2] < 0:
            theta = 2*np.pi - theta

    return h, e, theta, Omega, i, w


def propagate_orbit_dt(r0, v0, t0, te, dt):
    h, e, theta0, Omega, i, w = orbit_params_from_state(r0, v0)

    a = h**2 / (MU * (1 - e**2))
    T = 2 * np.pi * np.sqrt(a**3 / MU)
    n = 2 * np.pi / T

    E0 = eccentric_anomaly_from_true_anomaly(theta0, e)
    Me = mean_anomaly_from_eccentric_anomaly(E0, e)

    t = t0
    r_list, v_list, t_list = [], [], []

    while t <= te:
        # update mean anomaly
        Me = Me + n * dt
        Me = Me % (2 * np.pi)

        #Kepler + true anomaly
        E = eccentric_anomaly_from_mean_anomaly(Me, e)
        theta = true_anomaly_from_eccentric_anomaly(E, e)

        # state from orbit params
        r, v = state_from_orbit_params(h, e, theta, Omega, i, w)

        r_list.append(r)
        v_list.append(v)
        t_list.append(t)

        t += dt
    return np.array(t_list), np.array(r_list), np.array(v_list)

def epoch_to_julian_date(epoch):

    #  Extract year and day of year
    YY = int(epoch // 1000)
    DDD = epoch - YY * 1000

    # Convert YY to full year
    if YY < 57:
        year = 2000 + YY
    else:
        year = 1900 + YY

    # Convert day of year to month/day/hour/min/sec
    day_int = int(DDD)
    frac_day = DDD - day_int

    # Hours, minutes, seconds
    hours = frac_day * 24
    hour = int(hours)
    minutes = (hours - hour) * 60
    minute = int(minutes)
    seconds = (minutes - minute) * 60
    second = seconds

    # Convert day of year to month/day
    month = 1
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    day = day_int
    for dim in days_in_month:
        if day > dim:
            day -= dim
            month += 1
        else:
            break
    #Convert to Julian Date
    A = int((14 - month) / 12)
    Y = year + 4800 - A
    M = month + 12*A - 3

    JD = (day
          + int((153*M + 2)/5)
          + 365*Y
          + int(Y/4)
          - int(Y/100)
          + int(Y/400)
          - 32045)

    # Add fractional day
    JD = JD + (hour - 12)/24 + minute/1440 + second/86400
    return JD

def get_hubble_tle_params():
    return tle_params_from_degrees(
        i_deg=28.47,
        Omega_deg=339.47,
        e=0.00028,
        w_deg=279.94,
        Me_deg=75.12,
        revs_per_day=15.092
    )

def get_hubble_epoch():
    return 24073.12345678

def tle_params_from_degrees(i_deg, Omega_deg, e, w_deg, Me_deg, revs_per_day):
    i = i_deg * DTOR
    Omega = Omega_deg * DTOR
    w = w_deg * DTOR
    Me = Me_deg * DTOR
    return e, revs_per_day, Me, Omega, i, w

def get_iss_tle_params():
    return tle_params_from_degrees(
        i_deg=51.6435,
        Omega_deg=23.4567,
        e=0.0005678,
        w_deg=123.4567,
        Me_deg=321.9876,
        revs_per_day=15.50000000
    )

def get_iss_epoch():
    return 24073.51041667

def compute_initial_state_from_tle():
    e, revs_per_day, Me, Omega, i, w = get_iss_tle_params()  #esta es para la iss
    r0, v0 = state_from_tle_params(e, revs_per_day, Me, Omega, i, w)
    return r0, v0


def propagate_one_orbit(r0, v0, num_points=500):
    h, e, theta0, Omega, i, w = orbit_params_from_state(r0, v0)
    e_tle, revs_per_day, Me0, Omega_tle, i_tle, w_tle = tle_params_from_orbit_params(
        h, e, theta0, Omega, i, w
    )

    T = orbital_period_from_Revs_per_day(revs_per_day)

    t0 = 0
    te = T
    dt = T / num_points

    times, r_list, v_list = propagate_orbit_dt(r0, v0, t0, te, dt)
    return times, r_list, v_list


def plot_orbit_eci(r_list, r0):
    #graphs orbit in 3D ECI frame
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(r_list[:, 0], r_list[:, 1], r_list[:, 2], 'b', label='Orbit')

    ax.scatter(r0[0], r0[1], r0[2], color='red', label='Initial position')

    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    epoch = 24073.51041667

    JD = epoch_to_julian_date(epoch)
    theta0 = sidereal_angle(JD)

    x0 = R_E * np.outer(np.cos(u), np.sin(v))
    y0 = R_E * np.outer(np.sin(u), np.sin(v))
    z0 = R_E * np.outer(np.ones_like(u), np.cos(v))

    #initial earth rotation
    R = R3(theta0)
    xyz = np.stack([x0, y0, z0], axis=-1)
    xyz_rot = xyz @ R.T

    x = xyz_rot[:, :, 0]
    y = xyz_rot[:, :, 1]
    z = xyz_rot[:, :, 2]

    ax.plot_surface(x, y, z, color='lightblue', alpha=0.5)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Orbit propagation from TLE (ECI frame)')
    ax.legend()

    max_range = np.max(np.linalg.norm(r_list, axis=1))
    for axis in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        axis([-max_range, max_range])

    plt.show()

def quaternion_from_rotation_matrix(R):
    R = np.array(R, dtype=float)
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)
def orbit_frame_from_state(r_i, v_i):
    r = np.array(r_i, float)
    v = np.array(v_i, float)

    r_norm = np.linalg.norm(r)

    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec)

    # Right-handed LVLH / orbital frame:

    z_o = -r / r_norm
    y_o = -h_vec / h_norm
    x_o = np.cross(y_o, z_o)
    x_o = x_o / np.linalg.norm(x_o)

    R_io = np.column_stack((x_o, y_o, z_o))

    q_io = quaternion_from_rotation_matrix(R_io)

    if hasattr(orbit_frame_from_state, "q_io_prev"):
        if np.dot(orbit_frame_from_state.q_io_prev, q_io) < 0:
            q_io *= -1
    orbit_frame_from_state.q_io_prev = q_io.copy()

    # angular velocity of orbit frame
    w_i_io = h_vec / (r_norm**2)

    # angular acceleration of orbit frame 
    rdot = v
    factor = 2 * np.dot(r, rdot) / (r_norm**2)
    dw_i_io = factor * w_i_io

    return q_io, w_i_io, dw_i_io

class orbit_classic:
    def __init__(self, h, e, theta, Omega, i, w):
        self.h = h
        self.e = e
        self.theta = theta
        self.Omega = Omega
        self.i = i
        self.w = w

    def propagate(self, t_step):
        r, _ = state_from_orbit_params(self.h, self.e, self.theta,
                                       self.Omega, self.i, self.w)
        r_norm = np.linalg.norm(r)
        dot_theta = self.h / (r_norm**2)
        self.theta = (self.theta + dot_theta * t_step) % (2*np.pi)

    def get_params(self):
        return self.h, self.e, self.theta, self.Omega, self.i, self.w

    def get_state(self):
        return state_from_orbit_params(self.h, self.e, self.theta,
                                       self.Omega, self.i, self.w)

    def get_orbit_frame(self):
        r, v = self.get_state()
        return orbit_frame_from_state(r, v)


class orbit_tle:
    def __init__(self, n, e, Me, Omega, i, w):
        self.n = n          # rad/s
        self.e = e
        self.Me = Me
        self.Omega = Omega
        self.i = i
        self.w = w

    def propagate(self, t_step):
        self.Me = (self.Me + self.n * t_step) % (2*np.pi)

    def get_params(self):
        return self.n, self.e, self.Me, self.Omega, self.i, self.w

    def get_state(self):
        T = 2*np.pi / self.n

        a = (MU * (T / (2*np.pi))**2)**(1/3)
        h = np.sqrt(MU * a * (1 - self.e**2))

        E = eccentric_anomaly_from_mean_anomaly(self.Me, self.e)
        theta = true_anomaly_from_eccentric_anomaly(E, self.e)

        return state_from_orbit_params(h, self.e, theta,
                                   self.Omega, self.i, self.w)

    def get_orbit_frame(self):
        r, v = self.get_state()
        return orbit_frame_from_state(r, v)

def geocentric_from_xyz(rE):
    x, y, z = rE
    r = np.linalg.norm(rE)
    phi = np.arctan2(y, x)
    lam = np.arctan2(z, np.sqrt(x**2 + y**2))
    return phi, lam, r

def xyz_from_geocentric(phi, lam, r):
    x = r * np.cos(lam) * np.cos(phi)
    y = r * np.cos(lam) * np.sin(phi)
    z = r * np.sin(lam)
    return np.array([x, y, z])

def geodetic_from_xyz(rE):
    # WGS-84 constants
    RE = 6378.137  # km
    f = 1 / 298.257223563
    e2 = (2*f - f**2)

    x, y, z = rE
    r = np.linalg.norm(rE)

    # geocentric longitude and latitude
    phi = np.arctan2(y, x)
    lam = np.arctan2(z, np.sqrt(x**2 + y**2))

    # initial guess for geodetic latitude
    lat = lam
    #  iterate
    for _ in range(10):
        N = RE / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = np.sqrt(x**2 + y**2) / np.cos(lat) - N
        lat_new = np.arctan2(z + e2 * N * np.sin(lat), np.sqrt(x**2 + y**2))
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    return phi, lat, h

def xyz_from_geodetic(phi, lat, h):
    RE = 6378.137  # km
    f = 1 / 298.257223563
    e2 = (2*f - f**2)

    N = RE / np.sqrt(1 - e2 * np.sin(lat)**2)

    x = (N + h) * np.cos(lat) * np.cos(phi)
    y = (N + h) * np.cos(lat) * np.sin(phi)
    z = (N * (1 - e2) + h) * np.sin(lat)

    return np.array([x, y, z])

# WGS-84 constants
RE = 6378.137  # km
f = 1 / 298.257223563
J2 = 0.001082629821313
mu = 398600.4418  # km^3/s^2

class OrbitPKepler:
    def __init__(self, a, e, i, Omega, w, Me, n, ndot, nddot):
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.w = w
        self.Me = Me

        self.n = n              # rad/s
        self.ndot = ndot        # rad/s²
        self.nddot = nddot     # rad/s³

        self.mu = 398600.4418
        self.RE = 6378.137
        self.J2 = 0.001082629821313

        self.update_p()
        self.update_state()

    def update_p(self):
        self.p = self.a * (1 - self.e**2)

    def propagate(self, dt):

        # Update a 
        self.a = self.a - (2*self.a/(3*self.n)) * self.ndot * dt

        #  Update e 
        self.e = self.e - (self.ndot/self.n) * (1 - self.e) * dt

        # Clamp e y
        if self.e < 0: self.e = 0
        if self.e > 0.1: self.e = 0.1
        self.update_p()

        Omega_dot = -(3/2)*self.J2*(self.RE**2/self.p**2)*self.n*np.cos(self.i)
        w_dot     =  (3/4)*self.J2*(self.RE**2/self.p**2)*self.n*(4 - 5*np.sin(self.i)**2)

        self.Omega += Omega_dot * dt
        self.w     += w_dot * dt

        self.Omega = np.mod(self.Omega, 2*np.pi)
        self.w     = np.mod(self.w, 2*np.pi)

        #  Mean anomaly update
        self.Me = self.Me + self.n*dt + 0.5*self.ndot*dt**2 + (1/6)*self.nddot*dt**3
        self.Me = np.mod(self.Me, 2*np.pi)

        # Update state
        self.update_state()

    def update_state(self):
        E = eccentric_anomaly_from_mean_anomaly(self.Me, self.e)
        theta = true_anomaly_from_eccentric_anomaly(E, self.e)
        h = np.sqrt(self.mu * self.a * (1 - self.e**2))
        self.r_i, self.v_i = state_from_orbit_params(h, self.e, theta, self.Omega, self.i, self.w)

    def get_state(self):
        return self.r_i, self.v_i

    def get_orbit_frame(self):
        return orbit_frame_from_state(self.r_i, self.v_i)


class OrbitKeplerSimple:

    def __init__(self, a, e, i, Omega, w, Me, n, ndot, nddot):
        self.a = a
        self.e = e
        self.i = i
        self.Omega = Omega
        self.w = w
        self.Me = Me
        self.n = n
        self.ndot = ndot
        self.nddot = nddot
        self.mu = 398600.4418

    def propagate(self, dt):
        # Mean anomaly update
        self.Me = self.Me + self.n*dt + 0.5*self.ndot*dt**2 + (1/6)*self.nddot*dt**3
        # Solve Kepler
        E = self.solve_kepler(self.Me, self.e)
        # Position in orbital plane
        r_orb = self.a * np.array([
            np.cos(E) - self.e,
            np.sqrt(1 - self.e**2) * np.sin(E),
            0
        ])

        # Transform to ECI
        self.r_i = self.orbital_to_eci(r_orb)
        self.v_i = self.velocity_from_E(E)

    def solve_kepler(self, M, e):
        E = M
        for _ in range(10):
            E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        return E

    def orbital_to_eci(self, r_orb):
        R3_w = su.R3(self.w)
        R1_i = su.R1(self.i)
        R3_O = su.R3(self.Omega)
        return R3_O @ (R1_i @ (R3_w @ r_orb))

    def velocity_from_E(self, E):
        n = self.n
        e = self.e
        a = self.a

        r = a*(1 - e*np.cos(E))
        v_orb = np.array([
            -a*n*np.sin(E)/r,
            a*n*np.sqrt(1-e**2)*np.cos(E)/r,
            0
        ])

        R3_w = su.R3(self.w)
        R1_i = su.R1(self.i)
        R3_O = su.R3(self.Omega)
        return R3_O @ (R1_i @ (R3_w @ v_orb))

    def get_state(self):
        return self.r_i, self.v_i

    def get_orbit_frame(self):
        return orbit_frame_from_state(self.r_i, self.v_i)

def datetime_to_julian_date(year, month, day, hour=0, minute=0, second=0):
    if month <= 2:
        year -= 1
        month += 12

    A = int(year/100)
    B = 2 - A + int(A/4)

    JD_day = int(365.25*(year + 4716)) \
             + int(30.6001*(month + 1)) \
             + day + B - 1524.5

    JD_frac = (hour + minute/60 + second/3600) / 24

    return JD_day + JD_frac

def sun_vector(JD):
    T = (JD - 2451545.0) / 36525.0 #From julian time to Julian centuries sinceJ2000

    AM = 280.46 + 36000.771 * T          
    M  = 357.5291092 + 35999.05034 * T   # mean anomaly of sun (deg)
    eps = 23.439291 - 0.0130042 * T      # inclination of equator (deg)

    AM = AM % 360.0
    M = np.deg2rad(M % 360.0)
    eps = np.deg2rad(eps)

    lam = M + np.deg2rad(1.9146471) * np.sin(M) \
            + np.deg2rad(0.01994643) * np.sin(2*M)

    r = AU * (1.000140612
              - 0.016708617 * np.cos(M)
              - 0.000139589 * np.cos(2*M))

    x = r * np.cos(lam)
    y = r * np.cos(eps) * np.sin(lam)
    z = r * np.sin(eps) * np.sin(lam)

    return np.array([x, y, z])

def magnetic_field_dipole(r_i, JD):

    r = np.linalg.norm(r_i)
    if r == 0:
        return np.zeros(3)

    mx = np.cos(phi_m) * np.cos(lam_m)
    my = np.cos(phi_m) * np.sin(lam_m)
    mz = np.sin(phi_m)
    m_vec = m_earth * np.array([mx, my, mz])   # T·km^3

    m_dot_r = np.dot(m_vec, r_i)

    B = (-3 * m_dot_r * r_i + (r**2) * m_vec) / (r**5)
    return B

def eci_to_ecef(r_eci, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1]
    ])
    return R @ r_eci

def ecef_to_geodetic(r_ecef):
    phi, lat, h = geodetic_from_xyz(r_ecef)
    return lat, phi, h

def gravity_gradient(r_i, q_ib, J):
    r = np.linalg.norm(r_i)
    if r < 1e-6:
        return np.zeros(3)

    if not isinstance(q_ib, su.Quaternion):
        q_ib = su.Quaternion(q_ib)
    q_bi = q_ib.inverted()
    r_b = q_bi.rotate(r_i)

    Jr = J @ r_b
    tau_G = 3 * mu / r**5 * np.cross(r_b, Jr)
    return tau_G