import numpy as np

mu = 3.986004418e5   # km^3/s^2
MU= mu
R_E = 6378           # km
w_E = 7.2921e-5      # rad/s

# Degree/radian conversion
DTOR = np.pi / 180        # degrees → radians
RTOD = 180 / np.pi        # radians → degrees

def mean_anomaly_from_eccentric_anomaly(E, e):
    """
    Compute mean anomaly Me from eccentric anomaly E and eccentricity e.
    E, e in radians.
    """
    return E - e * np.sin(E)


def orbital_period_from_semi_major_axis(a):
    """
    Orbital period T from semi-major axis a (km).
    Returns T in seconds.
    """
    return 2 * np.pi * np.sqrt(a**3 / MU)


def orbital_period_from_Revs_per_day(revs_per_day):
    """
    Orbital period T from revolutions per day.
    Returns T in seconds.
    """
    return 86400 / revs_per_day


def eccentric_anomaly_from_true_anomaly(theta, e):
    """
    Eccentric anomaly E from true anomaly theta and eccentricity e.
    All angles in radians.
    """
    factor = np.sqrt((1 - e) / (1 + e))
    return 2 * np.arctan(factor * np.tan(theta / 2))


def true_anomaly_from_eccentric_anomaly(E, e):
    """
    True anomaly theta from eccentric anomaly E and eccentricity e.
    All angles in radians.
    """
    factor = np.sqrt((1 + e) / (1 - e))
    return 2 * np.arctan(factor * np.tan(E / 2))

def orbit_params_from_tle_params(e, revs_per_day, Me, Omega, i, w):
    """
    Convert TLE parameters (e, revs/day, Me, Ω, i, ω)
    into classical orbital parameters (h, e, θ, Ω, i, ω).
    All angles in radians.
    """
    # 1) Orbital period from revs/day
    T = orbital_period_from_Revs_per_day(revs_per_day)

    # 2) Semi-major axis from period
    a = (MU * (T / (2 * np.pi))**2) ** (1/3)

    # 3) Specific angular momentum
    h = np.sqrt(MU * a * (1 - e**2))

    # 4) Eccentric anomaly from mean anomaly
    E = eccentric_anomaly_from_mean_anomaly(Me, e)

    # 5) True anomaly from eccentric anomaly
    theta = true_anomaly_from_eccentric_anomaly(E, e)

    return h, e, theta, Omega, i, w


def tle_params_from_orbit_params(h, e, theta, Omega, i, w):
    """
    From classical orbital parameters to TLE-like parameters:
    returns (e, revs_per_day, Me, Omega, i, w).
    """
    # 1) Semi-major axis from h and e
    a = h**2 / (MU * (1 - e**2))

    # 2) Period and revs per day
    T = orbital_period_from_semi_major_axis(a)
    revs_per_day = 86400 / T

    # 3) Eccentric anomaly from true anomaly
    E = eccentric_anomaly_from_true_anomaly(theta, e)

    # 4) Mean anomaly from eccentric anomaly
    Me = mean_anomaly_from_eccentric_anomaly(E, e)

    return e, revs_per_day, Me, Omega, i, w

def rotation_matrix_from_classical_euler_sequence(Omega, i, w):
    """
    Rotation matrix from perifocal frame to ECI using classical Euler sequence:
    R = R3(Omega) * R1(i) * R3(w)
    """
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
    """
    Rotation matrix from roll-pitch-yaw (phi, theta, psi):
    R = R3(yaw) * R2(pitch) * R1(roll)
    """
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


def angle_wrap_radians(angle):
    """
    Wrap angle in radians to [0, 2π).
    """
    return angle % (2 * np.pi)


def angle_wrap_degrees(angle):
    """
    Wrap angle in degrees to [0, 360).
    """
    return angle % 360.0

def quat_from_axis_angle(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])

def quaternion_from_classical_euler_sequence(Omega, i, w):
    """
    Quaternion equivalent of R = R3(Omega) * R1(i) * R3(w)
    """
    qO = quat_from_axis_angle([0, 0, 1], Omega)
    qi = quat_from_axis_angle([1, 0, 0], i)
    qw = quat_from_axis_angle([0, 0, 1], w)

    # Quaternion multiplication: q_total = qO * qi * qw
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
    """
    Solve Kepler's equation Me = E - e*sin(E) for E using Newton's method.
    Me, e in radians.
    """
    # Initial guess
    if Me < np.pi:
        E = Me + e
    else:
        E = Me - e

    # Newton iterations
    for _ in range(N):
        f  = E - e*np.sin(E) - Me
        fp = 1 - e*np.cos(E)
        E_new = E - f/fp

        # Check convergence
        if abs(E_new - E) < tol:
            return E_new

        E = E_new

    # If not converged, return best estimate
    return E

def sidereal_angle(JD):
    """
    Compute Earth's sidereal angle θ_G (in radians) from Julian Date JD.
    Implements Algorithm 1 from Assignment 2.
    """

    # Step 1: Integer part of JD
    JD_int = np.floor(JD)

    # Step 2: Julian centuries since J2000.0
    T0 = (JD_int - 2451545.0) / 36525.0

    # Step 3: θ_G0 at 00:00 UTC (in degrees)
    theta_G0 = (100.4606184
                + 36000.77005361 * T0
                + 0.00038793 * T0**2
                - 2.6e-8 * T0**3)

    # Step 4: Add Earth's rotation since midnight
    # Fractional part of JD+0.5 gives UTC time since 00:00
    frac = (JD + 0.5) - np.floor(JD + 0.5)
    seconds_since_midnight = frac * 86400.0

    theta_G = theta_G0 + (w_E * seconds_since_midnight) * RTOD  # convert rad→deg

    # Step 5: Wrap to [0, 360)
    theta_G = theta_G % 360.0

    # Return in radians (ECI/ECEF rotations use radians)
    return theta_G * DTOR

def state_from_orbit_params(h, e, theta, Omega, i, w):
    """
    Compute position r and velocity v in ECI frame
    from classical orbital parameters.
    All angles in radians.
    """

    # Step 1: Position and velocity in perifocal frame
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

    # Step 2: Rotation matrix perifocal → ECI
    R = rotation_matrix_from_classical_euler_sequence(Omega, i, w)

    # Step 3: Transform to ECI
    r_eci = R @ r_p
    v_eci = R @ v_p

    return r_eci, v_eci

def state_from_tle_params(e, revs_per_day, Me, Omega, i, w):
    """
    Compute ECI position and velocity from TLE parameters.
    Implements Algorithm 3 from Assignment 2.
    All angles in radians.
    """

    # Step 1: Convert TLE parameters to classical orbital parameters
    h, e, theta, Omega, i, w = orbit_params_from_tle_params(
        e, revs_per_day, Me, Omega, i, w
    )

    # Step 2: Convert classical orbital parameters to ECI state
    r_eci, v_eci = state_from_orbit_params(h, e, theta, Omega, i, w)

    return r_eci, v_eci


def orbit_params_from_state(r, v):
    """
    Compute classical orbital parameters (h, e, theta, Omega, i, w)
    from position r and velocity v in ECI frame.
    """

    # Magnitudes
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    vr = np.dot(r, v) / r_mag

    # Step 2: Angular momentum
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    # Step 3: Inclination
    i = np.arccos(h_vec[2] / h)

    # Step 4: Node vector
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

    # Step 5: Eccentricity vector
    e_vec = (1/MU) * ((v_mag**2 - MU/r_mag)*r - vr*v)
    e = np.linalg.norm(e_vec)

    # Step 6: Argument of perigee
    if N_mag != 0 and e > 1e-12:
        w = np.arccos(np.dot(N, e_vec) / (N_mag * e))
        if e_vec[2] < 0:
            w = 2*np.pi - w
    else:
        w = 0.0  # circular or equatorial orbit

    # Step 7: True anomaly
    if e > 1e-12:
        theta = np.arccos(np.dot(e_vec, r) / (e * r_mag))
        if vr < 0:
            theta = 2*np.pi - theta
    else:
        # circular orbit: true anomaly undefined → use angle from N
        theta = np.arccos(np.dot(N, r) / (N_mag * r_mag))
        if r[2] < 0:
            theta = 2*np.pi - theta

    return h, e, theta, Omega, i, w

def R3(theta):
    """Rotation around Z-axis by angle theta (radians)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])



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
        # Step 3.1: update mean anomaly
        Me = Me + n * dt
        Me = Me % (2 * np.pi)

        # Step 3.2: Kepler + true anomaly
        E = eccentric_anomaly_from_mean_anomaly(Me, e)
        theta = true_anomaly_from_eccentric_anomaly(E, e)

        # Step 3.3: state from orbit params
        r, v = state_from_orbit_params(h, e, theta, Omega, i, w)

        r_list.append(r)
        v_list.append(v)
        t_list.append(t)

        t += dt

    return np.array(t_list), np.array(r_list), np.array(v_list)


def epoch_to_julian_date(epoch):
    """
    Convert TLE epoch (YYDDD.DDDDD) to Julian Date.
    Implements Algorithm 6 from Assignment 2.
    """

    # Step 1: Extract year and day of year
    YY = int(epoch // 1000)
    DDD = epoch - YY * 1000

    # Step 2: Convert YY to full year
    if YY < 57:
        year = 2000 + YY
    else:
        year = 1900 + YY

    # Step 3: Convert day-of-year to month/day/hour/min/sec
    day_int = int(DDD)
    frac_day = DDD - day_int

    # Hours, minutes, seconds
    hours = frac_day * 24
    hour = int(hours)
    minutes = (hours - hour) * 60
    minute = int(minutes)
    seconds = (minutes - minute) * 60
    second = seconds

    # Convert day-of-year to month/day
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

    # Step 4: Convert to Julian Date
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
