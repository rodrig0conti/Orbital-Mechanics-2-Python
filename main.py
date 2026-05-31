"""
=====================================================================
 STE-3605  Assignment 9 :  Hubble Space Telescope Simulation Study
=====================================================================

ONE modular driver for the whole assignment.  Pick what to run with the
MODE variable just below, then run:   python assignment9.py

    MODE = "p1t1"   Part 1 Task 1 : fill Table 1 at the TLE epoch
    MODE = "p1t2"   Part 1 Task 2 : PKepler decay study + 1-orbit comparison
    MODE = "p2t1"   Part 2 Task 1 : PD vs Sliding-Mode (gyro + 1 star tracker)
    MODE = "p2t2"   Part 2 Task 2 : 1 star tracker vs 3 (+ Davenport averaging)
    MODE = "custom" general-purpose run, fully configured by CUSTOM below

Everything uses the classes/functions already in the libraries:
    orbit_lib_9 :  OrbitPKepler, OrbitKeplerSimple, orbit_frame_from_state,
                   state_from_orbit_params, gravity_gradient, eci_to_ecef,
                   ecef_to_geodetic, geodetic_from_xyz, ...
    sat_lib_9   :  RigidBody6DOF, gyro, star_tracker, ADCS_PD, ADCS_SM,
                   Davenport, TRIAD, DavenportMultiST
    simutils_9  :  Quaternion, read_TLE_file
No new physics models are introduced (no drag / J2 beyond what the
libraries already implement).
"""

# =====================================================================
#  CHOOSE WHAT TO RUN
# =====================================================================
MODE = "p2t1"          # "p1t1" | "p1t2" | "p2t1" | "p2t2" | "custom"

# ---------------------------------------------------------------------
#  CUSTOM run configuration (only used when MODE == "custom").
#  The four assignment tasks just call this machinery with preset values.
# ---------------------------------------------------------------------
CUSTOM = {
    # --- which TLE in TLE.txt and which orbit propagator -------------
    "tle_name"        : "HST1",           # exact name of the CURRENT TLE in TLE.txt
    "ground_truth_tle": "HST2",           # old (Assignment-2) TLE for the p1t2 comparison
    "orbit_model"     : "pkepler",        # "pkepler" | "simple"

    # --- initial conditions (Part 2) --------------------------------
    # init_mode chooses how the *orbit* initial state is built; attitude
    # initial state is always (q_ib0, w_b_ib0) below.
    "init_mode"       : "tle",            # "tle" | "orbital" | "rv"
    "orbital"         : None,             # (a,e,i,Omega,w,Me) [rad] if init_mode=="orbital"
    "rv"              : None,             # (r_vec, v_vec) [km, km/s] if init_mode=="rv"
    "q_ib0"           : [1.0, 0.0, 0.0, 0.0],
    "w_b_ib0"         : [0.3e-3, -0.1e-3, 0.2e-3],

    # --- attitude control -------------------------------------------
    "controller"      : "PD",             # "PD" | "SM"
    "target"          : "inertial",       # "inertial" (hold q_id) | "nadir" (orbit frame)
    "q_id"            : [0.5, 0.5, 0.5, 0.5],
    "Kp"              : 1e-4, "Kd": 2e-2,            # PD gains (k1,k2)
    "k1"              : 3e-4, "k": 3e-3, "eps": 0.15,# SM gains (tuned, robust ~2x better than PD)
    "actuator_max"    : 1.13,             # per-axis actuator saturation [Nm], None to disable

    # --- attitude determination (SANDBOX: mix & match sensors) -------
    #  estimator:  "auto"            pick from whatever sensors are enabled
    #              "single_st"       1 star tracker, q_ib taken directly
    #              "davenport_multi" 3 star trackers fused by Davenport q-method
    #              "davenport_vec"   magnetometer + sun sensor(s) -> Davenport
    #              "triad"           magnetometer + sun sensor(s) -> TRIAD
    "estimator"       : "auto",
    "n_star_trackers" : 1,                # 0..3
    "use_magnetometer": False,            # enable the magnetometer
    "n_sun_sensors"   : 0,                # 0..6 (boresights +z,-z,+x,-x,+y,-y)
    "J"               : [[36046, -706, 1491],
                         [-706, 86868,  449],
                         [1491,   449, 93848]],

    # --- disturbances / noise ---------------------------------------
    "gravity_gradient": True,             # gravity-gradient torque (orbit_lib_9)
    "solar_disturb"   : True,             # solar-array disturbance d(t)
    "solar_axis"      : [0.0, 1.0, 0.0],  # body axis the scalar d(t) acts on
    "sensor_noise"    : True,             # False zeroes ALL sensor noise
    # ---- future physics hooks (NOT implemented -- not in the libraries yet):
    #   "drag"        : False,  # atmospheric drag force on the orbit + torque
    #   "j2_orbit"    : False,  # J2 already enters PKepler; a force model would go on the orbit
    #   "maneuvers"   : [],     # list of impulsive/finite burns (transfer maneuvers)
    # See the extension-hook comments in HSTAttitudeScenario.update().

    # --- run / output -----------------------------------------------
    "n_orbits"        : 4,
    "dt"              : 1.0,
    "visualise"       : False,            # True -> runs teacher's simulator.py (needs vispy)
    "viz_orbits"      : 2,                # how many orbits to animate when visualise=True
    "speed_factor"    : 120,              # larger = faster playback (fewer frames per sim-second)
    "show_orbit_line" : True,             # yellow Keplerian-orbit reference line in 3D view
    "plot"            : True,
    # ground-track background image: first path that exists is used. You almost
    # certainly already have "3DModels/earth.jpg" because the simulator needs it.
    "earth_image"     : ["earth_grid.jpg", "3DModels/earth.jpg", "earth.jpg"],
}

# =====================================================================
import os
import numpy as np

import simutils_9 as su          # NOTE: must come before orbit_lib_9 (circular import)
import orbit_lib_9 as ol
import sat_lib_9 as sl
from simutils_9 import Quaternion

MU   = 398600.4418          # km^3/s^2
R_E  = 6378.137             # km  (WGS-84 equatorial radius)
TLE_FILE = "TLE.txt"


# =====================================================================
#  small shared helpers (assignment-described quantities only)
# =====================================================================
def tle_to_kepler(n_revday, e, inc_deg, raan_deg, argp_deg, M_deg,
                  dn_revday2=0.0, ddn_revday3=0.0):
    """TLE mean elements -> SI/rad Kepler set used by the libraries."""
    n   = n_revday   * 2*np.pi / 86400.0
    dn  = dn_revday2 * 2*np.pi / 86400.0**2
    ddn = ddn_revday3* 2*np.pi / 86400.0**3
    a   = (MU / n**2)**(1.0/3.0)
    return (a, e, np.deg2rad(inc_deg), np.deg2rad(raan_deg),
            np.deg2rad(argp_deg), np.deg2rad(M_deg), n, dn, ddn)


def attitude_error_arcsec(q_err):
    """e_theta = (2*180*3600/pi) * arcsin(||q_v||)   (assignment formula)."""
    q = q_err.q if isinstance(q_err, Quaternion) else np.asarray(q_err)
    nv = np.clip(np.linalg.norm(q[1:4]), 0.0, 1.0)
    return (2.0 * 180.0 * 3600.0 / np.pi) * np.arcsin(nv)


def make_orbit(cfg):
    """Build an orbit object from TLE / orbital elements / r,v."""
    if cfg["init_mode"] == "rv":
        r0, v0 = (np.array(cfg["rv"][0], float), np.array(cfg["rv"][1], float))
        h, e, th, Om, i, w = ol.orbit_params_from_state(r0, v0)
        a = h**2 / (MU * (1 - e**2))
        n = np.sqrt(MU / a**3)
        E = ol.eccentric_anomaly_from_true_anomaly(th, e)
        Me = ol.mean_anomaly_from_eccentric_anomaly(E, e)
        elems = (a, e, i, Om, w, Me, n, 0.0, 0.0)
    elif cfg["init_mode"] == "orbital":
        a, e, i, Om, w, Me = cfg["orbital"]
        n = np.sqrt(MU / a**3)
        elems = (a, e, i, Om, w, Me, n, 0.0, 0.0)
    else:  # "tle"
        tle = su.read_TLE_file(TLE_FILE, cfg["tle_name"])[0]
        _, _, n_rev, dn, ddn, e, inc, raan, argp, M, _ = tle
        a, e, i, Om, w, Me, n, dn, ddn = tle_to_kepler(n_rev, e, inc, raan,
                                                       argp, M, dn, ddn)
        elems = (a, e, i, Om, w, Me, n, dn, ddn)

    if cfg["orbit_model"] == "simple":
        return ol.OrbitKeplerSimple(*elems)
    return ol.OrbitPKepler(*elems)


def solar_array_disturbance(t, axis):
    """d(t) = A1 sin(p1 t + phi1) + A2 sin(p2 t + phi2)  (flight-data model)."""
    p1, p2 = 0.14*np.pi, 1.22*np.pi
    A1, A2 = 0.2, 0.2
    f1, f2 = 0.31*np.pi, -0.05*np.pi
    d = A1*np.sin(p1*t + f1) + A2*np.sin(p2*t + f2)
    return d * np.asarray(axis, float)


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-9 else None


def _qbs_for_boresight(d):
    """Quaternion q_bs (sensor->body) that points the sensor +z axis along the
    body direction d.  Used to spread sun sensors over the body for coverage."""
    z = np.array([0.0, 0.0, 1.0]); d = np.asarray(d, float); d = d/np.linalg.norm(d)
    c = float(np.dot(z, d))
    if c >  0.9999:
        return Quaternion()
    if c < -0.9999:
        return Quaternion(np.pi, [1.0, 0.0, 0.0])
    axis = np.cross(z, d); axis = axis/np.linalg.norm(axis)
    return Quaternion(float(np.arccos(c)), list(axis))


def _resolve_estimator(cfg):
    """Decide which attitude-estimation method to use from the enabled sensors."""
    est = cfg.get("estimator", "auto")
    if est != "auto":
        return est
    nst  = int(cfg.get("n_star_trackers", 1))
    nsun = int(cfg.get("n_sun_sensors", 0))
    use_mag = bool(cfg.get("use_magnetometer", False))
    if nst >= 3:
        return "davenport_multi"
    if nst >= 1:
        return "single_st"
    if use_mag or nsun >= 1:
        return "davenport_vec"
    raise ValueError("estimator 'auto' needs at least one star tracker, or a "
                     "magnetometer plus a sun sensor; none are enabled in CONFIG.")


# =====================================================================
#  PART 1  -  TASK 1 :  Table 1 at the epoch
# =====================================================================
def run_p1t1(cfg):
    tle = su.read_TLE_file(TLE_FILE, cfg["tle_name"])[0]
    name, epoch, n_rev, dn, ddn, e, inc, raan, argp, M, bstar = tle
    a, e, i, Om, w, Me, n, dn_r, ddn_r = tle_to_kepler(n_rev, e, inc, raan,
                                                       argp, M, dn, ddn)
    h  = np.sqrt(MU * a * (1 - e**2))
    E  = ol.eccentric_anomaly_from_mean_anomaly(Me, e)
    th = ol.true_anomaly_from_eccentric_anomaly(E, e) % (2*np.pi)
    r_i, v_i = ol.state_from_orbit_params(h, e, th, Om, i, w)

    JD       = ol.epoch_to_julian_date(epoch)
    thetaG0  = ol.sidereal_angle(JD)
    q_io, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)

    r_ecef   = ol.eci_to_ecef(r_i, thetaG0)
    lat, lon, alt = ol.ecef_to_geodetic(r_ecef)

    rows = [
        ("Specific angular momentum h", h,                 "km^2/s"),
        ("True anomaly theta",         th,                 "rad"),
        ("Eccentric anomaly E",        E,                  "rad"),
        ("Semi-major axis a",          a,                  "km"),
        ("Mean motion n",              n,                  "rad/s"),
        ("dn",                         dn_r,               "rad/s^2"),
        ("ddn",                        ddn_r,              "rad/s^3"),
        ("Position r_i",               r_i,                "km"),
        ("Velocity v_i",               v_i,                "km/s"),
        ("Julian Date JD",             JD,                 "days"),
        ("Sidereal angle thetaG0",     thetaG0,            "rad"),
        ("Orbit frame q_io",           np.asarray(q_io),   "-"),
        ("Orbit ang. vel. w_i_io",     w_i_io,             "rad/s"),
        ("Orbit ang. acc. dw_i_io",    dw_i_io,            "rad/s^2"),
        ("Geodetic latitude",          np.rad2deg(lat),    "deg"),
        ("Geo longitude",              np.rad2deg(lon),    "deg"),
        ("Altitude",                   alt,                "km"),
    ]
    print("\n================ Table 1 (HST at epoch) ================")
    lines = []
    for nme, val, unit in rows:
        if np.ndim(val) == 0:
            s = f"{nme:30s} = {val: .10g}  [{unit}]"
        else:
            s = f"{nme:30s} = {np.array2string(np.asarray(val), precision=8)}  [{unit}]"
        print(s); lines.append(s)
    os.makedirs("data", exist_ok=True)
    with open("data/p1t1_table1.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nSaved -> data/p1t1_table1.txt")


# =====================================================================
#  PART 1  -  TASK 2 :  PKepler decay + 1-orbit comparison
# =====================================================================
def _geodetic_track(r_list, t_list):
    """ECI positions -> (lon_deg, lat_deg) ground track using the libraries."""
    lon, lat = [], []
    for r, t in zip(r_list, t_list):
        thetaE = ol.w_E * t
        r_ecef = ol.eci_to_ecef(np.asarray(r, float), thetaE)
        la, lo, _ = ol.ecef_to_geodetic(r_ecef)
        lat.append(np.rad2deg(la)); lon.append(np.rad2deg(lo))
    return np.array(lon), np.array(lat)


def run_p1t2(cfg):
    import matplotlib.pyplot as plt
    os.makedirs("data", exist_ok=True)

    tles = su.read_TLE_file(TLE_FILE, cfg["tle_name"])
    tle  = tles[0]
    name, epoch, n_rev, dn, ddn, e, inc, raan, argp, M, bstar = tle
    a, e, i, Om, w, Me, n, dn_r, ddn_r = tle_to_kepler(n_rev, e, inc, raan,
                                                       argp, M, dn, ddn)

    # ---- (1) 9-year forward propagation (orbital height) -----------
    years = 9.0
    step  = 6*3600.0                       # 6 h steps (good speed/accuracy tradeoff)
    n_steps = int(years*365.25*86400.0/step)
    orb = ol.OrbitPKepler(a, e, i, Om, w, Me, n, dn_r, ddn_r)
    t_days, alt_km = [], []
    for k in range(n_steps+1):
        r, v = orb.get_state()
        t_days.append(k*step/86400.0)
        alt_km.append(np.linalg.norm(r) - R_E)
        orb.propagate(step)
    t_days = np.array(t_days); alt_km = np.array(alt_km)
    np.savetxt("data/p1t2_altitude.txt", np.column_stack((t_days, alt_km)),
               header="t[days]  altitude[km]")
    print(f"Altitude at epoch        : {alt_km[0]:.2f} km")
    print(f"Altitude after {years:.0f} years : {alt_km[-1]:.2f} km")

    plt.figure()
    plt.plot(t_days/365.25, alt_km)
    plt.xlabel("time [years]"); plt.ylabel("altitude [km]")
    plt.title("HST altitude (PKepler, secular ndot only)")
    plt.grid(True); plt.savefig("data/p1t2_altitude.png", dpi=130)

    # ---- (2) one-orbit PKepler vs two-body 'truth' ------------------
    T  = 2*np.pi / n
    dt = 60.0
    t_arr = np.arange(0.0, T, dt)
    h_truth = np.sqrt(MU * a * (1 - e**2))
    orb = ol.OrbitPKepler(a, e, i, Om, w, Me, n, dn_r, ddn_r)
    r_pk, r_tr = [], []
    for t in t_arr:
        r_pk.append(orb.get_state()[0])
        Mt = Me + n*t
        Et = ol.eccentric_anomaly_from_mean_anomaly(Mt % (2*np.pi), e)
        tht = ol.true_anomaly_from_eccentric_anomaly(Et, e)
        r_tr.append(ol.state_from_orbit_params(h_truth, e, tht, Om, i, w)[0])
        orb.propagate(dt)
    r_pk = np.vstack(r_pk); r_tr = np.vstack(r_tr)
    dr = np.linalg.norm(r_pk - r_tr, axis=1)
    np.savetxt("data/p1t2_oneorbit_dr.txt", np.column_stack((t_arr, dr)),
               header="t[s]  ||dr||[km]")

    plt.figure()
    plt.plot(t_arr/60.0, dr)
    plt.xlabel("time [min]"); plt.ylabel("|| r_PKepler - r_truth || [km]")
    plt.title("Position difference over one orbit")
    plt.grid(True); plt.savefig("data/p1t2_oneorbit_dr.png", dpi=130)

    lon_p, lat_p = _geodetic_track(r_pk, t_arr)
    lon_t, lat_t = _geodetic_track(r_tr, t_arr)
    np.savetxt("data/p1t2_gt_pkepler.txt", np.column_stack((lon_p, lat_p)),
               header="lon[deg]  lat[deg]")
    np.savetxt("data/p1t2_gt_truth.txt",   np.column_stack((lon_t, lat_t)),
               header="lon[deg]  lat[deg]")
    fig, ax = plt.subplots()
    # Earth background: try each candidate; first one that exists wins.
    candidates = cfg.get("earth_image", ["earth_grid.jpg",
                                         "3DModels/earth.jpg", "earth.jpg"])
    if isinstance(candidates, str):
        candidates = [candidates]
    used = None
    try:
        from PIL import Image
        for path in candidates:
            if os.path.exists(path):
                with Image.open(path) as im:
                    ax.imshow(im, extent=[-180, 180, -90, 90])
                used = path
                break
    except Exception as ex:
        print(f"[p1t2] could not open background image: {ex}")
    if used:
        print(f"[p1t2] ground-track background: {used}")
    else:
        print(f"[p1t2] no Earth background image found (looked for {candidates}); "
              f"drop one of those files next to assignment9.py to get the map.")
        ax.set_facecolor("#d9e7f2")          # at least a sea-blue background
    ax.plot(lon_p, lat_p, ".", ms=3, label="PKepler")
    ax.plot(lon_t, lat_t, ".", ms=3, label="Truth (two-body)")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.grid(True)
    ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    ax.set_title("Ground track over one orbit"); ax.legend()
    plt.savefig("data/p1t2_groundtrack.png", dpi=130)

    # ---- (3) ground-truth comparison vs an OLD (Assignment 2) TLE ----
    gt_name = cfg.get("ground_truth_tle", "")
    old_list = su.read_TLE_file(TLE_FILE, gt_name) if gt_name else []
    if old_list:
        name_o, epoch_o, n_o, dn_o, ddn_o, e_o, inc_o, raan_o, argp_o, M_o, _ = old_list[0]
        ao, eo, io_, Omo, wo, Meo, no, dno, ddno = tle_to_kepler(
            n_o, e_o, inc_o, raan_o, argp_o, M_o, dn_o, ddn_o)
        # span in *seconds* must come from Julian dates (epoch is YYDDD.DDDD)
        dt_prop = (ol.epoch_to_julian_date(epoch)
                   - ol.epoch_to_julian_date(epoch_o)) * 86400.0
        orb = ol.OrbitPKepler(ao, eo, io_, Omo, wo, Meo, no, dno, ddno)
        big, tt = 6*3600.0, 0.0
        while tt < dt_prop:
            orb.propagate(min(big, dt_prop - tt)); tt += big
        r_prop, v_prop = orb.get_state()

        # ground truth = current TLE state at its own epoch
        r_true, v_true = ol.state_from_orbit_params(h_truth, e, th_from_M(Me, e),
                                                    Om, i, w)
        # propagated orbital elements vs current (the 'truth')
        h_p, e_p, th_p, Om_p, i_p, w_p = ol.orbit_params_from_state(r_prop, v_prop)
        a_p = h_p**2 / (MU*(1 - e_p**2))

        print(f"\n=== Ground-truth comparison: {name_o} -> epoch of {cfg['tle_name']} ===")
        print(f"  propagation span        : {dt_prop/86400.0:.2f} days "
              f"({dt_prop/86400.0/365.25:.2f} yr)")
        print(f"  state-vector error  |dr|: {np.linalg.norm(r_prop-r_true):.2f} km")
        print(f"                      |dv|: {np.linalg.norm(v_prop-v_true):.5f} km/s")
        print(f"  element errors  da      : {a_p - a:+.3f} km")
        print(f"                  de      : {e_p - e:+.6f}")
        print(f"                  di      : {np.rad2deg(i_p - i):+.4f} deg")
        print(f"                  dRAAN   : {np.rad2deg(ol.angle_wrap_radians(Om_p - Om)):+.4f} deg")
        print(f"                  dargp   : {np.rad2deg(ol.angle_wrap_radians(w_p - w)):+.4f} deg")
    else:
        print(f"\n[p1t2] ground-truth TLE '{gt_name}' not found in {TLE_FILE} - "
              f"skipping the propagation-error comparison.")

    print("\nSaved plots/data to data/p1t2_*.{png,txt}")
    if cfg.get("plot", True):
        plt.show()


def th_from_M(M, e):
    E = ol.eccentric_anomaly_from_mean_anomaly(M % (2*np.pi), e)
    return ol.true_anomaly_from_eccentric_anomaly(E, e)


# =====================================================================
#  PART 2  -  attitude scenario (duck-typed for the teacher's simulator)
# =====================================================================
class HSTAttitudeScenario:
    """Inertial-pointing (or nadir) attitude simulation built entirely from
    the library building blocks.  Works headless or inside simulator.py."""

    def __init__(self, cfg):
        self.cfg = cfg

    # -- simulator interface -----------------------------------------
    def init(self, t0):
        cfg = self.cfg
        self.J = np.array(cfg["J"], float)

        # orbit + JD bookkeeping
        self.orbit = make_orbit(cfg)
        try:
            tle = su.read_TLE_file(TLE_FILE, cfg["tle_name"])[0]
            self.JD0 = ol.epoch_to_julian_date(tle[1])
        except Exception:
            self.JD0 = 2451545.0
        sl.set_JD0(self.JD0)

        # rigid-body attitude dynamics (position taken from the orbit)
        r0, v0 = self.orbit.get_state()
        self.body = sl.RigidBody6DOF(r0, v0, 1.0,
                                     Quaternion(cfg["q_ib0"]),
                                     np.array(cfg["w_b_ib0"], float),
                                     self.J)

        # ----- sensors (configurable sandbox) -----------------------
        noise = cfg.get("sensor_noise", True)
        # rate sensor: always present (the controllers need omega_b_ib)
        self.gyro = sl.gyro(q_bs=Quaternion(), p_b=np.zeros(3), mu=0.0,
                            Q=(1e-6 if noise else 0.0), params={"bg": 0.0})

        self.method = _resolve_estimator(cfg)
        want_vec = self.method in ("davenport_vec", "triad")

        # star trackers (0..3)
        self.n_st = int(cfg.get("n_star_trackers", 1))
        Q_st = (1e-2 if noise else 0.0)
        self.star_trackers = [sl.star_tracker(q_bs=Quaternion(), p_b=np.zeros(3),
                                              mu=0.0, Q=Q_st)
                              for _ in range(self.n_st)]

        # magnetometer (the dipole field is ~4.5e-5 T at HST altitude, so the
        # library default Q=0.4e-8 -> sigma 6.3e-5 T drowns the signal; use a
        # small noise so vector determination is actually meaningful)
        self.use_mag = bool(cfg.get("use_magnetometer", False)) or want_vec
        Q_mag = (1e-12 if noise else 0.0)
        self.mag = (sl.magnetometer(q_bs=Quaternion(), p_b=np.zeros(3), mu=0.0,
                                    Q=Q_mag, params={"JD": self.JD0})
                    if self.use_mag else None)

        # sun sensors (0..6), boresights spread over the body for coverage
        n_sun = int(cfg.get("n_sun_sensors", 0))
        if want_vec and n_sun == 0:
            n_sun = 1
        Q_sun = (1e-4 if noise else 0.0)
        bores = [[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]]
        self.sun_sensors, self.sun_qbs = [], []
        for k in range(n_sun):
            qbs = _qbs_for_boresight(bores[k % 6])
            self.sun_sensors.append(
                sl.fine_sun_sensor(qbs, np.zeros(3), 0.0, Q_sun,
                                   params={"alpha": np.pi, "JD": self.JD0}))
            self.sun_qbs.append(qbs)

        # estimator instances (all from the library)
        self.multi = sl.DavenportMultiST()
        self.dav   = sl.Davenport()
        self.triad = sl.TRIAD()
        self.q_ib_last = Quaternion(cfg["q_ib0"])   # held when no estimate available

        # controller
        if cfg["controller"].upper() == "PD":
            self.ctrl = sl.ADCS_PD(Kp=cfg["Kp"], Kd=cfg["Kd"], J=self.J,
                                   attitude_estimator=None,
                                   tau_max=None)             # actuator sat done below
        else:
            self.ctrl = sl.ADCS_SM(k1=cfg["k1"], k=cfg["k"], eps=cfg["eps"],
                                   J=self.J, attitude_estimator=None,
                                   tau_max=None)
        self.ctrl.target = cfg["target"]
        self.ctrl.q_id   = Quaternion(cfg["q_id"]); self.ctrl.q_id.normalize()
        self.ctrl.w_d    = np.zeros(3)

        # logs
        self.t_log, self.e_log, self.tau_log = [], [], []
        self.etrue_log = []
        self.q_log = []

        # sidereal angle of Greenwich at t0 (for animating Earth rotation in 3D)
        self.theta_G = ol.sidereal_angle(self.JD0)

    def update(self, t, dt):
        cfg = self.cfg
        # advance Greenwich sidereal angle (for the animated Earth rotation)
        self.theta_G += ol.w_E * dt
        # orbit position for this step
        self.orbit.propagate(dt)
        r_i, v_i = self.orbit.get_state()
        _, _, q_ib, w_b_ib = self.body.get_state()

        # --- sensors ---
        self.gyro.update(t, dt, q_ib, w_b_ib, r_i, v_i)
        w_meas = self.gyro.output()

        qs = []
        for st in self.star_trackers:
            st.update(t, dt, q_ib, w_b_ib, r_i, v_i)
            qs.append(st.output())

        mag_meas = None
        if self.mag is not None:
            self.mag.update(t, dt, q_ib, w_b_ib, r_i, v_i)
            mag_meas = self.mag.output()             # field in body frame (q_bs=I)

        sun_body = []                                # sun unit vectors in body frame
        for ss, qbs in zip(self.sun_sensors, self.sun_qbs):
            ss.update(t, dt, q_ib, w_b_ib, r_i, v_i)
            u = _unit(ss.output())                   # unit sun in the SENSOR frame
            if u is not None:
                sun_body.append(qbs.rotate(u))       # rotate sensor->body

        # --- attitude estimate (chosen method, library estimators) ---
        q_ib_est = self._estimate(qs, mag_meas, sun_body, r_i, sl.jd_from_t(t))
        self.q_ib_last = q_ib_est

        # orbit frame (needed by the controller signature / nadir option)
        q_io, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)

        # --- controller (we hand it the finished estimate as a quaternion) ---
        self.ctrl.update(t=t, r_i=r_i, v_i=v_i, q_ib=q_ib, w_b_ib=w_b_ib,
                         q_io=Quaternion(q_io), w_i_io=w_i_io, dw_i_io=dw_i_io,
                         gyro_meas=w_meas, mag_meas=np.zeros(3),
                         sun_meas_list=[], JD=sl.jd_from_t(t),
                         star_meas=q_ib_est)
        tau_c = self.ctrl.get_control()

        # --- actuator saturation:  tau_a = 1.13 * sat(tau_c / 1.13) ---
        if cfg["actuator_max"] is not None:
            lim = cfg["actuator_max"]
            tau_a = lim * np.clip(tau_c/lim, -1.0, 1.0)
        else:
            tau_a = tau_c

        # --- environment torques ---
        tau = tau_a.copy()
        if cfg["gravity_gradient"]:
            tau = tau + ol.gravity_gradient(r_i, q_ib, self.J)
        if cfg["solar_disturb"]:
            tau = tau + solar_array_disturbance(t, cfg["solar_axis"])
        # ---- EXTENSION HOOKS (future personal-project additions) -------
        #  Add new *torques* here, e.g.:
        #     if cfg.get("drag"):     tau = tau + aero_torque(r_i, v_i, q_ib, ...)
        #     if cfg.get("j2_torque"):tau = tau + j2_gg_torque(r_i, q_ib, self.J)
        #  Add new *forces* (which change the orbit) by switching self.orbit to a
        #  numerically integrated propagator and applying them there, or by
        #  feeding f_ext into self.body.update() below instead of zeros.
        #  Impulsive transfer maneuvers: apply a delta-v to the orbit state at the
        #  scheduled time (e.g. rebuild self.orbit from r_i, v_i + dv).

        # --- integrate attitude (f_ext = 0; position comes from orbit) ---
        self.body.update(t, dt, np.zeros(3), tau)

        # --- log attitude error ---
        # Primary metric = the assignment's e_theta applied to the controller's
        # error quaternion q_err = q_id^-1 (x) q_ib_est  (the *measured* error,
        # which is what averaging the star trackers improves).  We also keep the
        # TRUE error (q_id vs the actual attitude) for discussion.
        _, _, q_now, _ = self.body.get_state()
        # measure the error against whatever the controller is actually tracking
        if cfg["target"] == "nadir":
            q_des_log = Quaternion(q_io)          # orbital frame
        else:
            q_des_log = self.ctrl.q_id            # fixed inertial attitude
        q_err_meas = q_des_log.inverted() @ q_ib_est
        q_err_true = q_des_log.inverted() @ q_now
        self.t_log.append(t)
        self.e_log.append(attitude_error_arcsec(q_err_meas))
        self.etrue_log.append(attitude_error_arcsec(q_err_true))
        self.tau_log.append(tau_a.copy())
        self.q_log.append(q_now.q.copy())

        # Cache the *orbit* position (and current attitude) so get() can serve
        # them to the simulator.  IMPORTANT: do NOT read position from the
        # rigid body - we integrate the body with f_ext=0 (the orbit is
        # propagated separately by OrbitPKepler), so the body's position
        # drifts in a straight line.
        self.r_i_cache = r_i
        self.v_i_cache = v_i
        self.q_ib_cache = Quaternion(q_now.q)

    def _estimate(self, qs, mag_meas, sun_body, r_i, JD):
        """Return q_ib estimate using the configured method and the library
        estimators.  Conventions (verified): a single star tracker and TRIAD
        return q_ib directly; Davenport and DavenportMultiST return q_bi, so
        they are inverted."""
        m = self.method
        if m == "single_st":
            return qs[0]
        if m == "davenport_multi":
            return self.multi.estimate_attitude(qs[:3]).inverted()

        # ---- vector estimators: magnetometer + sun (TRIAD / Davenport) ----
        B_i = ol.magnetic_field_dipole(r_i, JD)
        s_i = ol.sun_vector(JD)
        M_B, M_A = [], []
        bb, Bi = _unit(mag_meas), _unit(B_i)
        if bb is not None and Bi is not None:
            M_B.append(bb); M_A.append(Bi)
        if sun_body:
            sb = _unit(np.mean(np.vstack(sun_body), axis=0))   # average valid readings
            si = _unit(s_i)
            if sb is not None and si is not None:
                M_B.append(sb); M_A.append(si)
        if len(M_B) < 2:
            return self.q_ib_last            # e.g. sun not in view -> hold last
        if m == "triad":
            return self.triad.estimate_attitude(M_B, M_A)        # returns q_ib
        return self.dav.estimate_attitude(M_B, M_A).inverted()   # q_bi -> q_ib

    def get(self):
        # data shape the teacher's simulator expects:
        # [[name, position, quaternion], ...] for keys
        # 'satellite', 'earth', 'body frame', 'ECEF frame', 'ECI frame'
        # NOTE: position comes from the orbit cache, NOT from the rigid body
        # (which has f_ext=0 and would drift in a straight line).
        r_i  = getattr(self, "r_i_cache",  None)
        q_ib = getattr(self, "q_ib_cache", None)
        if r_i is None:                                # first call before any update
            r_i, _ = self.orbit.get_state()
            q_ib   = Quaternion(self.cfg["q_ib0"])
        q_E = Quaternion(self.theta_G, [0.0, 0.0, 1.0])    # Earth rotation about z
        return [
            ["satellite",  r_i,           q_ib],
            ["body frame", r_i,           q_ib],
            ["earth",      np.zeros(3),   q_E],
            ["ECEF frame", np.zeros(3),   q_E],
            ["ECI frame",  np.zeros(3),   Quaternion()],
        ]

    def post_process(self, t, dt):
        os.makedirs("data", exist_ok=True)
        tag = f"{self.cfg['controller'].upper()}_{self.n_st}ST"
        t_arr  = np.array(self.t_log)
        e_arr  = np.array(self.e_log)
        etrue  = np.array(self.etrue_log)
        tau_arr = np.vstack(self.tau_log)
        np.savetxt(f"data/p2_{tag}_error.txt",
                   np.column_stack((t_arr, e_arr, etrue)),
                   header="t[s]  measured_err[arcsec]  true_err[arcsec]")
        np.savetxt(f"data/p2_{tag}_torque.txt",
                   np.column_stack((t_arr, tau_arr)),
                   header="t[s]  taux  tauy  tauz [Nm]")
        # summary
        T = 2*np.pi/np.sqrt(MU/((np.linalg.norm(self.body.get_state()[0]))**3))
        last = t_arr >= (t_arr[-1] - T) if len(t_arr) else np.array([], bool)
        print(f"[{tag}] measured: max {e_arr.max():.0f} | mean(last orbit) "
              f"{e_arr[last].mean():.0f} arcsec   ||   true pointing: "
              f"mean(last orbit) {etrue[last].mean():.0f} arcsec "
              f"({etrue[last].mean()/3600:.2f} deg)")
        if self.cfg.get("plot", True):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(t_arr/60.0, e_arr, lw=0.7, label="measured (q_id vs estimate)")
            plt.plot(t_arr/60.0, etrue, lw=1.2, label="true (q_id vs actual)")
            plt.xlabel("time [min]"); plt.ylabel("attitude error [arcsec]")
            plt.title(f"Attitude error - {tag}"); plt.grid(True)
            plt.yscale("log"); plt.legend()
            plt.savefig(f"data/p2_{tag}_error.png", dpi=130)


def run_part2(cfg):
    """Run one or more attitude scenarios and compare them on one figure."""
    # ------------------------------------------------------------------
    # Visualisation path: animate ONE scenario in the teacher's simulator.
    # (We don't try to open two simultaneous 3D windows for a comparison;
    # for the comparison plots set visualise=False.)
    # ------------------------------------------------------------------
    if cfg.get("visualise"):
        viz_cfg = dict(cfg)
        # For p2t2 we want to animate the 3-ST case; for p2t1 the SMC case.
        if cfg.get("_compare") == "1_vs_3_st":
            viz_cfg["n_star_trackers"] = 3
            viz_cfg["controller"]      = "SM"
        elif cfg.get("_compare") == "pd_vs_sm":
            viz_cfg["controller"]      = "SM"
        viz_cfg["n_orbits"] = cfg.get("viz_orbits", cfg["n_orbits"])
        print(f"[viz] animating {viz_cfg['controller']}  "
              f"{viz_cfg['n_star_trackers']} ST  "
              f"for {viz_cfg['n_orbits']} orbits  "
              f"(speed_factor={viz_cfg.get('speed_factor')})")
        _run_with_simulator(viz_cfg)
        return

    # ------------------------------------------------------------------
    # Headless path: run the comparison scenarios, save data + comparison plot.
    # ------------------------------------------------------------------
    if cfg.get("_compare") == "pd_vs_sm":
        runs = [dict(cfg, controller="PD"), dict(cfg, controller="SM")]
    elif cfg.get("_compare") == "1_vs_3_st":
        runs = [dict(cfg, n_star_trackers=1), dict(cfg, n_star_trackers=3)]
    else:
        runs = [cfg]

    results = []
    for rc in runs:
        sc = HSTAttitudeScenario(rc)
        sc.init(0.0)
        T = 2*np.pi/np.sqrt(MU/(np.linalg.norm(sc.orbit.get_state()[0])**3))
        t_end = rc["n_orbits"]*T
        t, dt = 0.0, rc["dt"]
        while t < t_end:
            sc.update(t, dt); t += dt
        sc.post_process(t, dt)
        label = f"{rc['controller'].upper()} ({sc.n_st} ST)"
        results.append((label, np.array(sc.t_log),
                        np.array(sc.e_log), np.array(sc.etrue_log)))

    if results and cfg.get("plot", True):
        import matplotlib.pyplot as plt
        # PD-vs-SM is a controller comparison -> show TRUE pointing error;
        # 1-vs-3-ST is an estimator comparison -> show MEASURED error.
        use_true = (cfg.get("_compare") != "1_vs_3_st")
        which = "true pointing" if use_true else "measured (estimate)"
        plt.figure()
        for label, tarr, emeas, etrue in results:
            plt.plot(tarr/60.0, etrue if use_true else emeas, lw=0.9, label=label)
        plt.xlabel("time [min]")
        plt.ylabel(f"attitude error [arcsec] - {which}")
        plt.yscale("log"); plt.grid(True); plt.legend()
        plt.title("Attitude error comparison")
        plt.savefig("data/p2_comparison.png", dpi=130)
        plt.show()


def _run_with_simulator(cfg):
    """Use the teacher's simulator.py UNMODIFIED for 3D visualisation.
    Teacher's simulator.py does `import simutils / orbit_lib / sat_lib`
    without the _9 suffix, so we register the _9 modules under those names
    in sys.modules BEFORE importing it - the simulator file is not edited."""
    import sys
    for nm, mod in (("simutils", su), ("orbit_lib", ol), ("sat_lib", sl)):
        sys.modules.setdefault(nm, mod)
    import simulator as sim

    sc = HSTAttitudeScenario(cfg)
    # compute orbital period from the *initial* state (we don't reuse the
    # orbit object so the scenario starts fresh inside the simulator thread)
    T_orb = 2*np.pi/np.sqrt(MU/(np.linalg.norm(make_orbit(cfg).get_state()[0])**3))
    scale_factor = 1000
    sim_config = {
        "t_0":          0.0,
        "t_e":          cfg["n_orbits"] * T_orb,
        "t_step":       cfg["dt"],
        "speed_factor": cfg.get("speed_factor", 120),
        "anim_dt":      1.0/25.0,
        "scale_factor": scale_factor,     # same as teacher's example
        "visualise":    True,
    }

    # -- optional: yellow Keplerian-orbit reference line --------------
    # We subclass the teacher's SimCanvas at runtime to add a vispy Line
    # visual with the orbit points in ECI (so it does NOT rotate with the
    # Earth).  simulator.py itself is not modified.
    if cfg.get("show_orbit_line", True):
        try:
            from vispy.scene.visuals import Line
            from vispy.scene import MatrixTransform as Mat4
            # sample one full period
            n_pts = 720
            orb_pts = make_orbit(cfg)
            pts = []
            for _ in range(n_pts + 1):
                pts.append(orb_pts.get_state()[0])
                orb_pts.propagate(T_orb / n_pts)
            pts = np.vstack(pts) / scale_factor          # match canvas scaling
            _OriginalCanvas = sim.SimCanvas
            class _CanvasWithOrbit(_OriginalCanvas):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.unfreeze()
                    self.orbit_line = Line(pts, color=(1.0, 0.9, 0.0, 0.9),
                                           method='gl', width=2,
                                           parent=self.view.scene)
                    self.orbit_line.transform = Mat4()    # identity = stay in ECI
                    self.freeze()
            sim.SimCanvas = _CanvasWithOrbit
        except Exception as ex:
            print(f"[viz] could not add orbit-reference line ({ex}); continuing without it.")

    sim.create_and_start_simulation(sim_config, sc)


# =====================================================================
#  PRESETS for the four assignment tasks
# =====================================================================
def preset(mode):
    cfg = dict(CUSTOM)
    if mode == "p2t1":
        cfg.update(target="inertial", n_star_trackers=1, _compare="pd_vs_sm")
    elif mode == "p2t2":
        cfg.update(target="inertial", controller="SM", _compare="1_vs_3_st")
    return cfg


# =====================================================================
def main():
    if   MODE == "p1t1":  run_p1t1(CUSTOM)
    elif MODE == "p1t2":  run_p1t2(CUSTOM)
    elif MODE == "p2t1":  run_part2(preset("p2t1"))
    elif MODE == "p2t2":  run_part2(preset("p2t2"))
    elif MODE == "custom": run_part2(CUSTOM)
    else:
        raise ValueError(f"unknown MODE {MODE!r}")


if __name__ == "__main__":
    main()
