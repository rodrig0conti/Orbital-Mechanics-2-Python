import numpy as np
import simutils as su
import simulator as sim
import orbit_lib as ol

# ---------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------

mu   = 398600.4418      # km^3/s^2
R_E  = 6378.1363        # km
m_sat = 8000.0          # kg

z_c      = 1500.0       # km
r_target = R_E + z_c    # km

# Ganancias (ajustables, típicamente 1e-6 a 1e-3)
k1 = 5e-4
k2 = 5e-4


# ---------------------------------------------------------
# FUNCIONES ORBITALES (FORMULAS DEL DOCUMENTO)
# ---------------------------------------------------------

def eccentricity_vector(r, v):
    r_norm = np.linalg.norm(r)
    return (1.0/mu) * np.cross(v, np.cross(r, v)) - r / r_norm


def rp_ra(r, v, e_vec):
    e = np.linalg.norm(e_vec)
    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    ra = h**2 / (mu * (1.0 - e))
    rp = h**2 / (mu * (1.0 + e))

    return rp, ra


def cos_theta(e_vec, r):
    return np.dot(e_vec, r) / (np.linalg.norm(e_vec) * np.linalg.norm(r))


# ---------------------------------------------------------
# CONTROL DE EMPUJE (FORMULA DEL DOCUMENTO)
# ---------------------------------------------------------

def thrust_magnitude(r, v):
    e_vec = eccentricity_vector(r, v)
    rp, ra = rp_ra(r, v, e_vec)
    ct = cos_theta(e_vec, r)

    if ct > 0.9:
        return k1 * (r_target - ra)
    elif ct < -0.9:
        return k2 * (r_target - rp)
    else:
        return 0.0


# ---------------------------------------------------------
# DINÁMICA CON EMPUJE
# ---------------------------------------------------------

def dynamics_with_thrust(t, x):
    r = x[:3]
    v = x[3:]

    r_norm = np.linalg.norm(r)
    a_grav = -mu * r / r_norm**3

    T = thrust_magnitude(r, v)

    if T != 0.0:
        v_norm = np.linalg.norm(v)
        a_thrust = (T / m_sat) * (v / v_norm)
    else:
        a_thrust = np.zeros(3)

    a = a_grav + a_thrust

    if int(t) % 1000 == 0:
        e_vec = eccentricity_vector(r, v)
        rp, ra = rp_ra(r, v, e_vec)
        ct = cos_theta(e_vec, r)
        print(f"[dyn] t={t:6.0f} | cosθ={ct:6.3f} | T={T:10.3e} | rp={rp-R_E:8.1f} km | ra={ra-R_E:8.1f} km")

    return np.hstack((v, a))


# ---------------------------------------------------------
# ESCENARIO PARA EL SIMULADOR (ANIMACIÓN 3D)
# ---------------------------------------------------------

class ScenarioAssignment3(sim.BaseScenario):

    def init(self, t):
        r0 = np.array([7378.0, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 9.0])
        self.x = np.hstack((r0, v0))

        self.pos_plot = np.array([[t, *r0]])
        self.t = t

        self.theta_E = 0.0
        self.q_E = su.Quaternion([
            np.cos(self.theta_E/2.0),
            0.0,
            0.0,
            np.sin(self.theta_E/2.0)
        ])

    def update(self, t, dt):
        self.x = su.step_RK4(dt, t, self.x, dynamics_with_thrust)
        r = self.x[:3]
        v = self.x[3:]

        self.pos_plot = np.vstack((self.pos_plot, np.array([t, *r])))

        self.theta_E += ol.w_E * dt
        self.q_E = su.Quaternion([
            np.cos(self.theta_E/2.0),
            0.0,
            0.0,
            np.sin(self.theta_E/2.0)
        ])

        # Monitor rp y ra
        e_vec = eccentricity_vector(r, v)
        rp, ra = rp_ra(r, v, e_vec)

        if int(t) % 1000 == 0:
            print(f"[mon] t={t:6.0f} | rp={rp-R_E:8.1f} km | ra={ra-R_E:8.1f} km")

        # ---------------------------------------------------------
        # NUEVO: LOG FINAL (aunque no sea múltiplo de 1000)
        # ---------------------------------------------------------
        if t + dt >= 53000:
            print(f"[END] t={t:6.0f} | rp={rp-R_E:8.1f} km | ra={ra-R_E:8.1f} km")

    def get(self):
        r = self.x[:3]
        q_sat = su.Quaternion()
        return [
            ['satellite', r, q_sat],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
        ]

    def post_process(self, t, dt):
        su.log_pos("assignment3_position", self.pos_plot)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    scenario_config = {
        't_0': 0.0,
        't_e': 53000.0,
        't_step': 20.0,
        'speed_factor': 5.0,
        'anim_dt': 1/25,
        'scale_factor': 1000.0,
        'visualise': True
    }

    scenario = ScenarioAssignment3()
    sim.create_and_start_simulation(scenario_config, scenario)


if __name__ == "__main__":
    main()
