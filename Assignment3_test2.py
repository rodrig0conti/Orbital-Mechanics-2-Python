import numpy as np
import simutils as su
import simulator as sim
import orbit_lib as ol

# Earth parameters
mu = 398600.4418      # km^3/s^2
R_E = 6378.1363       # km
m_sat = 8000          # kg
r_target = R_E + 1500 # km (desired orbit)

# Gains (to be tuned)
k1 = 1e-5
k2 = 1e-5


# ---------------------------------------------------------
# ORBITAL ELEMENTS (AS DEFINED IN THE ASSIGNMENT)
# ---------------------------------------------------------

def eccentricity_vector(r, v):
    r_norm = np.linalg.norm(r)
    return (1/mu) * (np.cross(v, np.cross(r, v))) - (r / r_norm)


def rp_ra(r, v, e_vec):
    e = np.linalg.norm(e_vec)

    h_vec = np.cross(r, v)
    h = np.linalg.norm(h_vec)

    ra = h**2 / (mu * (1 - e))
    rp = h**2 / (mu * (1 + e))

    return rp, ra


def cos_theta(e_vec, v):
    return np.dot(e_vec, v) / (np.linalg.norm(e_vec) * np.linalg.norm(v))


# ---------------------------------------------------------
# THRUST CONTROLLER (EXACTLY AS IN THE ASSIGNMENT)
# ---------------------------------------------------------

def thrust_control(r, v):
    e_vec = eccentricity_vector(r, v)
    rp, ra = rp_ra(r, v, e_vec)
    ct = cos_theta(e_vec, v)

    if ct > 0.9:
        return k1 * (r_target - ra)
    elif ct < -0.9:
        return k2 * (r_target - rp)
    else:
        return 0.0


# ---------------------------------------------------------
# DYNAMICS WITH THRUST (THRUST ALIGNED WITH ACCELERATION)
# ---------------------------------------------------------

def dynamics_with_thrust(t, x):
    r = x[:3]
    v = x[3:]

    r_norm = np.linalg.norm(r)

    # Gravity
    a_grav = -mu * r / r_norm**3

    # Compute thrust magnitude
    T = thrust_control(r, v)

    # Thrust direction = direction of total acceleration (assignment definition)
    a_total = a_grav

    if T != 0:
        a_thrust = (T / m_sat) * (v / np.linalg.norm(v))

    else:
        a_thrust = np.zeros(3)

    a = a_grav + a_thrust

    # Debug print
    if int(t) % 1000 == 0:
        e_vec = eccentricity_vector(r, v)
        rp, ra = rp_ra(r, v, e_vec)
        ct = cos_theta(e_vec, v)
        print(f"[dyn] t={t:6.0f} | cosθ={ct:6.3f} | T={T:10.3e} | rp={rp-R_E:8.1f} km | ra={ra-R_E:8.1f} km")

    return np.hstack((v, a))


# ---------------------------------------------------------
# SCENARIO FOR SIMULATOR
# ---------------------------------------------------------

class ScenarioAssignment3(sim.BaseScenario):

    def init(self, t):
        # Initial orbit from assignment
        r0 = np.array([7378, 0, 0])
        v0 = np.array([0, 0, 9])
        self.x = np.hstack((r0, v0))

        self.pos_plot = np.array([[t, *r0]])
        self.t = t

        # Earth rotation
        self.theta_E = 0.0
        self.q_E = su.Quaternion([
            np.cos(self.theta_E/2),
            0,
            0,
            np.sin(self.theta_E/2)
        ])

    def update(self, t, dt):
        self.x = su.step_RK4(dt, t, self.x, dynamics_with_thrust)
        r = self.x[:3]
        v = self.x[3:]

        self.pos_plot = np.vstack((self.pos_plot, np.array([t, *r])))

        # Earth rotation
        self.theta_E += ol.w_E * dt
        self.q_E = su.Quaternion([
            np.cos(self.theta_E/2),
            0,
            0,
            np.sin(self.theta_E/2)
        ])

        # Monitor rp and ra
        e_vec = eccentricity_vector(r, v)
        rp, ra = rp_ra(r, v, e_vec)
        if int(t) % 1000 == 0:
            print(f"[mon] t={t:6.0f} | rp={rp-R_E:8.1f} km | ra={ra-R_E:8.1f} km")

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
        't_0': 0,
        't_e': 53000,
        't_step': 20,
        'speed_factor': 5,
        'anim_dt': 1/25,
        'scale_factor': 1000,
        'visualise': True
    }

    scenario = ScenarioAssignment3()
    sim.create_and_start_simulation(scenario_config, scenario)


if __name__ == "__main__":
    main()
