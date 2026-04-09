import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

USE_TLE = True  # False = órbita circular (Assignment 1), True = TLE (Assignment 2)
SATELLITE = "HUBBLE"   # options: "ISS", "HUBBLE"

def get_tle_and_epoch():
    if SATELLITE == "ISS":
        return ol.get_iss_tle_params(), ol.get_iss_epoch()
    elif SATELLITE == "HUBBLE":
        return ol.get_hubble_tle_params(), ol.get_hubble_epoch()
    else:
        raise ValueError("Unknown satellite selected")


class ScenarioAssignment4(sim.BaseScenario):

    def init(self, t):

        if USE_TLE:
            (e, revs_per_day, Me, Omega, i, w), epoch = get_tle_and_epoch()

            JD = ol.epoch_to_julian_date(epoch)
            theta0 = ol.sidereal_angle(JD)
            self.theta_E = theta0

        else:
            self.theta_E = 0.0

        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])

        # --- ACTITUD: ahora viene del RigidBody ---
        q0 = [1, 0, 0, 0]
        w0 = [0, 0, 5]
        J = np.array([
            [2,   1,   0],
            [1,  10, 0.1],
            [0, 0.1, 2.5]
        ])
        self.tau = np.array([0.0, 0.0, 0.0])
        self.rb = sl.RigidBody(q0, w0, J)

        if USE_TLE:
            r0, v0 = ol.state_from_tle_params(e, revs_per_day, Me, Omega, i, w)

            self.times, self.r_list, self.v_list = ol.propagate_orbit_dt(
                r0, v0, 0, 6000, 1
            )
            self.step_index = 0
            self.r_i = self.r_list[0]

            print("Epoch:", epoch)
            print("JD:", JD)
            print("theta0 (deg):", np.rad2deg(theta0))

        else:
            self.r = ol.R_E + 400
            self.T = 2 * np.pi * np.sqrt(self.r**3 / ol.mu)
            self.omega = 2 * np.pi / self.T
            self.theta = 0.0

            self.r_i = np.array([
                self.r*np.cos(self.theta),
                self.r*np.sin(self.theta),
                0
            ])

        self.pos_plot = np.array([[t,
                                   self.r_i[0],
                                   self.r_i[1],
                                   self.r_i[2]]])


    def update(self, t, dt):

        # 1. Posición del satélite (idéntico al Assignment 1)
        if USE_TLE:
            if self.step_index < len(self.r_list):
                self.r_i = self.r_list[self.step_index]
                self.step_index += 1
        else:
            self.theta += self.omega * dt
            self.r_i = np.array([
                self.r*np.cos(self.theta),
                self.r*np.sin(self.theta),
                0
            ])

        # 2. Actitud del satélite: integrar rigid-body SIEMPRE
        self.rb.update(t, dt, self.tau)

        # 3. Rotación de la Tierra (idéntico)
        earth_spin_factor = 1
        self.theta_E += ol.w_E * earth_spin_factor * dt

        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])

        # 4. Log de trayectoria
        self.pos_plot = np.vstack((self.pos_plot,
                                   np.array([t,
                                             self.r_i[0],
                                             self.r_i[1],
                                             self.r_i[2]])))


    def get(self):
        # Cuaternión del rigid-body convertido a simutils.Quaternion
        q_sat = su.Quaternion(self.rb.q.q)

        objects = [
            ['satellite', self.r_i, q_sat],
            ['body frame', self.r_i, q_sat],
            ['sat ref frame', self.r_i, su.Quaternion()],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
        ]

        if USE_TLE:
            objects.append(['orbit', self.r_list, su.Quaternion()])

        return objects


    def post_process(self, t, dt):
        su.log_pos('assignment4_rigidbody_position', self.pos_plot)


def main():
    scenario_config = {
        't_0': 0,
        't_e': 6000,        # 1 órbita
        't_step': 1,        # 1 segundo de simulación por paso
        'speed_factor': 10,
        'anim_dt': 1/25,
        'scale_factor': 2000,
        'visualise': True
    }

    scenario = ScenarioAssignment4()
    sim.create_and_start_simulation(scenario_config, scenario)

    import subprocess
    import sys
    print("Simulation finished. Plotting latest data...")
    subprocess.run([sys.executable, "plotter.py"])


if __name__ == "__main__":
    main()
