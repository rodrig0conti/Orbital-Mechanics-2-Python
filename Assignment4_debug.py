import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

USE_TLE = True
SATELLITE = "HUBBLE"

def get_tle_and_epoch():
    if SATELLITE == "ISS":
        return ol.get_iss_tle_params(), ol.get_iss_epoch()
    elif SATELLITE == "HUBBLE":
        return ol.get_hubble_tle_params(), ol.get_hubble_epoch()
    else:
        raise ValueError("Unknown satellite selected")


class ScenarioAssignment4Debug(sim.BaseScenario):

    def init(self, t):

        # -----------------------------
        # Tierra (idéntico a Assignment 1)
        # -----------------------------
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

        # -----------------------------
        # SATÉLITE (Assignment 4.2)
        # -----------------------------
        q0 = [1, 0, 0, 0]
        w0 = [0, 0, 5]
        J = np.diag([2, 10, 2.5])

        self.sat = sl.Satellite(q0, w0, J)

        # -----------------------------
        # Órbita (idéntico a Assignment 1/2)
        # -----------------------------
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

        self.pos_plot = np.array([[t, self.r_i[0], self.r_i[1], self.r_i[2]]])


    def update(self, t, dt):

        # -----------------------------
        # Órbita
        # -----------------------------
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

        # -----------------------------
        # Dinámica del satélite (Assignment 4.2)
        # -----------------------------
        self.sat.update(t, dt)

        # -----------------------------
        # Rotación de la Tierra
        # -----------------------------
        earth_spin_factor = 1
        self.theta_E += ol.w_E * earth_spin_factor * dt

        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])

        # -----------------------------
        # Log
        # -----------------------------
        self.pos_plot = np.vstack((self.pos_plot,
                                   np.array([t,
                                             self.r_i[0],
                                             self.r_i[1],
                                             self.r_i[2]])))

        # Debug
        if t % 200 == 0:
            q_sat, w_sat = self.sat.get_state()
            print(f"t={t} | q={q_sat.q} | w={w_sat}")


    def get(self):

        # Estado del satélite
        q_sat, w_sat = self.sat.get_state()

        # Convertir a Quaternion de simutils
        q_vis = su.Quaternion(q_sat.q)

        p_sat = self.r_i

        objects = [
            ['satellite', p_sat, q_vis],
            ['body frame', p_sat, q_vis],
            ['sat ref frame', p_sat, su.Quaternion()],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
        ]

        if USE_TLE:
            objects.append(['orbit', self.r_list, su.Quaternion()])

        return objects


    def post_process(self, t, dt):
        su.log_pos('assignment4_debug_position', self.pos_plot)


def main():
    scenario_config = {
        't_0': 0,
        't_e': 6000,
        't_step': 1,
        'speed_factor': 10,
        'anim_dt': 1/25,
        'scale_factor': 2000,
        'visualise': True
    }

    scenario = ScenarioAssignment4Debug()
    sim.create_and_start_simulation(scenario_config, scenario)


if __name__ == "__main__":
    main()
