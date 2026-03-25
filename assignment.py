import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim


class ScenarioAssignment1(sim.BaseScenario):

    def init(self, t):
        print("R_E =", ol.R_E)
        print("mu  =", ol.mu)
        self.theta_E = 0.0
        self.r = ol.R_E + 400
        self.T = 2 * np.pi * np.sqrt(self.r**3 / ol.mu)
        self.omega = 2 * np.pi / self.T

        print("r   =", self.r)
        print("T   =", self.T)
        print("omega =", self.omega)
        print("w_E =", ol.w_E)

        self.theta_E = 0.0  # ángulo de la Tierra (rad)


        # (Earth + 400 km)
        self.r = ol.R_E + 400 #run it in kilometers

        # Orbital period
        self.T = 2 * np.pi * np.sqrt(self.r**3 / ol.mu)


        # Angular velocity omega (radians per second)
        self.omega = 2 * np.pi / self.T

        # Initial angle
        self.theta = 0.0

        # Initial orientation (identity quaternion)
        self.q = su.Quaternion()
        angle = np.deg2rad(0) #ángulo inicial de la tierra
        self.q_E = su.Quaternion([np.cos(angle/2), 0, 0, np.sin(angle/2)])  # rotación 30° alrededor de z


        # Initial position in ECI (kilometers)
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
        # Update angle
        self.theta += self.omega * dt

        # Update quaternion (simple rotation around z)
        omega_vec = np.array([0, 0, self.omega])
        dq = 0.5 * (self.q * su.Quaternion([0, *omega_vec]))
        self.q = su.Quaternion(self.q.q + dt * dq.q)
        self.q.normalize()

        # Update satellite position in ECI (kilometers)
        self.r_i = np.array([
            self.r*np.cos(self.theta),
            self.r*np.sin(self.theta),
            0
        ])

        # debug to understand problem: print first 10 steps
        if t < 10:
            print(f"t={t:.1f}, theta={self.theta:.6f}, r_i={self.r_i}")

        # Log trajectory (in km)
        self.pos_plot = np.vstack((self.pos_plot,
                                   np.array([t,
                                             self.r_i[0],
                                             self.r_i[1],
                                             self.r_i[2]])))
        # Rotación de la Tierra (ECEF)
        earth_spin_factor = 1  # para que se note; luego lo bajas si quieres
        self.theta_E += ol.w_E * earth_spin_factor * dt

        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])
        if abs(self.theta - 2*np.pi) < 0.01:
            print("Órbita completa en t =", t)


        

        if t % 1000 < dt:  # cada 1000 s simulados aprox
        # ángulo alrededor de z a partir del cuaternión (aprox)
            angle = 2 * np.arccos(self.q_E.q[0])
            print(f"Earth angle ≈ {np.rad2deg(angle):.6f} deg")




    def get(self):
        return [
        ['satellite', self.r_i, self.q],
        ['body frame', self.r_i, self.q],
        ['sat ref frame', self.r_i, su.Quaternion()],  # ejes ECI en el satélite
        ['earth', np.zeros(3), self.q_E],
        ['ECEF frame', np.zeros(3), self.q_E],
        ['ECI frame', np.zeros(3), su.Quaternion()],
        ]


    def post_process(self, t, dt):
        su.log_pos('assignment1_position', self.pos_plot)


def main():
    scenario_config = {
        't_0': 0,
        't_e': 6000,        # 1 órbita
        't_step': 1,        # 1 segundo de simulación por paso
        'speed_factor': 10,  # 1 segundo real = 1 segundo simulado
        'anim_dt': 1/25,    # 25 FPS para no saturar GPU
        'scale_factor': 2000,
        'visualise': True
    }

    scenario = ScenarioAssignment1()
    sim.create_and_start_simulation(scenario_config, scenario)
    import subprocess
    import sys

    # After simulation ends, automatically plot the latest file
    print("Simulation finished. Plotting latest data...")

    subprocess.run([sys.executable, "plotter.py"])

    



if __name__ == "__main__":
    main()
