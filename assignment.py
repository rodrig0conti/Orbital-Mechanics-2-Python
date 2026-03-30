import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

USE_TLE = True  # False = órbita circular (Assignment 1), True = TLE (Assignment 2)
# Choose which satellite to simulate
SATELLITE = "HUBBLE"   # options: "ISS", "HUBBLE"

def get_tle_and_epoch():
    if SATELLITE == "ISS":
        return ol.get_iss_tle_params(), ol.get_iss_epoch()
    elif SATELLITE == "HUBBLE":
        return ol.get_hubble_tle_params(), ol.get_hubble_epoch()
    else:
        raise ValueError("Unknown satellite selected")

class ScenarioAssignment1(sim.BaseScenario):

    def init(self, t):

        if USE_TLE:
            # Obtener parámetros TLE y epoch del satélite seleccionado
            (e, revs_per_day, Me, Omega, i, w), epoch = get_tle_and_epoch()

            # Calcular ángulo sideral inicial de la Tierra
            JD = ol.epoch_to_julian_date(epoch)
            theta0 = ol.sidereal_angle(JD)
            self.theta_E = theta0

        else:
            # Modo órbita circular (Assignment 1)
            self.theta_E = 0.0

        # Cuaternión de la Tierra a partir de theta_E
        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])

        # Orientación inicial del satélite
        self.q = su.Quaternion()

        if USE_TLE:
            # Estado inicial del satélite desde TLE
            r0, v0 = ol.state_from_tle_params(e, revs_per_day, Me, Omega, i, w)

            # Propagación de la órbita
            self.times, self.r_list, self.v_list = ol.propagate_orbit_dt(
                r0, v0, 0, 6000, 1
            )
            self.step_index = 0
            self.r_i = self.r_list[0]

            # Debug opcional
            print("Epoch:", epoch)
            print("JD:", JD)
            print("theta0 (deg):", np.rad2deg(theta0))

        else:
            # --- MODO ASSIGNMENT 1: órbita circular simple ---
            self.r = ol.R_E + 400
            self.T = 2 * np.pi * np.sqrt(self.r**3 / ol.mu)
            self.omega = 2 * np.pi / self.T
            self.theta = 0.0

            self.r_i = np.array([
                self.r*np.cos(self.theta),
                self.r*np.sin(self.theta),
                0
            ])

        # Inicializar log de posiciones
        self.pos_plot = np.array([[t,
                                self.r_i[0],
                                self.r_i[1],
                                self.r_i[2]]])



    def update(self, t, dt):

        # -------------------------
        # 1. Actualizar posición del satélite
        # -------------------------
        if USE_TLE:
          

            # Usar la órbita propagada desde TLE
            if self.step_index < len(self.r_list):
                self.r_i = self.r_list[self.step_index]
                self.step_index += 1
        else:
            # Órbita circular clásica
            self.theta += self.omega * dt
            self.r_i = np.array([
                self.r*np.cos(self.theta),
                self.r*np.sin(self.theta),
                0
            ])

        # -------------------------
        # 2. Actualizar orientación del satélite
        # -------------------------
        if not USE_TLE:
            # Solo rotamos el satélite si estamos en órbita circular
            omega_vec = np.array([0, 0, self.omega])
            dq = 0.5 * (self.q * su.Quaternion([0, *omega_vec]))
            self.q = su.Quaternion(self.q.q + dt * dq.q)
            self.q.normalize()

        # -------------------------
        # 3. Debug (solo para órbita circular)
        # -------------------------
        if t < 10 and not USE_TLE:
            print(f"t={t:.1f}, theta={self.theta:.6f}, r_i={self.r_i}")

        # -------------------------
        # 4. Log de trayectoria
        self.pos_plot = np.vstack((self.pos_plot,
                               np.array([t,
                                         self.r_i[0],
                                         self.r_i[1],
                                         self.r_i[2]])))

        # -------------------------
        # 5. Rotación de la Tierra (ECEF)
        # -------------------------
        earth_spin_factor = 1
        self.theta_E += ol.w_E * earth_spin_factor * dt

        self.q_E = su.Quaternion([
            np.cos(self.theta_E / 2),
            0,
            0,
            np.sin(self.theta_E / 2)
        ])

        # -------------------------
        # 6. Mensaje de órbita completa (solo modo circular)
        # -------------------------
        if not USE_TLE and abs(self.theta - 2*np.pi) < 0.01:
            print("Órbita completa en t =", t)

        # -------------------------
        # 7. Debug de la Tierra
        # -------------------------
        if t % 1000 < dt:
            angle = 2 * np.arccos(self.q_E.q[0])
            print(f"Earth angle ≈ {np.rad2deg(angle):.6f} deg")



    def get(self):
        objects = [
            ['satellite', self.r_i, self.q],
            ['body frame', self.r_i, self.q],
            ['sat ref frame', self.r_i, su.Quaternion()],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
        ]

         # Añadir órbita completa si estamos en modo TLE
        if USE_TLE:
            objects.append(['orbit', self.r_list, su.Quaternion()])

        return objects



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
