import numpy as np
import simutils as su
import sat_lib as sl
import simulator as sim

class ScenarioAssignment42(sim.BaseScenario):

    def init(self, t):

        # Estado inicial
        q0 = [1, 0, 0, 0]
        w0 = [0, 0, 0]
        J  = np.diag([0.5, 0.5, 0.5])

        self.sat = sl.Satellite(q0, w0, J, k1=5.0, k2=2.0)

        # Referencia del enunciado
        q_d = [0.5, 0.5, 0.5, 0.5]
        w_d = [0.2, -0.1, 0.05]

        self.sat.set_reference(q_d, w_d)

    def update(self, t, dt):
        self.sat.update(t, dt)

    def get(self):
        q, w = self.sat.get_state()
        q_vis = su.Quaternion(q.q)
        p = np.zeros(3)

        return [
            ['satellite', p, q_vis],
            ['body frame', p, q_vis],
            ['ECI frame', p, su.Quaternion()]
        ]

def main():
    scenario_config = {
        't_0': 0,
        't_e': 500,
        't_step': 0.01,
        'speed_factor': 1,
        'anim_dt': 1/25,
        'scale_factor': 50,
        'visualise': True
    }

    scenario = ScenarioAssignment42()
    sim.create_and_start_simulation(scenario_config, scenario)

if __name__ == "__main__":
    main()

