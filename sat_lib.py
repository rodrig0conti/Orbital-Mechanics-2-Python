import numpy as np
from simutils import Quaternion, step_RK4
import simutils as su


class RigidBody:
    def __init__(self, q0, w0, J):
        """
        q0 : orientación inicial (lista de 4 o Quaternion)
        w0 : velocidad angular inicial (lista de 3)
        J  : matriz de inercia 3x3 (debe ser simétrica y positiva definida)
        """

        # Guardamos orientación como Quaternion
        if isinstance(q0, Quaternion):
            self.q = Quaternion(q0)
        else:
            self.q = Quaternion(q0)

        # Velocidad angular
        self.w = np.array(w0, dtype=float)

        # Matriz de inercia
        self.J = np.array(J, dtype=float)
        self.Jinv = np.linalg.inv(self.J)

        # Torque actual
        self.tau = np.zeros(3)

    def f(self, t, x):
        """
        Implementa f(t, x) para RK4.
        x = [q0, q1, q2, q3, wx, wy, wz]
        """

        # Extraemos cuaternión y velocidad angular
        q = Quaternion(x[:4])
        w = x[4:]

        # Cinemática del cuaternión: q_dot = 0.5 * q ⊗ w_quat
        w_quat = Quaternion([0.0, w[0], w[1], w[2]])
        dq = 0.5 * (q @ w_quat)
        dq = dq.q  # convertimos a array

        # Dinámica del cuerpo rígido: J w_dot = tau - w × (J w)
        Jw = self.J @ w
        dw = self.Jinv @ (self.tau - np.cross(w, Jw))

        return np.hstack((dq, dw))

    def update(self, t, dt, tau_ext):
        """
        Actualiza el estado del cuerpo rígido usando RK4.
        """

        # Guardamos torque externo
        self.tau = np.array(tau_ext, dtype=float)

        # Empaquetamos estado actual
        x = np.hstack((self.q.q, self.w))

        # Integramos con RK4
        x_next = step_RK4(dt, t, x, self.f)

        # Extraemos nueva orientación y velocidad angular
        q_next = Quaternion(x_next[:4])

        # Normalizamos con protección
        mag = q_next.magnitude()
        if mag < 1e-9 or not np.isfinite(mag):
            q_next = Quaternion()  # reset a identidad si explota
        else:
            q_next.normalize()

        self.q = q_next
        self.w = x_next[4:]
class Satellite:
    """
    Satélite con un cuerpo rígido interno y un controlador PD.
    """

    def __init__(self, q0, w0, J, k1=5.0, k2=2.0):
        self.rb = RigidBody(q0, w0, J)
        self.k1 = k1
        self.k2 = k2

        # Referencias (se pueden actualizar desde el escenario)
        self.q_d = Quaternion([1, 0, 0, 0])
        self.w_d = np.zeros(3)

    def set_reference(self, q_d, w_d):
        """Define la referencia deseada."""
        self.q_d = Quaternion(q_d)
        self.w_d = np.array(w_d, dtype=float)

    def compute_control_torque(self):
        """
        τ = -k1 * q_v - k2 * ω_db
        """

        # Error de cuaternión
        q_err = self.q_d.inverted() @ self.rb.q
        q_v = q_err.q[1:]   # parte vectorial

        # Error de velocidad angular
        R = su.quaternion_to_dcm(self.rb.q)
        w_d_body = R.T @ self.w_d
        w_err = self.rb.w - w_d_body

        # Control PD
        tau = -self.k1 * q_v - self.k2 * w_err
        return tau

    def update(self, t, dt):
        tau = self.compute_control_torque()
        self.rb.update(t, dt, tau)

    def get_state(self):
        return self.rb.q, self.rb.w
