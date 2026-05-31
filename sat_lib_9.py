from math import tau
import numpy as np
from simutils_9 import Quaternion, step_RK4
import simutils_9 as su
import orbit_lib_9 as ol
from orbit_lib_9 import magnetic_field_dipole, sun_vector
JD0 = None

def set_JD0(JD):
    global JD0
    JD0 = JD

def jd_from_t(t):
    if JD0 is None:
        raise ValueError("JD0 not initialized. Call set_JD0(epoch_JD).")
    return JD0 + t / 86400.0
   
class RigidBody6DOF:
    def __init__(self, p0, v0, m, q0, w0, J):
        self.p = np.array(p0, float)
        self.v = np.array(v0, float)
        self.m = float(m)

        self.q = Quaternion(q0)
        self.q.normalize()

        self.w = np.array(w0, float)

        self.J = np.array(J, float)
        self.Jinv = np.linalg.inv(self.J)

        self.f = np.zeros(3)
        self.tau = np.zeros(3)

        self.x = np.hstack((self.p, self.v, self.q.q, self.w))

    def f_state(self, t, x):
        p = x[0:3]
        v = x[3:6]
        q = Quaternion(x[6:10])
        q.normalize()
        w = x[10:13]

        dp = v
        dv = self.f / self.m

        wq = Quaternion([0, *w])
        dq = 0.5 * (q @ wq).q

        Jw = self.J @ w
        dw = self.Jinv @ (self.tau - np.cross(w, Jw))

        return np.hstack((dp, dv, dq, dw))

    def update(self, t, dt, f_ext, tau_ext):
        self.f = np.array(f_ext)
        self.tau = np.array(tau_ext)

        self.x = step_RK4(dt, t, self.x, self.f_state)

        self.p = self.x[0:3]
        self.v = self.x[3:6]

        q_next = Quaternion(self.x[6:10])
        q_next.normalize()
        self.q = q_next
        self.x[6:10] = self.q.q

        self.w = self.x[10:13]

    def get_state(self):
        return self.p, self.v, self.q, self.w


class ADCS_PD:
    def __init__(self, Kp, Kd, J, attitude_estimator, tau_max=None):
        self.Kp = Kp
        self.Kd = Kd
        self.J  = J
        self.estimator = attitude_estimator
        self.tau_max = tau_max          # per-axis controller-command limit (None = no clip)
        self.tau = np.zeros(3)

        # desired inertial attitude (can be changed by scenario)
        self.q_id = Quaternion([1.0, 0.0, 0.0, 0.0])
        self.w_d  = np.zeros(3)
        self.target = "nadir"   # "nadir"=track orbital frame; "inertial"=hold q_id

        self.debug = {}

        self.q_ib_prev = None

    def _safe_unit(self, v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        return v / n

    def update(self,
               t,
               r_i, v_i,
               q_ib, w_b_ib,
               q_io, w_i_io, dw_i_io,
               gyro_meas, mag_meas, sun_meas_list,
               JD,
               star_meas=None):

        if dw_i_io is None:
            dw_i_io = np.zeros(3)
        if not isinstance(q_io, Quaternion):
            q_io = Quaternion(q_io)

        # 1) Attitude estimate q_ib_est (body -> inertial)
        q_ib_est = self._estimate_q_ib(star_meas, mag_meas, sun_meas_list,
                                       r_i, JD)
        if q_ib_est is None:
            self.tau = np.zeros(3)
            self.debug = {"mode": "NO_DATA"}
            return

        # 2) Desired frame: nadir -> orbital frame, else inertial hold (q_id)
        if self.target == "nadir":
            q_des, w_des_i = q_io, w_i_io
        else:
            q_des = Quaternion(self.q_id.q); q_des.normalize()
            w_des_i = self.w_d

        # 3) Error quaternion q_e = q_des^-1 (x) q_ib_est  (identity at target)
        q_e = q_des.inverted() @ q_ib_est
        q_e.normalize()
        q0, qv = q_e.q[0], q_e.q[1:4]
        if q0 < 0:
            q0, qv = -q0, -qv

        # 4) Body-frame angular-rate error (feed-forward desired-frame rate)
        R_bi    = su.quaternion_to_dcm(q_ib_est).T     # inertial -> body
        w_des_b = R_bi @ w_des_i
        w_err   = gyro_meas - w_des_b

        # 5) PD control torque (inertia-scaled / feedback-linearized form).
        #    The assignment baseline gains k1=1e-4, k2=2e-2 are only sensible
        #    when scaled by J (J*k1 ~ 9 Nm/rad for HST):
        #    tau_c = w x Jw + J(-Kp q_v - Kd w_err)
        gyro_term = np.cross(gyro_meas, self.J @ gyro_meas)
        feedback  = -self.Kp * qv - self.Kd * w_err
        tau = gyro_term + self.J @ feedback
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.tau = tau

        self.debug = {
            "mode": self.target,
            "q_ib_est": q_ib_est.q.copy(),
            "q_err": q_e.q.copy(),
            "att_err_deg": np.degrees(2*np.arccos(np.clip(q0, -1.0, 1.0))),
            "w_error": w_err.copy(),
            "tau_c": tau.copy(),
        }

    def _estimate_q_ib(self, star_meas, mag_meas, sun_meas_list, r_i, JD):
        """q_ib (body->inertial): star tracker if present, else vector estimator
        with references expressed in the INERTIAL frame."""
        if isinstance(star_meas, Quaternion):
            q = Quaternion(star_meas.q); q.normalize()
            if self.q_ib_prev is not None and np.dot(self.q_ib_prev.q, q.q) < 0:
                q.q *= -1
            self.q_ib_prev = Quaternion(q.q)
            return q
        sun_i = sun_vector(JD)
        B_i   = magnetic_field_dipole(r_i, JD)
        M_B, M_A = [], []
        b_b = self._safe_unit(mag_meas); B_iu = self._safe_unit(B_i)
        if b_b is not None and B_iu is not None:
            M_B.append(b_b); M_A.append(B_iu)
        S_iu = self._safe_unit(sun_i)
        for s in sun_meas_list:
            s_b = self._safe_unit(s)
            if s_b is not None and S_iu is not None:
                M_B.append(s_b); M_A.append(S_iu)
        if len(M_B) < 2:
            return None
        q = self.estimator.estimate_attitude(M_B, M_A)
        if not isinstance(q, Quaternion):
            q = Quaternion(q)
        q = q.inverted()      # estimator returns q_bi; we want q_ib
        q.normalize()
        return q

    def get_control(self):
        return self.tau


class ADCS_SM:
    def __init__(self, k1, k, eps, J, attitude_estimator, tau_max=None):
        self.k1  = k1
        self.k   = k
        self.eps = eps
        self.J   = J
        self.estimator = attitude_estimator
        self.tau_max = tau_max          # per-axis controller-command limit (None = no clip)
        self.tau = np.zeros(3)

        self.q_id = Quaternion([1.0, 0.0, 0.0, 0.0])
        self.w_d  = np.zeros(3)
        self.target = "nadir"   # "nadir"=track orbital frame; "inertial"=hold q_id

        self.debug = {}
        self.q_ib_prev = None

    def _safe_unit(self, v):
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        return v / n

    def update(self,
               t,
               r_i, v_i,
               q_ib, w_b_ib,
               q_io, w_i_io, dw_i_io,
               gyro_meas, mag_meas, sun_meas_list,
               JD,
               star_meas=None):

        if dw_i_io is None:
            dw_i_io = np.zeros(3)
        if not isinstance(q_io, Quaternion):
            q_io = Quaternion(q_io)

        # 1) Attitude estimate q_ib_est (body -> inertial)
        q_ib_est = self._estimate_q_ib(star_meas, mag_meas, sun_meas_list,
                                       r_i, JD)
        if q_ib_est is None:
            self.tau = np.zeros(3)
            self.debug = {"mode": "NO_DATA"}
            return

        # 2) Desired frame
        if self.target == "nadir":
            q_des, w_des_i, dw_des_i = q_io, w_i_io, dw_i_io
        else:
            q_des = Quaternion(self.q_id.q); q_des.normalize()
            w_des_i, dw_des_i = self.w_d, np.zeros(3)

        # 3) Error quaternion (identity at target)
        q_e = q_des.inverted() @ q_ib_est
        q_e.normalize()
        q0, qv = q_e.q[0], q_e.q[1:4]
        if q0 < 0:
            q0, qv = -q0, -qv

        # 4) Body-frame rate / accel of desired frame
        R_bi     = su.quaternion_to_dcm(q_ib_est).T    # inertial -> body
        w_des_b  = R_bi @ w_des_i
        dw_des_b = R_bi @ dw_des_i
        w_err    = gyro_meas - w_des_b                 # body rate wrt desired, in body

        # 5) Sliding-mode (computed-torque) law with boundary layer.
        #    s = w_err + 2 k1 q_v ;  feedback-linearised so J/gyro cancel the plant
        s     = w_err + 2.0 * self.k1 * qv
        sat_s = np.clip(s / self.eps, -1.0, 1.0)
        wdot_des = (dw_des_b - np.cross(gyro_meas, w_des_b)
                    - self.k1 * qv
                    - self.k * sat_s)
        gyro_term = np.cross(gyro_meas, self.J @ gyro_meas)
        tau = gyro_term + self.J @ wdot_des
        if self.tau_max is not None:
            tau = np.clip(tau, -self.tau_max, self.tau_max)
        self.tau = tau

        self.debug = {
            "mode": self.target,
            "q_ib_est": q_ib_est.q.copy(),
            "q_err": q_e.q.copy(),
            "att_err_deg": np.degrees(2*np.arccos(np.clip(q0, -1.0, 1.0))),
            "w_error": w_err.copy(),
            "s": s.copy(),
            "tau_c": tau.copy(),
        }

    def _estimate_q_ib(self, star_meas, mag_meas, sun_meas_list, r_i, JD):
        if isinstance(star_meas, Quaternion):
            q = Quaternion(star_meas.q); q.normalize()
            if self.q_ib_prev is not None and np.dot(self.q_ib_prev.q, q.q) < 0:
                q.q *= -1
            self.q_ib_prev = Quaternion(q.q)
            return q
        sun_i = sun_vector(JD)
        B_i   = magnetic_field_dipole(r_i, JD)
        M_B, M_A = [], []
        b_b = self._safe_unit(mag_meas); B_iu = self._safe_unit(B_i)
        if b_b is not None and B_iu is not None:
            M_B.append(b_b); M_A.append(B_iu)
        S_iu = self._safe_unit(sun_i)
        for s in sun_meas_list:
            s_b = self._safe_unit(s)
            if s_b is not None and S_iu is not None:
                M_B.append(s_b); M_A.append(S_iu)
        if len(M_B) < 2:
            return None
        q = self.estimator.estimate_attitude(M_B, M_A)
        if not isinstance(q, Quaternion):
            q = Quaternion(q)
        q = q.inverted()
        q.normalize()
        return q

    def get_control(self):
        return self.tau
def angle_deg(a, b):
            a = a / np.linalg.norm(a)
            b = b / np.linalg.norm(b)
            return np.degrees(np.arccos(np.clip(np.dot(a,b), -1, 1)))


def disturbance_db(t):
    return np.array([
        1e-3 * np.sin(2*np.pi*t/600.0),
        1e-3 * np.cos(2*np.pi*t/700.0),
        1e-3 * np.sin(2*np.pi*t/800.0)
    ])

class Satellite:
    def __init__(self, q_ib, w_b_ib, J,
                 r=np.zeros(3), v=np.zeros(3), m=1.0,
                 orbit=None, substeps=0,
                 attitude_estimator=None,
                 controller_type="PD",
                 Kp=5e-2, Kd=6e-1,
                 k1=0.05, k_slide=0.5, eps=0.05):

        self.orbit = orbit

        if self.orbit is not None:
            r, v = self.orbit.get_state()

        self.body = RigidBody6DOF(r, v, m, q_ib, w_b_ib, J)
        self.N = substeps + 1

        self.estimator = attitude_estimator
        JD_init = jd_from_t(0.0)

        # ---------------- SENSORS ----------------
        self.sensors = []

        # gyro
        self.sensors.append(
            gyro(q_bs=Quaternion(), p_b=np.zeros(3),
                 mu=0.0, Q=0.0, params={"bg": 0.0})
        )

        # star tracker
        self.sensors.append(
            star_tracker(q_bs=Quaternion(), p_b=np.zeros(3),
                         mu=0.0, Q=0) #usually 1e-4
        )

        # magnetometer
        self.sensors.append(
            magnetometer(q_bs=Quaternion(), p_b=np.zeros(3),
                         mu=0.0, Q=0.4e-8, params={"JD": JD_init})
        )

        # fine sun sensors (6 faces)
        q_xp = Quaternion(np.pi/4,  [0,1,0])
        q_xm = Quaternion(-np.pi/4, [0,1,0])
        q_yp = Quaternion(-np.pi/4, [1,0,0])
        q_ym = Quaternion(np.pi/4,  [1,0,0])
        q_zp = Quaternion([1,0,0,0])
        q_zm = Quaternion(np.pi, [1,0,0])

        self.sensors.append(fine_sun_sensor(q_xp, np.array([0.1,0,0]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))
        self.sensors.append(fine_sun_sensor(q_xm, np.array([-0.1,0,0]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))
        self.sensors.append(fine_sun_sensor(q_yp, np.array([0,0.1,0]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))
        self.sensors.append(fine_sun_sensor(q_ym, np.array([0,-0.1,0]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))
        self.sensors.append(fine_sun_sensor(q_zp, np.array([0,0,0.1]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))
        self.sensors.append(fine_sun_sensor(q_zm, np.array([0,0,-0.1]),
                                            0.0, 0.0, params={"alpha": np.pi, "JD": JD_init}))

        # ---------------- CONTROLLER ----------------
        if controller_type == "PD":
            self.ADCS = ADCS_PD(Kp=Kp, Kd=Kd, J=J,
                                attitude_estimator=self.estimator)
        elif controller_type == "SM":
            self.ADCS = ADCS_SM(k1=k1, k=k_slide, eps=eps, J=J,
                                attitude_estimator=self.estimator)
        else:
            raise ValueError("controller_type must be 'PD' or 'SM'")

        self.logger = su.AttitudeLogger(name=f"satellite_{controller_type}")
        self.log_stride = 10   # log every 10 substeps
        self._log_counter = 0

    # ------------- helpers -------------

    def update_sensors(self, t, dt, q_ib, w_b_ib, r_i, v_i):
        measurements = []
        for s in self.sensors:
            s.update(t, dt, q_ib, w_b_ib, r_i, v_i)
            measurements.append(s.output(body_frame=True))
        return measurements

    def get_state(self):
        return self.body.get_state()

    def get_orbit_frame(self):
        if self.orbit is not None:
            return self.orbit.get_orbit_frame()
        else:
            r, v, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(r, v)

    # ------------- main update -------------

    def update(self, t_k, t_step):
        if self.orbit is not None:
            self.update_with_orbit(t_k, t_step)
        else:
            self.update_with_dynamics(t_k, t_step)

    def update_with_orbit(self, t_k, t_step):
        r_0, v_0 = self.orbit.get_state()
        self.orbit.propagate(t_step)
        r_1, v_1 = self.orbit.get_state()

        t_sub = t_step / self.N

        for n in range(self.N):
            r_i = r_0 + (n / self.N) * (r_1 - r_0)
            v_i = v_0 + (n / self.N) * (v_1 - v_0)

            _, _, q_ib, w_b_ib = self.body.get_state()

            q_io_arr, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)
            q_io = Quaternion(q_io_arr)

            meas = self.update_sensors(t_k, t_sub, q_ib, w_b_ib, r_i, v_i)
            gyro_meas = meas[0]
            star_meas = meas[1]
            mag_meas  = meas[2]
            sun_meas  = meas[3:]

            JD = jd_from_t(t_k)

            self.ADCS.update(
                t=t_k,
                r_i=r_i, v_i=v_i,
                q_ib=q_ib, w_b_ib=w_b_ib,
                q_io=q_io, w_i_io=w_i_io, dw_i_io=dw_i_io,
                gyro_meas=gyro_meas,
                mag_meas=mag_meas,
                sun_meas_list=sun_meas,
                JD=JD,
                star_meas=star_meas
            )
            tau_c = self.ADCS.get_control()

            tau_G  = ol.gravity_gradient(r_i, q_ib, self.body.J)
            tau_db = disturbance_db(t_k)
            tau_d  = tau_G + tau_db
            # ---------------- LOGGING ----------------
            self._log_counter += 1
            if self._log_counter % self.log_stride == 0:

                dbg = getattr(self.ADCS, "debug", {})

                data = {
                    "t": t_k,
                    "r_i": r_i,
                    "v_i": v_i,
                    "q_ib": q_ib.q,
                    "w_b_ib": w_b_ib,
                    "q_io": q_io.q,
                    "w_i_io": w_i_io,
                    "dw_i_io": dw_i_io,
                    "tau_c": tau_c,
                    "tau_G": tau_G,
                    "tau_db": tau_db,
                    "tau_d": tau_d,
                }

                # merge controller debug
                for k, v in dbg.items():
                    data[f"ctrl_{k}"] = v

                self.logger.log(data)
            # -----------------------------------------

            self.body.update(t_k, t_sub, np.zeros(3), tau_c + tau_d)
            t_k += t_sub

        self.body.p, self.body.v = self.orbit.get_state()

    def update_with_dynamics(self, t_k, t_step):
        t_sub = t_step / self.N
        for n in range(self.N):
            r_i, v_i, q_ib, w_b_ib = self.body.get_state()

            q_io_arr, w_i_io, dw_i_io = self.get_orbit_frame()
            q_io = Quaternion(q_io_arr)

            meas = self.update_sensors(t_k, t_sub, q_ib, w_b_ib, r_i, v_i)
            gyro_meas = meas[0]
            star_meas = meas[1]
            mag_meas  = meas[2]
            sun_meas  = meas[3:]

            JD = jd_from_t(t_k)

            self.ADCS.update(
                t=t_k,
                r_i=r_i, v_i=v_i,
                q_ib=q_ib, w_b_ib=w_b_ib,
                q_io=q_io, w_i_io=w_i_io, dw_i_io=dw_i_io,
                gyro_meas=gyro_meas,
                mag_meas=mag_meas,
                sun_meas_list=sun_meas,
                JD=JD,
                star_meas=star_meas
            )
            tau_c = self.ADCS.get_control()

            f = -ol.mu * r_i / np.linalg.norm(r_i)**3

            tau_G  = ol.gravity_gradient(r_i, q_ib, self.body.J)
            tau_db = disturbance_db(t_k)
            tau_d  = tau_G + tau_db
            # ---------------- LOGGING ----------------
            self._log_counter += 1
            if self._log_counter % self.log_stride == 0:

                dbg = getattr(self.ADCS, "debug", {})

                data = {
                    "t": t_k,
                    "r_i": r_i,
                    "v_i": v_i,
                    "q_ib": q_ib.q,
                    "w_b_ib": w_b_ib,
                    "q_io": q_io.q,
                    "w_i_io": w_i_io,
                    "dw_i_io": dw_i_io,
                    "tau_c": tau_c,
                    "tau_G": tau_G,
                    "tau_db": tau_db,
                    "tau_d": tau_d,
                }

                # merge controller debug
                for k, v in dbg.items():
                    data[f"ctrl_{k}"] = v

                self.logger.log(data)
            # -----------------------------------------

            self.body.update(t_k, t_sub, f, tau_c + tau_d)
            t_k += t_sub

class gyro:
    def __init__(self, q_bs=Quaternion(), p_b=np.zeros(3), mu=0, Q=0.0, params=None):
        self.q_bs = q_bs
        self.p_b = p_b
        self.mu = mu
        self.Q = Q
        self.bg = params.get("bg", 0) if params else 0
        self.z = np.zeros(3)

    def update(self, t, dt, q_ib, w_b_ib, r_i, v_i):
        noise = np.random.normal(self.mu, np.sqrt(self.Q), 3)
        self.z = w_b_ib + self.bg + noise

    def output(self, body_frame=True):
        return self.z


class star_tracker:
    """
    Clean, stable star tracker model (Assignment 9 Appendix C).
    """
    def __init__(self, q_bs=Quaternion(), p_b=np.zeros(3), mu=0.0, Q=1e-4):
        self.q_bs = q_bs
        self.p_b = p_b
        self.mu = mu
        self.Q = Q
        self.z = Quaternion([1,0,0,0])

    def update(self, t, dt, q_ib, w_b_ib, r_i, v_i):
        theta = np.random.normal(self.mu, np.sqrt(self.Q))

        x, y = np.random.rand(2)
        a = np.arccos(1 - 2*x)
        b = 2*np.pi*y
        u = np.array([np.cos(b)*np.sin(a),
                      np.sin(b)*np.sin(a),
                      np.cos(a)])

        q_e = Quaternion([np.cos(theta/2), *(np.sin(theta/2)*u)])

        self.z = q_ib @ self.q_bs @ q_e

    def output(self, body_frame=False):
        return self.z


class magnetometer:
    def __init__(self, q_bs=Quaternion(), p_b=np.zeros(3), mu=0, Q=0.4e-8, params=None):
        self.q_bs = q_bs
        self.p_b = p_b
        self.mu = mu
        self.Q = Q
        self.z = np.zeros(3)
        self.JD = params.get("JD") if params else None

    def update(self, t, dt, q_ib, w_b_ib, r_i, v_i):
        self.JD += dt/86400.0
        B_i = magnetic_field_dipole(r_i, self.JD)
        # field in the SENSOR frame: B_s = R_si B_i = (q_ib (x) q_bs)^-1 . B_i
        # (q_ib maps body->inertial, q_bs maps sensor->body, so q_ib@q_bs maps
        #  sensor->inertial; its inverse takes the inertial field into the sensor)
        q_is = q_ib @ self.q_bs
        B_s = q_is.inverted().rotate(B_i)
        noise = np.random.normal(self.mu, np.sqrt(self.Q), 3)
        self.z = B_s + noise

    def output(self, body_frame=True):
        return self.z


class fine_sun_sensor:
    def __init__(self, q_bs, p_b, mu, Q, params=None):
        self.q_bs = q_bs
        self.p_b = p_b
        self.mu = mu
        self.Q = Q
        self.alpha = params.get("alpha", np.pi)
        self.JD = params.get("JD")
        self.z = np.zeros(3)

    def update(self, t, dt, q_ib, w_b_ib, r_i, v_i):
        self.JD += dt/86400.0
        s_i = sun_vector(self.JD)
        s_b = q_ib.inverted().rotate(s_i)        # sun in BODY frame = R_bi s_i
        s_s = self.q_bs.inverted().rotate(s_b)   # sun in SENSOR frame = R_sb s_b

        x, y, z = s_s
        if z > 0 and np.arctan2(np.sqrt(x*x + y*y), z) < self.alpha/2:
            meas = np.array([x/z, y/z, 1.0])
            noise = np.random.normal(self.mu, np.sqrt(self.Q), 3)
            self.z = meas + noise
        else:
            self.z = np.zeros(3)

    def output(self, body_frame=True):
        return self.z


class TRIAD:
    def estimate_attitude(self, M_B, M_A):
        if len(M_B) < 2:
            return Quaternion([1,0,0,0])

        def safe(v):
            n = np.linalg.norm(v)
            return v/n if n > 1e-8 else None

        a_b = safe(M_B[0])
        b_b = safe(M_B[1])
        a_i = safe(M_A[0])
        b_i = safe(M_A[1])

        if any(x is None for x in (a_b, b_b, a_i, b_i)):
            return Quaternion([1,0,0,0])

        t1_b = a_b
        t2_b = np.cross(a_b, b_b)
        if np.linalg.norm(t2_b) < 1e-8:
            return Quaternion([1,0,0,0])
        t2_b /= np.linalg.norm(t2_b)
        t3_b = np.cross(t1_b, t2_b)

        t1_i = a_i
        t2_i = np.cross(a_i, b_i)
        if np.linalg.norm(t2_i) < 1e-8:
            return Quaternion([1,0,0,0])
        t2_i /= np.linalg.norm(t2_i)
        t3_i = np.cross(t1_i, t2_i)

        R_B = np.vstack([t1_b, t2_b, t3_b]).T
        R_A = np.vstack([t1_i, t2_i, t3_i]).T
        R_BA = R_B @ R_A.T

        q_arr = ol.quaternion_from_rotation_matrix(R_BA)
        return Quaternion(q_arr).conjugated()


class Davenport:
    def estimate_attitude(self, M_B, M_A):
        N = len(M_B)
        if N < 2:
            return Quaternion([1,0,0,0])

        B = np.zeros((3,3))
        z = np.zeros(3)
        a = 1.0/N

        for i in range(N):
            b = M_B[i] / np.linalg.norm(M_B[i])
            a_i = M_A[i] / np.linalg.norm(M_A[i])
            B += a * np.outer(b, a_i)
            z += a * np.cross(b, a_i)

        S = B + B.T
        sigma = np.trace(B)

        K = np.zeros((4,4))
        K[0,0] = sigma
        K[0,1:] = z
        K[1:,0] = z
        K[1:,1:] = S - sigma*np.eye(3)

        evals, evecs = np.linalg.eig(K)
        q = evecs[:, np.argmax(evals)]
        q /= np.linalg.norm(q)

        return Quaternion(q).conjugated()


class DavenportMultiST:
    def quat_rotate(self, q, v):
        return (q @ Quaternion([0,*v]) @ q.inverted()).q[1:]

    def estimate_attitude(self, q_list):
        e1 = np.array([1,0,0])
        e2 = np.array([0,1,0])
        e3 = np.array([0,0,1])

        a1, b1 = e1, e2
        a2, b2 = e2, e3
        a3, b3 = e3, e1

        q1, q2, q3 = q_list

        ahat1 = self.quat_rotate(q1, a1)
        bhat1 = self.quat_rotate(q1, b1)
        ahat2 = self.quat_rotate(q2, a2)
        bhat2 = self.quat_rotate(q2, b2)
        ahat3 = self.quat_rotate(q3, a3)
        bhat3 = self.quat_rotate(q3, b3)

        MB = np.vstack([ahat1, bhat1, ahat2, bhat2, ahat3, bhat3])
        MA = np.vstack([a1, b1, a2, b2, a3, b3])

        B = np.zeros((3,3))
        for i in range(6):
            B += np.outer(MB[i], MA[i])

        z = np.array([
            B[1,2] - B[2,1],
            B[2,0] - B[0,2],
            B[0,1] - B[1,0]
        ])

        K = np.zeros((4,4))
        K[0,0] = np.trace(B)
        K[0,1:] = z
        K[1:,0] = z
        K[1:,1:] = B + B.T - np.trace(B)*np.eye(3)

        evals, evecs = np.linalg.eig(K)
        q = evecs[:, np.argmax(evals)]
        q /= np.linalg.norm(q)

        return Quaternion(q)
