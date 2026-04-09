import numpy as np
import simutils as su

# Earth parameters
mu = 398600.4418  # km^3/s^2
R_E = 6378.1363   # km

def dynamics(t, x):
    r = x[:3]
    v = x[3:]
    norm_r = np.linalg.norm(r)
    a = -mu * r / norm_r**3
    return np.hstack((v, a))

# Initial conditions
r0 = np.array([R_E + 800, 0, 0])
v0 = np.array([0, 0, np.sqrt(mu / np.linalg.norm(r0))])
x0 = np.hstack((r0, v0))

# Simulation parameters
t0 = 0
t_end = 20000
h = 10

# Storage
times = [t0]
euler_traj = [x0]
leapfrog_traj = [x0]
verlet_traj = [x0]
rk4_traj = [x0]

# Verlet needs x_{-1} → use Euler for first step
x_minus1 = x0 - h * dynamics(t0, x0)

# Run simulation
t = t0
x_e = x0.copy()
x_l = x0.copy()
x_v = x0.copy()
x_rk = x0.copy()

while t < t_end:
    # Euler
    x_e = su.step_euler(h, t, x_e, dynamics)
    euler_traj.append(x_e)

    # Leapfrog
    x_l = su.step_leapfrog(h, t, x_l, dynamics)
    leapfrog_traj.append(x_l)

    # Verlet
    x_next = su.step_verlet(h, t, x_v[:3], x_minus1[:3], dynamics)
    x_minus1 = x_v.copy()
    x_v = np.hstack((x_next, dynamics(t, np.hstack((x_next, np.zeros(3))))[:3]))
    verlet_traj.append(x_v)

    # RK4
    x_rk = su.step_RK4(h, t, x_rk, dynamics)
    rk4_traj.append(x_rk)

    t += h
    times.append(t)

print("Simulation finished!")
print("Final Euler position:", euler_traj[-1][:3])
print("Final Leapfrog position:", leapfrog_traj[-1][:3])
print("Final Verlet position:", verlet_traj[-1][:3])
print("Final RK4 position:", rk4_traj[-1][:3])

##

