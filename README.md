# Spacecraft Dynamics Sandbox (Python)

> Modular Python framework for orbital propagation and spacecraft attitude control, originally built around an HST (Hubble Space Telescope) pointing scenario and extended into a configurable sandbox for further experimentation.

<p align="center">
  <img src="images/hst_visualization.png" alt="HST attitude control 3D visualization" width="70%">
  <br>
  <em>3D visualization of the HST in low Earth orbit, body and orbit frames rendered with vispy.</em>
</p>

---

## Overview

This codebase combines an orbital propagator, a spacecraft model with realistic sensor and actuator dynamics, and a set of attitude controllers into a single simulator. The original goal was to study inertial pointing of the **Hubble Space Telescope** under realistic disturbances, sensor noise and actuator saturation. The architecture was then generalized into a **custom sandbox mode** that lets you toggle propagators, sensors, perturbations and controllers individually, so the same codebase can be reused for further spacecraft-dynamics studies.

The project originated in the *Spacecraft Engineering* coursework at UiT — The Arctic University of Norway and was extended independently to add the sandbox mode and modular architecture beyond what the coursework required.

---

## Capabilities

**Orbital dynamics**
- Two-body Keplerian propagator (closed-form)
- PKepler propagator with secular $J_2$ effects on $\Omega$, $\omega$ and mean motion
- Multi-year decay studies (PKepler vs SGP4 reference)
- Frame conversions: ECI ↔ ECEF (with sidereal rotation), body, orbit (RSW)
- TLE-based initialization

**Attitude dynamics**
- Full 6-DOF rigid-body dynamics with quaternion kinematics
- HST inertia tensor with off-diagonal coupling
- Disturbance torques: gravity gradient + solar-array (sinusoidal)
- Actuator model: reaction wheels with saturation $\tau_a = 1.13\,\mathrm{sat}(\tau_c / 1.13)$

**Sensors**
- 3-axis gyroscope (bias + white noise)
- Star tracker(s), configurable count, with quaternion measurement noise
- Sun sensors (multiple, distributed in body frame)
- Magnetometer (dipole Earth field model)

**Attitude determination**
- Single star-tracker passthrough
- TRIAD algorithm (two-vector deterministic)
- Davenport's q-method (multi-vector least-squares)
- Multi-star-tracker fusion via Davenport

**Control**
- PD baseline controller, $\tau = -k_1\,q_v - k_2\,\omega_\text{err}$
- Sliding Mode Control with boundary-layer smoothing

**Visualization & analysis**
- 3D real-time visualization (vispy): Earth with sidereal rotation, spacecraft body frame, orbit-frame triad, optional Keplerian-orbit reference line
- Ground track plotting
- Pointing-error time histories

---

## Scenarios

The main entry point is `hst_attitude_control.py`. Available presets:

| Preset | Description |
|--------|-------------|
| `p1t1` | Orbital frame `q_io` derivation and verification against RSW definition |
| `p1t2` | PKepler vs SGP4 long-term decay comparison toward mid-2030s HST reentry |
| `p2t1` | Inertial pointing using a single star tracker (weak baseline) |
| `p2t2` | Inertial pointing with 3 star trackers + Davenport fusion (improved) |
| `custom` | Full sandbox — choose propagator, sensors, controllers, disturbances, ICs, etc. |

Run a preset:

```bash
python hst_attitude_control.py --preset p2t2
```

Run the sandbox with custom options (see `--help` for the full list):

```bash
python hst_attitude_control.py --preset custom \
    --propagator pkepler \
    --controller smc \
    --sensors star_tracker,sun,gyro \
    --disturbances grav_grad,solar_array \
    --visualise
```

---

## Architecture

```
orbit_lib.py    Orbital propagators (Kepler, PKepler), frame conversions, TLE parsing
sat_lib.py     Sensor models, actuator model, attitude estimators, controllers
simutils.py    Quaternion math, rotations, generic numerical helpers
plotter.py     Headless plotting utilities (matplotlib) for ground tracks, errors, etc.
simulator.py   3D real-time visualization framework (vispy) — see credit header inside
hst_attitude_control.py    Scenario definitions, presets, custom-mode dispatcher
```

The scenario file ties everything together: it constructs the spacecraft state, registers sensors and controllers, advances the simulation step by step, and either runs headless (producing plots) or hands the state over to `simulator.py` for 3D animation.

---

## Quick Start

```bash
git clone https://github.com/rodrig0conti/spacecraft-dynamics-sandbox.git
cd spacecraft-dynamics-sandbox
pip install -r requirements.txt
python hst_attitude_control.py --preset p2t2 --visualise
```

For systems without OpenGL / vispy support, drop the `--visualise` flag and the run produces matplotlib plots instead.

---

## Example Results

<p align="center">
  <img src="images/pointing_error.png" alt="Pointing error" width="48%">
  <img src="images/ground_track.png" alt="Ground track" width="48%">
  <br>
  <em>Left: pointing-error time history comparing PD baseline vs SMC under realistic disturbances. Right: HST ground track over one orbit.</em>
</p>

Typical results obtained from closed-loop runs:

| Configuration | True pointing error |
|---|---:|
| Single star tracker, PD | 1.10° |
| 3 star trackers + Davenport, PD | 1.09° |
| TRIAD, PD | 1.14° |
| Magnetometer + 2 sun sensors (no ST) | 1.12° |
| Nadir pointing, full sensor suite | 0.82° |
| Noise-free, no disturbances | ≈ 0° (verification) |

The expected pointing-error floor is dominated by star-tracker noise ($\sigma^2 \approx 10^{-2}$ rad²); multi-tracker fusion reduces it as expected.

---

## Tech Stack

- **Python 3.10+**
- `numpy`, `scipy` — numerical core
- `matplotlib` — headless plotting
- `vispy` — 3D real-time visualization
- `sgp4` — reference propagator for verification

See `requirements.txt` for exact versions.

---

## Limitations & Roadmap

**Current limitations**
- Atmospheric drag not modelled (relevant for long-term LEO decay studies)
- $J_2$ enters as analytic secular drift in PKepler only; no $J_2$ torque on attitude
- No third-body, SRP, or higher-order gravity terms
- Magnetometer-based determination has known coverage gaps below the noise floor
- No transfer-maneuver capability (instantaneous $\Delta v$ or finite-burn)

**Planned extensions**
- Atmospheric drag (NRLMSISE-00 or Jacchia–Bowman with $F_{10.7}$ / $K_p$)
- $J_2$ disturbance torque on attitude
- Impulsive and finite-burn transfer maneuvers
- SGP4 native integration in PKepler comparisons
- Integration with the [Orbital Mechanics MATLAB Toolbox](https://github.com/rodrig0conti/Orbital-Mechanics-Toolbox) for transfer optimization

---

## Credits

- `simulator.py` was provided as course material at UiT (3D visualization framework). All credit for that file goes to the course instructor; it is reproduced here with attribution. See header inside the file.
- All other modules in this repository are my own implementation.
- HST inertia tensor, controller gain definitions and disturbance models follow standard references on spacecraft attitude dynamics (Sidi, *Spacecraft Dynamics and Control*; Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control*).
- Hubble deorbit timeline reference: NASA Astrophysics Division (M. Clampin) statements on HST orbital decay through the mid-2030s.

---

## Related Work

- [Orbital Mechanics Toolbox (MATLAB)](https://github.com/rodrig0conti/Orbital-Mechanics-Toolbox) — final bachelor project on orbital transfer optimization and Δv minimization. This Python sandbox extends and reimplements parts of that work with an object-oriented architecture and realistic spacecraft dynamics.
- [CFD Nozzle Study](https://github.com/rodrig0conti/CFD-Nozzle-Study) — aerothermodynamic analysis of rocket nozzles.

---

## Author

**Rodrigo Conti Gallenti**
MSc Aerospace Engineering
