"""
montecarlo_pointing.py

test monte carlo del true pointing error dell'HST.
corre lo stesso scenario N volte con semi diversi (ma fissi, quindi
riproducibili) e calcola media e deviazione standard del pointing error
sull'ultima orbita. serve a mostrare che la conclusione PD vs SMC non
dipende da una singola realizzazione del rumore.

uso:  python montecarlo_pointing.py
"""

import numpy as np
import main as m   # riusa tutta la macchina gia' scritta in main.py

MU = m.MU


def run_one(cfg, seed):
    """corre uno scenario con un seme fisso e ritorna il true pointing
    error medio sull'ultima orbita [arcsec]."""
    np.random.seed(seed)              # rende la corsa riproducibile

    sc = m.HSTAttitudeScenario(cfg)
    sc.init(0.0)
    T = 2*np.pi/np.sqrt(MU/(np.linalg.norm(sc.orbit.get_state()[0])**3))
    t_end = cfg["n_orbits"]*T
    t, dt = 0.0, cfg["dt"]
    while t < t_end:
        sc.update(t, dt)
        t += dt

    # media del true pointing error sull'ultima orbita
    t_arr    = np.array(sc.t_log)
    etrue    = np.array(sc.etrue_log)
    last     = t_arr >= (t_arr[-1] - T)
    return etrue[last].mean()          # [arcsec]


def montecarlo(controller="SM", n_star_trackers=1, N=20):
    """corre N semi e stampa la statistica."""
    # configurazione base: parto da CUSTOM, spengo grafica e plot,
    # e forzo controllore e numero di star tracker richiesti.
    cfg = dict(m.CUSTOM)
    cfg.update(target="inertial",
               controller=controller,
               n_star_trackers=n_star_trackers,
               visualise=False,
               plot=False)
    cfg.pop("_compare", None)          # niente confronto: un solo scenario

    rms_list = []
    for k in range(N):
        e = run_one(cfg, seed=k)
        rms_list.append(e)
        print(f"  seed {k:2d}:  {e:8.1f} arcsec  ({e/3600:.3f} deg)")

    arr = np.array(rms_list)
    print(f"\n{controller}, {n_star_trackers} ST, over {N} seeds:")
    print(f"  mean = {arr.mean():.1f} arcsec ({arr.mean()/3600:.3f} deg)")
    print(f"  std  = {arr.std():.1f} arcsec ({arr.std()/3600:.3f} deg)")
    return arr


if __name__ == "__main__":
    # esempio: confronto PD vs SMC su N semi
    N = 10
    print(" PD baseline ")
    pd = montecarlo(controller="PD", n_star_trackers=1, N=N)
    print("\n SMC ")
    sm = montecarlo(controller="SM", n_star_trackers=1, N=N)

    print(f"\n--- confronto ---")
    print(f"PD  mean: {pd.mean()/3600:.3f} deg")
    print(f"SMC mean: {sm.mean()/3600:.3f} deg")
    print(f"SMC/PD ratio: {sm.mean()/pd.mean():.2f}")
