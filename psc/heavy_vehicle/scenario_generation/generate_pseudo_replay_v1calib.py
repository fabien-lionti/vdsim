#!/usr/bin/env python3
"""
Variante pseudo-replay avec DOF10 V1 calibré (sans relaxation pneu ni combined slip).
Teste si les améliorations V2 sont nécessaires au-delà de la calibration.
"""
import sys
import csv
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from psc.heavy_vehicle.params.vehicle_params import make_heavy_vehicle

# Reuse OU / events generation
import generate_pseudo_replay as gen

OUTPUT_DIR = SCRIPT_DIR / "data_v6b_pseudoreplay_v1calib"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DT = gen.DT
N_SCENARIOS = 400


def run_sim_v1(vx_ref, delta_f, mu=0.85):
    """DOF10 V1 (16 états) avec PID vitesse + δf injecté."""
    model = make_heavy_vehicle(mu)
    n_steps = len(vx_ref)
    x = np.zeros(16)
    x[1] = vx_ref[0]; x[4] = model.h
    w0 = max(vx_ref[0], 1.0) / model.r
    x[12:16] = w0

    vx_out = np.zeros(n_steps); vy_out = np.zeros(n_steps); psi_dot_out = np.zeros(n_steps)
    phi_out = np.zeros(n_steps); theta_out = np.zeros(n_steps); delta_f_out = np.zeros(n_steps)
    ltr_f = np.zeros(n_steps); ltr_r = np.zeros(n_steps)
    kp, ki = 2100.0, 21.0; integral = 0.0

    for i in range(n_steps):
        err = vx_ref[i] - x[1]; integral += err * DT
        torque = np.clip(kp * err + ki * integral, -8500, 8500) / 4.0
        u = np.array([torque, torque, torque, torque, delta_f[i], 0.0])
        try:
            dx, outputs = model.get_dx__dt(x, u)
        except Exception:
            return None
        vx_out[i] = x[1]; vy_out[i] = x[3]; psi_dot_out[i] = x[11]
        phi_out[i] = x[8]; theta_out[i] = x[6]; delta_f_out[i] = delta_f[i]
        fz = outputs["Fz"]; eps = 1e-12
        ltr_f[i] = abs((fz[1] - fz[0]) / (fz[1] + fz[0] + eps))
        ltr_r[i] = abs((fz[3] - fz[2]) / (fz[3] + fz[2] + eps))
        x = x + dx * DT
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e8):
            return None

    delta_f_dot = np.gradient(delta_f_out, DT)
    ltr_max = np.maximum(ltr_f, ltr_r)
    return {
        'time': np.arange(n_steps) * DT,
        'vx': vx_out, 'vy': vy_out, 'psi_dot': psi_dot_out,
        'phi': phi_out, 'theta': theta_out,
        'delta_f': delta_f_out, 'delta_f_dot': delta_f_dot,
        'LTR_avant': ltr_f, 'LTR_arriere': ltr_r, 'LTRmax': ltr_max,
    }


def main():
    print("=" * 60)
    print("PSEUDO-REPLAY — DOF10 V1 CALIBRÉ (sans relax/combined slip)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    rng_master = np.random.default_rng(42)
    generated, skipped = 0, 0
    ltr_peaks = []
    attempts = 0
    while generated < N_SCENARIOS and attempts < N_SCENARIOS * 5:
        attempts += 1
        seed = int(rng_master.integers(0, 10_000_000))
        rng = np.random.default_rng(seed)
        duration = rng.uniform(*gen.DURATION_RANGE)
        mu = rng.uniform(0.7, 1.0)
        vx = gen.generate_vx(duration, rng)
        delta_f = gen.generate_delta_f(duration, rng)
        result = run_sim_v1(vx, delta_f, mu=mu)
        if result is None:
            skipped += 1; continue
        ltr_peak = float(np.max(result['LTRmax']))
        # Rejet strict : LTR > 1.0 est non physique (roue avec Fz négatif).
        # On garde seulement les scénarios physiquement valides.
        if ltr_peak > 1.0:
            skipped += 1; continue
        ltr_peaks.append(ltr_peak)
        gen.save_csv(result, OUTPUT_DIR / f"pr_v1_{seed:08d}.csv")
        generated += 1
        if generated % 50 == 0:
            p = np.array(ltr_peaks)
            print(f"  {generated}/{N_SCENARIOS} — LTR p50={np.percentile(p,50):.2f}, "
                  f"p90={np.percentile(p,90):.2f}, >0.7: {(p>0.7).sum()}", flush=True)

    p = np.array(ltr_peaks) if ltr_peaks else np.array([0.0])
    print(f"\nDone: {generated} generated, {skipped} skipped (attempts: {attempts})")
    print(f"LTR: p10={np.percentile(p,10):.2f}, p50={np.percentile(p,50):.2f}, "
          f"p90={np.percentile(p,90):.2f}, max={p.max():.2f}")


if __name__ == "__main__":
    main()
