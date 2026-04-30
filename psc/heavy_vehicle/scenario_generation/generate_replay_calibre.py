#!/usr/bin/env python3
"""Replay DXD avec DOF10 calibré (make_heavy_vehicle, sans relax/Kamm).
Sortie: data_replay_calibre/."""

import sys, json, csv, numpy as np
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from psc.heavy_vehicle.params.vehicle_params import make_heavy_vehicle

DXD_DIR = Path("/Users/zak/Desktop/selected_dxd_json_resampled/")
OUTPUT_DIR = SCRIPT_DIR / "data_replay_calibre"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DT = 0.01

def load_dxd_inputs(filepath):
    with open(filepath) as f: data = json.load(f)
    ch = data["channels"]
    if "VelX (km/h)" not in ch or "WheelSteer_S1 (_)" not in ch: return None
    def get(name):
        return np.array([v if v is not None else np.nan for v in ch[name]["values"]], dtype=float)
    vx = get("VelX (km/h)") / 3.6
    steer_raw = get("WheelSteer_S1 (_)")
    if "VehYaw_W_Actl (rad/s)" in ch:
        yaw_raw = get("VehYaw_W_Actl (rad/s)")
        psi_dot = yaw_raw if np.nanstd(yaw_raw) > 0.01 else get("AngVelZ_body (_/s)") * np.pi/180
    elif "AngVelZ_body (_/s)" in ch:
        psi_dot = get("AngVelZ_body (_/s)") * np.pi/180
    else: return None
    if np.nanmin(steer_raw) >= -0.001:
        delta_f = np.sign(psi_dot) * np.abs(steer_raw)
    else:
        delta_f = -steer_raw
    n = min(len(vx), len(delta_f))
    valid = ~np.isnan(vx[:n]) & ~np.isnan(delta_f[:n])
    if valid.sum() < 500: return None
    idx = np.where(valid)[0]; s, e = idx[0], idx[-1] + 1
    return {'vx': vx[s:e], 'delta_f': delta_f[s:e]}

def run_replay_v1(inputs, mu=0.8):
    """DOF10 calibré (16 états)."""
    model = make_heavy_vehicle(mu)
    n = len(inputs["vx"])
    x = np.zeros(16); x[1] = inputs["vx"][0]; x[4] = model.h
    w0 = max(inputs["vx"][0], 1.0) / model.r; x[12:16] = w0
    vx_out = np.zeros(n); vy_out = np.zeros(n); psi_dot_out = np.zeros(n)
    phi_out = np.zeros(n); theta_out = np.zeros(n); delta_f_out = np.zeros(n)
    ltr_f = np.zeros(n); ltr_r = np.zeros(n)
    kp, ki = 2100.0, 21.0; integral = 0.0
    for i in range(n):
        vx_t = inputs["vx"][i]; df = inputs["delta_f"][i]
        err = vx_t - x[1]; integral += err * DT
        torque = np.clip(kp*err + ki*integral, -8500, 8500) / 4.0
        u = np.array([torque, torque, torque, torque, df, 0.0])
        try: dx, out = model.get_dx__dt(x, u)
        except Exception: return None
        vx_out[i]=x[1]; vy_out[i]=x[3]; psi_dot_out[i]=x[11]
        phi_out[i]=x[8]; theta_out[i]=x[6]; delta_f_out[i]=df
        fz = out["Fz"]; eps = 1e-12
        ltr_f[i] = abs((fz[1]-fz[0])/(fz[1]+fz[0]+eps))
        ltr_r[i] = abs((fz[3]-fz[2])/(fz[3]+fz[2]+eps))
        x = x + dx * DT
        if np.any(np.isnan(x)) or np.any(np.abs(x) > 1e8): return None
    delta_f_dot = np.gradient(delta_f_out, DT); ltr_max = np.maximum(ltr_f, ltr_r)
    return {'time': np.arange(n)*DT, 'vx': vx_out, 'vy': vy_out, 'psi_dot': psi_dot_out,
            'phi': phi_out, 'theta': theta_out, 'delta_f': delta_f_out, 'delta_f_dot': delta_f_dot,
            'LTR_avant': ltr_f, 'LTR_arriere': ltr_r, 'LTRmax': ltr_max}

def save_csv(data, fp):
    cols = ['time','vx','vy','psi_dot','phi','theta','delta_f','delta_f_dot','LTR_avant','LTR_arriere','LTRmax']
    with open(fp, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(len(data['time'])): w.writerow([data[c][i] for c in cols])

def main():
    print("=" * 60)
    print("REPLAY DXD avec DOF10 CALIBRÉ")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    files = sorted(DXD_DIR.glob("*.json"))
    print(f"\n{len(files)} DXD files")
    gen, skip = 0, 0
    for i, f in enumerate(files):
        if f.stat().st_size > 15_000_000: skip += 1; continue
        inputs = load_dxd_inputs(f)
        if inputs is None: skip += 1; continue
        r = run_replay_v1(inputs, mu=0.8)
        if r is None: skip += 1; continue
        save_csv(r, OUTPUT_DIR / f"replay_{f.stem}.csv")
        gen += 1
        if gen % 25 == 0:
            print(f"  {gen} generated ({i+1}/{len(files)} processed), peak LTR: {np.max(r['LTRmax']):.3f}", flush=True)
    print(f"\nDone: {gen} generated, {skip} skipped")

if __name__ == "__main__":
    main()
