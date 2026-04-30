#!/usr/bin/env python3
"""Calibre les paramètres Pacejka des pneus à partir des données WFT réelles."""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

DXD_DIR = Path("/Users/zak/Desktop/selected_dxd_json_resampled/")
OUTPUT_DIR = Path(__file__).parent

# Vehicle geometry (from vehicle_params.py)
LF = 1.802
LR = 1.364
L1 = 0.814
L2 = 0.814

DT = 0.01
MIN_VX = 3.0     # m/s — minimum speed for valid slip angle
MAX_ALPHA = 0.25  # rad (~14 deg) — clip extreme values


def pacejka_unsigned(alpha_abs, By, Cy, Ey, mu):
    """Unsigned Pacejka lateral force coefficient: |Fy/Fz| = f(|alpha|)."""
    phi = By * alpha_abs
    return mu * np.sin(Cy * np.arctan(phi - Ey * (phi - np.arctan(phi))))


def residuals(params, alpha_abs, mu_y_abs):
    """Residuals for least_squares fit."""
    By, Cy, Ey, mu = params
    pred = pacejka_unsigned(alpha_abs, By, Cy, Ey, mu)
    return pred - mu_y_abs


def extract_tire_data_from_file(filepath):
    """Extract (alpha, Fy, Fz) per wheel from a DXD JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    ch = data["channels"]

    required = ["VelX (km/h)", "VelY (km/h)", "WheelSteer_S1 (_)",
                "LF_Fy_1 (N)", "LF_Fz_1 (N)", "RF_Fy_2 (N)", "RF_Fz_2 (N)",
                "LR_Fy_3 (N)", "LR_Fz_3 (N)", "RR_Fy_4 (N)", "RR_Fz_4 (N)"]
    for r in required:
        if r not in ch:
            return None

    def get(name):
        vals = ch[name]["values"]
        return np.array([v if v is not None else np.nan for v in vals], dtype=float)

    vx = get("VelX (km/h)") / 3.6
    vy = get("VelY (km/h)") / 3.6
    steer = get("WheelSteer_S1 (_)")

    if "VehYaw_W_Actl (rad/s)" in ch:
        yaw_raw = get("VehYaw_W_Actl (rad/s)")
        if np.nanstd(yaw_raw) > 0.01:
            psi_dot = yaw_raw
        else:
            psi_dot = get("AngVelZ_body (_/s)") * np.pi / 180.0
    elif "AngVelZ_body (_/s)" in ch:
        psi_dot = get("AngVelZ_body (_/s)") * np.pi / 180.0
    else:
        return None

    # Steering sign: DXD positive = right, sim positive = left → negate
    if np.nanmin(steer) >= -0.001:
        delta_f = np.sign(psi_dot) * np.abs(steer)
    else:
        delta_f = -steer

    # Wheel forces
    fy = {
        "FL": get("LF_Fy_1 (N)"), "FR": get("RF_Fy_2 (N)"),
        "RL": get("LR_Fy_3 (N)"), "RR": get("RR_Fy_4 (N)"),
    }
    fz = {
        "FL": get("LF_Fz_1 (N)"), "FR": get("RF_Fz_2 (N)"),
        "RL": get("LR_Fz_3 (N)"), "RR": get("RR_Fz_4 (N)"),
    }

    n = min(len(vx), len(vy), len(psi_dot), len(delta_f),
            *(len(v) for v in fy.values()), *(len(v) for v in fz.values()))

    # Compute slip angles in vehicle frame: alpha = delta - arctan(vy_wheel / vx_wheel)
    # Front wheels
    vy_front = vy[:n] + LF * psi_dot[:n]
    vx_fl = vx[:n] - L1 * psi_dot[:n]
    vx_fr = vx[:n] + L1 * psi_dot[:n]

    alpha_fl = delta_f[:n] - np.arctan2(vy_front, np.maximum(np.abs(vx_fl), 0.5) * np.sign(vx_fl + 1e-10))
    alpha_fr = delta_f[:n] - np.arctan2(vy_front, np.maximum(np.abs(vx_fr), 0.5) * np.sign(vx_fr + 1e-10))

    # Rear wheels (delta_r = 0)
    vy_rear = vy[:n] - LR * psi_dot[:n]
    vx_rl = vx[:n] - L2 * psi_dot[:n]
    vx_rr = vx[:n] + L2 * psi_dot[:n]

    alpha_rl = -np.arctan2(vy_rear, np.maximum(np.abs(vx_rl), 0.5) * np.sign(vx_rl + 1e-10))
    alpha_rr = -np.arctan2(vy_rear, np.maximum(np.abs(vx_rr), 0.5) * np.sign(vx_rr + 1e-10))

    # Filter: minimum speed
    speed_ok = vx[:n] > MIN_VX

    result = {"front": [], "rear": []}
    for wheel, alpha, fy_w, fz_w in [
        ("front", alpha_fl, fy["FL"][:n], fz["FL"][:n]),
        ("front", alpha_fr, fy["FR"][:n], fz["FR"][:n]),
        ("rear", alpha_rl, fy["RL"][:n], fz["RL"][:n]),
        ("rear", alpha_rr, fy["RR"][:n], fz["RR"][:n]),
    ]:
        valid = speed_ok & ~np.isnan(alpha) & ~np.isnan(fy_w) & ~np.isnan(fz_w) & (fz_w > 500)
        valid &= (np.abs(alpha) < MAX_ALPHA)

        if valid.sum() > 0:
            result[wheel].append(np.column_stack([
                np.abs(alpha[valid]),
                np.abs(fy_w[valid]),
                fz_w[valid],
            ]))

    for axle in result:
        if result[axle]:
            result[axle] = np.vstack(result[axle])
        else:
            result[axle] = np.empty((0, 3))

    return result


def fit_pacejka(alpha_abs, mu_y_abs):
    """Fit Pacejka parameters to |mu_y| = f(|alpha|) data."""
    # Initial guess
    x0 = [8.5, 1.3, -1.0, 0.8]
    bounds = ([3, 0.8, -3, 0.3], [20, 2.5, 0.5, 1.5])

    try:
        result = least_squares(residuals, x0, args=(alpha_abs, mu_y_abs),
                               bounds=bounds, method='trf', max_nfev=5000)
        By, Cy, Ey, mu = result.x
        pred = pacejka_unsigned(alpha_abs, By, Cy, Ey, mu)
        ss_res = np.sum((mu_y_abs - pred) ** 2)
        ss_tot = np.sum((mu_y_abs - mu_y_abs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"By": By, "Cy": Cy, "Ey": Ey, "mu": mu, "r2": r2, "n": len(alpha_abs)}
    except Exception as e:
        return {"error": str(e)}


def main():
    print("=" * 70)
    print("TIRE CALIBRATION FROM WFT DATA")
    print("=" * 70)

    all_files = sorted(DXD_DIR.glob("*.json"))
    # Prioritize dynamic tests
    priority = ["slalom", "erdv", "ld_", "sinus", "anr", "vir"]
    dynamic = [f for f in all_files if any(k in f.name for k in priority) and f.stat().st_size < 10_000_000]

    print(f"Loading {len(dynamic)} dynamic test files...")

    front_data = []
    rear_data = []
    loaded = 0

    for i, f in enumerate(dynamic[:150]):
        try:
            result = extract_tire_data_from_file(f)
            if result is None:
                continue
            if len(result["front"]) > 0:
                front_data.append(result["front"])
            if len(result["rear"]) > 0:
                rear_data.append(result["rear"])
            loaded += 1
        except Exception:
            pass

    print(f"Loaded: {loaded} files")

    front_all = np.vstack(front_data) if front_data else np.empty((0, 3))
    rear_all = np.vstack(rear_data) if rear_data else np.empty((0, 3))

    print(f"Front data points: {len(front_all)}")
    print(f"Rear data points: {len(rear_all)}")

    # Fit per axle
    for axle, data in [("FRONT", front_all), ("REAR", rear_all)]:
        if len(data) < 100:
            print(f"\n{axle}: Not enough data")
            continue

        alpha_abs = data[:, 0]
        fy_abs = data[:, 1]
        fz = data[:, 2]
        mu_y = fy_abs / fz

        print(f"\n{'='*50}")
        print(f"{axle} AXLE")
        print(f"{'='*50}")
        print(f"  Points: {len(data)}")
        print(f"  |alpha|: [{alpha_abs.min():.4f}, {alpha_abs.max():.4f}] rad "
              f"({np.degrees(alpha_abs.min()):.1f} to {np.degrees(alpha_abs.max()):.1f} deg)")
        print(f"  Fz: [{fz.min():.0f}, {fz.max():.0f}] N (mean={fz.mean():.0f})")
        print(f"  |mu_y|: [{mu_y.min():.4f}, {mu_y.max():.4f}] (mean={mu_y.mean():.4f})")

        # Global fit (all Fz together)
        print(f"\n  --- Global fit ---")
        result = fit_pacejka(alpha_abs, mu_y)
        if "error" in result:
            print(f"  FIT FAILED: {result['error']}")
        else:
            print(f"  By={result['By']:.2f}, Cy={result['Cy']:.2f}, "
                  f"Ey={result['Ey']:.2f}, mu={result['mu']:.3f}")
            print(f"  R² = {result['r2']:.4f}")

        # Fit per Fz bin
        print(f"\n  --- Per Fz bin ---")
        fz_bins = [(2000, 5000), (5000, 8000), (8000, 12000), (12000, 17000)]
        for fz_lo, fz_hi in fz_bins:
            mask = (fz >= fz_lo) & (fz < fz_hi)
            if mask.sum() < 50:
                print(f"  Fz [{fz_lo}-{fz_hi}] N: {mask.sum()} pts (skip)")
                continue
            r = fit_pacejka(alpha_abs[mask], mu_y[mask])
            if "error" in r:
                print(f"  Fz [{fz_lo}-{fz_hi}] N: FIT FAILED")
            else:
                print(f"  Fz [{fz_lo}-{fz_hi}] N: By={r['By']:.2f} Cy={r['Cy']:.2f} "
                      f"Ey={r['Ey']:.2f} mu={r['mu']:.3f} R²={r['r2']:.3f} (n={r['n']})")

    # Summary
    print(f"\n{'='*70}")
    print("RECOMMENDED PARAMETERS")
    print(f"{'='*70}")
    print("(Compare with current: By=8.5, Cy=1.3, Ey=-1.0, mu=0.8)")

    # Save results
    results_path = OUTPUT_DIR / "tire_calibration_results.json"
    # Re-run fits for saving
    save = {}
    for axle, data in [("front", front_all), ("rear", rear_all)]:
        if len(data) < 100:
            continue
        mu_y = data[:, 1] / data[:, 2]
        r = fit_pacejka(data[:, 0], mu_y)
        save[axle] = r

    with open(results_path, "w") as f:
        json.dump(save, f, indent=2, default=float)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
