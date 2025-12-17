import os
import csv
import numpy as np

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireParams
from controllers import SpeedPIDController, StanleyController
from simulation.closed_loop_runner import ClosedLoopRunner

from trajectories import CircleTrajectory, DoubleLaneChangeTrajectory, SlalomTrajectory
from typing import List, Optional, Dict, Any


# -----------------------------
# Utils
# -----------------------------
def kmh_to_ms(v_kmh: float) -> float:
    return float(v_kmh) / 3.6


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# -----------------------------
# Vehicle config (comme tes exemples)
# -----------------------------
def make_vehicle_model(mu: float = 1.05) -> DOF10:
    params = VehiclePhysicalParams10DOF(
        g=9.81,
        m=1500.0,
        ms=1300.0,
        lf=1.6,
        lr=1.6,
        h=0.55,
        L1=0.75,
        L2=0.75,
        r=0.3,

        ix=400.0,
        iy=1200.0,
        iz=2500.0,
        ir=1.2,

        ra=12.0,
        s=2.2,
        cx=0.32,

        ks1=30000.0,
        ks2=30000.0,
        ks3=30000.0,
        ks4=30000.0,

        ds1=3500.0,
        ds2=3500.0,
        ds3=3500.0,
        ds4=3500.0
    )

    # Charges nominales statiques AV/AR
    L = params.lf + params.lr
    Fzf0 = params.m * params.g * (params.lr / L)  # essieu avant
    Fzr0 = params.m * params.g * (params.lf / L)  # essieu arrière

    fz0_fl = 0.5 * Fzf0
    fz0_fr = 0.5 * Fzf0
    fz0_rl = 0.5 * Fzr0
    fz0_rr = 0.5 * Fzr0

    def make_pacejka_params(fz0_wheel: float) -> SimplifiedPacejkaTireParams:
        Dx0 = mu * fz0_wheel
        Dy0 = mu * fz0_wheel
        return SimplifiedPacejkaTireParams(
            # Lateral
            By=10.0, Cy=1.3, Dy=Dy0, Ey=-1.0,
            # Longitudinal
            Bx=12.0, Cx=1.6, Dx=Dx0, Ex=-0.5,
        )

    tire1 = make_pacejka_params(fz0_fl)  # FL
    tire2 = make_pacejka_params(fz0_fr)  # FR
    tire3 = make_pacejka_params(fz0_rl)  # RL
    tire4 = make_pacejka_params(fz0_rr)  # RR

    config = VehicleConfig10DOF(
        vehicle=params,
        tire1=tire1, tire2=tire2, tire3=tire3, tire4=tire4
    )
    return DOF10(config)


# -----------------------------
# CSV timeseries writer
# -----------------------------
def save_timeseries_csv(
    path: str,
    time: np.ndarray,
    result,
    model,
    traj_name: str,
    v_kmh: float,
    steer_idx=(4, 5),  # <-- AJUSTER SI BESOIN
):
    """
    Sauvegarde séries temporelles dans un CSV (1 ligne = 1 pas de temps):
    - états véhicule sélectionnés
    - commandes braquage avant/arrière (delta_f, delta_r)
    - forces pneus Fx/Fy/Fz (4 roues)
    - meta (traj_name, v_kmh)
    """
    keys = model.state_keys
    x = result.vehicle.x  # (N, nx)
    u = result.vehicle.u  # (N, nu)

    # états qu'on veut sortir (ajoute/enlève des champs ici si tu veux)
    # <-- AJUSTER SI BESOIN (noms dans model.state_keys)
    state_map = {
        "x": keys["x"],
        "y": keys["y"],
        "vx": keys["vx"],
        "vy": keys.get("vy", None),
        "psi": keys.get("psi", keys.get("yaw", None)),
        "phi": keys.get("phi", None),
        "theta": keys.get("theta", None),
    }

    # Forces pneus (doivent exister)
    Fx = result.tires.Fx  # (N,4)
    Fy = result.tires.Fy  # (N,4)
    Fz = result.tires.Fz  # (N,4)

    eps = 1e-12
    LTR_f = (Fz[:, 1] - Fz[:, 0]) / (Fz[:, 1] + Fz[:, 0] + eps)  # FR-FL / (FR+FL)
    LTR_r = (Fz[:, 3] - Fz[:, 2]) / (Fz[:, 3] + Fz[:, 2] + eps)  # RR-RL / (RR+RL)

    delta_f = u[:, steer_idx[0]] if u.shape[1] > steer_idx[0] else np.full_like(time, np.nan)
    delta_r = u[:, steer_idx[1]] if u.shape[1] > steer_idx[1] else np.full_like(time, np.nan)

    header = [
    "time",
    *state_map.keys(),
    "delta_f", "delta_r",
    "LTR_f", "LTR_r",
    "Fx_FL", "Fx_FR", "Fx_RL", "Fx_RR",
    "Fy_FL", "Fy_FR", "Fy_RL", "Fy_RR",
    "Fz_FL", "Fz_FR", "Fz_RL", "Fz_RR",
    "traj_name", "v_kmh",
]


    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        N = len(time)
        for k in range(1, N):
            row = [float(time[k])]

            # états
            for _, idx in state_map.items():
                row.append(float(x[k, idx]) if idx is not None else np.nan)

            # commandes
            row += [float(delta_f[k]), float(delta_r[k])]

            #LTR
            row += [float(LTR_f[k]), float(LTR_r[k])]

            # forces pneus
            row += [float(v) for v in Fx[k]] + [float(v) for v in Fy[k]] + [float(v) for v in Fz[k]]

            # meta
            row += [traj_name, float(v_kmh)]

            w.writerow(row)

# -----------------------------
# Metrics (pour summary.csv)
# -----------------------------
def compute_run_metrics(time: np.ndarray, result, model, traj):
    idx_x = model.state_keys["x"]
    idx_y = model.state_keys["y"]
    idx_psi = model.state_keys.get("psi", model.state_keys.get("yaw", None))

    sim_xy = result.vehicle.x[:, [idx_x, idx_y]]
    sim_psi = result.vehicle.x[:, idx_psi] if idx_psi is not None else None

    ref = traj.sample_array(time)
    ref_xy = ref[:, 1:3]
    ref_psi = ref[:, 3] if ref.shape[1] > 3 else None

    e_xy = sim_xy - ref_xy
    e_norm = np.linalg.norm(e_xy, axis=1)

    rmse_xy = float(np.sqrt(np.mean(e_norm**2)))
    max_xy = float(np.max(e_norm))

    # lat error approx: projection sur normale au cap ref
    if ref_psi is not None:
        nx = -np.sin(ref_psi)
        ny = np.cos(ref_psi)
        e_lat = e_xy[:, 0] * nx + e_xy[:, 1] * ny
        rmse_lat = float(np.sqrt(np.mean(e_lat**2)))
        max_lat = float(np.max(np.abs(e_lat)))
    else:
        rmse_lat = np.nan
        max_lat = np.nan

    if sim_psi is not None and ref_psi is not None:
        dpsi = wrap_pi(sim_psi - ref_psi)
        rmse_psi_deg = float(np.sqrt(np.mean(dpsi**2)) * 180.0 / np.pi)
        max_psi_deg = float(np.max(np.abs(dpsi)) * 180.0 / np.pi)
    else:
        rmse_psi_deg = np.nan
        max_psi_deg = np.nan

    # braquage (indices comme tes plots)
    delta_f = result.vehicle.u[:, 4] if result.vehicle.u.shape[1] > 4 else None
    delta_r = result.vehicle.u[:, 5] if result.vehicle.u.shape[1] > 5 else None
    max_delta_f_deg = float(np.max(np.abs(delta_f)) * 180.0 / np.pi) if delta_f is not None else np.nan
    max_delta_r_deg = float(np.max(np.abs(delta_r)) * 180.0 / np.pi) if delta_r is not None else np.nan

    # forces max (utile debug)
    Fx = result.tires.Fx
    Fy = result.tires.Fy
    Fz = result.tires.Fz
    max_Fx = float(np.max(np.abs(Fx[1:])))
    max_Fy = float(np.max(np.abs(Fy[1:])))
    min_Fz = float(np.min(Fz[1:]))

    return {
        "rmse_xy": rmse_xy,
        "max_xy": max_xy,
        "rmse_lat": rmse_lat,
        "max_lat": max_lat,
        "rmse_psi_deg": rmse_psi_deg,
        "max_psi_deg": max_psi_deg,
        "max_delta_f_deg": max_delta_f_deg,
        "max_delta_r_deg": max_delta_r_deg,
        "max_abs_Fx": max_Fx,
        "max_abs_Fy": max_Fy,
        "min_Fz": min_Fz,
    }


# -----------------------------
# One run
# -----------------------------
def run_one(model, speed_ctrl, steer_ctrl, traj, time_array: np.ndarray, vref_ms: float):
    # init cohérent avec trajectoire (comme ton script cercle)
    p0 = traj.sample(0.0)

    # <-- AJUSTER SI BESOIN : ordre exact de ton état x0 (ici je reprends tes scripts)
    x0 = np.array([
        p0.x, vref_ms,
        p0.y, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        p0.psi, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ], dtype=float)

    runner = ClosedLoopRunner(
        vehicle_model=model,
        speed_controller=speed_ctrl,
        steering_controller=steer_ctrl,
        trajectory=traj,
    )

    result = runner.run(x0, time_array, method="euler")
    return result


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = "datasets/three_trajectories_csv"
    ensure_dir(out_dir)

    # vitesses km/h
    v_kmh_list = list(range(35, 105, 5))  # 35..100
    # horizon & pas simu
    T = 15.0
    dt = 0.0002
    time_array = np.linspace(0.0, T, int(T / dt) + 1)

    # modèle + contrôleurs
    model = make_vehicle_model(mu=1.05)
    speed_ctrl = SpeedPIDController(kp=1000, ki=10, kd=0.0)
    steer_ctrl = StanleyController(k=0.2)  # <-- AJUSTER SI BESOIN

    # cercle: rayon
    R_circle = 50.0

    # summary
    summary_header = [
        "traj_name", "v_kmh", "v_ms",
        "rmse_xy", "max_xy", "rmse_lat", "max_lat",
        "rmse_psi_deg", "max_psi_deg",
        "max_delta_f_deg", "max_delta_r_deg",
        "max_abs_Fx", "max_abs_Fy", "min_Fz",
        "timeseries_csv"
    ]
    summary_rows = []

    for v_kmh in v_kmh_list:
        v_ms = kmh_to_ms(v_kmh)
        print(f"\n=== vref = {v_kmh} km/h ({v_ms:.3f} m/s) ===")

        # fabrique les trajectoires pour cette vitesse (en m/s)
        traj_list = [
            ("circle", lambda: CircleTrajectory(v_ref=v_ms, R=R_circle)),
            ("double_lane_change", lambda: DoubleLaneChangeTrajectory(v_ref=v_ms)),
            ("slalom", lambda: SlalomTrajectory(v_ref=v_ms)),  # si ta signature est SlalomTrajectory(vref), change ici
        ]

        for traj_name, make_traj in traj_list:
            traj = make_traj()

            result = run_one(
                model=model,
                speed_ctrl=speed_ctrl,
                steer_ctrl=steer_ctrl,
                traj=traj,
                time_array=time_array,
                vref_ms=v_ms
            )

            # CSV timeseries
            ts_name = f"{traj_name}_v{v_kmh:03d}kmh_timeseries.csv"
            ts_path = os.path.join(out_dir, ts_name)

            save_timeseries_csv(
                path=ts_path,
                time=time_array,
                result=result,
                model=model,
                traj_name=traj_name,
                v_kmh=v_kmh,
                steer_idx=(4, 5)  # <-- AJUSTER SI BESOIN
            )

            # metrics summary
            m = compute_run_metrics(time_array, result, model, traj)

            summary_rows.append([
                traj_name, v_kmh, f"{v_ms:.6f}",
                f"{m['rmse_xy']:.6f}", f"{m['max_xy']:.6f}",
                f"{m['rmse_lat']:.6f}", f"{m['max_lat']:.6f}",
                f"{m['rmse_psi_deg']:.6f}", f"{m['max_psi_deg']:.6f}",
                f"{m['max_delta_f_deg']:.6f}", f"{m['max_delta_r_deg']:.6f}",
                f"{m['max_abs_Fx']:.3f}", f"{m['max_abs_Fy']:.3f}", f"{m['min_Fz']:.3f}",
                ts_name
            ])

            print(
                f"  {traj_name:18s}  rmse_xy={m['rmse_xy']:.3f} m  "
                f"max_xy={m['max_xy']:.3f} m  "
                f"max|Fy|={m['max_abs_Fy']:.0f} N"
            )

    summary_path = os.path.join(out_dir, "summary.csv")
    write_csv(summary_path, summary_header, summary_rows)

    print(f"\n Fait. Dossier: {out_dir}")
    print(f" Summary: {summary_path}")


if __name__ == "__main__":
    main()
