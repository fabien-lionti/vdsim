#!/usr/bin/env python3
"""
Generation des scenarios pour l'entrainement du modele de prediction LTR.

Ce script genere des trajectoires de 7 types:
- circle: virage circulaire continu
- single: virage simple
- lemniscate: figure en 8 (ancien "slalom")
- slalom: vrai slalom sinusoidal
- dlc: double lane change (evitement obstacle avec retour)
- waypoint: trajectoires aleatoires par waypoints

Contraintes:
- Distribution par type equilibree
- Distribution LTR equilibree avec quotas par plage
- Maximum 3 pics LTR par scenario
- Majorite des scenarios avec 1 pic (realiste)
- MAX_STEERING_RATE: 25 deg/s
- Braquage lisse (pas d'echelon) grace aux trajectoires smooth

Objectif: ~700 scenarios

Auteur: Zak
Date: Janvier 2025
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireParams
from controllers import SpeedPIDController, StanleyController
from simulation.closed_loop_runner import ClosedLoopRunner
from trajectories import (
    CircleTrajectory,
    LemniscateTrajectory,
    DoubleLaneChangeTrajectory,
    SlalomTrajectory,
    WaypointTrajectory
)

DT = 0.01
T_SIMU = 15.0
MAX_STEERING_RATE = np.radians(25.0) 

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent / "data_new"

# Quotas par type (objectif ~700 scenarios)

TYPE_QUOTAS = {
    "circle": 150,
    "single": 100,
    "lemniscate": 60,    # Figure en 8
    "slalom": 80,        # Vrai slalom sinusoidal
    "dlc": 100,
    "waypoint": 150      # Trajectoires aleatoires par waypoints
}

# Quotas par plage LTR 
LTR_QUOTAS = {
    (0.0, 0.3): 50,
    (0.3, 0.5): 70,
    (0.5, 0.7): 100,
    (0.7, 0.8): 120,
    (0.8, 0.9): 130,
    (0.9, 1.0): 89
}


def make_vehicle(mu: float) -> DOF10:
    """Cree le modele vehicule 10DOF."""
    params = VehiclePhysicalParams10DOF(
        g=9.81, m=1500.0, ms=1300.0,
        lf=1.6, lr=1.6, h=0.55,
        L1=0.75, L2=0.75, r=0.3,
        ix=400.0, iy=1200.0, iz=2500.0, ir=1.2,
        ra=12.0, s=2.2, cx=0.32,
        ks1=30000.0, ks2=30000.0, ks3=30000.0, ks4=30000.0,
        ds1=3500.0, ds2=3500.0, ds3=3500.0, ds4=3500.0
    )

    L = params.lf + params.lr
    Fzf0 = params.m * params.g * (params.lr / L)
    Fzr0 = params.m * params.g * (params.lf / L)

    def make_tire(fz0):
        return SimplifiedPacejkaTireParams(
            By=10.0, Cy=1.3, Dy=mu * fz0, Ey=-1.0,
            Bx=12.0, Cx=1.6, Dx=mu * fz0, Ex=-0.5
        )

    config = VehicleConfig10DOF(
        vehicle=params,
        tire1=make_tire(0.5 * Fzf0),
        tire2=make_tire(0.5 * Fzf0),
        tire3=make_tire(0.5 * Fzr0),
        tire4=make_tire(0.5 * Fzr0)
    )
    return DOF10(config)


def run_simulation(model, traj, v_ref: float):
    """Execute une simulation en boucle fermee."""
    time_array = np.arange(0, T_SIMU + DT, DT)

    speed_ctrl = SpeedPIDController(kp=1000, ki=10, kd=0.0)
    steer_ctrl = StanleyController(k=0.2)  # Comme les anciens scenarios

    runner = ClosedLoopRunner(
        vehicle_model=model,
        speed_controller=speed_ctrl,
        steering_controller=steer_ctrl,
        trajectory=traj
    )

    # Placer le vehicule sur la trajectoire a t=0
    p0 = traj.sample(0.0)
    x0 = np.array([
        p0.x, v_ref,   # x, vx
        p0.y, 0.0,     # y, vy
        0.55, 0.0,     # z, vz (hauteur CG)
        0.0, 0.0,      # phi, phi_dot (roulis)
        0.0, 0.0,      # theta, theta_dot (tangage)
        p0.psi, 0.0,   # psi, psi_dot (lacet)
        0.0, 0.0, 0.0, 0.0  # roues
    ], dtype=float)

    result = runner.run(x0, time_array, method="euler", max_steering_rate=MAX_STEERING_RATE)

    keys = model.state_keys

    # Pas de lissage - garder le braquage brut comme les anciens scenarios
    delta_f_raw = result.vehicle.u[:, 4]

    Fz = result.tires.Fz
    Fx = result.tires.Fx
    Fy = result.tires.Fy
    eps = 1e-12
    LTR_f = (Fz[:, 1] - Fz[:, 0]) / (Fz[:, 1] + Fz[:, 0] + eps)
    LTR_r = (Fz[:, 3] - Fz[:, 2]) / (Fz[:, 3] + Fz[:, 2] + eps)
    LTR_max = np.maximum(np.abs(LTR_f), np.abs(LTR_r))

    return {
        'time': time_array,
        'x': result.vehicle.x[:, keys['x']],
        'y': result.vehicle.x[:, keys['y']],
        'vx': result.vehicle.x[:, keys['vx']],
        'vy': result.vehicle.x[:, keys['vy']],
        'psi': result.vehicle.x[:, keys['psi']],
        'phi': result.vehicle.x[:, keys['phi']],
        'theta': result.vehicle.x[:, keys['theta']],
        'psi_dot': result.vehicle.x[:, keys['psidt']],
        'delta_f': delta_f_raw,
        'delta_r': np.zeros(len(time_array)),  # Pas de braquage arriere
        'LTR_avant': LTR_f,
        'LTR_arriere': LTR_r,
        'LTRmax': LTR_max,
        'Fx_FL': Fx[:, 0],
        'Fx_FR': Fx[:, 1],
        'Fx_RL': Fx[:, 2],
        'Fx_RR': Fx[:, 3],
        'Fy_FL': Fy[:, 0],
        'Fy_FR': Fy[:, 1],
        'Fy_RL': Fy[:, 2],
        'Fy_RR': Fy[:, 3],
        'Fz_FL': Fz[:, 0],
        'Fz_FR': Fz[:, 1],
        'Fz_RL': Fz[:, 2],
        'Fz_RR': Fz[:, 3]
    }


def save_csv(data: dict, filepath: Path):
    """Sauvegarde les donnees en CSV (26 colonnes comme raw_v8)."""
    cols = [
        'time', 'x', 'y', 'vx', 'vy', 'psi', 'phi', 'theta', 'psi_dot',
        'delta_f', 'delta_r', 'LTR_avant', 'LTR_arriere', 'LTRmax',
        'Fx_FL', 'Fx_FR', 'Fx_RL', 'Fx_RR',
        'Fy_FL', 'Fy_FR', 'Fy_RL', 'Fy_RR',
        'Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR'
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        # Ignorer t=0 (NaN), commencer a t=0.01 comme raw_v8
        for i in range(1, len(data['time'])):
            row = [data[c][i] for c in cols]
            writer.writerow(row)


def count_peaks(ltr, threshold=0.4):
    """Compte le nombre de pics significatifs au-dessus du seuil."""
    above = np.abs(ltr) > threshold
    crossings = np.diff(above.astype(int))
    return max(1, np.sum(crossings == 1))


def get_ltr_bin(ltr_max):
    """Retourne la plage LTR pour un scenario."""
    for (lo, hi) in LTR_QUOTAS.keys():
        if lo <= ltr_max < hi:
            return (lo, hi)
    return None


def is_valid(data: dict, max_peaks: int = 3) -> tuple:
    """
    Verifie si le scenario est valide:
    - Pas de NaN apres la premiere seconde
    - LTR max < 1.0 (pas de renversement)
    - Maximum N pics

    Retourne (valide, raison, ltr_max, n_peaks, ltr_bin)
    """
    start = 100  # Ignorer premiere seconde
    ltr = data['LTRmax'][start:]
    delta = data['delta_f'][start:]

    if np.any(np.isnan(ltr)):
        return False, "NaN LTR", 0, 0, None
    if np.any(np.isnan(delta)):
        return False, "NaN delta", 0, 0, None

    ltr_max = np.max(ltr)
    if ltr_max > 1.0:
        return False, f"LTR>1.0 ({ltr_max:.2f})", ltr_max, 0, None

    n_peaks = count_peaks(ltr)
    if n_peaks > max_peaks:
        return False, f"Trop de pics ({n_peaks})", ltr_max, n_peaks, None

    ltr_bin = get_ltr_bin(ltr_max)
    if ltr_bin is None:
        return False, "LTR hors plage", ltr_max, n_peaks, None

    return True, "OK", ltr_max, n_peaks, ltr_bin


def generate_all_scenarios():
    """Genere tous les scenarios avec controle des quotas par type et LTR."""
    print("=" * 60)
    print("GENERATION DES SCENARIOS LTR")
    print("=" * 60)
    print(f"Sortie: {OUTPUT_DIR}")
    print(f"Vitesse de braquage max: {np.degrees(MAX_STEERING_RATE):.0f} deg/s")

    print(f"\nQuotas par type:")
    for t, q in TYPE_QUOTAS.items():
        print(f"  {t}: {q}")

    print(f"\nQuotas par plage LTR:")
    for (lo, hi), quota in LTR_QUOTAS.items():
        print(f"  [{lo:.1f}, {hi:.1f}): {quota}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compteurs
    type_counts = defaultdict(int)
    bin_counts = defaultdict(int)
    peak_counts = defaultdict(int)
    total_tested = 0
    total_generated = 0

    # Quotas par bin LTR par type (repartition equilibree)
    type_bin_quotas = {
        "circle": {(0.0,0.3): 15, (0.3,0.5): 20, (0.5,0.7): 30, (0.7,0.8): 30, (0.8,0.9): 30, (0.9,1.0): 25},
        "single": {(0.0,0.3): 10, (0.3,0.5): 15, (0.5,0.7): 20, (0.7,0.8): 20, (0.8,0.9): 20, (0.9,1.0): 15},
        "lemniscate": {(0.0,0.3): 5, (0.3,0.5): 8, (0.5,0.7): 12, (0.7,0.8): 12, (0.8,0.9): 12, (0.9,1.0): 11},
        "slalom": {(0.0,0.3): 8, (0.3,0.5): 12, (0.5,0.7): 15, (0.7,0.8): 15, (0.8,0.9): 15, (0.9,1.0): 15},
        "dlc": {(0.0,0.3): 10, (0.3,0.5): 15, (0.5,0.7): 20, (0.7,0.8): 20, (0.8,0.9): 20, (0.9,1.0): 15},
        "waypoint": {(0.0,0.3): 15, (0.3,0.5): 20, (0.5,0.7): 30, (0.7,0.8): 30, (0.8,0.9): 30, (0.9,1.0): 25}
    }
    type_bin_counts = {t: defaultdict(int) for t in TYPE_QUOTAS.keys()}

    # Parametres optimises pour differentes plages de LTR
    # ay = v^2/R determine le LTR: ay>10 => LTR>0.9, ay~6-8 => LTR~0.7, ay<3 => LTR<0.3
    directions = [-1, 1]
    mu_values = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    T_ramps = [3, 4, 5]

    def try_scenario(traj_type, model, traj, v_ref, filename, target_bin=None):
        nonlocal total_tested, total_generated

        # Verifier quota type global
        if type_counts[traj_type] >= TYPE_QUOTAS.get(traj_type, 0):
            return False, None

        # Verifier quota bin si specifie
        if target_bin and type_bin_counts[traj_type][target_bin] >= type_bin_quotas[traj_type].get(target_bin, 0):
            return False, None

        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            return True, None

        total_tested += 1
        try:
            data = run_simulation(model, traj, v_ref)
            valid, reason, ltr_max, n_peaks, ltr_bin = is_valid(data)

            if valid:
                # Verifier quota bin
                if type_bin_counts[traj_type][ltr_bin] >= type_bin_quotas[traj_type].get(ltr_bin, 999):
                    return False, ltr_bin  # Retourner le bin pour info

                save_csv(data, filepath)
                bin_counts[ltr_bin] += 1
                type_bin_counts[traj_type][ltr_bin] += 1
                type_counts[traj_type] += 1
                peak_counts[n_peaks] += 1
                total_generated += 1

                if total_generated % 50 == 0:
                    print(f"  {total_generated} scenarios generes...")
                return True, ltr_bin
        except Exception as e:
            pass
        return False, None

    # ===== CIRCLES =====
    # Parametres organises par plage de LTR cible (basé sur ay = v²/R)
    # ay < 3 => LTR < 0.3
    # ay 3-5 => LTR 0.3-0.5
    # ay 5-7 => LTR 0.5-0.7
    # ay 7-9 => LTR 0.7-0.8
    # ay 9-11 => LTR 0.8-0.9
    # ay > 11 => LTR 0.9-1.0
    circle_params = [
        # (v, R) pour ay faible (LTR < 0.5)
        (10, 60), (10, 50), (10, 44), (12, 60), (12, 50), (10, 36), (12, 44),
        # ay moyen-faible (LTR 0.5-0.7)
        (14, 44), (14, 36), (16, 44), (16, 40), (18, 52), (18, 48), (20, 60),
        # ay moyen (LTR 0.7-0.8)
        (14, 28), (16, 32), (18, 36), (20, 44), (22, 52), (24, 60), (26, 68),
        # ay moyen-haut (LTR 0.8-0.9)
        (14, 24), (16, 28), (18, 32), (20, 36), (22, 44), (24, 52), (26, 60),
        # ay haut (LTR 0.9+)
        (14, 16), (14, 20), (16, 24), (18, 28), (20, 32), (22, 40), (24, 48), (26, 56),
    ]
    print("\nGeneration cercles...")
    for v, R in circle_params:
        if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
            break
        for d in directions:
            if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
                break
            for mu in mu_values:
                if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
                    break
                mu_int = int(round(mu * 100))
                filename = f"circle_v{v}_R{R}_d{d}_mu{mu_int}.csv"
                model = make_vehicle(mu)
                clockwise = (d == -1)
                traj = CircleTrajectory(v_ref=float(v), R=float(R), clockwise=clockwise, L_straight=20.0)
                try_scenario("circle", model, traj, float(v), filename)

    print(f"  Cercles: {type_counts['circle']}")

    # ===== SINGLE TURNS =====
    # Parametres pour single - virages simples avec entree longue
    single_params = [
        # ay faible (LTR < 0.5)
        (10, 50), (10, 45), (10, 40), (12, 55), (12, 50), (12, 45),
        # ay moyen-faible (LTR 0.5-0.7)
        (12, 35), (14, 40), (14, 35), (16, 45), (16, 40), (18, 50),
        # ay moyen (LTR 0.7-0.8)
        (14, 28), (16, 32), (18, 38), (18, 35), (20, 42), (20, 38),
        # ay moyen-haut (LTR 0.8-0.9)
        (14, 22), (14, 20), (16, 26), (18, 30), (20, 36), (22, 42),
        # ay haut (LTR 0.9+)
        (12, 16), (14, 18), (16, 22), (18, 26), (20, 32), (22, 38),
    ]
    print("\nGeneration virages simples...")
    for v, R in single_params:
        if type_counts["single"] >= TYPE_QUOTAS["single"]:
            break
        for d in directions:
            if type_counts["single"] >= TYPE_QUOTAS["single"]:
                break
            for mu in mu_values:
                if type_counts["single"] >= TYPE_QUOTAS["single"]:
                    break
                mu_int = int(round(mu * 100))
                filename = f"single_v{v}_R{R}_d{d}_mu{mu_int}.csv"
                model = make_vehicle(mu)
                clockwise = (d == -1)
                traj = CircleTrajectory(v_ref=float(v), R=float(R), clockwise=clockwise, L_straight=50.0)
                try_scenario("single", model, traj, float(v), filename)

    print(f"  Single: {type_counts['single']}")

    # ===== LEMNISCATE (Figure en 8) =====
    # Ancien "slalom" - maintenant correctement nomme
    lemniscate_params = [
        # (v, a, n_periods) - vitesse, amplitude, nombre de 8 par simulation
        # LTR faible (<0.5)
        (6, 2, 1.0), (8, 2, 1.0), (8, 2, 1.5), (10, 2, 1.0),
        # LTR moyen-faible (0.5-0.7)
        (8, 3, 1.0), (10, 2, 1.5), (10, 3, 1.0), (12, 2, 1.0),
        # LTR moyen (0.7-0.8)
        (10, 3, 1.5), (10, 4, 1.0), (12, 3, 1.0), (12, 3, 1.5),
        # LTR moyen-haut (0.8-0.9) - attention rollover
        (10, 4, 1.5), (12, 4, 1.0), (8, 5, 1.0),
        # LTR haut (0.9+) - tres risque
        (10, 5, 1.0), (12, 4, 1.5),
    ]
    print("\nGeneration lemniscates (figure-8)...")
    for v, a, n in lemniscate_params:
        if type_counts["lemniscate"] >= TYPE_QUOTAS["lemniscate"]:
            break
        for mu in mu_values:
            if type_counts["lemniscate"] >= TYPE_QUOTAS["lemniscate"]:
                break
            mu_int = int(round(mu * 100))
            filename = f"lemniscate_v{v}_a{a}_n{int(n*10)}_mu{mu_int}.csv"
            model = make_vehicle(mu)
            T_period = T_SIMU / n
            A = float(a * 3)
            B = float(a * 3)
            traj = LemniscateTrajectory(v_ref=float(v), A=A, B=B, T_period=T_period, L_straight=10.0)
            try_scenario("lemniscate", model, traj, float(v), filename)

    print(f"  Lemniscates: {type_counts['lemniscate']}")

    # ===== SLALOM SINUSOIDAL (vrai slalom) =====
    # Utilise SlalomTrajectory - oscillations sinusoidales autour d'une ligne droite
    slalom_params = [
        # (v, L, A, n_waves) - vitesse, longueur, amplitude, nombre d'oscillations
        # LTR faible (<0.5)
        (10, 200, 2.0, 2), (12, 200, 2.0, 2), (10, 180, 2.5, 2), (12, 180, 2.5, 2),
        (8, 150, 2.0, 3), (10, 150, 2.0, 3),
        # LTR moyen-faible (0.5-0.7)
        (12, 180, 3.0, 2), (14, 200, 2.5, 2), (14, 180, 3.0, 2), (16, 200, 2.5, 2),
        (10, 150, 3.0, 3), (12, 150, 3.0, 3), (10, 120, 2.5, 3),
        # LTR moyen (0.7-0.8)
        (14, 160, 3.5, 2), (16, 180, 3.0, 2), (16, 160, 3.5, 2), (18, 200, 3.0, 2),
        (12, 120, 3.5, 3), (14, 150, 3.0, 3), (12, 100, 3.0, 3),
        # LTR moyen-haut (0.8-0.9)
        (16, 140, 4.0, 2), (18, 160, 3.5, 2), (18, 140, 4.0, 2), (20, 180, 3.5, 2),
        (14, 120, 4.0, 3), (16, 140, 3.5, 3),
        # LTR haut (0.9+)
        (18, 120, 4.5, 2), (20, 150, 4.0, 2), (20, 130, 4.5, 2),
        (16, 100, 4.5, 3), (18, 120, 4.0, 3),
    ]
    print("\nGeneration slaloms sinusoidaux...")
    for v, L, A, n_waves in slalom_params:
        if type_counts["slalom"] >= TYPE_QUOTAS["slalom"]:
            break
        for mu in mu_values:
            if type_counts["slalom"] >= TYPE_QUOTAS["slalom"]:
                break
            mu_int = int(round(mu * 100))
            filename = f"slalom_v{v}_L{L}_A{int(A*10)}_n{n_waves}_mu{mu_int}.csv"
            model = make_vehicle(mu)
            traj = SlalomTrajectory(
                v_ref=float(v),
                L=float(L),
                A=float(A),
                n_waves=n_waves,
                L_straight=20.0,
                T_ramp=1.0
            )
            try_scenario("slalom", model, traj, float(v), filename)

    print(f"  Slaloms: {type_counts['slalom']}")

    # ===== DOUBLE LANE CHANGE =====
    # Parametres: (v, A, L_total) - vitesse, amplitude laterale, longueur totale
    # LTR depend de la courbure: plus v est grand et L_total petit, plus LTR est eleve
    # A = amplitude laterale (typiquement 2-5m)
    # L_total = longueur de la manoeuvre (60-150m)
    dlc_params = [
        # LTR faible (<0.5) - manoeuvres douces, longues
        (12, 2.0, 150), (14, 2.0, 150), (12, 2.5, 140), (14, 2.5, 140), (16, 2.0, 150), (16, 2.5, 140),
        (10, 2.0, 120), (10, 2.5, 120), (12, 2.0, 120), (14, 2.0, 130),
        # LTR moyen-faible (0.5-0.7)
        (16, 3.0, 130), (18, 2.5, 130), (18, 3.0, 120), (20, 2.5, 130), (20, 3.0, 120),
        (14, 3.0, 110), (16, 3.0, 110), (18, 3.0, 110), (12, 3.0, 100), (14, 3.5, 110),
        # LTR moyen (0.7-0.8)
        (18, 3.5, 100), (20, 3.0, 100), (20, 3.5, 100), (22, 3.0, 110), (22, 3.5, 100),
        (16, 3.5, 90), (18, 3.5, 90), (20, 3.5, 90), (24, 3.0, 110), (24, 3.5, 100),
        # LTR moyen-haut (0.8-0.9)
        (20, 4.0, 90), (22, 3.5, 90), (22, 4.0, 85), (24, 3.5, 90), (24, 4.0, 85),
        (18, 4.0, 80), (20, 4.0, 80), (22, 4.0, 80), (26, 3.5, 95), (26, 4.0, 90),
        # LTR haut (0.9+) - manoeuvres agressives
        (22, 4.5, 75), (24, 4.0, 75), (24, 4.5, 70), (26, 4.0, 80), (26, 4.5, 75),
        (20, 4.5, 70), (22, 4.5, 70), (24, 4.5, 65), (28, 4.0, 85), (28, 4.5, 80),
    ]
    print("\nGeneration Double Lane Change...")
    for v, A, L in dlc_params:
        if type_counts["dlc"] >= TYPE_QUOTAS["dlc"]:
            break
        for mu in mu_values:
            if type_counts["dlc"] >= TYPE_QUOTAS["dlc"]:
                break
            mu_int = int(round(mu * 100))
            filename = f"dlc_v{v}_A{int(A*10)}_L{L}_mu{mu_int}.csv"
            model = make_vehicle(mu)
            traj = DoubleLaneChangeTrajectory(v_ref=float(v), A=float(A), L_total=float(L))
            try_scenario("dlc", model, traj, float(v), filename)

    print(f"  DLC: {type_counts['dlc']}")

    # ===== WAYPOINT (trajectoires aleatoires) =====
    # Genere des trajectoires aleatoires par waypoints et garde celles avec LTR raisonnable
    # PAS de quota par bin LTR - on accepte tout ce qui est valide
    print("\nGeneration waypoints aleatoires...")
    np.random.seed(42)  # Reproductibilite

    waypoint_generated = 0
    waypoint_attempts = 0
    max_waypoint_attempts = 1000

    def try_waypoint(traj, v_ref, filename, mu):
        """Essaie un waypoint sans contrainte de bin LTR."""
        nonlocal waypoint_generated, total_tested, total_generated

        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            return True

        total_tested += 1
        try:
            data = run_simulation(model, traj, v_ref)
            valid, reason, ltr_max, n_peaks, ltr_bin = is_valid(data)

            if valid and ltr_max < 0.98:  # Accepter tout LTR < 0.98
                save_csv(data, filepath)
                bin_counts[ltr_bin] += 1
                type_bin_counts["waypoint"][ltr_bin] += 1
                type_counts["waypoint"] += 1
                peak_counts[n_peaks] += 1
                total_generated += 1
                waypoint_generated += 1
                return True
        except Exception:
            pass
        return False

    while waypoint_generated < TYPE_QUOTAS["waypoint"] and waypoint_attempts < max_waypoint_attempts:
        waypoint_attempts += 1

        # Parametres aleatoires avec plus de variete
        v_ref = np.random.uniform(8, 24)  # Plus large: 8-24 m/s
        mu = np.random.choice(mu_values)
        mu_int = int(round(mu * 100))

        # Type de trajectoire aleatoire
        traj_type = np.random.choice(["scurve", "chicane", "evitement", "ondulation", "zigzag"])

        t_points = np.linspace(0, T_SIMU, 100)
        x_points = v_ref * t_points

        if traj_type == "scurve":
            amp = np.random.uniform(1.5, 7)
            freq = np.random.uniform(0.3, 2.0)
            y_points = amp * np.sin(2 * np.pi * freq * t_points / T_SIMU)
            filename = f"waypoint_scurve_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}.csv"

        elif traj_type == "chicane":
            amp = np.random.uniform(1.5, 6)
            t_chicane = np.random.uniform(2, 7)
            trans_time = np.random.uniform(1, 3)
            y_points = np.zeros_like(t_points)
            mask1 = (t_points > t_chicane) & (t_points < t_chicane + trans_time)
            mask2 = t_points >= t_chicane + trans_time
            y_points[mask1] = amp * (t_points[mask1] - t_chicane) / trans_time
            y_points[mask2] = amp
            if np.random.random() > 0.4:
                t_return = t_chicane + trans_time + np.random.uniform(1, 3)
                if t_return + trans_time < T_SIMU:
                    mask3 = (t_points > t_return) & (t_points < t_return + trans_time)
                    mask4 = t_points >= t_return + trans_time
                    y_points[mask3] = amp - amp * (t_points[mask3] - t_return) / trans_time
                    y_points[mask4] = 0
            filename = f"waypoint_chicane_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}.csv"

        elif traj_type == "evitement":
            amp = np.random.uniform(2, 7)
            t_start = np.random.uniform(1, 6)
            t_dur = np.random.uniform(0.8, 2.5)
            y_points = np.zeros_like(t_points)
            mask = (t_points > t_start) & (t_points < t_start + t_dur)
            y_points[mask] = amp * np.sin(np.pi * (t_points[mask] - t_start) / t_dur)
            y_points[t_points >= t_start + t_dur] = 0
            filename = f"waypoint_evitement_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}.csv"

        elif traj_type == "zigzag":
            # Zigzag: changements de direction multiples
            n_turns = np.random.randint(2, 5)
            amp = np.random.uniform(2, 5)
            y_points = np.zeros_like(t_points)
            turn_times = np.sort(np.random.uniform(2, T_SIMU-2, n_turns))
            current_y = 0
            direction = np.random.choice([-1, 1])
            for i, tt in enumerate(turn_times):
                if i < len(turn_times) - 1:
                    mask = (t_points >= tt) & (t_points < turn_times[i+1])
                else:
                    mask = t_points >= tt
                target_y = direction * amp
                y_points[mask] = current_y + (target_y - current_y) * (t_points[mask] - tt) / 1.5
                y_points[mask] = np.clip(y_points[mask], -amp, amp)
                current_y = target_y
                direction *= -1
            filename = f"waypoint_zigzag_{waypoint_attempts}_v{int(v_ref)}_n{n_turns}_a{int(amp*10)}_mu{mu_int}.csv"

        else:  # ondulation
            n_waves = np.random.randint(1, 6)
            amp = np.random.uniform(1.5, 6)
            y_points = amp * np.sin(2 * np.pi * n_waves * t_points / T_SIMU)
            filename = f"waypoint_ondulation_{waypoint_attempts}_v{int(v_ref)}_n{n_waves}_a{int(amp*10)}_mu{mu_int}.csv"

        try:
            traj = WaypointTrajectory(t_points, x_points, y_points)
            model = make_vehicle(mu)
            try_waypoint(traj, v_ref, filename, mu)
        except Exception:
            pass

        if waypoint_attempts % 100 == 0:
            print(f"    Waypoints: {waypoint_generated}/{TYPE_QUOTAS['waypoint']} (essai {waypoint_attempts})")

    print(f"  Waypoints: {waypoint_generated} (sur {waypoint_attempts} essais)")

    # Rapport final
    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)
    print(f"Scenarios testes: {total_tested}")
    print(f"Scenarios generes: {total_generated}")

    print(f"\nPar type:")
    for t in ["circle", "single", "lemniscate", "slalom", "dlc", "waypoint"]:
        quota = TYPE_QUOTAS.get(t, 0)
        count = type_counts[t]
        status = "OK" if count >= quota else f"MANQUE {quota - count}"
        print(f"  {t}: {count}/{quota} - {status}")

    print(f"\nDistribution LTR:")
    for (lo, hi), quota in LTR_QUOTAS.items():
        count = bin_counts[(lo, hi)]
        print(f"  [{lo:.1f}, {hi:.1f}): {count}")

    print(f"\nDistribution des pics:")
    for p in sorted(peak_counts.keys()):
        count = peak_counts[p]
        pct = 100 * count / total_generated if total_generated > 0 else 0
        print(f"  {p} pic(s): {count} ({pct:.1f}%)")

    return total_generated


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    total = generate_all_scenarios()
    print(f"\nTermine: {total} scenarios generes")
