#!/usr/bin/env python3
"""Génère les scénarios V2 pour l'entraînement du modèle de prédiction LTR."""

import os
import sys
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "params"))

from vehicle_params import make_heavy_vehicle, H, PID_KP, PID_KI, TORQUE_LIMITS
from controllers import SpeedPIDController, StanleyController
from simulation.closed_loop_runner import ClosedLoopRunner
from trajectories import (
    CircleTrajectory,
    LemniscateTrajectory,
    DoubleLaneChangeTrajectory,
    SlalomTrajectory,
    WaypointTrajectory,
    SmoothRandomTrajectory,
)

# ============== Configuration ==============

DT = 0.01
T_SIMU = 15.0
MAX_STEERING_RATE = np.radians(25.0)

# Configuration du timing des pics LTR
# Pour scenarios AVEC warm-up: on veut que le pic arrive apres MIN_TIME_TO_PEAK
MIN_TIME_TO_PEAK = 4.0  # secondes (1.5s input + marge pour prediction)
LTR_EARLY_THRESHOLD = 0.5  # Seuil pour detecter un pic "trop tot"

# Proportion de scenarios SANS warm-up (LTR peut etre eleve des le debut)
# Ces scenarios sont utiles pour apprendre "LTR est deja haut"
RATIO_NO_WARMUP = 0.35  # 35% sans warm-up, 65% avec warm-up

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "data"

# Quotas par type (objectif ~750 scenarios)
TYPE_QUOTAS = {
    "circle": 150,
    "single": 120,
    "lemniscate": 80,
    "slalom": 100,
    "dlc": 120,
    "waypoint": 180,
    "smooth_random": 200,
}

# Quotas globaux par plage LTR (augmentes pour ne pas bloquer la generation)
# Les quotas servent principalement a garantir une distribution minimale,
# pas a limiter strictement
LTR_QUOTAS = {
    (0.0, 0.3): 100,
    (0.3, 0.5): 130,
    (0.5, 0.7): 170,
    (0.7, 0.8): 150,
    (0.8, 0.9): 130,
    (0.9, 1.0): 80,   # Plus difficile à atteindre sans rollover sur véhicule lourd
}

# Longueurs de ligne droite initiale
L_STRAIGHT_LONG = 80.0   # metres (~5s a 16m/s) - avec warm-up
L_STRAIGHT_SHORT = 10.0  # metres (~0.6s) - sans warm-up (demarrage rapide)


# make_vehicle remplacé par make_heavy_vehicle importé depuis vehicle_params.py


def run_simulation(model, traj, v_ref: float):
    """Execute une simulation en boucle fermee."""
    time_array = np.arange(0, T_SIMU + DT, DT)

    speed_ctrl = SpeedPIDController(kp=PID_KP, ki=PID_KI, kd=0.0)
    steer_ctrl = StanleyController(k=0.2)

    runner = ClosedLoopRunner(
        vehicle_model=model,
        speed_controller=speed_ctrl,
        steering_controller=steer_ctrl,
        trajectory=traj
    )

    p0 = traj.sample(0.0)
    x0 = np.array([
        p0.x, v_ref,
        p0.y, 0.0,
        H, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        p0.psi, 0.0,
        0.0, 0.0, 0.0, 0.0
    ], dtype=float)

    result = runner.run(x0, time_array, method="euler", max_steering_rate=MAX_STEERING_RATE)

    keys = model.state_keys
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
        'delta_r': np.zeros(len(time_array)),
        'LTR_avant': LTR_f,
        'LTR_arriere': LTR_r,
        'LTRmax': LTR_max,
        'Fx_FL': Fx[:, 0], 'Fx_FR': Fx[:, 1], 'Fx_RL': Fx[:, 2], 'Fx_RR': Fx[:, 3],
        'Fy_FL': Fy[:, 0], 'Fy_FR': Fy[:, 1], 'Fy_RL': Fy[:, 2], 'Fy_RR': Fy[:, 3],
        'Fz_FL': Fz[:, 0], 'Fz_FR': Fz[:, 1], 'Fz_RL': Fz[:, 2], 'Fz_RR': Fz[:, 3]
    }


def save_csv(data: dict, filepath: Path):
    """Sauvegarde les donnees en CSV."""
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


def get_time_to_high_ltr(data: dict, threshold: float = LTR_EARLY_THRESHOLD) -> float:
    """
    Retourne le temps (en secondes) avant que LTR depasse le seuil pour la premiere fois.
    Retourne T_SIMU si jamais depasse.
    """
    ltr = data['LTRmax']
    time = data['time']

    indices = np.where(np.abs(ltr) > threshold)[0]
    if len(indices) == 0:
        return T_SIMU

    return time[indices[0]]


def is_valid(data: dict, max_peaks: int = 3, allow_early_peak: bool = False) -> tuple:
    """
    Verifie si le scenario est valide:
    - Pas de NaN apres la premiere seconde
    - LTR max < 1.0 (pas de renversement)
    - Maximum N pics
    - Pic LTR pas trop tot (sauf si allow_early_peak=True)

    Args:
        data: donnees de simulation
        max_peaks: nombre max de pics autorises
        allow_early_peak: si True, accepte les scenarios ou LTR est eleve des le debut

    Retourne (valide, raison, ltr_max, n_peaks, ltr_bin, time_to_peak)
    """
    start = 100  # Ignorer premiere seconde
    ltr = data['LTRmax'][start:]
    delta = data['delta_f'][start:]

    if np.any(np.isnan(ltr)):
        return False, "NaN LTR", 0, 0, None, 0
    if np.any(np.isnan(delta)):
        return False, "NaN delta", 0, 0, None, 0

    ltr_max = np.max(ltr)
    if ltr_max > 1.0:
        return False, f"LTR>1.0 ({ltr_max:.2f})", ltr_max, 0, None, 0

    n_peaks = count_peaks(ltr)
    if n_peaks > max_peaks:
        return False, f"Trop de pics ({n_peaks})", ltr_max, n_peaks, None, 0

    ltr_bin = get_ltr_bin(ltr_max)
    if ltr_bin is None:
        return False, "LTR hors plage", ltr_max, n_peaks, None, 0

    # Verifier que le pic n'arrive pas trop tot (sauf si autorise)
    time_to_peak = get_time_to_high_ltr(data, LTR_EARLY_THRESHOLD)

    if not allow_early_peak:
        # Pour les scenarios a LTR eleve, on veut que le pic arrive apres MIN_TIME_TO_PEAK
        if ltr_max > LTR_EARLY_THRESHOLD and time_to_peak < MIN_TIME_TO_PEAK:
            return False, f"Pic trop tot ({time_to_peak:.1f}s)", ltr_max, n_peaks, ltr_bin, time_to_peak

    return True, "OK", ltr_max, n_peaks, ltr_bin, time_to_peak


def generate_all_scenarios():
    """Genere tous les scenarios avec controle des quotas et timing."""
    print("=" * 60)
    print("GENERATION DES SCENARIOS LTR - V2")
    print("=" * 60)
    print(f"Sortie: {OUTPUT_DIR}")
    print(f"Vitesse de braquage max: {np.degrees(MAX_STEERING_RATE):.0f} deg/s")
    print(f"Temps minimum avant pic LTR (avec warm-up): {MIN_TIME_TO_PEAK}s")
    print(f"L_straight: {L_STRAIGHT_SHORT}m (sans warmup) / {L_STRAIGHT_LONG}m (avec warmup)")
    print(f"Ratio sans warm-up: {RATIO_NO_WARMUP*100:.0f}%")

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
    rejected_reasons = defaultdict(int)
    total_tested = 0
    total_generated = 0

    # Parametres
    directions = [-1, 1]
    mu_values = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def try_scenario(traj_type, model, traj, v_ref, filename, allow_early_peak=False, max_peaks=3):
        nonlocal total_tested, total_generated

        # Verifier quota type global
        if type_counts[traj_type] >= TYPE_QUOTAS.get(traj_type, 0):
            return False, None

        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            return True, None

        total_tested += 1
        try:
            data = run_simulation(model, traj, v_ref)
            valid, reason, ltr_max, n_peaks, ltr_bin, time_to_peak = is_valid(
                data, max_peaks=max_peaks, allow_early_peak=allow_early_peak
            )

            if not valid:
                rejected_reasons[reason] += 1
                return False, None

            # Note: on ne limite plus par quota LTR pour maximiser la diversite des types
            # La distribution LTR sera controlee a posteriori si necessaire

            save_csv(data, filepath)
            bin_counts[ltr_bin] += 1
            type_counts[traj_type] += 1
            peak_counts[n_peaks] += 1
            total_generated += 1

            if total_generated % 50 == 0:
                print(f"  {total_generated} scenarios generes...")
            return True, ltr_bin
        except Exception as e:
            rejected_reasons[f"Exception: {type(e).__name__}"] += 1
            return False, None

    def get_l_straight(use_warmup: bool) -> float:
        """Retourne la longueur de ligne droite selon le mode."""
        return L_STRAIGHT_LONG if use_warmup else L_STRAIGHT_SHORT

    # ===== CIRCLES =====
    # L_straight augmente pour warm-up
    # Parametres (v, R) reorganises: HIGH LTR EN PREMIER pour equilibrer la distribution
    # Plages recalibrées pour véhicule lourd (SSF ~0.95)
    # Calibration empirique: rollover à ay~5-6 m/s² (transitoire)
    # LTR 0.5 à (v=10, R=40), LTR 0.8 à (v=11, R=30), LTR 0.9 à (v=10, R=20, mu bas)
    # Mu bas (0.55-0.65) permet d'atteindre LTR élevé sans rollover (slide avant tipover)
    circle_params = [
        # === LTR faible (<0.3): BEAUCOUP de scenarios pour D1/D2 train ===
        (6, 80), (6, 70), (6, 60), (7, 80), (7, 70), (7, 60),
        (7, 90), (6, 90), (8, 80), (8, 90), (5, 60), (5, 70),
        (6, 100), (7, 100), (8, 100), (5, 80), (5, 50),
        # === LTR faible-moyen (0.3-0.5): aussi beaucoup ===
        (8, 50), (8, 55), (8, 45), (9, 55), (9, 60), (9, 50),
        (7, 45), (7, 40), (8, 40), (9, 45), (7, 50), (8, 60),
        (9, 65), (8, 48), (7, 42), (9, 52), (10, 60),
        # === LTR moyen (0.5-0.7): zone intermédiaire ===
        (10, 40), (9, 36), (10, 44), (9, 40), (8, 32),
        (10, 42), (9, 38), (8, 34), (10, 38), (9, 34),
        (10, 46), (9, 42), (8, 36), (10, 48),
        # === LTR moyen-haut (0.7-0.8) ===
        (10, 32), (11, 34), (10, 36), (11, 38), (10, 34),
        (9, 28), (10, 30), (11, 36), (9, 26), (10, 33),
        # === LTR haut (0.8-0.9) ===
        (11, 28), (10, 26), (11, 30), (10, 24), (11, 26),
        (10, 28), (11, 32), (10, 25), (11, 27), (10, 30),
        # === LTR très haut (0.9+) - mu BAS pour éviter rollover ===
        (10, 20), (10, 22), (11, 24), (10, 18), (11, 20),
        (10, 19), (11, 22), (10, 21), (11, 23), (10, 17),
    ]

    print("\nGeneration cercles...")
    scenario_idx = 0

    for v, R in circle_params:
        if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
            break
        for d in directions:
            if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
                break
            for mu in mu_values:
                if type_counts["circle"] >= TYPE_QUOTAS["circle"]:
                    break

                # Alterner entre avec et sans warm-up
                use_warmup = (scenario_idx % 3 != 0)  # ~65% avec warm-up
                allow_early = not use_warmup
                L_straight = get_l_straight(use_warmup)
                warmup_tag = "w" if use_warmup else "nw"

                mu_int = int(round(mu * 100))
                filename = f"circle_v{v}_R{R}_d{d}_mu{mu_int}_{warmup_tag}.csv"
                model = make_heavy_vehicle(mu)
                clockwise = (d == -1)
                traj = CircleTrajectory(v_ref=float(v), R=float(R), clockwise=clockwise, L_straight=L_straight)
                try_scenario("circle", model, traj, float(v), filename, allow_early_peak=allow_early)
                scenario_idx += 1

    print(f"  Cercles: {type_counts['circle']}/{TYPE_QUOTAS['circle']}")

    # ===== SINGLE TURNS =====
    # Comme circle mais avec L_straight encore plus long
    # HIGH LTR EN PREMIER
    single_params = [
        # === LTR faible (<0.3): beaucoup pour D1/D2 ===
        (6, 80), (6, 70), (7, 80), (7, 70), (7, 60),
        (8, 80), (8, 90), (6, 60), (5, 70), (6, 90),
        (7, 100), (8, 100), (5, 60), (5, 80),
        # === LTR faible-moyen (0.3-0.5) ===
        (8, 50), (8, 55), (9, 55), (9, 60), (7, 45),
        (7, 40), (8, 40), (9, 45), (8, 48), (9, 52),
        # === LTR moyen (0.5-0.7) ===
        (10, 40), (9, 36), (10, 44), (9, 40),
        (10, 42), (9, 38), (8, 32), (10, 46),
        # === LTR moyen-haut (0.7-0.8) ===
        (10, 32), (11, 34), (10, 36), (9, 28), (10, 30),
        # === LTR haut (0.8-0.9) ===
        (11, 28), (10, 26), (11, 30), (10, 24), (11, 26),
        # === LTR très haut (0.9+) ===
        (10, 20), (10, 22), (11, 24), (10, 18), (11, 20),
    ]

    print("\nGeneration virages simples...")
    scenario_idx = 0

    for v, R in single_params:
        if type_counts["single"] >= TYPE_QUOTAS["single"]:
            break
        for d in directions:
            if type_counts["single"] >= TYPE_QUOTAS["single"]:
                break
            for mu in mu_values:
                if type_counts["single"] >= TYPE_QUOTAS["single"]:
                    break

                use_warmup = (scenario_idx % 3 != 0)  # ~65% avec warm-up
                allow_early = not use_warmup
                L_straight = get_l_straight(use_warmup) + 20.0  # Single = plus long
                warmup_tag = "w" if use_warmup else "nw"

                mu_int = int(round(mu * 100))
                filename = f"single_v{v}_R{R}_d{d}_mu{mu_int}_{warmup_tag}.csv"
                model = make_heavy_vehicle(mu)
                clockwise = (d == -1)
                traj = CircleTrajectory(v_ref=float(v), R=float(R), clockwise=clockwise, L_straight=L_straight)
                try_scenario("single", model, traj, float(v), filename, allow_early_peak=allow_early)
                scenario_idx += 1

    print(f"  Single: {type_counts['single']}/{TYPE_QUOTAS['single']}")

    # ===== LEMNISCATE (Figure en 8) =====
    # Avec L_straight pour warm-up et T_period ajuste
    # HIGH LTR EN PREMIER (periodes courtes, grandes amplitudes)
    lemniscate_params = [
        # (v, A, B, T_period) — calibré pour véhicule lourd
        # === LTR faible (<0.3) ===
        (6, 8, 5, 24), (6, 7, 5, 22), (7, 8, 6, 24), (7, 9, 6, 26),
        (5, 7, 5, 24), (6, 9, 6, 26),
        # === LTR faible-moyen (0.3-0.5) ===
        (7, 8, 6, 20), (8, 9, 7, 22), (7, 10, 7, 22), (8, 10, 7, 24),
        (8, 8, 6, 20), (7, 9, 7, 20),
        # === LTR moyen (0.5-0.7) ===
        (8, 10, 8, 18), (9, 10, 8, 20), (8, 12, 9, 18), (9, 12, 9, 20),
        (9, 10, 8, 16), (8, 10, 8, 16),
        # === LTR moyen-haut (0.7-0.8) ===
        (9, 12, 9, 14), (10, 12, 9, 16), (10, 14, 10, 16), (9, 10, 8, 14),
        (10, 12, 9, 14), (10, 10, 8, 14),
        # === LTR haut (0.8-0.9) ===
        (10, 14, 10, 12), (11, 14, 10, 14), (10, 12, 9, 12), (11, 12, 9, 12),
        # === LTR très haut (0.9+) ===
        (10, 14, 10, 10), (11, 14, 10, 12), (10, 12, 9, 10),
    ]

    print("\nGeneration lemniscates...")
    scenario_idx = 0

    for v, A, B, T_period in lemniscate_params:
        if type_counts["lemniscate"] >= TYPE_QUOTAS["lemniscate"]:
            break
        for mu in mu_values:
            if type_counts["lemniscate"] >= TYPE_QUOTAS["lemniscate"]:
                break

            # Lemniscate: 50% avec warm-up (la figure en 8 commence vite de toute facon)
            use_warmup = (scenario_idx % 2 == 0)
            allow_early = not use_warmup
            L_straight = get_l_straight(use_warmup)
            warmup_tag = "w" if use_warmup else "nw"

            mu_int = int(round(mu * 100))
            filename = f"lemniscate_v{v}_A{A}_B{B}_T{T_period}_mu{mu_int}_{warmup_tag}.csv"
            model = make_heavy_vehicle(mu)
            traj = LemniscateTrajectory(
                v_ref=float(v), A=float(A), B=float(B),
                T_period=float(T_period), L_straight=L_straight
            )
            # Lemniscates ont plusieurs oscillations naturellement
            try_scenario("lemniscate", model, traj, float(v), filename, allow_early_peak=allow_early, max_peaks=8)
            scenario_idx += 1

    print(f"  Lemniscates: {type_counts['lemniscate']}/{TYPE_QUOTAS['lemniscate']}")

    # ===== SLALOM =====
    # Avec L_straight et T_ramp pour montee progressive
    # HIGH LTR EN PREMIER - parametres ajustes pour eviter renversement
    slalom_params = [
        # (v, L, A, n_waves, T_ramp) — calibré pour véhicule lourd
        # === LTR faible (<0.3) ===
        (6, 180, 2.0, 2, 2.0), (7, 200, 2.0, 2, 2.0), (6, 200, 2.0, 3, 2.0),
        (7, 180, 1.8, 2, 2.0), (5, 160, 2.0, 2, 2.0),
        # === LTR faible-moyen (0.3-0.5) ===
        (8, 160, 2.5, 2, 2.0), (8, 150, 2.5, 2, 2.0), (9, 170, 2.5, 2, 2.0),
        (7, 140, 2.5, 2, 2.0), (8, 140, 2.3, 2, 2.0), (9, 160, 2.3, 2, 2.0),
        # === LTR moyen (0.5-0.7) ===
        (9, 130, 2.8, 2, 2.0), (10, 140, 2.8, 2, 2.0), (9, 120, 2.8, 2, 2.0),
        (10, 130, 2.5, 2, 2.0), (9, 120, 2.5, 3, 2.0), (10, 120, 2.8, 2, 2.0),
        # === LTR moyen-haut (0.7-0.8) ===
        (10, 110, 3.0, 2, 2.0), (11, 120, 3.0, 2, 2.0), (10, 100, 3.0, 2, 2.0),
        (11, 110, 2.8, 2, 2.0), (10, 100, 2.8, 2, 2.0), (11, 100, 3.0, 2, 2.0),
        # === LTR haut (0.8-0.9) ===
        (11, 100, 3.3, 2, 2.0), (10, 90, 3.3, 2, 2.0), (11, 90, 3.0, 2, 2.0),
        (10, 85, 3.0, 2, 2.0),
        # === LTR très haut (0.9+) ===
        (11, 80, 3.5, 2, 2.0), (10, 80, 3.5, 2, 2.0), (11, 85, 3.3, 2, 2.0),
    ]

    print("\nGeneration slaloms...")
    scenario_idx = 0

    for v, L, A, n_waves, T_ramp in slalom_params:
        if type_counts["slalom"] >= TYPE_QUOTAS["slalom"]:
            break
        for mu in mu_values:
            if type_counts["slalom"] >= TYPE_QUOTAS["slalom"]:
                break

            # Slalom: 40% sans warm-up (le T_ramp gere deja la montee progressive)
            use_warmup = (scenario_idx % 5 != 0) and (scenario_idx % 5 != 2)  # ~60% avec warm-up
            allow_early = not use_warmup
            L_straight = get_l_straight(use_warmup)
            warmup_tag = "w" if use_warmup else "nw"

            mu_int = int(round(mu * 100))
            filename = f"slalom_v{v}_L{L}_A{int(A*10)}_n{n_waves}_Tr{int(T_ramp*10)}_mu{mu_int}_{warmup_tag}.csv"
            model = make_heavy_vehicle(mu)
            traj = SlalomTrajectory(
                v_ref=float(v), L=float(L), A=float(A),
                n_waves=n_waves, L_straight=L_straight, T_ramp=T_ramp
            )
            # Slaloms ont naturellement beaucoup de pics (oscillations)
            try_scenario("slalom", model, traj, float(v), filename, allow_early_peak=allow_early, max_peaks=12)
            scenario_idx += 1

    print(f"  Slaloms: {type_counts['slalom']}/{TYPE_QUOTAS['slalom']}")

    # ===== DLC (Double Lane Change) =====
    # Utilise DoubleLaneChangeTrajectory avec L_straight pour warm-up
    # HIGH LTR EN PREMIER - parametres ajustes pour eviter renversement
    dlc_params = [
        # (v, A, L_total) — calibré pour véhicule lourd
        # === LTR faible (<0.3) ===
        (6, 2.0, 140), (7, 2.0, 150), (6, 1.8, 130), (7, 1.8, 140),
        (5, 2.0, 130), (6, 2.0, 150),
        # === LTR faible-moyen (0.3-0.5) ===
        (8, 2.5, 120), (8, 2.3, 110), (9, 2.5, 130), (7, 2.5, 110),
        (8, 2.3, 120), (9, 2.3, 120),
        # === LTR moyen (0.5-0.7) ===
        (9, 2.8, 100), (10, 2.8, 110), (9, 3.0, 100), (10, 3.0, 110),
        (9, 2.5, 95), (10, 2.5, 100),
        # === LTR moyen-haut (0.7-0.8) ===
        (10, 3.0, 90), (11, 3.0, 100), (10, 3.3, 90), (11, 3.3, 100),
        (10, 2.8, 85), (11, 2.8, 95),
        # === LTR haut (0.8-0.9) ===
        (11, 3.5, 85), (10, 3.5, 80), (11, 3.3, 80), (10, 3.3, 75),
        # === LTR très haut (0.9+) ===
        (11, 3.8, 75), (10, 3.8, 70), (11, 3.5, 70),
    ]

    print("\nGeneration DLC...")
    scenario_idx = 0

    for v, A, L_total in dlc_params:
        if type_counts["dlc"] >= TYPE_QUOTAS["dlc"]:
            break
        for mu in mu_values:
            if type_counts["dlc"] >= TYPE_QUOTAS["dlc"]:
                break

            # DLC: alterner entre avec et sans warm-up
            use_warmup = (scenario_idx % 3 != 0)  # ~65% avec warm-up
            allow_early = not use_warmup
            L_straight = get_l_straight(use_warmup)
            warmup_tag = "w" if use_warmup else "nw"

            mu_int = int(round(mu * 100))
            filename = f"dlc_v{v}_A{int(A*10)}_L{L_total}_Ls{int(L_straight)}_mu{mu_int}_{warmup_tag}.csv"
            model = make_heavy_vehicle(mu)
            try:
                traj = DoubleLaneChangeTrajectory(
                    v_ref=float(v), A=float(A), L_total=float(L_total),
                    L_straight=L_straight
                )
                # DLC peut avoir plusieurs oscillations LTR lors des transitions
                try_scenario("dlc", model, traj, float(v), filename, allow_early_peak=allow_early, max_peaks=10)
            except Exception as e:
                rejected_reasons[f"DLC Exception: {type(e).__name__}"] += 1
            scenario_idx += 1

    print(f"  DLC: {type_counts['dlc']}/{TYPE_QUOTAS['dlc']}")

    # ===== WAYPOINT (trajectoires aleatoires) =====
    print("\nGeneration waypoints aleatoires...")
    np.random.seed(42)

    waypoint_generated = 0
    waypoint_attempts = 0
    max_waypoint_attempts = 2500

    while waypoint_generated < TYPE_QUOTAS["waypoint"] and waypoint_attempts < max_waypoint_attempts:
        waypoint_attempts += 1

        v_ref = np.random.uniform(6, 12)
        mu = np.random.choice(mu_values)
        mu_int = int(round(mu * 100))

        # Mix: 60% avec delai, 40% sans delai
        use_warmup = np.random.random() < 0.6
        allow_early = not use_warmup

        if use_warmup:
            T_delay = np.random.uniform(4.0, 6.0)
            warmup_tag = "w"
        else:
            T_delay = np.random.uniform(0.0, 1.0)  # Commence quasi immediatement
            warmup_tag = "nw"

        traj_type = np.random.choice(["scurve", "chicane", "evitement", "ondulation", "zigzag"])

        t_points = np.linspace(0, T_SIMU, 300)
        x_points = v_ref * t_points
        y_points = np.zeros_like(t_points)

        if traj_type == "scurve":
            amp = np.random.uniform(2.0, 6.0)
            freq = np.random.uniform(0.3, 1.5)
            mask = t_points > T_delay
            t_shifted = t_points[mask] - T_delay
            y_points[mask] = amp * np.sin(2 * np.pi * freq * t_shifted / (T_SIMU - T_delay + 0.01))
            filename = f"waypoint_scurve_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}_{warmup_tag}.csv"

        elif traj_type == "chicane":
            amp = np.random.uniform(2.0, 5.0)
            trans_time = np.random.uniform(1.5, 3.0)
            t_chicane = T_delay
            mask1 = (t_points > t_chicane) & (t_points < t_chicane + trans_time)
            mask2 = t_points >= t_chicane + trans_time
            y_points[mask1] = amp * (t_points[mask1] - t_chicane) / trans_time
            y_points[mask2] = amp
            if np.random.random() > 0.4:
                t_return = t_chicane + trans_time + np.random.uniform(2, 4)
                if t_return + trans_time < T_SIMU:
                    mask3 = (t_points > t_return) & (t_points < t_return + trans_time)
                    mask4 = t_points >= t_return + trans_time
                    y_points[mask3] = amp - amp * (t_points[mask3] - t_return) / trans_time
                    y_points[mask4] = 0
            filename = f"waypoint_chicane_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}_{warmup_tag}.csv"

        elif traj_type == "evitement":
            amp = np.random.uniform(2.5, 6.0)
            t_dur = np.random.uniform(1.0, 2.5)
            mask = (t_points > T_delay) & (t_points < T_delay + t_dur)
            y_points[mask] = amp * np.sin(np.pi * (t_points[mask] - T_delay) / t_dur)
            filename = f"waypoint_evitement_{waypoint_attempts}_v{int(v_ref)}_a{int(amp*10)}_mu{mu_int}_{warmup_tag}.csv"

        elif traj_type == "zigzag":
            n_turns = np.random.randint(2, 4)
            amp = np.random.uniform(2.0, 4.5)
            t_start = max(T_delay, 0.5)
            turn_times = t_start + np.sort(np.random.uniform(0.5, T_SIMU - t_start - 2, n_turns))
            current_y = 0
            direction = np.random.choice([-1, 1])
            for i, tt in enumerate(turn_times):
                if i < len(turn_times) - 1:
                    mask = (t_points >= tt) & (t_points < turn_times[i+1])
                else:
                    mask = t_points >= tt
                target_y = direction * amp
                trans = np.minimum((t_points[mask] - tt) / 1.5, 1.0)
                y_points[mask] = current_y + (target_y - current_y) * trans
                current_y = target_y
                direction *= -1
            filename = f"waypoint_zigzag_{waypoint_attempts}_v{int(v_ref)}_n{n_turns}_a{int(amp*10)}_mu{mu_int}_{warmup_tag}.csv"

        else:  # ondulation
            n_waves = np.random.randint(1, 4)
            amp = np.random.uniform(2.0, 5.0)
            mask = t_points > T_delay
            t_shifted = t_points[mask] - T_delay
            denom = max(T_SIMU - T_delay, 0.1)
            y_points[mask] = amp * np.sin(2 * np.pi * n_waves * t_shifted / denom)
            filename = f"waypoint_ondulation_{waypoint_attempts}_v{int(v_ref)}_n{n_waves}_a{int(amp*10)}_mu{mu_int}_{warmup_tag}.csv"

        try:
            traj = WaypointTrajectory(t_points, x_points, y_points)
            model = make_heavy_vehicle(mu)
            # Waypoints peuvent avoir plusieurs oscillations
            success, _ = try_scenario("waypoint", model, traj, v_ref, filename, allow_early_peak=allow_early, max_peaks=10)
            if success:
                waypoint_generated += 1
        except Exception:
            pass

        if waypoint_attempts % 200 == 0:
            print(f"    Waypoints: {waypoint_generated}/{TYPE_QUOTAS['waypoint']} (essai {waypoint_attempts})")

    print(f"  Waypoints: {type_counts['waypoint']}/{TYPE_QUOTAS['waypoint']}")

    # ===== SMOOTH RANDOM (trajectoires aléatoires réalistes) =====
    print("\nGeneration smooth random...")
    sr_seed_base = 10000
    sr_generated = 0
    sr_attempts = 0
    max_sr_attempts = 3000

    while sr_generated < TYPE_QUOTAS["smooth_random"] and sr_attempts < max_sr_attempts:
        sr_attempts += 1
        seed = sr_seed_base + sr_attempts

        rng_sr = np.random.RandomState(seed)

        # Paramètres randomisés pour couvrir tout le spectre LTR
        v_ref = rng_sr.uniform(8.0, 24.0)
        y_max_val = rng_sr.uniform(2.0, 8.0)
        d_min_val = rng_sr.uniform(30.0, 80.0)
        alpha_val = rng_sr.uniform(0.3, 0.8)
        v_min_ratio_val = rng_sr.uniform(0.3, 0.7)
        n_wp = rng_sr.randint(5, 13)  # 5-12 waypoints
        mu = float(rng_sr.choice(mu_values))
        mu_int = int(round(mu * 100))

        # Warm-up: 65% avec, 35% sans
        use_warmup = rng_sr.random() < 0.65
        allow_early = not use_warmup
        L_straight = get_l_straight(use_warmup)
        warmup_tag = "w" if use_warmup else "nw"

        filename = (
            f"smooth_v{int(v_ref)}_ym{int(y_max_val*10)}_dm{int(d_min_val)}"
            f"_a{int(alpha_val*10)}_n{n_wp}_mu{mu_int}_{warmup_tag}_s{seed}.csv"
        )

        try:
            traj = SmoothRandomTrajectory(
                v_ref=v_ref,
                n_waypoints=n_wp,
                d_min=d_min_val,
                y_max=y_max_val,
                alpha=alpha_val,
                v_min_ratio=v_min_ratio_val,
                L_straight=L_straight,
                seed=seed,
                kappa_max_reject=0.25,
            )
            model = make_heavy_vehicle(mu)
            # Smooth random peut avoir plusieurs oscillations (virages successifs)
            success, _ = try_scenario(
                "smooth_random", model, traj, v_ref, filename,
                allow_early_peak=allow_early, max_peaks=10,
            )
            if success:
                sr_generated += 1
        except Exception:
            pass

        if sr_attempts % 200 == 0:
            print(f"    Smooth random: {sr_generated}/{TYPE_QUOTAS['smooth_random']} (essai {sr_attempts})")

    print(f"  Smooth random: {type_counts['smooth_random']}/{TYPE_QUOTAS['smooth_random']}")

    # ===== Rapport final =====
    print("\n" + "=" * 60)
    print("RAPPORT FINAL")
    print("=" * 60)
    print(f"Scenarios testes: {total_tested}")
    print(f"Scenarios generes: {total_generated}")

    print(f"\nPar type:")
    for t in TYPE_QUOTAS.keys():
        quota = TYPE_QUOTAS.get(t, 0)
        count = type_counts[t]
        pct = 100 * count / quota if quota > 0 else 0
        status = "OK" if count >= quota * 0.8 else f"MANQUE {quota - count}"
        print(f"  {t}: {count}/{quota} ({pct:.0f}%) - {status}")

    print(f"\nDistribution LTR:")
    for (lo, hi), quota in LTR_QUOTAS.items():
        count = bin_counts[(lo, hi)]
        pct = 100 * count / total_generated if total_generated > 0 else 0
        print(f"  [{lo:.1f}, {hi:.1f}): {count} ({pct:.1f}%)")

    print(f"\nDistribution des pics:")
    for p in sorted(peak_counts.keys()):
        count = peak_counts[p]
        pct = 100 * count / total_generated if total_generated > 0 else 0
        print(f"  {p} pic(s): {count} ({pct:.1f}%)")

    print(f"\nRaisons de rejet (top 10):")
    for reason, count in sorted(rejected_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count}")

    return total_generated


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    total = generate_all_scenarios()
    print(f"\nTermine: {total} scenarios generes")
