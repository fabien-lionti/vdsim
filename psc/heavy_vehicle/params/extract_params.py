#!/usr/bin/env python3
"""Extrait les paramètres physiques du véhicule lourd depuis les fichiers DXD JSON."""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/Users/zak/Desktop/selected_dxd_json_resampled/")
OUTPUT_DIR = Path(__file__).parent


def load_json(filepath):
    """Charge un fichier JSON DXD et retourne les channels."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def has_channel(data, name):
    """Vérifie si un canal existe."""
    return name in data["channels"]


def get_channel(data, name):
    """Extrait un canal comme array numpy, remplaçant None par NaN."""
    if name not in data["channels"]:
        return None
    values = data["channels"][name]["values"]
    return np.array([v if v is not None else np.nan for v in values], dtype=float)


def get_files_by_type(data_dir):
    """Classe les fichiers par type de test."""
    files = sorted(data_dir.glob("*.json"))
    by_type = defaultdict(list)
    for f in files:
        name = f.stem
        if "vide" in name:
            load_state = "vide"
        elif "charge" in name:
            load_state = "charge"
        else:
            load_state = "unknown"
        by_type[load_state].append(f)
    return by_type, files


# =============================================================================
# 1. MASSE ET REPARTITION F/R
# =============================================================================

def extract_mass(files, max_speed_kmh=3.0):
    """
    Estime la masse depuis ΣFz/g à basse vitesse.
    Retourne masse moyenne, std, et répartition F/R.
    """
    masses = []
    front_pcts = []
    left_pcts = []
    fz_per_wheel = defaultdict(list)

    required = ["Vel (km/h)", "LF_Fz_1 (N)", "RF_Fz_2 (N)", "LR_Fz_3 (N)", "RR_Fz_4 (N)"]
    for f in files[:50]:  # 50 fichiers suffisent pour une estimation robuste
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        vel = get_channel(data, "Vel (km/h)")
        fz_lf = get_channel(data, "LF_Fz_1 (N)")
        fz_rf = get_channel(data, "RF_Fz_2 (N)")
        fz_lr = get_channel(data, "LR_Fz_3 (N)")
        fz_rr = get_channel(data, "RR_Fz_4 (N)")

        # Points à basse vitesse
        mask = np.abs(vel) < max_speed_kmh
        if np.sum(mask) < 10:
            continue

        avg_lf = np.nanmean(fz_lf[mask])
        avg_rf = np.nanmean(fz_rf[mask])
        avg_lr = np.nanmean(fz_lr[mask])
        avg_rr = np.nanmean(fz_rr[mask])
        total = avg_lf + avg_rf + avg_lr + avg_rr

        if total < 5000:  # Filtre aberrant
            continue

        masses.append(total / 9.81)
        front_pcts.append((avg_lf + avg_rf) / total * 100)
        left_pcts.append((avg_lf + avg_lr) / total * 100)
        fz_per_wheel["LF"].append(avg_lf)
        fz_per_wheel["RF"].append(avg_rf)
        fz_per_wheel["LR"].append(avg_lr)
        fz_per_wheel["RR"].append(avg_rr)

    result = {
        "mass_mean": np.mean(masses),
        "mass_std": np.std(masses),
        "mass_n": len(masses),
        "front_pct": np.mean(front_pcts),
        "left_pct": np.mean(left_pcts),
        "fz_lf": np.mean(fz_per_wheel["LF"]),
        "fz_rf": np.mean(fz_per_wheel["RF"]),
        "fz_lr": np.mean(fz_per_wheel["LR"]),
        "fz_rr": np.mean(fz_per_wheel["RR"]),
    }
    return result


# =============================================================================
# 2. RAYON PNEU
# =============================================================================

def extract_tire_radius(files, min_speed=30, max_speed=80):
    """
    Estime le rayon de roulement depuis R = V / ω_roue en ligne droite.
    Utilise les 4 roues et moyenne.
    """
    all_radii = []
    required = ["Vel (km/h)", "WhlFl_W_Meas (rad/s)", "WhlFr_W_Meas (rad/s)",
                "WhlRl_W_Meas (rad/s)", "WhlRr_W_Meas (rad/s)", "VehLat_A_Actl (m/s^2)"]
    # Priorité aux fichiers en ligne droite, limiter à 30
    priority = [f for f in files if any(k in f.name for k in ["ligne", "acceler", "rodage", "libre"])]
    subset = (priority + [f for f in files if f not in priority])[:30]

    for f in subset:
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        vel = get_channel(data, "Vel (km/h)")
        whl_fl = get_channel(data, "WhlFl_W_Meas (rad/s)")
        whl_fr = get_channel(data, "WhlFr_W_Meas (rad/s)")
        whl_rl = get_channel(data, "WhlRl_W_Meas (rad/s)")
        whl_rr = get_channel(data, "WhlRr_W_Meas (rad/s)")
        ay = get_channel(data, "VehLat_A_Actl (m/s^2)")

        # Points en ligne droite (ay faible) et vitesse dans la plage
        mask = (vel > min_speed) & (vel < max_speed) & (np.abs(ay) < 1.0)
        mask &= (whl_fl > 1) & (whl_fr > 1) & (whl_rl > 1) & (whl_rr > 1)

        if np.sum(mask) < 20:
            continue

        vel_ms = vel[mask] / 3.6
        for whl in [whl_fl[mask], whl_fr[mask], whl_rl[mask], whl_rr[mask]]:
            radii = vel_ms / whl
            # Filtre valeurs aberrantes
            valid = (radii > 0.25) & (radii < 0.55)
            all_radii.extend(radii[valid].tolist())

    return {
        "tire_radius_mean": np.mean(all_radii),
        "tire_radius_std": np.std(all_radii),
        "tire_radius_n": len(all_radii),
    }


# =============================================================================
# 3. HAUTEUR CdG
# =============================================================================

def extract_cg_height(files, assumed_track=1.65):
    """
    Estime h_cg depuis le transfert de charge latéral en virage.
    ΔFz_lat = m * ay * h / (track/2)
    => h = ΔFz_lat * (track/2) / (m * ay)

    On utilise aussi une approche alternative:
    h = (Fz_left - Fz_right) * track / (2 * Fz_total / g * ay)
    """
    h_estimates = []
    required = ["Vel (km/h)", "VehLat_A_Actl (m/s^2)", "LF_Fz_1 (N)", "RF_Fz_2 (N)", "LR_Fz_3 (N)", "RR_Fz_4 (N)"]
    # Priorité aux tests dynamiques, limiter à 40
    priority = [f for f in files if any(k in f.name for k in ["erdv", "slalom", "ld_", "feco"])]
    subset = (priority + [f for f in files if f not in priority])[:40]

    for f in subset:
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        vel = get_channel(data, "Vel (km/h)")
        ay = get_channel(data, "VehLat_A_Actl (m/s^2)")
        fz_lf = get_channel(data, "LF_Fz_1 (N)")
        fz_rf = get_channel(data, "RF_Fz_2 (N)")
        fz_lr = get_channel(data, "LR_Fz_3 (N)")
        fz_rr = get_channel(data, "RR_Fz_4 (N)")

        # Points avec bonne accélération latérale et vitesse
        mask = (np.abs(ay) > 3.0) & (vel > 20)

        for i in np.where(mask)[0]:
            fz_left = fz_lf[i] + fz_lr[i]
            fz_right = fz_rf[i] + fz_rr[i]
            fz_total = fz_left + fz_right
            m = fz_total / 9.81

            if m < 1000:
                continue

            delta_fz = fz_left - fz_right
            h = abs(delta_fz) * assumed_track / (2 * m * abs(ay[i]))

            if 0.3 < h < 1.5:
                h_estimates.append(h)

    return {
        "cg_height_mean": np.mean(h_estimates),
        "cg_height_std": np.std(h_estimates),
        "cg_height_median": np.median(h_estimates),
        "cg_height_n": len(h_estimates),
        "assumed_track": assumed_track,
    }


# =============================================================================
# 4. VOIE (TRACK WIDTH)
# =============================================================================

def extract_track_width(files, assumed_h=0.85):
    """
    Estime la voie en inversant la formule de transfert de charge.
    track = 2 * m * ay * h / ΔFz_lat

    Utilise des points en virage stabilisé (ay quasi-constant).
    """
    track_estimates = []
    required = ["Vel (km/h)", "VehLat_A_Actl (m/s^2)", "LF_Fz_1 (N)", "RF_Fz_2 (N)", "LR_Fz_3 (N)", "RR_Fz_4 (N)"]
    # Priorité aux virages stabilisés, limiter à 30
    priority = [f for f in files if any(k in f.name for k in ["rpc", "vir", "anneau", "anr"])]
    subset = (priority + [f for f in files if f not in priority])[:30]

    for f in subset:
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        vel = get_channel(data, "Vel (km/h)")
        ay = get_channel(data, "VehLat_A_Actl (m/s^2)")
        fz_lf = get_channel(data, "LF_Fz_1 (N)")
        fz_rf = get_channel(data, "RF_Fz_2 (N)")
        fz_lr = get_channel(data, "LR_Fz_3 (N)")
        fz_rr = get_channel(data, "RR_Fz_4 (N)")

        # Virage stabilisé : ay > 2 et relativement constant
        mask = (np.abs(ay) > 2.0) & (vel > 20)

        for i in np.where(mask)[0]:
            fz_left = fz_lf[i] + fz_lr[i]
            fz_right = fz_rf[i] + fz_rr[i]
            fz_total = fz_left + fz_right
            delta_fz = abs(fz_left - fz_right)
            m = fz_total / 9.81

            if m < 1000 or delta_fz < 100:
                continue

            track = 2 * m * abs(ay[i]) * assumed_h / delta_fz

            if 1.2 < track < 2.2:
                track_estimates.append(track)

    return {
        "track_width_mean": np.mean(track_estimates),
        "track_width_std": np.std(track_estimates),
        "track_width_median": np.median(track_estimates),
        "track_width_n": len(track_estimates),
        "assumed_h": assumed_h,
    }


# =============================================================================
# 5. EMPATTEMENT (WHEELBASE)
# =============================================================================

def extract_wheelbase(files):
    """
    Estime l'empattement via géométrie d'Ackermann à basse vitesse.
    L = delta_roue * R

    Utilise WheelSteer_S1 (angle braquage roue) et Radius (rayon GPS).
    """
    wb_estimates = []
    required = ["Vel (km/h)", "Radius (m)", "WheelSteer_S1 (_)"]
    # Priorité virages basse vitesse, limiter à 30
    priority = [f for f in files if any(k in f.name for k in ["vir_place", "vir_l", "vir_e", "rpc", "slalom"])]
    subset = (priority + [f for f in files if f not in priority])[:30]

    for f in subset:
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        vel = get_channel(data, "Vel (km/h)")
        radius = get_channel(data, "Radius (m)")
        steer = get_channel(data, "WheelSteer_S1 (_)")

        # Basse vitesse + virage (rayon fini et braquage non nul)
        mask = (vel > 5) & (vel < 30) & (np.abs(radius) > 5) & (np.abs(radius) < 200) & (np.abs(steer) > 0.02)

        if np.sum(mask) < 20:
            continue

        # Ackermann: tan(delta) = L / R => L = R * tan(delta)
        # steer est en radians d'après l'exploration
        for i in np.where(mask)[0]:
            L = abs(radius[i]) * abs(np.tan(steer[i]))
            if 2.0 < L < 5.0:
                wb_estimates.append(L)

    return {
        "wheelbase_mean": np.mean(wb_estimates) if wb_estimates else None,
        "wheelbase_std": np.std(wb_estimates) if wb_estimates else None,
        "wheelbase_median": np.median(wb_estimates) if wb_estimates else None,
        "wheelbase_n": len(wb_estimates),
    }


# =============================================================================
# 6. INERTIES (ESTIMATIONS)
# =============================================================================

def estimate_inertias(mass, wheelbase, track_width):
    """
    Estime les inerties par formules empiriques.
    - Iz (lacet) : Iz ≈ 0.127 * m * L² (formule d'Abe)
    - Ix (roulis) : Ix ≈ 0.35 * m * T² / 4 (distribution mass latérale)
    - Iy (tangage) : Iy ≈ 0.75 * Iz (empirique)
    """
    iz = 0.127 * mass * wheelbase ** 2
    ix = 0.35 * mass * track_width ** 2 / 4
    iy = 0.75 * iz

    return {
        "iz_empirical": iz,
        "ix_empirical": ix,
        "iy_empirical": iy,
        "method": "Abe (Iz), lateral distribution (Ix), 0.75*Iz (Iy)",
    }


# =============================================================================
# 7. INERTIES DEPUIS DYNAMIQUE (BONUS)
# =============================================================================

def extract_yaw_inertia(files, mass, wheelbase_half_front, wheelbase_half_rear, track_half):
    """
    Tente d'estimer Iz depuis la dynamique de lacet lors de transitoires rapides.
    Iz * yaw_accel ≈ lf * ΣFy_front - lr * ΣFy_rear
    """
    iz_estimates = []
    required = ["VehYaw_W_Actl (rad/s)", "LF_Fy_1 (N)", "RF_Fy_2 (N)", "LR_Fy_3 (N)", "RR_Fy_4 (N)"]

    for f in files:
        data = load_json(f)
        if not all(has_channel(data, ch) for ch in required):
            continue
        yaw_rate = get_channel(data, "VehYaw_W_Actl (rad/s)")
        fy_lf = get_channel(data, "LF_Fy_1 (N)")
        fy_rf = get_channel(data, "RF_Fy_2 (N)")
        fy_lr = get_channel(data, "LR_Fy_3 (N)")
        fy_rr = get_channel(data, "RR_Fy_4 (N)")

        # Calcul de l'accélération angulaire en lacet
        dt = 0.01  # 100 Hz
        yaw_accel = np.gradient(yaw_rate, dt)

        # Points avec forte accélération yaw (transitoires)
        mask = (np.abs(yaw_accel) > 5.0) & (~np.isnan(yaw_accel))

        for i in np.where(mask)[0]:
            fy_front = fy_lf[i] + fy_rf[i]
            fy_rear = fy_lr[i] + fy_rr[i]

            # Moment de lacet depuis les forces latérales
            mz = wheelbase_half_front * fy_front - wheelbase_half_rear * fy_rear

            iz = abs(mz / yaw_accel[i])
            if 1000 < iz < 20000:
                iz_estimates.append(iz)

    if iz_estimates:
        return {
            "iz_dynamic_mean": np.mean(iz_estimates),
            "iz_dynamic_std": np.std(iz_estimates),
            "iz_dynamic_median": np.median(iz_estimates),
            "iz_dynamic_n": len(iz_estimates),
        }
    return {"iz_dynamic_mean": None, "iz_dynamic_n": 0}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXTRACTION DES PARAMÈTRES PHYSIQUES — VÉHICULE LOURD")
    print("=" * 70)

    files_by_type, all_files = get_files_by_type(DATA_DIR)
    print(f"\nFichiers: {len(all_files)} total")
    print(f"  - vide: {len(files_by_type['vide'])}")
    print(f"  - charge: {len(files_by_type['charge'])}")
    print(f"  - unknown: {len(files_by_type['unknown'])}")

    # ---- 1. Masse ----
    print("\n" + "=" * 50)
    print("1. MASSE ET RÉPARTITION")
    print("=" * 50)

    mass_vide = extract_mass(files_by_type["vide"])
    mass_charge = extract_mass(files_by_type["charge"])

    print(f"\nÀ VIDE (n={mass_vide['mass_n']} fichiers):")
    print(f"  Masse: {mass_vide['mass_mean']:.0f} ± {mass_vide['mass_std']:.0f} kg")
    print(f"  F/R: {mass_vide['front_pct']:.1f}% / {100-mass_vide['front_pct']:.1f}%")
    print(f"  G/D: {mass_vide['left_pct']:.1f}% / {100-mass_vide['left_pct']:.1f}%")
    print(f"  Fz par roue: LF={mass_vide['fz_lf']:.0f} RF={mass_vide['fz_rf']:.0f} "
          f"LR={mass_vide['fz_lr']:.0f} RR={mass_vide['fz_rr']:.0f} N")

    print(f"\nCHARGÉ (n={mass_charge['mass_n']} fichiers):")
    print(f"  Masse: {mass_charge['mass_mean']:.0f} ± {mass_charge['mass_std']:.0f} kg")
    print(f"  F/R: {mass_charge['front_pct']:.1f}% / {100-mass_charge['front_pct']:.1f}%")
    print(f"  Delta charge: {mass_charge['mass_mean']-mass_vide['mass_mean']:.0f} kg")

    m_ref = (mass_vide["mass_mean"] + mass_charge["mass_mean"]) / 2
    fr_ref = (mass_vide["front_pct"] + mass_charge["front_pct"]) / 2
    print(f"\n→ Masse de référence (midpoint): {m_ref:.0f} kg")
    print(f"→ Répartition F/R de référence: {fr_ref:.1f}% / {100-fr_ref:.1f}%")

    # ---- 2. Rayon pneu ----
    print("\n" + "=" * 50)
    print("2. RAYON PNEU")
    print("=" * 50)

    tire = extract_tire_radius(all_files)
    print(f"  Rayon: {tire['tire_radius_mean']:.4f} ± {tire['tire_radius_std']:.4f} m (n={tire['tire_radius_n']})")
    print(f"  Diamètre: {tire['tire_radius_mean']*2*1000:.0f} mm")

    # ---- 3. Hauteur CdG ----
    print("\n" + "=" * 50)
    print("3. HAUTEUR CENTRE DE GRAVITÉ")
    print("=" * 50)

    cg = extract_cg_height(all_files, assumed_track=1.65)
    print(f"  h_cg: {cg['cg_height_mean']:.3f} ± {cg['cg_height_std']:.3f} m (n={cg['cg_height_n']})")
    print(f"  Médiane: {cg['cg_height_median']:.3f} m")
    print(f"  (voie supposée: {cg['assumed_track']} m)")

    # ---- 4. Voie ----
    print("\n" + "=" * 50)
    print("4. VOIE (TRACK WIDTH)")
    print("=" * 50)

    track = extract_track_width(all_files, assumed_h=cg["cg_height_mean"])
    print(f"  Voie: {track['track_width_mean']:.3f} ± {track['track_width_std']:.3f} m (n={track['track_width_n']})")
    print(f"  Médiane: {track['track_width_median']:.3f} m")
    print(f"  (h_cg supposé: {track['assumed_h']:.3f} m)")

    # Itération : recalculer h avec la voie estimée
    print("\n  → Itération : recalcul h_cg avec voie estimée...")
    cg2 = extract_cg_height(all_files, assumed_track=track["track_width_mean"])
    print(f"  h_cg (itéré): {cg2['cg_height_mean']:.3f} ± {cg2['cg_height_std']:.3f} m")

    track2 = extract_track_width(all_files, assumed_h=cg2["cg_height_mean"])
    print(f"  Voie (itérée): {track2['track_width_mean']:.3f} ± {track2['track_width_std']:.3f} m")

    h_final = cg2["cg_height_mean"]
    track_final = track2["track_width_mean"]
    print(f"\n→ Valeurs convergées: h={h_final:.3f} m, voie={track_final:.3f} m")

    # ---- 5. Empattement ----
    print("\n" + "=" * 50)
    print("5. EMPATTEMENT (WHEELBASE)")
    print("=" * 50)

    wb = extract_wheelbase(all_files)
    if wb["wheelbase_mean"]:
        print(f"  Empattement: {wb['wheelbase_mean']:.3f} ± {wb['wheelbase_std']:.3f} m (n={wb['wheelbase_n']})")
        print(f"  Médiane: {wb['wheelbase_median']:.3f} m")
        L_total = wb["wheelbase_median"]
    else:
        print("  Empattement non estimable depuis les données. Utilisation de la valeur par défaut.")
        L_total = 3.2

    # lf, lr depuis répartition F/R
    lf = L_total * (1 - fr_ref / 100)  # lr/(lf+lr) = front_pct => lf/(lf+lr) = 1-front_pct...
    # Non : front_pct = Fz_front/Fz_total = (m*g*lr/L) / (m*g) = lr/L
    # Donc lr = L * front_pct / 100, et lf = L - lr
    lr = L_total * fr_ref / 100
    lf = L_total - lr
    print(f"  lf (CdG→avant): {lf:.3f} m")
    print(f"  lr (CdG→arrière): {lr:.3f} m")
    print(f"  Vérif F/R: {lr/L_total*100:.1f}% / {lf/L_total*100:.1f}%")

    # ---- 6. Inerties ----
    print("\n" + "=" * 50)
    print("6. INERTIES")
    print("=" * 50)

    inertias_emp = estimate_inertias(m_ref, L_total, track_final)
    print(f"  Formules empiriques (m={m_ref:.0f} kg, L={L_total:.2f} m, T={track_final:.2f} m):")
    print(f"    Iz (lacet): {inertias_emp['iz_empirical']:.0f} kg·m²")
    print(f"    Ix (roulis): {inertias_emp['ix_empirical']:.0f} kg·m²")
    print(f"    Iy (tangage): {inertias_emp['iy_empirical']:.0f} kg·m²")

    # Tentative estimation dynamique de Iz
    erdv_files = [f for f in all_files if "erdv" in f.stem]
    if erdv_files:
        iz_dyn = extract_yaw_inertia(erdv_files[:20], m_ref, lf, lr, track_final / 2)
        if iz_dyn["iz_dynamic_mean"]:
            print(f"\n  Estimation dynamique Iz (depuis erdv, n={iz_dyn['iz_dynamic_n']}):")
            print(f"    Iz: {iz_dyn['iz_dynamic_mean']:.0f} ± {iz_dyn['iz_dynamic_std']:.0f} kg·m²")
            print(f"    Médiane: {iz_dyn['iz_dynamic_median']:.0f} kg·m²")

    # ---- SSF ----
    ssf = track_final / (2 * h_final)
    print("\n" + "=" * 50)
    print("7. STATIC STABILITY FACTOR")
    print("=" * 50)
    print(f"  SSF = voie / (2·h) = {track_final:.3f} / (2·{h_final:.3f}) = {ssf:.3f}")
    risk = "HIGH" if ssf < 1.04 else "MODERATE" if ssf < 1.12 else "LOW"
    print(f"  Rollover risk (NHTSA): {risk}")

    # ---- Résumé ----
    print("\n" + "=" * 70)
    print("RÉSUMÉ — PARAMÈTRES POUR LE SIMULATEUR")
    print("=" * 70)

    summary = {
        "m": round(m_ref, 0),
        "ms": round(m_ref * 0.925, 0),  # ~7.5% unsprung
        "h": round(h_final, 3),
        "L1": round(track_final / 2, 3),
        "L2": round(track_final / 2, 3),
        "lf": round(lf, 3),
        "lr": round(lr, 3),
        "r": round(tire["tire_radius_mean"], 4),
        "ix": round(inertias_emp["ix_empirical"], 0),
        "iy": round(inertias_emp["iy_empirical"], 0),
        "iz": round(inertias_emp["iz_empirical"], 0),
        "ir": 2.5,
        "SSF": round(ssf, 3),
    }

    for k, v in summary.items():
        print(f"  {k:6s} = {v}")

    # Sauvegarder le résumé
    output_file = OUTPUT_DIR / "extracted_params.json"
    import json as json_mod
    with open(output_file, "w") as f:
        json_mod.dump(summary, f, indent=2)
    print(f"\n→ Paramètres sauvegardés dans {output_file}")


if __name__ == "__main__":
    main()
