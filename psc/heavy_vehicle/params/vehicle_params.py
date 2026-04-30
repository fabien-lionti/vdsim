#!/usr/bin/env python3
"""Paramètres du véhicule lourd (~3200 kg) extraits des données réelles DXD."""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireParams

# =============================================================================
# PARAMÈTRES EXTRAITS DES DONNÉES RÉELLES
# =============================================================================

# Masse (midpoint vide 3001 kg / chargé 3380 kg)
M = 3190.0          # kg — masse totale
MS = 2951.0          # kg — masse suspendue (~92.5% de M)

# Géométrie
H = 0.68             # m — hauteur CdG (grid search H/zeta/ARB sur erdv+slalom, Transit ~0.70-0.75)
L1 = 0.814           # m — demi-voie (voie totale = 1.629 m)
L2 = 0.814           # m — demi-voie
LF = 1.802           # m — distance CdG → essieu avant
LR = 1.364           # m — distance CdG → essieu arrière (F/R = 43/57)
R = 0.389            # m — rayon de roulement pneu (diam ~779 mm)

# Inerties (formules empiriques : Abe pour Iz, distribution latérale pour Ix)
IX = 2100.0          # kg·m² — roulis (corrigé: 0.25 × m × track²)
IY = 3045.0          # kg·m² — tangage (≈ 0.75 × Iz)
IZ = 4060.0          # kg·m² — lacet (Abe: 0.127 × m × L²)
IR = 2.5             # kg·m² — roue (proportionnel à r⁴, plus gros pneus)

# Aérodynamique
RA = 15.0            # N·s/m — résistance au roulement (scaling avec masse)
S = 2.8              # m² — surface frontale (véhicule plus large)
CX = 0.35            # — coefficient de traînée

# Suspension — préserver fréquence naturelle ~1.1 Hz pour véhicule lourd
# omega_n = 2*pi*1.1 = 6.91 rad/s
# ms_corner = MS / 4 = 737.8 kg
# ks = omega_n² * ms_corner = 47.7 * 737.8 = 35,225 N/m
# Différenciation AV/AR proportionnelle à la charge statique (43/57)
_OMEGA_N = 2 * np.pi * 1.47  # rad/s — fréquence naturelle (grid search: 1.55×0.9≈1.47, Transit ~1.1-1.5 Hz)
_MS_CORNER = MS / 4
_KS_BASE = _OMEGA_N ** 2 * _MS_CORNER  # ~35,225 N/m
_ZETA = 0.30  # ratio d'amortissement (Transit typique 0.20-0.40, calibré grid search)
_DS_BASE = 2 * _ZETA * np.sqrt(_KS_BASE * _MS_CORNER)  # ~5,100 N·s/m

# Front légèrement plus souple (43% charge), rear plus raide (57%)
KS_F = round(_KS_BASE * 0.92, 0)   # ~32,400 N/m
KS_R = round(_KS_BASE * 1.08, 0)   # ~38,050 N/m
DS_F = round(_DS_BASE * 0.92, 0)    # ~4,690 N·s/m
DS_R = round(_DS_BASE * 1.08, 0)    # ~5,510 N·s/m

# Pneus Pacejka — shape factors adaptés pour pneus plus gros sous charge élevée
PACEJKA_BY = 4.0     # stiffness factor latéral (recalibré: sweep sur slalom/erdv/sinus/ld)
PACEJKA_CY = 1.3     # shape factor latéral
PACEJKA_EY = -1.0    # curvature factor latéral
PACEJKA_BX = 10.0    # stiffness factor longitudinal
PACEJKA_CX = 1.6     # shape factor longitudinal
PACEJKA_EX = -0.5    # curvature factor longitudinal

# SSF calculé
SSF = (L1 + L2) / (2 * H)  # ≈ 0.95

# PID speed controller — scaling linéaire avec la masse
PID_KP = 2100.0      # proportionnel (1000 × 3190/1500)
PID_KI = 21.0        # intégral
PID_KD = 0.0
TORQUE_LIMITS = (-8500.0, 8500.0)

# Stanley steering controller
STANLEY_K = 0.2       # gain (inchangé, à valider)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def make_heavy_vehicle(mu: float) -> DOF10:
    """
    Crée le modèle véhicule 10DOF pour le véhicule lourd.

    Args:
        mu: coefficient de friction pneumatique (0.55 - 1.0 typique)

    Returns:
        DOF10 model prêt pour la simulation
    """
    params = VehiclePhysicalParams10DOF(
        g=9.81, m=M, ms=MS,
        lf=LF, lr=LR, h=H,
        L1=L1, L2=L2, r=R,
        ix=IX, iy=IY, iz=IZ, ir=IR,
        ra=RA, s=S, cx=CX,
        ks1=KS_F, ks2=KS_F, ks3=KS_R, ks4=KS_R,
        ds1=DS_F, ds2=DS_F, ds3=DS_R, ds4=DS_R,
    )

    L = params.lf + params.lr
    Fzf0 = params.m * params.g * (params.lr / L)  # Charge statique essieu avant
    Fzr0 = params.m * params.g * (params.lf / L)  # Charge statique essieu arrière

    def make_tire(fz0):
        return SimplifiedPacejkaTireParams(
            By=PACEJKA_BY, Cy=PACEJKA_CY, Dy=mu * fz0, Ey=PACEJKA_EY,
            Bx=PACEJKA_BX, Cx=PACEJKA_CX, Dx=mu * fz0, Ex=PACEJKA_EX,
        )

    config = VehicleConfig10DOF(
        vehicle=params,
        tire1=make_tire(0.5 * Fzf0),  # Front-left
        tire2=make_tire(0.5 * Fzf0),  # Front-right
        tire3=make_tire(0.5 * Fzr0),  # Rear-left
        tire4=make_tire(0.5 * Fzr0),  # Rear-right
    )
    return DOF10(config)


# =============================================================================
# DISPLAY
# =============================================================================

if __name__ == "__main__":
    print("=== Véhicule lourd — Paramètres ===")
    print(f"Masse: {M} kg (suspendue: {MS} kg)")
    print(f"Empattement: {LF+LR:.3f} m (lf={LF}, lr={LR})")
    print(f"Voie: {L1+L2:.3f} m")
    print(f"CdG: h={H} m")
    print(f"Rayon pneu: {R} m")
    print(f"Inerties: Ix={IX}, Iy={IY}, Iz={IZ} kg·m²")
    print(f"Suspension: ks_f={KS_F}, ks_r={KS_R} N/m | ds_f={DS_F}, ds_r={DS_R} N·s/m")
    print(f"Freq naturelle: {_OMEGA_N/(2*np.pi):.2f} Hz | Amort: {_ZETA}")
    print(f"SSF: {SSF:.3f}")
    print(f"PID: kp={PID_KP}, ki={PID_KI}")
    print(f"\nTest make_heavy_vehicle(mu=0.8)...")
    model = make_heavy_vehicle(0.8)
    print(f"  OK — model type: {type(model).__name__}")
