#!/usr/bin/env python3
"""
Test : pretrain LSTM sur pseudo-replay synthétique → évalué sur réel.

Données : scénarios v6 générés synthétiquement (OU process pour vx + δf avec
événements injectés), puis injectés dans DOF10-V2 en open-loop.

Référence attendue (entre V5 pur et Replay réel) :
  Sim V1 pur               : AUC-PR 0.086
  Sim V5 HumanDriver + DR  : AUC-PR 0.442 (D4 h2s)
  Pseudo-replay (ce test)  : ??? (cible : 0.45-0.60)
  Replay réel DOF10-V2     : AUC-PR 0.624
  Real-only full           : AUC-PR 0.916
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time as time_module

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    LSTM, create_sequences_max, load_all_scenarios,
    DT, N_FEATURES, DATASET_CONFIGS, device,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)

PSEUDO_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/scenario_generation/data_v6_pseudoreplay/")
SEQ_LEN = 150
MODEL_OUT = Path(__file__).parent.parent.parent / "models" / "lstm_pseudoreplay_v6.pt"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)


def load_sim_scenarios(directory, max_files=None):
    files = sorted(directory.glob("*.csv"))
    if max_files:
        files = files[:max_files]
    print(f"  Chargement {len(files)} scénarios depuis {directory.name}...", flush=True)
    scenarios = []
    for i, fp in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(files)}...", flush=True)
        try:
            if fp.stat().st_size > 15_000_000:
                continue
            df = pd.read_csv(fp)
            required = ['vx', 'vy', 'psi_dot', 'phi', 'theta', 'delta_f', 'LTRmax']
            if not all(c in df.columns for c in required):
                continue
            delta_f = df['delta_f'].values
            ltr = np.clip(np.abs(df['LTRmax'].values), 0, 1)
            built = pd.DataFrame({
                'vx': df['vx'].values, 'vy': df['vy'].values,
                'psi_dot': df['psi_dot'].values,
                'phi': df['phi'].values, 'theta': df['theta'].values,
                'delta_f': delta_f,
                'delta_f_dot': np.gradient(delta_f, DT),
                'LTR_current': ltr, 'dLTR_dt': np.gradient(ltr, DT),
                'LTRmax': ltr,
            }).dropna().reset_index(drop=True)
            if len(built) > SEQ_LEN + 500:
                scenarios.append({
                    'df': built,
                    'max_ltr': float(built['LTRmax'].max()),
                    'name': fp.stem,
                })
        except Exception:
            continue
    print(f"  → {len(scenarios)} scénarios valides", flush=True)
    return scenarios


def build_sequences(scenarios, horizon_steps):
    Xl, yl = [], []
    for s in scenarios:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if X is not None:
            Xl.append(X); yl.append(y)
    if not Xl:
        return None, None
    return np.concatenate(Xl), np.concatenate(yl)


def main():
    print("=" * 70)
    print("TEST : pretrain pseudo-replay synthétique → Real")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print("\n[1/3] Chargement pseudo-replay...")
    sim_scs = load_sim_scenarios(PSEUDO_DIR)
    if not sim_scs:
        print("ERROR: aucun scénario v6. Lance d'abord generate_pseudo_replay.py.")
        return

    # Stats distribution LTR peaks
    peaks = np.array([s['max_ltr'] for s in sim_scs])
    print(f"  LTR peaks: p10={np.percentile(peaks,10):.2f}, p50={np.percentile(peaks,50):.2f}, "
          f"p90={np.percentile(peaks,90):.2f}, >0.7: {(peaks>0.7).sum()}")

    sim_train_sc, sim_val_sc = split_train_val(sim_scs, 0.2, seed=42)
    print(f"  Train: {len(sim_train_sc)}, Val: {len(sim_val_sc)}")

    X_train, y_train = build_sequences(sim_train_sc, horizon_steps)
    X_val, y_val = build_sequences(sim_val_sc, horizon_steps)
    print(f"  Sequences: train {X_train.shape}, val {X_val.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape)

    model = LSTM(N_FEATURES)
    t0 = time_module.time()
    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    )
    print(f"  Pretrain done: {hist['n_epochs']} epochs, best val {hist['best_val_loss']:.4f} "
          f"({time_module.time()-t0:.0f}s)")

    # Sauvegarde modèle + scaler
    torch.save({'state_dict': model.state_dict(), 'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_, 'history': hist}, MODEL_OUT)
    print(f"  Model saved: {MODEL_OUT}")

    # Eval sur réel
    print("\n[2/3] Chargement real...")
    real_scs, _ = load_all_scenarios(use_cache=True)

    results = {}
    for ds in ['D4', 'D2']:
        cfg = DATASET_CONFIGS[ds]
        test_sc = [s for s in real_scs if s['max_ltr'] > cfg['threshold']]
        X_test, y_test = build_sequences(test_sc, horizon_steps)
        X_test = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)
        met = evaluate(model, X_test, y_test, device)
        results[ds] = met

        print(f"\n{ds} h2s (test={len(test_sc)} scénarios, {len(X_test)} séquences):")
        print(f"  R²     = {met['r2']:.3f}")
        print(f"  AUC-PR = {met['auc_pr']:.3f}")
        print(f"  Recall = {met['recall']:.0%}")

    print("\n" + "=" * 70)
    print("COMPARAISON :")
    print(f"  Sim V1 pur            : AUC-PR = 0.086")
    print(f"  Sim V5 HD+DR          : AUC-PR = 0.442 (D4 h2s)")
    print(f"  Pseudo-replay (ici)   : AUC-PR = {results['D4']['auc_pr']:.3f} (D4 h2s)")
    print(f"  Replay DOF10-V2       : AUC-PR = 0.624")
    print(f"  Real-only full        : AUC-PR = 0.916 (D4 h2s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
