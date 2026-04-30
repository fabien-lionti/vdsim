#!/usr/bin/env python3
"""
Ablation SIM→REAL (pseudo-replay) : pretrain synthétique + finetune X% réel.

Pendant du `data_ablation_finetune.py` mais avec le pretrain issu du
**pseudo-replay** (data_v6_pseudoreplay) au lieu de replay réel.

Question DGA : un pipeline 100% synthétique (pseudo-replay) + peu de données
réelles peut-il approcher les performances real-only ?

Baseline (via ce pipeline) :
  Pseudo-replay seul (0% réel)   : AUC-PR 0.684 (D4 h2s)
  Real-only full                 : AUC-PR 0.916 (D4 h2s)

Protocole :
  1. Charger le modèle pseudo-replay déjà pré-entraîné (lstm_pseudoreplay_v6.pt).
  2. Pour chaque fraction X ∈ {0%, 25%, 50%, 75%, 100%} du train réel :
     - Finetune avec X% du train réel, lr réduit, early stopping sur val réel.
  3. Tester sur le test réel OOD.

Configs : D2 h2s + D4 h2s. Seeds : 3.
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time as time_module

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    LSTM, create_sequences_max, load_all_scenarios,
    DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)

PSEUDO_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/scenario_generation/data_v6_pseudoreplay/")
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models/lstm_pseudoreplay_v6.pt")
SEQ_LEN = 150

FRACTIONS_REAL = [0.0, 0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]
CONFIGS = [('D2', 2), ('D4', 2)]

VAL_FRACTION = 0.2
VAL_SEED = 42
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 40
FINETUNE_PATIENCE = 10

OUTPUT_FILE = OUTPUT_DIR / "data_ablation_pseudoreplay_finetune_results.json"
PREDS_DIR = OUTPUT_DIR / "preds_ablation_pseudoreplay_finetune"
PREDS_DIR.mkdir(parents=True, exist_ok=True)


def build_sequences(scenarios, horizon_steps):
    Xl, yl = [], []
    for s in scenarios:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if X is not None:
            Xl.append(X); yl.append(y)
    if not Xl:
        return None, None
    return np.concatenate(Xl), np.concatenate(yl)


def load_pretrained():
    """Reload saved pseudo-replay LSTM + scaler (from pseudo_replay_loader.py)."""
    ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
    model = LSTM(N_FEATURES)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device)

    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt['scaler_mean'])
    scaler.scale_ = np.array(ckpt['scaler_scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = N_FEATURES
    scaler.n_samples_seen_ = 1

    return model, scaler


def build_test_split(scenarios, dataset_name, horizon_steps):
    cfg = DATASET_CONFIGS[dataset_name]
    train_full = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scenarios if s['max_ltr'] > cfg['threshold']]
    if not train_full or not test_sc:
        return None, None, None, None
    train_pool, val_sc = split_train_val(train_full, val_fraction=VAL_FRACTION, seed=VAL_SEED)
    if not train_pool or not val_sc:
        return None, None, None, None
    X_test_raw, y_test = build_sequences(test_sc, horizon_steps)
    return train_pool, val_sc, X_test_raw, y_test


def sample_uniform(pool, n, seed):
    rng = np.random.default_rng(seed)
    n = min(max(1, n), len(pool))
    idx = rng.choice(len(pool), n, replace=False)
    return [pool[i] for i in idx]


def run_finetune_point(pretrained_sd, scaler_sim, train_pool, val_sc,
                       X_test_raw, y_test, horizon_steps,
                       fraction_real, seed, save_preds_key=None):
    if fraction_real == 0.0:
        model = LSTM(N_FEATURES)
        model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})
        model = model.to(device)
        X_test = scaler_sim.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
        met = evaluate(model, X_test, y_test, device)
        met['n_train_scenarios'] = 0
        met['fraction_real'] = 0.0
        met['seed'] = int(seed)
        met['finetune_epochs'] = 0

        if save_preds_key is not None:
            model.eval()
            with torch.no_grad():
                preds = np.concatenate([
                    model(torch.FloatTensor(X_test[i:i + 512].astype(np.float32)).to(device)).cpu().numpy()
                    for i in range(0, len(X_test), 512)
                ])
            preds = np.clip(preds, 0, 1.5)
            np.savez_compressed(PREDS_DIR / f"{save_preds_key}.npz",
                                preds=preds.astype(np.float32),
                                targets=y_test.astype(np.float32),
                                fraction_real=0.0, seed=int(seed))
        return met

    n_sample = max(1, int(fraction_real * len(train_pool)))
    train_sampled = sample_uniform(train_pool, n_sample, seed)

    X_train_raw, y_train = build_sequences(train_sampled, horizon_steps)
    X_val_raw, y_val = build_sequences(val_sc, horizon_steps)
    if X_train_raw is None or X_val_raw is None:
        return None

    X_train = scaler_sim.transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val = scaler_sim.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_test = scaler_sim.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTM(N_FEATURES)
    model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})

    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
        lr=FINETUNE_LR, batch_size=BATCH_SIZE,
    )

    model.eval()
    with torch.no_grad():
        preds = np.concatenate([
            model(torch.FloatTensor(X_test[i:i + 512].astype(np.float32)).to(device)).cpu().numpy()
            for i in range(0, len(X_test), 512)
        ])
    preds = np.clip(preds, 0, 1.5)

    met = evaluate(model, X_test, y_test, device)
    met['n_train_scenarios'] = int(len(train_sampled))
    met['n_train_samples'] = int(len(y_train))
    met['fraction_real'] = float(fraction_real)
    met['seed'] = int(seed)
    met['finetune_epochs'] = int(hist.get('n_epochs', 0))
    met['best_val_loss'] = float(hist.get('best_val_loss', float('nan')))

    if save_preds_key is not None:
        np.savez_compressed(PREDS_DIR / f"{save_preds_key}.npz",
                            preds=preds.astype(np.float32),
                            targets=y_test.astype(np.float32),
                            fraction_real=float(fraction_real),
                            seed=int(seed))
    return met


def clean_for_json(met):
    out = {}
    for k, v in met.items():
        if isinstance(v, (np.floating,)):
            fv = float(v)
            out[k] = None if np.isnan(fv) else fv
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def main():
    print("=" * 70)
    print("ABLATION SIM→REAL : pretrain PSEUDO-REPLAY + finetune X% réel")
    print(f"Device: {device}  Fractions: {FRACTIONS_REAL}  Seeds: {SEEDS}  Configs: {CONFIGS}")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print(f"\n[1/4] Chargement modèle pré-entraîné pseudo-replay depuis {PRETRAIN_PATH}...")
    if not PRETRAIN_PATH.exists():
        raise RuntimeError(f"Modèle introuvable. Lance d'abord pseudo_replay_loader.py")
    pretrained_model, scaler_sim = load_pretrained()
    pretrained_sd = {k: v.cpu().clone() for k, v in pretrained_model.state_dict().items()}
    print(f"  Modèle + scaler chargés.")

    print(f"\n[2/4] Chargement des scénarios réels (cache)...")
    real_scenarios, _ = load_all_scenarios(use_cache=True)
    print(f"  → {len(real_scenarios)} scénarios réels")

    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"  Resume: {sum(len(v) for cfg in results.values() for v in cfg.values())} runs sauvegardés")

    total_runs = len(CONFIGS) * len(FRACTIONS_REAL) * len(SEEDS)
    run_idx = 0

    for ds_name, h in CONFIGS:
        hkey = f"h{h}s"
        cfg_key = f"{ds_name}_{hkey}"
        print(f"\n[3/4] Config {cfg_key}")

        train_pool, val_sc, X_test_raw, y_test = build_test_split(real_scenarios, ds_name, horizon_steps)
        if train_pool is None:
            continue
        print(f"  Train pool: {len(train_pool)}, Val: {len(val_sc)}, Test: {len(y_test)} échantillons")

        if cfg_key not in results:
            results[cfg_key] = {}

        for frac in FRACTIONS_REAL:
            frac_key = f"{int(frac * 100)}pct_real"
            if frac_key not in results[cfg_key]:
                results[cfg_key][frac_key] = []
            done_seeds = {r['seed'] for r in results[cfg_key][frac_key]}

            for seed in SEEDS:
                run_idx += 1
                if seed in done_seeds:
                    print(f"  [{run_idx}/{total_runs}] {frac_key} seed={seed}: skip")
                    continue

                n_train = int(frac * len(train_pool))
                t0 = time_module.time()
                print(f"  [{run_idx}/{total_runs}] {frac_key} seed={seed} "
                      f"(n_train_real={n_train})...", flush=True)

                save_key = (f"{cfg_key}_{frac_key}_seed{seed}"
                            if seed == 0 else None)
                met = run_finetune_point(
                    pretrained_sd, scaler_sim, train_pool, val_sc,
                    X_test_raw, y_test, horizon_steps, frac, seed,
                    save_preds_key=save_key,
                )
                if met is None:
                    print(f"       → ÉCHEC")
                    continue

                auc = met.get('auc_pr')
                auc_s = f"{auc:.3f}" if auc is not None and not (isinstance(auc, float) and np.isnan(auc)) else "N/A"
                dt = time_module.time() - t0
                print(f"       → R²={met['r2']:.3f} AUC-PR={auc_s} "
                      f"Recall={met['recall']:.0%} ({dt:.0f}s)")

                results[cfg_key][frac_key].append(clean_for_json(met))
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(results, f, indent=2)

    print(f"\n[4/4] Résumé\n" + "=" * 70)
    for cfg_key, fracs in results.items():
        print(f"\n{cfg_key}:")
        for frac_key in sorted(fracs.keys(), key=lambda k: int(k.split('pct')[0])):
            runs = fracs[frac_key]
            if not runs:
                continue
            r2s = [r['r2'] for r in runs]
            aucs = [r['auc_pr'] for r in runs if r.get('auc_pr') is not None]
            recalls = [r['recall'] for r in runs]
            auc_s = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}" if aucs else "N/A"
            print(f"  {frac_key:>14}: R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}  "
                  f"AUC-PR={auc_s}  Rec={np.mean(recalls):.0%} ({len(runs)} seeds)")


if __name__ == "__main__":
    main()
