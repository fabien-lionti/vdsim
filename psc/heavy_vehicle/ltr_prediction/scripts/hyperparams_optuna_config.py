#!/usr/bin/env python3
"""Test de la config Optuna sur pseudo-replay + finetune X% réel (D4 h2s)."""

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    DT, N_FEATURES, FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, PATIENCE, load_all_scenarios,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)
from data_ablation_pseudoreplay_finetune import (
    build_test_split, sample_uniform, clean_for_json,
    FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE,
)

# Optuna config
OPTUNA_HIDDEN = 64
OPTUNA_DROPOUT = 0.4
OPTUNA_SEQ_LEN = 200
OPTUNA_LR = 3e-4
OPTUNA_ACTIVATION = 'gelu'

FRACTIONS = [0.0, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
CONFIG = ('D4', 2)

PSEUDO_CACHE = OUTPUT_DIR / "pseudoreplay_cache.pkl"
OUTPUT_FILE = OUTPUT_DIR / f"hyperparams_optuna_results.json"
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models") / f"lstm_pseudoreplay_v6_optuna.pt"
PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)


class LSTMOptuna(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def create_sequences_custom(df, horizon_steps, seq_len):
    """Variante de create_sequences_max avec seq_len paramétré."""
    features = df[FEATURES].values
    ltr = df['LTRmax'].values
    X, y = [], []
    for i in range(0, len(df) - seq_len - horizon_steps, 10):
        X.append(features[i:i + seq_len])
        y.append(np.max(ltr[i + seq_len:i + seq_len + horizon_steps]))
    if not X:
        return None, None
    return np.array(X), np.array(y)


def build_sequences_custom(scenarios, horizon_steps, seq_len):
    Xl, yl = [], []
    for s in scenarios:
        X, y = create_sequences_custom(s['df'], horizon_steps, seq_len)
        if X is not None:
            Xl.append(X); yl.append(y)
    if not Xl:
        return None, None
    return np.concatenate(Xl), np.concatenate(yl)


def build_test_split_custom(scenarios, dataset_name, horizon_steps, seq_len):
    cfg = DATASET_CONFIGS[dataset_name]
    train_full = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scenarios if s['max_ltr'] > cfg['threshold']]
    train_pool, val_sc = split_train_val(train_full, val_fraction=0.2, seed=42)
    X_test_raw, y_test = build_sequences_custom(test_sc, horizon_steps, seq_len)
    return train_pool, val_sc, X_test_raw, y_test


def pretrain():
    horizon_steps = int(2 / DT)
    if PRETRAIN_PATH.exists():
        print(f"  Reload pretrain {PRETRAIN_PATH.name}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
        model = LSTMOptuna(N_FEATURES, hidden_size=OPTUNA_HIDDEN, dropout=OPTUNA_DROPOUT)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt['scaler_mean'])
        scaler.scale_ = np.array(ckpt['scaler_scale'])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = N_FEATURES
        scaler.n_samples_seen_ = 1
        return model, scaler

    print(f"  Loading pseudo-replay cache...")
    with open(PSEUDO_CACHE, 'rb') as f:
        sim_scs = pickle.load(f)

    train_sc, val_sc = split_train_val(sim_scs, 0.2, seed=42)
    X_train, y_train = build_sequences_custom(train_sc, horizon_steps, OPTUNA_SEQ_LEN)
    X_val, y_val = build_sequences_custom(val_sc, horizon_steps, OPTUNA_SEQ_LEN)
    print(f"  Sequences: train {X_train.shape}, val {X_val.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape)

    model = LSTMOptuna(N_FEATURES, hidden_size=OPTUNA_HIDDEN, dropout=OPTUNA_DROPOUT)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    t0 = time.time()
    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=EPOCHS, patience=PATIENCE, lr=OPTUNA_LR, batch_size=BATCH_SIZE,
    )
    print(f"  Pretrain done: {hist['n_epochs']} epochs, best val {hist['best_val_loss']:.4f} "
          f"({time.time()-t0:.0f}s)")

    torch.save({'state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, PRETRAIN_PATH)
    return model, scaler


def run_finetune(pretrained_sd, scaler, train_pool, val_sc,
                 X_test_raw, y_test, horizon_steps, frac, seed):
    if frac == 0.0:
        model = LSTMOptuna(N_FEATURES, hidden_size=OPTUNA_HIDDEN, dropout=OPTUNA_DROPOUT)
        model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})
        model = model.to(device)
        X_test = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
        met = evaluate(model, X_test, y_test, device)
        met.update({'n_train_scenarios': 0, 'fraction_real': 0.0,
                    'seed': int(seed), 'finetune_epochs': 0})
        return met

    n_sample = max(1, int(frac * len(train_pool)))
    train_sampled = sample_uniform(train_pool, n_sample, seed)
    X_train_raw, y_train = build_sequences_custom(train_sampled, horizon_steps, OPTUNA_SEQ_LEN)
    X_val_raw, y_val = build_sequences_custom(val_sc, horizon_steps, OPTUNA_SEQ_LEN)

    X_train = scaler.transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)

    torch.manual_seed(seed); np.random.seed(seed)
    model = LSTMOptuna(N_FEATURES, hidden_size=OPTUNA_HIDDEN, dropout=OPTUNA_DROPOUT)
    model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})

    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
        lr=OPTUNA_LR * 0.33, batch_size=BATCH_SIZE,  # finetune lr = pretrain lr / 3
    )
    met = evaluate(model, X_test, y_test, device)
    met.update({'n_train_scenarios': int(len(train_sampled)),
                'n_train_samples': int(len(y_train)),
                'fraction_real': float(frac), 'seed': int(seed),
                'finetune_epochs': int(hist.get('n_epochs', 0)),
                'best_val_loss': float(hist.get('best_val_loss', float('nan')))})
    return met


def main():
    print("=" * 70)
    print(f"HYPERPARAMS OPTUNA : h={OPTUNA_HIDDEN}, dropout={OPTUNA_DROPOUT}, "
          f"GELU, seq_len={OPTUNA_SEQ_LEN}, lr={OPTUNA_LR}")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print(f"\n[1/3] Pretrain...")
    model, scaler = pretrain()
    pretrained_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[2/3] Load real + build test split...")
    scs, _ = load_all_scenarios(use_cache=True)
    ds, h = CONFIG
    cfg_key = f"{ds}_h{h}s"
    train_pool, val_sc, X_test_raw, y_test = build_test_split_custom(scs, ds, horizon_steps, OPTUNA_SEQ_LEN)
    print(f"  Train pool: {len(train_pool)}, Val: {len(val_sc)}, Test: {len(y_test)} séquences")

    results = {cfg_key: {}}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)

    print(f"\n[3/3] Ablation finetune")
    total = len(FRACTIONS) * len(SEEDS)
    idx = 0
    for frac in FRACTIONS:
        frac_key = f"{int(frac*100)}pct_real"
        if frac_key not in results[cfg_key]:
            results[cfg_key][frac_key] = []
        done = {r['seed'] for r in results[cfg_key][frac_key]}

        for seed in SEEDS:
            idx += 1
            if seed in done:
                print(f"  [{idx}/{total}] {frac_key} seed={seed}: skip")
                continue
            t0 = time.time()
            print(f"  [{idx}/{total}] {frac_key} seed={seed}...", flush=True)
            met = run_finetune(pretrained_sd, scaler, train_pool, val_sc,
                               X_test_raw, y_test, horizon_steps, frac, seed)
            auc = met.get('auc_pr')
            print(f"       → R²={met['r2']:.3f} AUC-PR={auc:.3f} "
                  f"Recall={met['recall']:.0%} ({time.time()-t0:.0f}s)")
            results[cfg_key][frac_key].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # Summary vs baseline
    print(f"\n[résumé] {cfg_key}, Optuna config")
    baseline_file = OUTPUT_DIR / "data_ablation_pseudoreplay_finetune_results.json"
    with open(baseline_file) as f:
        baseline = json.load(f).get(cfg_key, {})
    print(f"{'frac':>10}  {'baseline':>11}  {'optuna':>11}  {'Δ':>8}")
    for frac in FRACTIONS:
        k = f"{int(frac*100)}pct_real"
        b = baseline.get(k, [])
        n = results[cfg_key].get(k, [])
        bmean = np.mean([r['auc_pr'] for r in b]) if b else float('nan')
        nmean = np.mean([r['auc_pr'] for r in n]) if n else float('nan')
        delta = nmean - bmean if not (np.isnan(bmean) or np.isnan(nmean)) else float('nan')
        print(f"{k:>10}  {bmean:>11.3f}  {nmean:>11.3f}  {delta:>+8.3f}")


if __name__ == "__main__":
    main()
