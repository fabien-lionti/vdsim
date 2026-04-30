#!/usr/bin/env python3
"""Test hyperparams : pseudo-replay + finetune avec LSTM hidden=128 (D4 h2s)."""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    create_sequences_max, load_all_scenarios,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)
from pseudo_replay_loader import load_sim_scenarios as load_pseudo_raw
from data_ablation_pseudoreplay_finetune import (
    build_sequences, build_test_split, sample_uniform, clean_for_json,
    PSEUDO_DIR, FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE,
)
import pickle
PSEUDO_CACHE = OUTPUT_DIR / "pseudoreplay_cache.pkl"


def load_pseudo(path):
    if PSEUDO_CACHE.exists():
        print(f"  Reload cache {PSEUDO_CACHE.name}...")
        with open(PSEUDO_CACHE, 'rb') as f:
            return pickle.load(f)
    return load_pseudo_raw(path)

HIDDEN = 128
FRACTIONS = [0.0, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
CONFIG = ('D4', 2)

OUTPUT_FILE = OUTPUT_DIR / f"hyperparams_pseudoreplay_h{HIDDEN}_results.json"
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models") / f"lstm_pseudoreplay_v6_h{HIDDEN}.pt"
PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)


class LSTMBig(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_size))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_big(horizon_steps):
    if PRETRAIN_PATH.exists():
        print(f"  Reload pretrained from {PRETRAIN_PATH}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
        model = LSTMBig(N_FEATURES, hidden_size=HIDDEN)
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt['scaler_mean'])
        scaler.scale_ = np.array(ckpt['scaler_scale'])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = N_FEATURES
        scaler.n_samples_seen_ = 1
        return model, scaler

    print(f"  Pretrain LSTM hidden={HIDDEN} on pseudo-replay...")
    sim_scs = load_pseudo(PSEUDO_DIR)
    train_sc, val_sc = split_train_val(sim_scs, 0.2, seed=42)
    X_train, y_train = build_sequences(train_sc, horizon_steps)
    X_val, y_val = build_sequences(val_sc, horizon_steps)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape)

    model = LSTMBig(N_FEATURES, hidden_size=HIDDEN)
    print(f"  Model params: {count_params(model):,}")

    t0 = time.time()
    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    )
    print(f"  Pretrain done: {hist['n_epochs']} epochs, best val {hist['best_val_loss']:.4f} "
          f"({time.time()-t0:.0f}s)")

    torch.save({'state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
                'hidden': HIDDEN, 'history': hist}, PRETRAIN_PATH)
    print(f"  Saved: {PRETRAIN_PATH}")
    return model, scaler


def run_finetune(pretrained_sd, scaler_sim, train_pool, val_sc,
                 X_test_raw, y_test, horizon_steps, frac, seed):
    if frac == 0.0:
        model = LSTMBig(N_FEATURES, hidden_size=HIDDEN)
        model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})
        model = model.to(device)
        X_test = scaler_sim.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
        met = evaluate(model, X_test, y_test, device)
        met['n_train_scenarios'] = 0
        met['fraction_real'] = 0.0
        met['seed'] = int(seed)
        met['finetune_epochs'] = 0
        return met

    n_sample = max(1, int(frac * len(train_pool)))
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
    model = LSTMBig(N_FEATURES, hidden_size=HIDDEN)
    model.load_state_dict({k: v.clone() for k, v in pretrained_sd.items()})

    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
        lr=FINETUNE_LR, batch_size=BATCH_SIZE,
    )
    met = evaluate(model, X_test, y_test, device)
    met['n_train_scenarios'] = int(len(train_sampled))
    met['n_train_samples'] = int(len(y_train))
    met['fraction_real'] = float(frac)
    met['seed'] = int(seed)
    met['finetune_epochs'] = int(hist.get('n_epochs', 0))
    met['best_val_loss'] = float(hist.get('best_val_loss', float('nan')))
    return met


def main():
    print("=" * 70)
    print(f"HYPERPARAMS pseudo-replay : LSTM hidden={HIDDEN}")
    print(f"Device: {device}  Fractions: {FRACTIONS}  Seeds: {SEEDS}  Config: {CONFIG}")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print(f"\n[1/3] Pretrain...")
    model, scaler = pretrain_big(horizon_steps)
    pretrained_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[2/3] Chargement real...")
    scs, _ = load_all_scenarios(use_cache=True)

    ds, h = CONFIG
    cfg_key = f"{ds}_h{h}s"
    train_pool, val_sc, X_test_raw, y_test = build_test_split(scs, ds, horizon_steps)
    print(f"  Train pool: {len(train_pool)}, Val: {len(val_sc)}, Test: {len(y_test)} échantillons")

    results = {cfg_key: {}}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        print(f"  Resume: {sum(len(v) for cfg in results.values() for v in cfg.values())} runs saved")
    if cfg_key not in results:
        results[cfg_key] = {}

    print(f"\n[3/3] Ablation finetune hidden={HIDDEN}")
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
            if met is None:
                print(f"       → FAIL"); continue
            auc = met.get('auc_pr')
            print(f"       → R²={met['r2']:.3f} AUC-PR={auc:.3f} "
                  f"Recall={met['recall']:.0%} ({time.time()-t0:.0f}s)")
            results[cfg_key][frac_key].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # Summary vs baseline
    print(f"\n[résumé] Config {cfg_key}, hidden={HIDDEN}")
    baseline_file = OUTPUT_DIR / "data_ablation_pseudoreplay_finetune_results.json"
    with open(baseline_file) as f:
        baseline = json.load(f).get(cfg_key, {})
    print(f"{'frac':>10}  {'AUC-PR h64':>13}  {'AUC-PR h128':>13}  {'Δ':>7}")
    for frac in FRACTIONS:
        k = f"{int(frac*100)}pct_real"
        b = baseline.get(k, [])
        n = results[cfg_key].get(k, [])
        bmean = np.mean([r['auc_pr'] for r in b]) if b else float('nan')
        nmean = np.mean([r['auc_pr'] for r in n]) if n else float('nan')
        delta = nmean - bmean if not (np.isnan(bmean) or np.isnan(nmean)) else float('nan')
        print(f"{k:>10}  {bmean:>13.3f}  {nmean:>13.3f}  {delta:>+7.3f}")


if __name__ == "__main__":
    main()
