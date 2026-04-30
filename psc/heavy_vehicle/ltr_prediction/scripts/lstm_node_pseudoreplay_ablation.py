#!/usr/bin/env python3
"""Ablation LSTM + features NODE, pretrain pseudo-replay puis finetune X% réel."""

import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    DT, N_FEATURES, FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
    create_sequences_max, load_all_scenarios,
)
from training_utils import split_train_val, weighted_mse_loss

from lstm_node_real import (
    load_node_backbone, LSTMPhysicsWrapper, PHYSICS_PROJ_DIM,
    build_sequences, normalize_for_node, train_physics_model,
    evaluate_physics, clean_for_json,
)

from data_ablation_pseudoreplay_finetune import (
    sample_uniform, FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE,
)

PSEUDO_CACHE = OUTPUT_DIR / "pseudoreplay_cache.pkl"
FRACTIONS = [0.0, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
CONFIG = ('D4', 2)

OUTPUT_FILE = OUTPUT_DIR / "lstm_node_pseudoreplay_ablation_results.json"
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models") / "lstm_node_pseudoreplay.pt"
PRETRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)


def pretrain_lstm_node_on_pseudo(node_backbone, node_scaler, node_n_input, horizon_steps):
    """Pretrain LSTM+NODE sur 400 scénarios pseudo-replay."""
    if PRETRAIN_PATH.exists():
        print(f"  Reload pretrain {PRETRAIN_PATH.name}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
        model = LSTMPhysicsWrapper(
            node_backbone=node_backbone, node_n_input=node_n_input,
            physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48,
        )
        model.load_state_dict(ckpt['state_dict'])
        model = model.to(device)
        base_scaler = StandardScaler()
        base_scaler.mean_ = np.array(ckpt['scaler_mean'])
        base_scaler.scale_ = np.array(ckpt['scaler_scale'])
        base_scaler.var_ = base_scaler.scale_ ** 2
        base_scaler.n_features_in_ = N_FEATURES
        base_scaler.n_samples_seen_ = 1
        return model, base_scaler

    print(f"  Loading pseudo-replay cache...")
    with open(PSEUDO_CACHE, 'rb') as f:
        sim_scs = pickle.load(f)
    train_sc, val_sc = split_train_val(sim_scs, 0.2, seed=42)
    print(f"  Pretrain: {len(train_sc)} train / {len(val_sc)} val scenarios")

    X_train_raw, y_train = build_sequences(train_sc, horizon_steps)
    X_val_raw, y_val = build_sequences(val_sc, horizon_steps)
    print(f"  Sequences: train {X_train_raw.shape}, val {X_val_raw.shape}")

    base_scaler = StandardScaler()
    X_train_base = base_scaler.fit_transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val_base = base_scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)

    X_train_node = normalize_for_node(X_train_raw, node_scaler, node_n_input)
    X_val_node = normalize_for_node(X_val_raw, node_scaler, node_n_input)

    model = LSTMPhysicsWrapper(
        node_backbone=node_backbone, node_n_input=node_n_input,
        physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    t0 = time.time()
    model, hist = train_physics_model(
        model, X_train_base, X_train_node, y_train,
        X_val_base, X_val_node, y_val,
        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    )
    print(f"  Pretrain done: {hist['n_epochs']} epochs, best val {hist['best_val_loss']:.4f} "
          f"({time.time()-t0:.0f}s)")

    torch.save({'state_dict': model.state_dict(),
                'scaler_mean': base_scaler.mean_, 'scaler_scale': base_scaler.scale_}, PRETRAIN_PATH)
    return model, base_scaler


def run_point(pretrained_sd, base_scaler, node_backbone, node_scaler, node_n_input,
              train_pool, val_sc, X_test_base, X_test_node, y_test,
              horizon_steps, frac, seed):
    """Évalue ou finetune selon la fraction."""
    # Build the pretrained model with frozen NODE backbone reattached
    model = LSTMPhysicsWrapper(
        node_backbone=node_backbone, node_n_input=node_n_input,
        physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48,
    )
    # Load pretrained weights (proj + LSTM + FC), ignore NODE backbone (already attached)
    model.load_state_dict(pretrained_sd, strict=False)
    model = model.to(device)

    if frac == 0.0:
        met = evaluate_physics(model, X_test_base, X_test_node, y_test)
        met['n_train_scenarios'] = 0
        met['fraction_real'] = 0.0
        met['seed'] = int(seed)
        met['finetune_epochs'] = 0
        return met

    # Sample real scenarios
    n_sample = max(1, int(frac * len(train_pool)))
    train_sampled = sample_uniform(train_pool, n_sample, seed)

    X_train_raw, y_train = build_sequences(train_sampled, horizon_steps)
    X_val_raw, y_val = build_sequences(val_sc, horizon_steps)

    X_train_base = base_scaler.transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val_base = base_scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_train_node = normalize_for_node(X_train_raw, node_scaler, node_n_input)
    X_val_node = normalize_for_node(X_val_raw, node_scaler, node_n_input)

    torch.manual_seed(seed); np.random.seed(seed)
    model, hist = train_physics_model(
        model, X_train_base, X_train_node, y_train,
        X_val_base, X_val_node, y_val,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
        lr=FINETUNE_LR, batch_size=BATCH_SIZE,
    )

    met = evaluate_physics(model, X_test_base, X_test_node, y_test)
    met['n_train_scenarios'] = int(len(train_sampled))
    met['n_train_samples'] = int(len(y_train))
    met['fraction_real'] = float(frac)
    met['seed'] = int(seed)
    met['finetune_epochs'] = int(hist.get('n_epochs', 0))
    met['best_val_loss'] = float(hist.get('best_val_loss', float('nan')))
    return met


def main():
    print("=" * 70)
    print("ABLATION : LSTM+NODE pretrain pseudo-replay + finetune X% réel (D4 h2s)")
    print(f"Fractions: {FRACTIONS}  Seeds: {SEEDS}")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print(f"\n[1/4] NODE backbone...")
    node_backbone, node_scaler, node_n_input = load_node_backbone()

    print(f"\n[2/4] Pretrain LSTM+NODE on pseudo-replay...")
    model_pretrain, base_scaler = pretrain_lstm_node_on_pseudo(
        node_backbone, node_scaler, node_n_input, horizon_steps)
    pretrained_sd = {k: v.cpu().clone() for k, v in model_pretrain.state_dict().items()}

    print(f"\n[3/4] Load real D4 h2s test split...")
    scs, _ = load_all_scenarios(use_cache=True)
    ds, h = CONFIG
    cfg = DATASET_CONFIGS[ds]
    train_full = [s for s in scs if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scs if s['max_ltr'] > cfg['threshold']]
    train_pool, val_sc = split_train_val(train_full, val_fraction=0.2, seed=42)
    print(f"  Train pool: {len(train_pool)}, Val: {len(val_sc)}, Test: {len(test_sc)}")

    X_test_raw, y_test = build_sequences(test_sc, horizon_steps)
    X_test_base = base_scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
    X_test_node = normalize_for_node(X_test_raw, node_scaler, node_n_input)

    results = {'D4_h2s': {}}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
    cfg_key = 'D4_h2s'
    if cfg_key not in results:
        results[cfg_key] = {}

    print(f"\n[4/4] Ablation")
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
                print(f"  [{idx}/{total}] {frac_key} seed={seed}: skip"); continue
            t0 = time.time()
            print(f"  [{idx}/{total}] {frac_key} seed={seed}...", flush=True)
            met = run_point(pretrained_sd, base_scaler, node_backbone, node_scaler,
                            node_n_input, train_pool, val_sc,
                            X_test_base, X_test_node, y_test,
                            horizon_steps, frac, seed)
            auc = met.get('auc_pr')
            dt = time.time() - t0
            print(f"       → R²={met['r2']:.3f} AUC-PR={auc:.3f} "
                  f"Recall={met['recall']:.0%} ({dt:.0f}s)")
            results[cfg_key][frac_key].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2)

    # Summary
    print(f"\n[Résumé] {cfg_key}, LSTM+NODE pretrain pseudo-replay")
    baseline_file = OUTPUT_DIR / "data_ablation_pseudoreplay_finetune_results.json"
    with open(baseline_file) as f:
        baseline = json.load(f).get(cfg_key, {})
    print(f"{'frac':>10}  {'pseudo (h64)':>14}  {'node+pseudo':>13}  {'Δ':>8}")
    for frac in FRACTIONS:
        k = f"{int(frac*100)}pct_real"
        b = baseline.get(k, [])
        n = results[cfg_key].get(k, [])
        bmean = np.mean([r['auc_pr'] for r in b]) if b else float('nan')
        nmean = np.mean([r['auc_pr'] for r in n]) if n else float('nan')
        delta = nmean - bmean if not (np.isnan(bmean) or np.isnan(nmean)) else float('nan')
        print(f"{k:>10}  {bmean:>14.3f}  {nmean:>13.3f}  {delta:>+8.3f}")


if __name__ == "__main__":
    main()
