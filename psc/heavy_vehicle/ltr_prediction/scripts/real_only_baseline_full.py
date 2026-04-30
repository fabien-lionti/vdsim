#!/usr/bin/env python3
"""
Baseline real-only × D2+D4 × h1/h2/h4 × 3 seeds.
Pour tracer la ligne de référence sur les plots.
"""
import sys, json, time, numpy as np, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (LSTM, DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
                             BATCH_SIZE, EPOCHS, LR, PATIENCE, load_all_scenarios)
from training_utils import train_model_clean, evaluate, split_train_val, weighted_mse_loss
from pseudo_replay_loader import build_sequences
from data_ablation_pseudoreplay_finetune import clean_for_json

OUTPUT_FILE = OUTPUT_DIR / "real_only_baseline_full_results.json"
SEEDS = [0, 1, 2]
# Configs manquantes uniquement : D4 h4s, D2 h1s, D2 h4s
# (D4 h1s, D4 h2s, D2 h2s déjà connus depuis rapport intermédiaire)
MISSING_CONFIGS = [('D4', 4), ('D2', 1), ('D2', 4)]
DATASETS = sorted({d for d, _ in MISSING_CONFIGS})
HORIZONS = sorted({h for _, h in MISSING_CONFIGS})


def main():
    print("=" * 70)
    print(f"REAL-ONLY BASELINE × {DATASETS} × h{HORIZONS} × {len(SEEDS)} seeds")
    print("=" * 70)
    scs, _ = load_all_scenarios(use_cache=True)
    print(f"{len(scs)} scenarios")

    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f: results = json.load(f)

    for ds, h_sec in MISSING_CONFIGS:
        h_steps = int(h_sec / DT)
        cfg_key = f"{ds}_h{h_sec}s"
        if cfg_key in results and len(results[cfg_key]) >= 3:
            print(f"{cfg_key}: done, skip"); continue
        if cfg_key not in results: results[cfg_key] = []
        cfg = DATASET_CONFIGS[ds]
        train_full = [s for s in scs if s['max_ltr'] <= cfg['threshold']]
        test_sc = [s for s in scs if s['max_ltr'] > cfg['threshold']]
        tr_pool, val_sc = split_train_val(train_full, 0.2, seed=42)
        X_te, y_te = build_sequences(test_sc, h_steps)
        X_va_raw, y_va = build_sequences(val_sc, h_steps)
        if X_te is None or X_va_raw is None: print(f"{cfg_key}: no data"); continue
        X_tr_raw, y_tr = build_sequences(tr_pool, h_steps)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw.reshape(-1, N_FEATURES)).reshape(X_tr_raw.shape)
        X_va = scaler.transform(X_va_raw.reshape(-1, N_FEATURES)).reshape(X_va_raw.shape)
        X_te_n = scaler.transform(X_te.reshape(-1, N_FEATURES)).reshape(X_te.shape)
        print(f"\n{cfg_key}: train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

        done = {r['seed'] for r in results[cfg_key]}
        for seed in SEEDS:
            if seed in done:
                print(f"  seed={seed}: skip"); continue
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            m = LSTM(N_FEATURES)
            m, _ = train_model_clean(m, X_tr, y_tr, X_va, y_va, device=device,
                                      loss_fn=weighted_mse_loss, epochs=EPOCHS,
                                      patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE)
            met = evaluate(m, X_te_n, y_te, device)
            met['seed'] = int(seed)
            print(f"  seed={seed} → R²={met['r2']:.3f} AUC={met['auc_pr']:.3f} ({time.time()-t0:.0f}s)")
            results[cfg_key].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=2)

    print(f"\nRésumé:")
    for k, runs in results.items():
        if runs:
            a = [r['auc_pr'] for r in runs]; r2 = [r['r2'] for r in runs]
            print(f"  {k}: AUC={np.mean(a):.3f}±{np.std(a):.3f} R²={np.mean(r2):.2f}")


if __name__ == "__main__":
    main()
