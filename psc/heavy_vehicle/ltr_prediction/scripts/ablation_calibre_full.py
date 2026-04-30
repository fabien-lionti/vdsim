#!/usr/bin/env python3
"""Ablation pseudo-replay calibré + finetune X% réel sur D2/D4 × h1/h2/h4."""
import sys, pickle, json, time, numpy as np, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (LSTM, DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
                             BATCH_SIZE, EPOCHS, LR, PATIENCE, load_all_scenarios)
from training_utils import train_model_clean, evaluate, split_train_val, weighted_mse_loss
from pseudo_replay_loader import load_sim_scenarios, build_sequences
from data_ablation_pseudoreplay_finetune import (sample_uniform, clean_for_json,
                                                   FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE)

PSEUDO_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/scenario_generation/data_v6b_pseudoreplay_v1calib/")
MODEL_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models/")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ablation_calibre_full_results.json"
PREDS_DIR = OUTPUT_DIR / "preds_calibre_full"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]
DATASETS = ['D2', 'D4']
HORIZONS = [1, 2, 4]


def pretrain_for_horizon(sim_scs, h_seconds):
    """Pretrain un LSTM sur pseudo-replay pour un horizon donné."""
    pretrain_path = MODEL_DIR / f"lstm_pseudoreplay_calibre_h{h_seconds}s.pt"
    if pretrain_path.exists():
        print(f"  Reload {pretrain_path.name}")
        ckpt = torch.load(pretrain_path, map_location=device, weights_only=False)
        model = LSTM(N_FEATURES); model.load_state_dict(ckpt['state_dict']); model = model.to(device)
        sc = StandardScaler(); sc.mean_ = np.array(ckpt['scaler_mean']); sc.scale_ = np.array(ckpt['scaler_scale'])
        sc.var_ = sc.scale_**2; sc.n_features_in_ = N_FEATURES; sc.n_samples_seen_ = 1
        return model, sc
    h_steps = int(h_seconds / DT)
    tr, va = split_train_val(sim_scs, 0.2, seed=42)
    X_tr, y_tr = build_sequences(tr, h_steps); X_va, y_va = build_sequences(va, h_steps)
    sc = StandardScaler()
    X_tr_n = sc.fit_transform(X_tr.reshape(-1, N_FEATURES)).reshape(X_tr.shape)
    X_va_n = sc.transform(X_va.reshape(-1, N_FEATURES)).reshape(X_va.shape)
    model = LSTM(N_FEATURES)
    t0 = time.time()
    model, hist = train_model_clean(model, X_tr_n, y_tr, X_va_n, y_va, device=device,
                                    loss_fn=weighted_mse_loss, epochs=EPOCHS,
                                    patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE)
    print(f"  h{h_seconds}s pretrain {hist['n_epochs']}ep val {hist['best_val_loss']:.4f} ({time.time()-t0:.0f}s)")
    torch.save({'state_dict': model.state_dict(), 'scaler_mean': sc.mean_, 'scaler_scale': sc.scale_},
               pretrain_path)
    return model, sc


def save_preds(model, X, y, fp):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            preds.append(model(torch.FloatTensor(X[i:i+512].astype(np.float32)).to(device)).cpu().numpy())
    preds = np.clip(np.concatenate(preds), 0, 1.5)
    np.savez_compressed(fp, preds=preds.astype(np.float32), targets=y.astype(np.float32))


def main():
    print("=" * 70)
    print(f"ABLATION FULL : pseudo-replay calibré × {DATASETS} × h{HORIZONS}")
    print(f"Fractions {FRACTIONS}  Seeds {SEEDS}")
    print(f"Total: {len(DATASETS) * len(HORIZONS) * len(FRACTIONS) * len(SEEDS)} runs")
    print("=" * 70)

    print("\n[1] Load pseudo-replay calibré...")
    sim_scs = load_sim_scenarios(PSEUDO_DIR)
    print(f"  {len(sim_scs)} scenarios")

    print("\n[2] Load real scenarios...")
    real_scs, _ = load_all_scenarios(use_cache=True)
    print(f"  {len(real_scs)} scenarios")

    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f: results = json.load(f)

    print("\n[3] Ablation")
    for h_sec in HORIZONS:
        h_steps = int(h_sec / DT)
        print(f"\n{'='*50}\n  Horizon h{h_sec}s\n{'='*50}")
        print(f"  Pretrain LSTM for h{h_sec}s...")
        model, scaler = pretrain_for_horizon(sim_scs, h_sec)
        sd = {k: v.cpu().clone() for k,v in model.state_dict().items()}

        for ds in DATASETS:
            cfg_key = f"{ds}_h{h_sec}s"
            print(f"\n  --- {cfg_key} ---")
            cfg = DATASET_CONFIGS[ds]
            train_full = [s for s in real_scs if s['max_ltr'] <= cfg['threshold']]
            test_sc = [s for s in real_scs if s['max_ltr'] > cfg['threshold']]
            pool, val_sc = split_train_val(train_full, 0.2, seed=42)
            X_test_raw, y_test = build_sequences(test_sc, h_steps)
            if X_test_raw is None:
                print(f"  {cfg_key}: no test, skip"); continue
            X_test = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
            X_val_raw, y_val = build_sequences(val_sc, h_steps)
            X_val = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
            print(f"    Pool {len(pool)}, Val {len(val_sc)}, Test {len(y_test)} seq")

            if cfg_key not in results: results[cfg_key] = {}

            for frac in FRACTIONS:
                fk = f"{int(frac*100)}pct_real"
                if fk not in results[cfg_key]: results[cfg_key][fk] = []
                done = {r['seed'] for r in results[cfg_key][fk]}
                for seed in SEEDS:
                    if seed in done:
                        print(f"    {fk} seed={seed}: skip"); continue
                    t0 = time.time()
                    print(f"    {fk} seed={seed}...", flush=True)
                    if frac == 0.0:
                        m = LSTM(N_FEATURES); m.load_state_dict(sd); m = m.to(device)
                        met = evaluate(m, X_test, y_test, device)
                    else:
                        n_s = max(1, int(frac * len(pool)))
                        sampled = sample_uniform(pool, n_s, seed)
                        X_tr_r, y_tr = build_sequences(sampled, h_steps)
                        X_tr = scaler.transform(X_tr_r.reshape(-1, N_FEATURES)).reshape(X_tr_r.shape)
                        torch.manual_seed(seed); np.random.seed(seed)
                        m = LSTM(N_FEATURES); m.load_state_dict({k: v.clone() for k,v in sd.items()})
                        m, _ = train_model_clean(m, X_tr, y_tr, X_val, y_val, device=device,
                                                  loss_fn=weighted_mse_loss, epochs=FINETUNE_EPOCHS,
                                                  patience=FINETUNE_PATIENCE, lr=FINETUNE_LR, batch_size=BATCH_SIZE)
                        met = evaluate(m, X_test, y_test, device)
                    if seed == 0:
                        save_preds(m, X_test, y_test, PREDS_DIR / f"{cfg_key}_{fk}_seed{seed}.npz")
                    met.update({'fraction_real': float(frac), 'seed': int(seed)})
                    print(f"      → R²={met['r2']:.3f} AUC={met['auc_pr']:.3f} rec={met['recall']:.0%} ({time.time()-t0:.0f}s)")
                    results[cfg_key][fk].append(clean_for_json(met))
                    with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=2)

    # Résumé
    print(f"\n\n{'='*70}\nRésumé\n{'='*70}")
    for cfg_key in sorted(results.keys()):
        print(f"\n{cfg_key}:")
        for fk in sorted(results[cfg_key].keys(), key=lambda k: int(k.split('pct')[0])):
            runs = results[cfg_key][fk]
            if runs:
                a = [r['auc_pr'] for r in runs]; r2 = [r['r2'] for r in runs]
                print(f"  {fk}: AUC={np.mean(a):.3f}±{np.std(a):.3f} R²={np.mean(r2):.2f}")


if __name__ == "__main__":
    main()
