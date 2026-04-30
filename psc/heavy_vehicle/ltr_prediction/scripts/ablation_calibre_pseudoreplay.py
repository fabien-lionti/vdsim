#!/usr/bin/env python3
"""
Ablation pseudo-replay + finetune X% réel avec DOF10 CALIBRÉ
(make_heavy_vehicle(), sans relax/Kamm). D4 h2s × 5 fractions × 3 seeds.
"""
import sys, pickle, json, time, numpy as np, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (LSTM, DT, N_FEATURES, DATASET_CONFIGS, device,
                             BATCH_SIZE, EPOCHS, LR, PATIENCE, load_all_scenarios, OUTPUT_DIR)
from training_utils import train_model_clean, evaluate, split_train_val, weighted_mse_loss
from pseudo_replay_loader import load_sim_scenarios, build_sequences
from data_ablation_pseudoreplay_finetune import (sample_uniform, clean_for_json,
                                                   FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE)

PSEUDO_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/scenario_generation/data_v6b_pseudoreplay_v1calib/")
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models/lstm_pseudoreplay_calibre.pt")
OUTPUT_FILE = OUTPUT_DIR / "ablation_calibre_pseudoreplay_results.json"
PREDS_DIR = OUTPUT_DIR / "preds_calibre_pseudoreplay"
PREDS_DIR.mkdir(parents=True, exist_ok=True)
FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2]


def save_preds_npz(model, X, y, filepath):
    model.eval()
    with torch.no_grad():
        preds = np.concatenate([
            model(torch.FloatTensor(X[i:i+512].astype(np.float32)).to(device)).cpu().numpy()
            for i in range(0, len(X), 512)
        ])
    preds = np.clip(preds, 0, 1.5)
    np.savez_compressed(filepath, preds=preds.astype(np.float32), targets=y.astype(np.float32))


def pretrain():
    h = int(2/DT)
    if PRETRAIN_PATH.exists():
        print(f"  Reload {PRETRAIN_PATH.name}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
        model = LSTM(N_FEATURES); model.load_state_dict(ckpt['state_dict']); model = model.to(device)
        sc = StandardScaler(); sc.mean_ = np.array(ckpt['scaler_mean']); sc.scale_ = np.array(ckpt['scaler_scale'])
        sc.var_ = sc.scale_**2; sc.n_features_in_ = N_FEATURES; sc.n_samples_seen_ = 1
        return model, sc
    sim = load_sim_scenarios(PSEUDO_DIR)
    tr, va = split_train_val(sim, 0.2, seed=42)
    X_tr, y_tr = build_sequences(tr, h); X_va, y_va = build_sequences(va, h)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr.reshape(-1, N_FEATURES)).reshape(X_tr.shape)
    X_va = sc.transform(X_va.reshape(-1, N_FEATURES)).reshape(X_va.shape)
    model = LSTM(N_FEATURES)
    t0 = time.time()
    model, hist = train_model_clean(model, X_tr, y_tr, X_va, y_va, device=device,
                                    loss_fn=weighted_mse_loss, epochs=EPOCHS,
                                    patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE)
    print(f"  Pretrain {hist['n_epochs']}ep, val {hist['best_val_loss']:.4f} ({time.time()-t0:.0f}s)")
    torch.save({'state_dict': model.state_dict(), 'scaler_mean': sc.mean_, 'scaler_scale': sc.scale_},
               PRETRAIN_PATH)
    return model, sc


def main():
    print("=" * 70)
    print("ABLATION pseudo-replay CALIBRÉ + finetune X% réel — D4 h2s")
    print("=" * 70)
    h = int(2/DT)
    model, scaler = pretrain()
    sd = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    scs, _ = load_all_scenarios(use_cache=True)
    cfg = DATASET_CONFIGS['D4']
    train_full = [s for s in scs if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scs if s['max_ltr'] > cfg['threshold']]
    pool, val_sc = split_train_val(train_full, 0.2, seed=42)
    X_test_raw, y_test = build_sequences(test_sc, h)
    X_test = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
    print(f"Pool {len(pool)}, Val {len(val_sc)}, Test {len(y_test)} seq")

    results = {'D4_h2s': {}}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f: results = json.load(f)
    if 'D4_h2s' not in results: results['D4_h2s'] = {}

    X_val_raw, y_val = build_sequences(val_sc, h)
    X_val = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)

    for frac in FRACTIONS:
        fk = f"{int(frac*100)}pct_real"
        if fk not in results['D4_h2s']: results['D4_h2s'][fk] = []
        done = {r['seed'] for r in results['D4_h2s'][fk]}
        for seed in SEEDS:
            if seed in done:
                print(f"  {fk} seed={seed}: skip"); continue
            t0 = time.time()
            print(f"  {fk} seed={seed}...", flush=True)
            if frac == 0.0:
                m = LSTM(N_FEATURES); m.load_state_dict(sd); m = m.to(device)
                met = evaluate(m, X_test, y_test, device)
            else:
                n_s = max(1, int(frac * len(pool)))
                sampled = sample_uniform(pool, n_s, seed)
                X_tr_raw, y_tr = build_sequences(sampled, h)
                X_tr = scaler.transform(X_tr_raw.reshape(-1, N_FEATURES)).reshape(X_tr_raw.shape)
                torch.manual_seed(seed); np.random.seed(seed)
                m = LSTM(N_FEATURES); m.load_state_dict({k: v.clone() for k,v in sd.items()})
                m, _ = train_model_clean(m, X_tr, y_tr, X_val, y_val, device=device,
                                         loss_fn=weighted_mse_loss, epochs=FINETUNE_EPOCHS,
                                         patience=FINETUNE_PATIENCE, lr=FINETUNE_LR, batch_size=BATCH_SIZE)
                met = evaluate(m, X_test, y_test, device)

            # Sauvegarde preds pour plots ultérieurs (seed 0 uniquement)
            if seed == 0:
                save_preds_npz(m, X_test, y_test, PREDS_DIR / f"D4_h2s_{fk}_seed{seed}.npz")

            met.update({'fraction_real': float(frac), 'seed': int(seed),
                        'n_train_scenarios': 0 if frac == 0.0 else max(1, int(frac*len(pool)))})
            print(f"    → R²={met['r2']:.3f} AUC-PR={met['auc_pr']:.3f} ({time.time()-t0:.0f}s)")
            results['D4_h2s'][fk].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=2)

    print("\n[Résumé] Pseudo-replay CALIBRÉ D4 h2s")
    for fk in sorted(results['D4_h2s'].keys(), key=lambda k: int(k.split('pct')[0])):
        runs = results['D4_h2s'][fk]
        if runs:
            a = [r['auc_pr'] for r in runs]; r2 = [r['r2'] for r in runs]
            print(f"  {fk}: AUC-PR={np.mean(a):.3f}±{np.std(a):.3f} R²={np.mean(r2):.2f}")


if __name__ == "__main__":
    main()
