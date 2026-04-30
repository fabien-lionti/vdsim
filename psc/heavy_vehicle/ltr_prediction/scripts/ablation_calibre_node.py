#!/usr/bin/env python3
"""
Ablation NODE + pseudo-replay calibré + finetune X% réel.
NODE features (pretrained sim V2 backbone, figé) + LSTM entraîné sur
pseudo-replay calibré (DOF10 sans relax/Kamm), puis finetune réel.
"""
import sys, pickle, json, time, numpy as np, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
                             BATCH_SIZE, EPOCHS, LR, PATIENCE, load_all_scenarios)
from training_utils import split_train_val, weighted_mse_loss
from pseudo_replay_loader import load_sim_scenarios, build_sequences
from lstm_node_real import (load_node_backbone, LSTMPhysicsWrapper, PHYSICS_PROJ_DIM,
                            normalize_for_node, train_physics_model, evaluate_physics,
                            clean_for_json)
from data_ablation_pseudoreplay_finetune import (sample_uniform, FINETUNE_LR,
                                                   FINETUNE_EPOCHS, FINETUNE_PATIENCE)

PSEUDO_DIR = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/scenario_generation/data_v6b_pseudoreplay_v1calib/")
PRETRAIN_PATH = Path("/Users/zak/Documents/vdsim/psc/heavy_vehicle/models/lstm_node_pseudoreplay_calibre.pt")
OUTPUT_FILE = OUTPUT_DIR / "ablation_calibre_node_pseudoreplay_results.json"
PREDS_DIR = OUTPUT_DIR / "preds_calibre_node_pseudoreplay"
PREDS_DIR.mkdir(parents=True, exist_ok=True)
FRACTIONS = [0.0, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]


def save_preds(model, X_base, X_node, y, fp):
    model.eval()
    X_bt = torch.FloatTensor(X_base.astype(np.float32))
    X_nt = torch.FloatTensor(X_node)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_bt), 512):
            preds.append(model(X_bt[i:i+512].to(device), X_nt[i:i+512].to(device)).cpu().numpy())
    preds = np.clip(np.concatenate(preds), 0, 1.5)
    np.savez_compressed(fp, preds=preds.astype(np.float32), targets=y.astype(np.float32))


def pretrain(node_backbone, node_scaler, node_n_input):
    h = int(2/DT)
    if PRETRAIN_PATH.exists():
        print(f"  Reload {PRETRAIN_PATH.name}")
        ckpt = torch.load(PRETRAIN_PATH, map_location=device, weights_only=False)
        model = LSTMPhysicsWrapper(node_backbone=node_backbone, node_n_input=node_n_input,
                                   physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48)
        model.load_state_dict(ckpt['state_dict']); model = model.to(device)
        sc = StandardScaler(); sc.mean_ = np.array(ckpt['scaler_mean']); sc.scale_ = np.array(ckpt['scaler_scale'])
        sc.var_ = sc.scale_**2; sc.n_features_in_ = N_FEATURES; sc.n_samples_seen_ = 1
        return model, sc

    sim = load_sim_scenarios(PSEUDO_DIR)
    tr, va = split_train_val(sim, 0.2, seed=42)
    X_tr_r, y_tr = build_sequences(tr, h); X_va_r, y_va = build_sequences(va, h)
    base = StandardScaler()
    X_tr_b = base.fit_transform(X_tr_r.reshape(-1, N_FEATURES)).reshape(X_tr_r.shape)
    X_va_b = base.transform(X_va_r.reshape(-1, N_FEATURES)).reshape(X_va_r.shape)
    X_tr_n = normalize_for_node(X_tr_r, node_scaler, node_n_input)
    X_va_n = normalize_for_node(X_va_r, node_scaler, node_n_input)

    model = LSTMPhysicsWrapper(node_backbone=node_backbone, node_n_input=node_n_input,
                               physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48)
    t0 = time.time()
    model, hist = train_physics_model(model, X_tr_b, X_tr_n, y_tr, X_va_b, X_va_n, y_va,
                                       epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE)
    print(f"  Pretrain {hist['n_epochs']}ep, val {hist['best_val_loss']:.4f} ({time.time()-t0:.0f}s)")
    torch.save({'state_dict': model.state_dict(), 'scaler_mean': base.mean_, 'scaler_scale': base.scale_},
               PRETRAIN_PATH)
    return model, base


def main():
    print("=" * 70)
    print("ABLATION NODE + pseudo-replay CALIBRÉ + finetune X% réel — D4 h2s")
    print("=" * 70)
    h = int(2/DT)
    print("\n[1] Load NODE backbone...")
    node_backbone, node_scaler, node_n_input = load_node_backbone()
    print("\n[2] Pretrain LSTM+NODE sur pseudo-replay calibré...")
    model, scaler = pretrain(node_backbone, node_scaler, node_n_input)
    sd = {k: v.cpu().clone() for k,v in model.state_dict().items()}

    print("\n[3] Load real D4 h2s...")
    scs, _ = load_all_scenarios(use_cache=True)
    cfg = DATASET_CONFIGS['D4']
    train_full = [s for s in scs if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scs if s['max_ltr'] > cfg['threshold']]
    pool, val_sc = split_train_val(train_full, 0.2, seed=42)
    X_test_raw, y_test = build_sequences(test_sc, h)
    X_test_base = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)
    X_test_node = normalize_for_node(X_test_raw, node_scaler, node_n_input)
    X_val_raw, y_val = build_sequences(val_sc, h)
    X_val_base = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_val_node = normalize_for_node(X_val_raw, node_scaler, node_n_input)
    print(f"Pool {len(pool)}, Val {len(val_sc)}, Test {len(y_test)} seq")

    results = {'D4_h2s': {}}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f: results = json.load(f)
    if 'D4_h2s' not in results: results['D4_h2s'] = {}

    print("\n[4] Ablation")
    for frac in FRACTIONS:
        fk = f"{int(frac*100)}pct_real"
        if fk not in results['D4_h2s']: results['D4_h2s'][fk] = []
        done = {r['seed'] for r in results['D4_h2s'][fk]}
        for seed in SEEDS:
            if seed in done:
                print(f"  {fk} seed={seed}: skip"); continue
            t0 = time.time()
            print(f"  {fk} seed={seed}...", flush=True)
            torch.manual_seed(seed); np.random.seed(seed)
            m = LSTMPhysicsWrapper(node_backbone=node_backbone, node_n_input=node_n_input,
                                   physics_proj_dim=PHYSICS_PROJ_DIM, base_feat_dim=N_FEATURES, dropout=0.48)
            m.load_state_dict(sd, strict=False); m = m.to(device)
            if frac == 0.0:
                met = evaluate_physics(m, X_test_base, X_test_node, y_test)
            else:
                n_s = max(1, int(frac * len(pool)))
                sampled = sample_uniform(pool, n_s, seed)
                X_tr_r, y_tr = build_sequences(sampled, h)
                X_tr_b = scaler.transform(X_tr_r.reshape(-1, N_FEATURES)).reshape(X_tr_r.shape)
                X_tr_n = normalize_for_node(X_tr_r, node_scaler, node_n_input)
                m, _ = train_physics_model(m, X_tr_b, X_tr_n, y_tr, X_val_base, X_val_node, y_val,
                                            epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
                                            lr=FINETUNE_LR, batch_size=BATCH_SIZE)
                met = evaluate_physics(m, X_test_base, X_test_node, y_test)
            if seed == 0:
                save_preds(m, X_test_base, X_test_node, y_test, PREDS_DIR / f"D4_h2s_{fk}_seed{seed}.npz")
            met.update({'fraction_real': float(frac), 'seed': int(seed)})
            print(f"    → R²={met['r2']:.3f} AUC-PR={met['auc_pr']:.3f} rec={met['recall']:.0%} ({time.time()-t0:.0f}s)")
            results['D4_h2s'][fk].append(clean_for_json(met))
            with open(OUTPUT_FILE, 'w') as f: json.dump(results, f, indent=2)

    print("\n[Résumé]")
    for fk in sorted(results['D4_h2s'].keys(), key=lambda k: int(k.split('pct')[0])):
        r = results['D4_h2s'][fk]
        if r:
            a = [x['auc_pr'] for x in r]; r2 = [x['r2'] for x in r]
            print(f"  {fk}: AUC-PR={np.mean(a):.3f}±{np.std(a):.3f} R²={np.mean(r2):.2f}")


if __name__ == "__main__":
    main()
