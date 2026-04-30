#!/usr/bin/env python3
"""Entraîne un LSTM sur données réelles DXD avec weighted MSE loss."""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, average_precision_score
import pickle
import time as time_module
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    train_model_clean, evaluate as evaluate_on_test, split_train_val,
    weighted_mse_loss as wmse,
)

DXD_DIR = Path("/Users/zak/Desktop/selected_dxd_json_resampled/")
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
MODEL_DIR = Path(__file__).parent.parent / "models" / "real"
FIG_DIR = OUTPUT_DIR / "figures"

for d in [OUTPUT_DIR, MODEL_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DT = 0.01
SEQ_LEN = 150       # 1.5s lookback
HORIZONS = [1, 2, 4] # seconds
BATCH_SIZE = 64
EPOCHS = 80
LR = 0.001
PATIENCE = 15
DANGER = 0.7

FEATURES = ['vx', 'vy', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot', 'LTR_current', 'dLTR_dt']
N_FEATURES = len(FEATURES)  # 9 = 7 original + LTR_current + dLTR_dt

DATASET_CONFIGS = {
    'D1': {'threshold': 0.7, 'ood': False, 'desc': 'In-distribution (LTR <= 0.7)'},
    'D2': {'threshold': 0.7, 'ood': True, 'desc': 'OOD (train <= 0.7, test > 0.7)'},
    'D3': {'threshold': 0.8, 'ood': True, 'desc': 'OOD (train <= 0.8, test > 0.8)'},
    'D4': {'threshold': 0.9, 'ood': True, 'desc': 'OOD critique (train <= 0.9, test > 0.9)'},
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ============== Models ==============

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, output_size))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def weighted_mse_loss(pred, target):
    """Weighted MSE: high LTR samples count 10x more than low LTR."""
    weight = 1.0 + 9.0 * target
    return (weight * (pred - target) ** 2).mean()


# ============== Data loading ==============

def load_dxd_scenario(filepath):
    """Convertit un fichier DXD JSON en DataFrame avec 8 features + LTR."""
    with open(filepath) as f:
        data = json.load(f)

    ch = data["channels"]
    required = ["VelX (km/h)", "VelY (km/h)", "LF_Fz_1 (N)", "RF_Fz_2 (N)",
                "LR_Fz_3 (N)", "RR_Fz_4 (N)", "Roll (_)", "Pitch (_)", "WheelSteer_S1 (_)"]
    for r in required:
        if r not in ch:
            return None

    def get(name):
        vals = ch[name]["values"]
        return np.array([v if v is not None else np.nan for v in vals], dtype=float)

    vx = get("VelX (km/h)") / 3.6
    vy = get("VelY (km/h)") / 3.6

    # psi_dot
    if "VehYaw_W_Actl (rad/s)" in ch:
        yaw_raw = get("VehYaw_W_Actl (rad/s)")
        if np.nanstd(yaw_raw) > 0.01:
            psi_dot = yaw_raw
        else:
            psi_dot = get("AngVelZ_body (_/s)") * np.pi / 180.0
    elif "AngVelZ_body (_/s)" in ch:
        psi_dot = get("AngVelZ_body (_/s)") * np.pi / 180.0
    else:
        return None

    # psi removed — cumsum(psi_dot) drifts on real data, useless feature

    # MAPPING CORRIGÉ: phi simu = tangage → Pitch réel
    #                   theta simu = roulis → Roll réel
    phi = get("Pitch (_)") * np.pi / 180.0     # phi = tangage
    theta = get("Roll (_)") * np.pi / 180.0    # theta = roulis

    delta_f = get("WheelSteer_S1 (_)")
    delta_f_dot = np.gradient(delta_f, DT)

    # LTR ground truth: max(front axle, rear axle) — same as simulation
    fz_lf = get("LF_Fz_1 (N)")
    fz_rf = get("RF_Fz_2 (N)")
    fz_lr = get("LR_Fz_3 (N)")
    fz_rr = get("RR_Fz_4 (N)")
    eps = 1e-12
    # Only compute LTR when both wheels have meaningful load (>200N)
    fz_f_sum = fz_rf + fz_lf
    fz_r_sum = fz_rr + fz_lr
    ltr_front = np.where(fz_f_sum > 400, np.abs((fz_rf - fz_lf) / (fz_f_sum + eps)), 0.0)
    ltr_rear = np.where(fz_r_sum > 400, np.abs((fz_rr - fz_lr) / (fz_r_sum + eps)), 0.0)
    ltr_abs = np.clip(np.maximum(ltr_front, ltr_rear), 0.0, 1.0)

    n = min(len(vx), len(vy), len(psi_dot), len(phi), len(theta), len(delta_f), len(ltr_abs))
    dLTR_dt = np.gradient(ltr_abs[:n], DT)
    df = pd.DataFrame({
        'vx': vx[:n], 'vy': vy[:n], 'psi_dot': psi_dot[:n],
        'phi': phi[:n], 'theta': theta[:n],
        'delta_f': delta_f[:n], 'delta_f_dot': delta_f_dot[:n],
        'LTR_current': ltr_abs[:n], 'dLTR_dt': dLTR_dt,
        'LTRmax': ltr_abs[:n],
    })
    df = df.dropna().reset_index(drop=True)
    return df


SCENARIOS_CACHE = OUTPUT_DIR / "scenarios_cache.pkl"


def load_all_scenarios(max_files=488, use_cache=True):
    """Charge les fichiers DXD et retourne une liste de scénarios.

    Priorise les fichiers dynamiques (erdv, slalom, ld, feco, etc.)
    pour maximiser la couverture LTR avec moins de fichiers.

    Si `use_cache=True`, sauvegarde/charge depuis SCENARIOS_CACHE (pickle).
    Accélère les re-runs: <5s au lieu de ~5 min.
    """
    if use_cache and SCENARIOS_CACHE.exists():
        print(f"  Cache détecté: {SCENARIOS_CACHE}", flush=True)
        try:
            with open(SCENARIOS_CACHE, 'rb') as f:
                payload = pickle.load(f)
            scenarios = payload['scenarios']
            skipped = payload['skipped']
            print(f"  → {len(scenarios)} scénarios rechargés depuis le cache "
                  f"({skipped} skipped à l'origine)", flush=True)
            return scenarios, skipped
        except Exception as e:
            print(f"  Cache corrompu ({e}), rechargement from scratch", flush=True)

    all_files = sorted(DXD_DIR.glob("*.json"))

    # Priorise les tests dynamiques
    priority_keys = ['erdv', 'slalom', 'ld_', 'feco', 'fuld', 'devers', 'sinus',
                     'anr', 'vir', 'rpc', 'slalom_croi', 'slalom_decroi']
    priority = [f for f in all_files if any(k in f.name for k in priority_keys)]
    other = [f for f in all_files if f not in priority]
    files_to_load = (priority + other)[:min(max_files, len(priority) + len(other))]

    print(f"  Chargement de {len(files_to_load)}/{len(all_files)} fichiers...", flush=True)
    scenarios = []
    skipped = 0
    MAX_SIZE = 15_000_000  # 15 MB — cohérent avec le dataset initial (392 scénarios)
    for i, f in enumerate(files_to_load):
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(files_to_load)} traités  ({len(scenarios)} retenus, "
                  f"{skipped} skipped)", flush=True)
        try:
            sz = f.stat().st_size
            if sz > MAX_SIZE:
                skipped += 1
                continue
            df = load_dxd_scenario(f)
            if df is not None and len(df) > SEQ_LEN + 500:
                ltr_max = df['LTRmax'].max()
                scenarios.append({'df': df, 'max_ltr': ltr_max, 'name': f.stem})
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if use_cache:
        try:
            with open(SCENARIOS_CACHE, 'wb') as f:
                pickle.dump({'scenarios': scenarios, 'skipped': skipped}, f)
            print(f"  Cache sauvegardé: {SCENARIOS_CACHE}", flush=True)
        except Exception as e:
            print(f"  Échec sauvegarde cache: {e}", flush=True)

    return scenarios, skipped


def create_sequences_max(df, horizon_steps):
    """Crée X, y pour prédiction du max LTR sur l'horizon."""
    features = df[FEATURES].values
    ltr = df['LTRmax'].values
    X, y = [], []
    for i in range(0, len(df) - SEQ_LEN - horizon_steps, 10):
        X.append(features[i:i + SEQ_LEN])
        y.append(np.max(ltr[i + SEQ_LEN:i + SEQ_LEN + horizon_steps]))
    if not X:
        return None, None
    return np.array(X), np.array(y)


def load_data(scenarios, dataset_name, horizon_steps, val_fraction=0.2, val_seed=42):
    """Split D1-D4 puis train -> train'/val. Crée toutes les séquences.

    Le val set sert UNIQUEMENT à l'early stopping.
    Le test set n'est évalué qu'une fois à la fin (jamais vu pendant le train).
    """
    cfg = DATASET_CONFIGS[dataset_name]

    if cfg['ood']:
        train_sc_full = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
        test_sc = [s for s in scenarios if s['max_ltr'] > cfg['threshold']]
    else:
        eligible = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
        np.random.seed(42)
        perm = np.random.permutation(len(eligible))
        split = int(0.8 * len(eligible))
        train_sc_full = [eligible[i] for i in perm[:split]]
        test_sc = [eligible[i] for i in perm[split:]]

    if not train_sc_full or not test_sc:
        return None

    # Train -> train' + val (disjoints par scénario)
    train_sc, val_sc = split_train_val(train_sc_full, val_fraction=val_fraction, seed=val_seed)
    if not train_sc or not val_sc:
        return None

    def build(scs):
        Xl, yl = [], []
        for s in scs:
            X, y = create_sequences_max(s['df'], horizon_steps)
            if X is not None:
                Xl.append(X); yl.append(y)
        if not Xl:
            return None, None
        return np.concatenate(Xl), np.concatenate(yl)

    X_train, y_train = build(train_sc)
    X_val,   y_val   = build(val_sc)
    X_test,  y_test  = build(test_sc)

    if X_train is None or X_val is None or X_test is None:
        return None

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, N_FEATURES)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
        'n_train_sc': len(train_sc),
        'n_val_sc':   len(val_sc),
        'n_test_sc':  len(test_sc),
    }


# ============== Training ==============

def train_model(model, X_train, y_train, X_val, y_val):
    """Entraîne un LSTM avec weighted MSE et early stopping sur VAL set.

    Le test set n'est JAMAIS vu ici (corrigé par rapport à la version initiale
    qui utilisait le test set pour l'early stopping, introduisant un biais
    optimiste).

    Returns: model (avec best val state chargé)
    """
    model, _hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    )
    return model


def compute_metrics(y, preds):
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    y_bin = (y >= DANGER).astype(int)
    n_danger = y_bin.sum()
    auc_pr = float('nan')
    if 0 < n_danger < len(y_bin):
        auc_pr = average_precision_score(y_bin, preds)
    # Recall/precision at best F1 threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.2, 0.8, 0.05):
        det = preds >= thresh
        tp = np.sum(det & (y_bin == 1))
        rec = tp / max(n_danger, 1)
        prec = tp / max(np.sum(det), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    det = preds >= best_thresh
    tp = np.sum(det & (y_bin == 1))
    recall = tp / max(n_danger, 1)
    precision = tp / max(np.sum(det), 1)
    return {'rmse': rmse, 'r2': r2, 'auc_pr': auc_pr,
            'recall': recall, 'precision': precision, 'f1': best_f1,
            'threshold': best_thresh, 'n_test': len(y), 'n_danger': int(n_danger)}


# ============== Main ==============

def main():
    print("=" * 70)
    print("ENTRAINEMENT SUR DONNÉES RÉELLES (DXD)")
    print(f"Device: {device}")
    print("=" * 70)

    # Load all scenarios
    print("\nChargement des fichiers DXD...")
    scenarios, skipped = load_all_scenarios()
    print(f"Scénarios: {len(scenarios)} valides, {skipped} skipped")

    ltr_maxes = [s['max_ltr'] for s in scenarios]
    print(f"LTR max range: [{min(ltr_maxes):.3f}, {max(ltr_maxes):.3f}]")
    print(f"LTR > 0.7: {sum(1 for l in ltr_maxes if l > 0.7)} scénarios")
    print(f"LTR > 0.8: {sum(1 for l in ltr_maxes if l > 0.8)} scénarios")
    print(f"LTR > 0.9: {sum(1 for l in ltr_maxes if l > 0.9)} scénarios")

    all_results = {}
    models_list = ['LSTM']

    for ds_name, ds_cfg in DATASET_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} - {ds_cfg['desc']}")
        print(f"{'='*60}")

        for h in HORIZONS:
            horizon_steps = int(h / DT)
            hkey = f"h{h}s"
            print(f"\n  --- Horizon {h}s ({horizon_steps} pas) ---")

            data = load_data(scenarios, ds_name, horizon_steps)
            if data is None:
                print(f"    Pas assez de données pour {ds_name} {hkey}")
                continue

            print(f"    Train: {data['n_train_sc']} scénarios, {len(data['y_train'])} échantillons")
            print(f"    Val:   {data['n_val_sc']} scénarios, {len(data['y_val'])} échantillons  (early stopping)")
            print(f"    Test:  {data['n_test_sc']} scénarios, {len(data['y_test'])} échantillons")
            print(f"    LTR test: [{data['y_test'].min():.2f}, {data['y_test'].max():.2f}]")

            for model_name in models_list:
                t0 = time_module.time()

                model = LSTM(N_FEATURES)

                model = train_model(
                    model, data['X_train'], data['y_train'],
                    data['X_val'], data['y_val'])

                # Evaluate on test ONLY at the end (never seen during training)
                met = evaluate_on_test(model, data['X_test'], data['y_test'], device)
                dt = time_module.time() - t0

                # Save model + scaler
                torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_{ds_name}_{hkey}.pt")
                with open(MODEL_DIR / f"scaler_{ds_name}_{hkey}.pkl", 'wb') as f:
                    pickle.dump(data['scaler'], f)

                # Store results
                if ds_name not in all_results:
                    all_results[ds_name] = {}
                if hkey not in all_results[ds_name]:
                    all_results[ds_name][hkey] = {}
                all_results[ds_name][hkey][model_name] = met

                auc_s = f"{met['auc_pr']:.3f}" if not np.isnan(met['auc_pr']) else "N/A"
                print(f"    {model_name}... R²={met['r2']:.3f} RMSE={met['rmse']:.3f} "
                      f"AUC-PR={auc_s} F1={met['f1']:.3f} Rec={met['recall']:.0%} "
                      f"Prec={met['precision']:.0%} ({dt:.0f}s)")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Dataset':<6} {'H':<5} {'Model':<10} {'RMSE':<8} {'R²':<8} {'AUC-PR':<8} {'Recall':<8} {'Prec':<8}")
    print("-" * 80)
    for ds in ['D1', 'D2', 'D3', 'D4']:
        if ds not in all_results: continue
        for hk in ['h1s', 'h2s', 'h4s']:
            if hk not in all_results[ds]: continue
            for m in models_list:
                if m not in all_results[ds][hk]: continue
                r = all_results[ds][hk][m]
                auc = f"{r['auc_pr']:.3f}" if not np.isnan(r['auc_pr']) else "N/A"
                print(f"{ds:<6} {hk:<5} {m:<10} {r['rmse']:<8.3f} {r['r2']:<8.3f} "
                      f"{auc:<8} {r['recall']:<8.0%} {r['precision']:<8.0%}")

    # Save results
    results_out = {}
    for ds in all_results:
        for hk in all_results[ds]:
            for m in all_results[ds][hk]:
                key = f"{m}_{ds}_{hk}"
                results_out[key] = {k: float(v) if not isinstance(v, int) else v
                                    for k, v in all_results[ds][hk][m].items()}
    with open(OUTPUT_DIR / "results_real_data.json", 'w') as f:
        json.dump(results_out, f, indent=2)

    print(f"\nModèles: {MODEL_DIR}")
    print(f"Résultats: {OUTPUT_DIR / 'results_real_data.json'}")
    print("TERMINÉ!")


if __name__ == "__main__":
    main()
