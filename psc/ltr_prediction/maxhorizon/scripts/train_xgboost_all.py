#!/usr/bin/env python3
"""Entraîne les modèles XGBoost Q90 pour la prédiction du max(|LTR|) sur [t, t+h]."""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
import time

# Paths
PROJECT_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/maxhorizon")
DATA_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Config
DT = 0.01
SEQ_LEN = 150
FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = len(FEATURES)
HORIZONS = [1, 2, 4, 6, 8]
DATASETS = ['D1', 'D2', 'D3', 'D4']

DATASET_CONFIGS = {
    'D1': {'threshold': 0.7, 'test_above': False},
    'D2': {'threshold': 0.7, 'test_above': True},
    'D3': {'threshold': 0.8, 'test_above': True},
    'D4': {'threshold': 0.9, 'test_above': True},
}


def load_scenario(filepath):
    """Charge un scenario CSV et calcule les features derivees."""
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def extract_features(window):
    """Extrait les features statistiques d'une fenetre temporelle."""
    feats = []
    for i in range(window.shape[1]):
        col = window[:, i]
        feats.extend([
            col.mean(), col.std(), col.min(), col.max(),
            col[-1] - col[0],  # Delta
            np.abs(np.diff(col)).mean()  # Mean absolute change
        ])
    return np.array(feats)


def get_data_split(config_name):
    """Recupere le split train/test pour une configuration donnee."""
    config = DATASET_CONFIGS[config_name]
    all_files = sorted(DATA_DIR.glob("*.csv"))

    low, high = [], []
    for f in all_files:
        df = pd.read_csv(f)
        ltr_max = np.max(np.abs(df['LTRmax'].values))
        if ltr_max <= config['threshold']:
            low.append(f)
        else:
            high.append(f)

    if config['test_above']:
        return low, high
    else:
        np.random.seed(42)
        idx = np.random.permutation(len(low))
        split = int(0.8 * len(low))
        return [low[i] for i in idx[:split]], [low[i] for i in idx[split:]]


def build_dataset(files, horizon_steps, step=15):
    """
    Construit le dataset de features pour XGBoost.

    Difference avec horizon_exact: ici on predit le MAX sur [t, t+h],
    pas la valeur a l'instant exact t+h.
    """
    X_list, y_list = [], []
    for f in files:
        df = load_scenario(f)
        data = df[FEATURES].values
        ltr = np.abs(df['LTRmax'].values)
        for i in range(0, len(data) - SEQ_LEN - horizon_steps, step):
            X_list.append(extract_features(data[i:i + SEQ_LEN]))
            # Target = MAX du LTR sur l'horizon [t, t+h]
            y_list.append(np.max(ltr[i + SEQ_LEN:i + SEQ_LEN + horizon_steps]))
    return np.array(X_list), np.array(y_list)


def train_and_save_xgboost(config_name, horizon):
    """Entraine et sauvegarde un modele XGBoost Q90."""
    horizon_steps = int(horizon / DT)
    horizon_key = f"h{horizon}s"

    train_files, test_files = get_data_split(config_name)

    # Build datasets
    X_train, y_train = build_dataset(train_files, horizon_steps, step=15)
    X_test, y_test = build_dataset(test_files, horizon_steps, step=10)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Train XGBoost Q90
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:quantileerror',
        quantile_alpha=0.9,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Precision/Recall pour plusieurs seuils
    results = {
        'rmse': float(rmse),
        'r2': float(r2),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }

    for threshold in [0.7, 0.8, 0.9]:
        true_danger = y_test >= threshold
        pred_danger = y_pred >= threshold

        if np.sum(true_danger) > 0:
            recall = np.sum(true_danger & pred_danger) / np.sum(true_danger)
        else:
            recall = np.nan

        if np.sum(pred_danger) > 0:
            precision = np.sum(true_danger & pred_danger) / np.sum(pred_danger)
        else:
            precision = np.nan

        results[f'recall_{threshold}'] = float(recall) if not np.isnan(recall) else None
        results[f'precision_{threshold}'] = float(precision) if not np.isnan(precision) else None

    # Save model
    model_path = MODEL_DIR / f"XGBoost_{config_name}_{horizon_key}.json"
    model.save_model(str(model_path))

    return results


def main():
    print("=" * 60)
    print("Entrainement des modeles XGBoost Q90 - MAX HORIZON")
    print("Tache: max(|LTR|) sur horizons 1s, 2s, 4s, 6s, 8s")
    print("=" * 60)

    all_results = {}

    for ds in DATASETS:
        print(f"\n{'='*20} {ds} {'='*20}")
        all_results[ds] = {}

        for h in HORIZONS:
            t0 = time.time()
            result = train_and_save_xgboost(ds, h)
            dt = time.time() - t0

            if result:
                all_results[ds][f'h{h}s'] = result
                recall_str = f"{result['recall_0.7']*100:.1f}%" if result['recall_0.7'] else "N/A"
                print(f"  h={h}s: RMSE={result['rmse']:.3f}, R2={result['r2']:.3f}, "
                      f"Recall@0.7={recall_str} ({dt:.1f}s)")

    # Sauvegarde des resultats
    with open(OUTPUT_DIR / "results_xgboost.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESUME DES RESULTATS")
    print("=" * 60)

    print(f"\n{'Dataset':<6} {'Horizon':<8} {'RMSE':<8} {'R2':<8} {'Recall@0.7':<12}")
    print("-" * 50)

    for ds in DATASETS:
        for h in HORIZONS:
            hk = f'h{h}s'
            if hk in all_results.get(ds, {}):
                r = all_results[ds][hk]
                recall_str = f"{r['recall_0.7']*100:.1f}%" if r['recall_0.7'] else "N/A"
                print(f"{ds:<6} {hk:<8} {r['rmse']:<8.3f} {r['r2']:<8.3f} {recall_str:<12}")

    print(f"\nModeles sauvegardes dans {MODEL_DIR}")
    print(f"Resultats sauvegardes dans {OUTPUT_DIR / 'results_xgboost.json'}")


if __name__ == '__main__':
    main()
