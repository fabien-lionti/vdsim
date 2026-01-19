#!/usr/bin/env python3
"""
Entraînement des modèles XGBoost Q90 pour toutes les configurations.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score
import time

# Paths
PROJECT_DIR = Path("/Users/zak/Documents/vdsim/zak/ltr_prediction/horizon_exact")
DATA_DIR = Path("/Users/zak/Documents/vdsim/zak/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

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
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def extract_features(window):
    """Extrait les features statistiques d'une fenêtre temporelle."""
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
    """Récupère le split train/test pour une configuration donnée."""
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
    """Construit le dataset de features pour XGBoost."""
    X_list, y_list = [], []
    for f in files:
        df = load_scenario(f)
        data = df[FEATURES].values
        ltr = np.abs(df['LTRmax'].values)
        for i in range(0, len(data) - SEQ_LEN - horizon_steps, step):
            X_list.append(extract_features(data[i:i + SEQ_LEN]))
            y_list.append(ltr[i + SEQ_LEN + horizon_steps - 1])
    return np.array(X_list), np.array(y_list)


def train_and_save_xgboost(config_name, horizon):
    """Entraîne et sauvegarde un modèle XGBoost Q90."""
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

    true_danger = y_test >= 0.7
    pred_danger = y_pred >= 0.7

    if np.sum(true_danger) > 0:
        recall = np.sum(true_danger & pred_danger) / np.sum(true_danger)
    else:
        recall = np.nan

    if np.sum(pred_danger) > 0:
        precision = np.sum(true_danger & pred_danger) / np.sum(pred_danger)
    else:
        precision = np.nan

    # Save model
    model_path = MODEL_DIR / f"XGBoost_{config_name}_{horizon_key}.json"
    model.save_model(str(model_path))

    return {
        'rmse': float(rmse),
        'r2': float(r2),
        'recall': float(recall) if not np.isnan(recall) else None,
        'precision': float(precision) if not np.isnan(precision) else None,
        'n_train': len(y_train),
        'n_test': len(y_test)
    }


def main():
    print("=" * 60)
    print("Entraînement des modèles XGBoost Q90")
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
                recall_str = f"{result['recall']*100:.1f}%" if result['recall'] else "N/A"
                print(f"  h={h}s: RMSE={result['rmse']:.3f}, Recall={recall_str} ({dt:.1f}s)")

    # Sauvegarde des résultats
    with open(OUTPUT_DIR / "results_xgboost.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nModèles sauvegardés dans {MODEL_DIR}")
    print(f"Résultats sauvegardés dans {OUTPUT_DIR / 'results_xgboost.json'}")


if __name__ == '__main__':
    main()
