#!/usr/bin/env python3
"""Entraîne MLP, LSTM, PatchTST et XGBoost pour la prédiction du max(|LTR|) sur [t, t+h]."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import json
import time

# ============== Configuration ==============

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DT = 0.01
SEQUENCE_LENGTH = 150  # 1.5s lookback

# Horizons de prediction (en secondes)
HORIZONS = [1, 2, 4, 6, 8]

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15

FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = len(FEATURES)

# Configurations des datasets
DATASET_CONFIGS = {
    'D1': {'train_threshold': 0.7, 'test_above': False, 'desc': 'In-distribution (LTR <= 0.7)'},
    'D2': {'train_threshold': 0.7, 'test_above': True, 'desc': 'OOD (train <= 0.7, test > 0.7)'},
    'D3': {'train_threshold': 0.8, 'test_above': True, 'desc': 'OOD (train <= 0.8, test > 0.8)'},
    'D4': {'train_threshold': 0.9, 'test_above': True, 'desc': 'OOD critique (train <= 0.9, test > 0.9)'},
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ============== Architectures ==============

class MLP(nn.Module):
    """MLP identique a horizon_exact."""
    def __init__(self, input_size, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class LSTM(nn.Module):
    """LSTM identique a horizon_exact."""
    def __init__(self, input_size, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PatchTST(nn.Module):
    """PatchTST identique a horizon_exact."""
    def __init__(self, seq_len, n_features, patch_size=15, d_model=64, nhead=4, num_layers=2, output_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size

        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x):
        B = x.shape[0]
        x = x[:, :self.n_patches * self.patch_size, :].reshape(B, self.n_patches, -1)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.head(self.transformer(x)[:, 0])


# ============== Loss ==============

def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """Perte quantile pour regression multi-quantile."""
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.mean(torch.max(q * err, (q - 1) * err)))
    return sum(losses)


# ============== Data ==============

def load_scenario(filepath):
    """Charge un scenario CSV et calcule les features derivees."""
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def create_sequences_max(df, horizon_steps):
    """
    Cree des sequences avec max(LTR) sur l'horizon.

    Difference avec horizon_exact: ici on predit le MAX sur [t, t+h],
    pas la valeur a l'instant exact t+h.
    """
    features = df[FEATURES].values
    ltr = np.abs(df['LTRmax'].values)
    X, y = [], []

    for i in range(0, len(df) - SEQUENCE_LENGTH - horizon_steps, 10):  # stride 10
        X.append(features[i:i + SEQUENCE_LENGTH])
        # Target = MAX du LTR sur l'horizon [t, t+h]
        y.append(np.max(ltr[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + horizon_steps]))

    return np.array(X), np.array(y)


def load_data(dataset_name, horizon_steps):
    """Charge les donnees pour un dataset et un horizon."""
    config = DATASET_CONFIGS[dataset_name]
    files = sorted(DATA_DIR.glob("*.csv"))

    scenarios = []
    for f in files:
        df = load_scenario(f)
        scenarios.append({'path': f, 'df': df, 'max_ltr': np.max(np.abs(df['LTRmax'].values))})

    # Split train/test selon configuration
    if config['test_above']:
        train_sc = [s for s in scenarios if s['max_ltr'] <= config['train_threshold']]
        test_sc = [s for s in scenarios if s['max_ltr'] > config['train_threshold']]
    else:
        eligible = [s for s in scenarios if s['max_ltr'] <= config['train_threshold']]
        np.random.seed(42)
        np.random.shuffle(eligible)
        split = int(0.8 * len(eligible))
        train_sc, test_sc = eligible[:split], eligible[split:]

    if len(train_sc) == 0 or len(test_sc) == 0:
        return None, None, None, None, None, 0, 0

    # Creer les sequences
    X_train_list, y_train_list = [], []
    for s in train_sc:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if len(X) > 0:
            X_train_list.append(X)
            y_train_list.append(y)

    X_test_list, y_test_list = [], []
    for s in test_sc:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if len(X) > 0:
            X_test_list.append(X)
            y_test_list.append(y)

    if not X_train_list or not X_test_list:
        return None, None, None, None, None, 0, 0

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test, scaler, len(train_sc), len(test_sc)


# ============== Training ==============

def train_nn_model(model, train_loader, val_loader):
    """Entraine un modele NN avec early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = quantile_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(quantile_loss(model(X.to(device)), y.to(device)).item()
                           for X, y in val_loader) / len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, no_improve = val_loss, model.state_dict().copy(), 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


# ============== Evaluation ==============

def evaluate_nn(model, X_test, y_test):
    """Evalue un modele NN."""
    model.eval()
    with torch.no_grad():
        preds = np.concatenate([
            model(torch.FloatTensor(X_test[i:i + 512]).to(device)).cpu().numpy()
            for i in range(0, len(X_test), 512)
        ])

    pred_q10 = np.clip(preds[:, 0], 0, 1.5)
    pred_q50 = np.clip(preds[:, 1], 0, 1.5)
    pred_q90 = np.clip(preds[:, 2], 0, 1.5)

    return compute_metrics(y_test, pred_q50, pred_q90)


def compute_metrics(y_test, pred_q50, pred_q90):
    """Calcule les metriques de performance."""
    r2 = r2_score(y_test, pred_q50)
    rmse = np.sqrt(mean_squared_error(y_test, pred_q50))
    mae = np.mean(np.abs(y_test - pred_q50))

    # Coverage Q90 (pour max_horizon, on veut Q90 >= y_test)
    coverage_90 = np.mean(y_test <= pred_q90)

    # Precision/Recall pour plusieurs seuils
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coverage_90': coverage_90,
    }

    for threshold in [0.7, 0.8, 0.9]:
        pred_danger = pred_q90 >= threshold
        true_danger = y_test >= threshold

        if pred_danger.sum() > 0:
            precision = np.sum(pred_danger & true_danger) / np.sum(pred_danger)
        else:
            precision = 0

        if true_danger.sum() > 0:
            recall = np.sum(pred_danger & true_danger) / np.sum(true_danger)
        else:
            recall = 1.0  # Pas de danger = parfait

        metrics[f'precision_{threshold}'] = precision
        metrics[f'recall_{threshold}'] = recall

    return metrics


# ============== Main ==============

def main():
    print("=" * 70)
    print("ENTRAINEMENT MAX HORIZON - MLP, LSTM, PatchTST")
    print("Tache: max(|LTR|) sur horizons 1s, 2s, 4s, 6s, 8s")
    print(f"Device: {device}")
    print("=" * 70)

    all_results = {}

    for dataset_name in ['D1', 'D2', 'D3', 'D4']:
        config = DATASET_CONFIGS[dataset_name]
        all_results[dataset_name] = {}

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} - {config['desc']}")
        print(f"{'='*60}")

        for horizon in HORIZONS:
            horizon_key = f'h{horizon}s'
            horizon_steps = int(horizon / DT)
            all_results[dataset_name][horizon_key] = {}

            print(f"\n  --- Horizon {horizon}s ({horizon_steps} pas) ---")

            # Charger les donnees
            X_train, y_train, X_test, y_test, scaler, n_train, n_test = load_data(
                dataset_name, horizon_steps
            )

            if X_train is None:
                print(f"    Pas assez de donnees!")
                continue

            print(f"    Train: {n_train} scenarios, {len(X_train)} echantillons")
            print(f"    Test: {n_test} scenarios, {len(X_test)} echantillons")
            print(f"    LTR test: [{y_test.min():.2f}, {y_test.max():.2f}]")

            # Sauvegarder scaler
            with open(MODEL_DIR / f"scaler_{dataset_name}_{horizon_key}.pkl", 'wb') as f:
                pickle.dump(scaler, f)

            # DataLoaders pour NN
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
                batch_size=BATCH_SIZE
            )

            # Entrainer les modeles NN
            for model_name in ['MLP', 'LSTM', 'PatchTST']:
                print(f"    {model_name}...", end=" ", flush=True)
                t0 = time.time()

                if model_name == 'MLP':
                    model = MLP(SEQUENCE_LENGTH * N_FEATURES)
                elif model_name == 'LSTM':
                    model = LSTM(N_FEATURES)
                else:
                    model = PatchTST(SEQUENCE_LENGTH, N_FEATURES)

                model = train_nn_model(model, train_loader, val_loader)
                metrics = evaluate_nn(model, X_test, y_test)

                torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_{dataset_name}_{horizon_key}.pt")

                all_results[dataset_name][horizon_key][model_name] = {
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'r2': float(metrics['r2']),
                    'coverage_90': float(metrics['coverage_90']),
                    'precision_0.7': float(metrics['precision_0.7']),
                    'recall_0.7': float(metrics['recall_0.7']),
                    'precision_0.9': float(metrics['precision_0.9']),
                    'recall_0.9': float(metrics['recall_0.9']),
                }

                dt = time.time() - t0
                print(f"RMSE={metrics['rmse']:.3f} R2={metrics['r2']:.3f} "
                      f"Recall@0.7={metrics['recall_0.7']:.1%} ({dt:.1f}s)")

    # Sauvegarder resultats
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESUME DES RESULTATS")
    print("=" * 70)

    # Tableau recapitulatif
    print(f"\n{'Dataset':<6} {'Horizon':<8} {'Model':<10} {'RMSE':<8} {'R2':<8} {'Recall@0.7':<12}")
    print("-" * 60)

    for ds in ['D1', 'D2', 'D3', 'D4']:
        for h in HORIZONS:
            hk = f'h{h}s'
            if hk in all_results.get(ds, {}):
                for m in ['MLP', 'LSTM', 'PatchTST']:
                    if m in all_results[ds][hk]:
                        r = all_results[ds][hk][m]
                        print(f"{ds:<6} {hk:<8} {m:<10} {r['rmse']:<8.3f} {r['r2']:<8.3f} {r['recall_0.7']:<12.1%}")

    print(f"\nModeles sauvegardes dans: {MODEL_DIR}")
    print(f"Resultats sauvegardes dans: {OUTPUT_DIR / 'results.json'}")
    print("\nTERMINE!")


if __name__ == '__main__':
    main()
