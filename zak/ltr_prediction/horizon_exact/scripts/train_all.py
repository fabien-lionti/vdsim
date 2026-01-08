#!/usr/bin/env python3
"""
Entrainement des modeles MLP, LSTM et PatchTST pour la prediction du LTR.

Tache: Prediction de LTR(t+h) a l'instant exact t+h.
Horizons: h = 1, 2, 4, 6, 8 secondes

Architecture des modeles:
- MLP: Reseau fully-connected avec BatchNorm et Dropout
- LSTM: Reseau recurrent (1 couche, 64 unites)
- PatchTST: Transformer avec tokenisation par patches temporels

Configuration:
- Entree: 8 features (vx, vy, psi, psi_dot, phi, theta, delta_f, delta_f_dot)
- Sequence: 1.5s (150 pas de temps a 100Hz)
- Sortie: 3 quantiles (Q10, Q50, Q90) pour estimation d'incertitude
- Donnees: 484 scenarios depuis data_new/

Jeux de donnees:
- D1: Distribution normale (train et test <= 0.7, split 80/20)
- D4: Hors-distribution (train <= 0.9, test > 0.9) - cas critiques

Auteur: Zak
Date: Janvier 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import json


# ============== Configuration des chemins ==============

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path("/Users/zak/Documents/vdsim/zak/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== Hyperparametres ==============

DT = 0.01                    # Pas de temps (10ms = 100Hz)
SEQUENCE_LENGTH = 150        # Lookback: 1.5s

# Horizons de prediction en secondes
HORIZONS_SEC = [1, 2, 4, 6, 8]
HORIZONS_STEPS = {h: int(h / DT) for h in HORIZONS_SEC}  # {1: 100, 2: 200, ...}

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15                # Early stopping patience

# 8 features d'entree
FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = len(FEATURES)

# Jeux de donnees a entrainer
DATASETS = ['D1', 'D4']

# Configuration des splits train/test
DATASET_CONFIGS = {
    'D1': {
        'train_threshold': 0.7,
        'test_above': False,
        'desc': 'In-distribution (train/test LTR <= 0.7)'
    },
    'D4': {
        'train_threshold': 0.9,
        'test_above': True,
        'desc': 'OOD critique (train <= 0.9, test > 0.9)'
    },
}

# Selection automatique du device (MPS pour Mac M1/M2, sinon CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Features ({N_FEATURES}): {FEATURES}")
print(f"Sequence: {SEQUENCE_LENGTH} pas ({SEQUENCE_LENGTH*DT:.1f}s)")
print(f"Horizons: {HORIZONS_SEC} secondes")


# ============== Architectures des modeles ==============

class MLP(nn.Module):
    """
    Perceptron multicouche pour la prediction du LTR.
    Architecture: 1200 -> 512 -> 256 -> 128 -> 3
    """

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
    """
    LSTM pour la prediction du LTR.
    Architecture: LSTM(8 -> 64) puis FC(64 -> 32 -> 3)
    """

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
    """
    Transformer avec tokenisation par patches pour series temporelles.
    """

    def __init__(self, seq_len, n_features, patch_size=15, d_model=64, nhead=4,
                 num_layers=2, output_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size

        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=d_model*4,
                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x):
        B, T, C = x.shape
        x = x[:, :self.n_patches * self.patch_size, :]
        x = x.reshape(B, self.n_patches, -1)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.transformer(x)
        return self.head(x[:, 0])


# ============== Fonction de perte ==============

def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """
    Perte quantile pour regression multi-quantile.
    """
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.mean(torch.max(q * err, (q - 1) * err)))
    return sum(losses)


# ============== Chargement des donnees ==============

def load_scenario(filepath):
    """Charge un scenario CSV et calcule les features derivees."""
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def create_sequences_exact_horizon(df, horizon_steps):
    """
    Cree des sequences pour prediction a l'horizon exact.

    Args:
        df: DataFrame du scenario
        horizon_steps: Nombre de pas pour l'horizon (ex: 200 pour 2s)

    Returns:
        X: Sequences d'entree (n_samples, seq_len, n_features)
        y: LTR a l'instant t+horizon (n_samples,)
    """
    data = df[FEATURES].values
    ltr = np.abs(df['LTRmax'].values)

    X, y = [], []
    max_start = len(data) - SEQUENCE_LENGTH - horizon_steps

    for i in range(0, max_start, 10):  # Stride de 10 pour eviter trop de correlation
        X.append(data[i:i + SEQUENCE_LENGTH])
        # Target = LTR a l'instant exact t + horizon
        y.append(ltr[i + SEQUENCE_LENGTH + horizon_steps - 1])

    return np.array(X), np.array(y)


def prepare_dataset(dataset_name, horizon_steps):
    """Prepare les donnees pour un dataset et un horizon donnes."""
    config = DATASET_CONFIGS[dataset_name]

    all_files = sorted(DATA_DIR.glob("*.csv"))
    print(f"\n{dataset_name}: {config['desc']}")
    print(f"Horizon: {horizon_steps * DT:.0f}s ({horizon_steps} pas)")

    # Classifier les scenarios par LTR max
    scenarios_by_ltr = {'low': [], 'high': []}
    for f in all_files:
        df = pd.read_csv(f)
        ltr_max = np.max(np.abs(df['LTRmax'].values))
        if ltr_max <= config['train_threshold']:
            scenarios_by_ltr['low'].append(f)
        else:
            scenarios_by_ltr['high'].append(f)

    print(f"  Scenarios LTR <= {config['train_threshold']}: {len(scenarios_by_ltr['low'])}")
    print(f"  Scenarios LTR > {config['train_threshold']}: {len(scenarios_by_ltr['high'])}")

    # Preparer train/test selon la config
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    if config['test_above']:
        # Train sur low, test sur high (OOD)
        for f in scenarios_by_ltr['low']:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_train_list.append(X)
                y_train_list.append(y)

        for f in scenarios_by_ltr['high']:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)
    else:
        # Train/test split 80/20 sur les scenarios low
        np.random.seed(42)
        low_files = scenarios_by_ltr['low'].copy()
        np.random.shuffle(low_files)
        split_idx = int(0.8 * len(low_files))

        for f in low_files[:split_idx]:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_train_list.append(X)
                y_train_list.append(y)

        for f in low_files[split_idx:]:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Normalisation
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, N_FEATURES)
    X_test_flat = X_test.reshape(-1, N_FEATURES)

    scaler.fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test = scaler.transform(X_test_flat).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test, scaler


# ============== Entrainement ==============

def train_model(model, train_loader, val_loader, model_name):
    """Entraine un modele avec early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = quantile_loss(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss += quantile_loss(model(X), y).item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader, danger_threshold=0.7):
    """Evalue un modele et retourne les metriques."""
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)

    # Metriques sur Q50 (mediane)
    q50 = pred[:, 1]
    rmse = np.sqrt(mean_squared_error(true, q50))
    mae = np.mean(np.abs(true - q50))
    r2 = r2_score(true, q50)

    # Coverage des quantiles
    coverage_90 = np.mean((true >= pred[:, 0]) & (true <= pred[:, 2]))

    # Precision et Rappel pour detection de danger (LTR > seuil)
    pred_danger = q50 > danger_threshold
    true_danger = true > danger_threshold

    tp = np.sum(pred_danger & true_danger)
    fp = np.sum(pred_danger & ~true_danger)
    fn = np.sum(~pred_danger & true_danger)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coverage_90': coverage_90,
        'precision': precision,
        'recall': recall,
        'pred': pred,
        'true': true
    }


# ============== Main ==============

def main():
    results = {}

    for dataset_name in DATASETS:
        results[dataset_name] = {}

        for horizon_sec in HORIZONS_SEC:
            horizon_steps = HORIZONS_STEPS[horizon_sec]
            horizon_key = f"h{horizon_sec}s"
            results[dataset_name][horizon_key] = {}

            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name}, Horizon: {horizon_sec}s")
            print('='*60)

            # Preparer les donnees
            X_train, y_train, X_test, y_test, scaler = prepare_dataset(
                dataset_name, horizon_steps
            )

            # Sauvegarder le scaler
            scaler_path = MODEL_DIR / f"scaler_{dataset_name}_{horizon_key}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            # DataLoaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.FloatTensor(y_test)
            )

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            # Entrainer chaque modele
            models = {
                'MLP': MLP(SEQUENCE_LENGTH * N_FEATURES).to(device),
                'LSTM': LSTM(N_FEATURES).to(device),
                'PatchTST': PatchTST(SEQUENCE_LENGTH, N_FEATURES).to(device),
            }

            for model_name, model in models.items():
                print(f"\n  Training {model_name}...")
                model = train_model(model, train_loader, test_loader, model_name)

                # Sauvegarder le modele
                model_path = MODEL_DIR / f"{model_name}_{dataset_name}_{horizon_key}.pt"
                torch.save(model.state_dict(), model_path)

                # Evaluer
                metrics = evaluate_model(model, test_loader)
                results[dataset_name][horizon_key][model_name] = {
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae']),
                    'r2': float(metrics['r2']),
                    'coverage_90': float(metrics['coverage_90']),
                    'precision': float(metrics['precision']),
                    'recall': float(metrics['recall'])
                }

                print(f"    RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}, "
                      f"Prec: {metrics['precision']:.2%}, Recall: {metrics['recall']:.2%}")

    # Sauvegarder les resultats
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResultats sauvegardes dans {OUTPUT_DIR / 'results.json'}")

    # Generer les graphiques
    plot_results(results)

    return results


def plot_results(results):
    """Genere les graphiques de comparaison."""

    # Figure 1: RMSE par horizon et modele
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparaison des modeles - RMSE par horizon', fontsize=14, fontweight='bold')

    for idx, dataset_name in enumerate(DATASETS):
        ax = axes[idx]

        model_names = ['MLP', 'LSTM', 'PatchTST']
        x = np.arange(len(HORIZONS_SEC))
        width = 0.25

        for i, model_name in enumerate(model_names):
            rmse_values = [
                results[dataset_name][f"h{h}s"][model_name]['rmse']
                for h in HORIZONS_SEC
            ]
            ax.bar(x + i * width, rmse_values, width, label=model_name)

        ax.set_xlabel('Horizon (s)')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{dataset_name}: {DATASET_CONFIGS[dataset_name]["desc"]}')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{h}s' for h in HORIZONS_SEC])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparaison_rmse.png", dpi=150)
    plt.close()

    # Figure 2: R2 par horizon
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Comparaison des modeles - R2 par horizon', fontsize=14, fontweight='bold')

    for idx, dataset_name in enumerate(DATASETS):
        ax = axes[idx]

        for model_name in ['MLP', 'LSTM', 'PatchTST']:
            r2_values = [
                results[dataset_name][f"h{h}s"][model_name]['r2']
                for h in HORIZONS_SEC
            ]
            ax.plot(HORIZONS_SEC, r2_values, 'o-', label=model_name, linewidth=2, markersize=8)

        ax.set_xlabel('Horizon (s)')
        ax.set_ylabel('R2')
        ax.set_title(f'{dataset_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparaison_r2.png", dpi=150)
    plt.close()

    # Figure 3: Tableau recapitulatif avec meilleurs resultats en gras
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.axis('off')

    # Creer le tableau avec 4 metriques
    headers = ['Dataset', 'Horizon'] + [f'{m}\nRMSE/R2/Prec/Rec' for m in ['MLP', 'LSTM', 'PatchTST']]
    table_data = []
    best_cells = []  # (row_idx, col_idx) des meilleures cellules

    model_names = ['MLP', 'LSTM', 'PatchTST']
    row_idx = 0

    for dataset_name in DATASETS:
        for h in HORIZONS_SEC:
            row = [dataset_name, f'{h}s']

            # Trouver le meilleur modele (RMSE min)
            rmse_values = [results[dataset_name][f"h{h}s"][m]['rmse'] for m in model_names]
            best_model_idx = np.argmin(rmse_values)

            for i, model_name in enumerate(model_names):
                r = results[dataset_name][f"h{h}s"][model_name]
                prec = r.get('precision', 0)
                rec = r.get('recall', 0)
                row.append(f"{r['rmse']:.3f} / {r['r2']:.2f}\n{prec:.0%} / {rec:.0%}")
                if i == best_model_idx:
                    best_cells.append((row_idx + 1, i + 2))

            table_data.append(row)
            row_idx += 1

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#f0f0f0']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.2)

    # Mettre en evidence les meilleures cellules
    for (r, c) in best_cells:
        cell = table[r, c]
        cell.set_facecolor('#90EE90')  # Vert clair
        cell.set_text_props(fontweight='bold')

    plt.title('Resultats: Prediction LTR(t+h)\nRMSE / R2 / Precision / Recall (seuil danger: LTR>0.7)\nMeilleur RMSE en vert',
              fontsize=12, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / "tableau_resultats.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Graphiques sauvegardes dans {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
