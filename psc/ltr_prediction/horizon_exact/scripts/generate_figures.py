#!/usr/bin/env python3
"""Génère les figures du rapport LTR prediction (horizon exact)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import xgboost as xgb

# =============================================================================
# Configuration
# =============================================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

PROJECT_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/horizon_exact")
DATA_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DT = 0.01
SEQ_LEN = 150
FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = len(FEATURES)
HORIZONS = [1, 2, 4, 6, 8]

COLORS = {'MLP': '#e74c3c', 'LSTM': '#3498db', 'PatchTST': '#27ae60', 'XGBoost': '#9b59b6'}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

DATASET_CONFIGS = {
    'D1': {'threshold': 0.7, 'test_above': False},
    'D2': {'threshold': 0.7, 'test_above': True},
    'D3': {'threshold': 0.8, 'test_above': True},
    'D4': {'threshold': 0.9, 'test_above': True},
}

# Trajectoires pour figures 5 et 6
TRAJECTORIES = {
    'Oscillations rapides': 'single_v14_R28_d-1_mu100.csv',
    'Lemniscate (8 répétés)': 'lemniscate_v10_a5_n10_mu100.csv',
    'Transition progressive': 'single_v18_R35_d1_mu90.csv',
}


# =============================================================================
# Définitions des modèles
# =============================================================================
class MLP(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x.reshape(x.size(0), -1))


class LSTM(nn.Module):
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


# =============================================================================
# Fonctions utilitaires
# =============================================================================
def load_scenario(filepath):
    """Charge un scénario CSV et calcule les features dérivées."""
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def extract_features_xgb(window):
    """Extrait les features statistiques d'une fenêtre temporelle pour XGBoost."""
    feats = []
    for i in range(window.shape[1]):
        col = window[:, i]
        feats.extend([
            col.mean(), col.std(), col.min(), col.max(),
            col[-1] - col[0], np.abs(np.diff(col)).mean()
        ])
    return np.array(feats)


def get_test_data(config_name, horizon):
    """Récupère les données de test pour une configuration donnée."""
    config = DATASET_CONFIGS[config_name]
    horizon_steps = int(horizon / DT)

    all_files = sorted(DATA_DIR.glob("*.csv"))
    test_files = []

    for f in all_files:
        df = pd.read_csv(f)
        ltr_max = np.max(np.abs(df['LTRmax'].values))
        if config['test_above']:
            if ltr_max > config['threshold']:
                test_files.append(f)
        else:
            if ltr_max <= config['threshold']:
                test_files.append(f)

    if not config['test_above']:
        np.random.seed(42)
        np.random.shuffle(test_files)
        test_files = test_files[int(0.8 * len(test_files)):]

    X_list, y_list = [], []
    for f in test_files:
        df = load_scenario(f)
        data = df[FEATURES].values
        ltr = np.abs(df['LTRmax'].values)
        for i in range(0, len(data) - SEQ_LEN - horizon_steps, 10):
            X_list.append(data[i:i + SEQ_LEN])
            y_list.append(ltr[i + SEQ_LEN + horizon_steps - 1])

    return np.array(X_list), np.array(y_list)


def predict_nn(model_name, config_name, horizon, X):
    """Prédit avec un modèle de réseau de neurones."""
    horizon_key = f"h{horizon}s"
    scaler_path = MODEL_DIR / f"scaler_{config_name}_{horizon_key}.pkl"
    model_path = MODEL_DIR / f"{model_name}_{config_name}_{horizon_key}.pt"

    if not model_path.exists():
        return None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    if model_name == 'MLP':
        model = MLP(SEQ_LEN * N_FEATURES).to(device)
    elif model_name == 'LSTM':
        model = LSTM(N_FEATURES).to(device)
    else:
        model = PatchTST(SEQ_LEN, N_FEATURES).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_scaled = scaler.transform(X.reshape(-1, N_FEATURES)).reshape(X.shape)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled)), batch_size=64)

    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device)).cpu().numpy())

    return np.concatenate(preds)


def predict_xgb(config_name, horizon, X):
    """Prédit avec un modèle XGBoost."""
    horizon_key = f"h{horizon}s"
    model_path = MODEL_DIR / f"XGBoost_{config_name}_{horizon_key}.json"

    if not model_path.exists():
        return None

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    X_feats = np.array([extract_features_xgb(x) for x in X])
    return model.predict(X_feats)


def predict_on_trajectory(model_name, config, horizon, df):
    """Prédit le LTR sur toute la trajectoire."""
    horizon_key = f"h{horizon}s"
    horizon_steps = int(horizon / DT)

    data = df[FEATURES].values
    ltr = np.abs(df['LTRmax'].values)

    times, preds, targets = [], [], []

    if model_name == 'XGBoost':
        model_path = MODEL_DIR / f"XGBoost_{config}_{horizon_key}.json"
        if not model_path.exists():
            return None, None, None

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))

        for i in range(0, len(data) - SEQ_LEN - horizon_steps, 5):
            window = data[i:i + SEQ_LEN]
            feats = extract_features_xgb(window).reshape(1, -1)
            pred = model.predict(feats)[0]
            times.append((i + SEQ_LEN) * DT)
            preds.append([pred * 0.85, pred * 0.95, pred])  # Approximation Q10, Q50, Q90
            targets.append(ltr[i + SEQ_LEN + horizon_steps - 1])
    else:
        scaler_path = MODEL_DIR / f"scaler_{config}_{horizon_key}.pkl"
        model_path = MODEL_DIR / f"{model_name}_{config}_{horizon_key}.pt"

        if not model_path.exists():
            return None, None, None

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        if model_name == 'MLP':
            model = MLP(SEQ_LEN * N_FEATURES).to(device)
        elif model_name == 'LSTM':
            model = LSTM(N_FEATURES).to(device)
        else:
            model = PatchTST(SEQ_LEN, N_FEATURES).to(device)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        for i in range(0, len(data) - SEQ_LEN - horizon_steps, 5):
            window = data[i:i + SEQ_LEN].copy()
            window_scaled = scaler.transform(window)

            with torch.no_grad():
                x = torch.FloatTensor(window_scaled).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy()[0]

            times.append((i + SEQ_LEN) * DT)
            preds.append(pred)
            targets.append(ltr[i + SEQ_LEN + horizon_steps - 1])

    return np.array(times), np.array(preds), np.array(targets)


# =============================================================================
# Figure 1: Tableau comparatif complet
# =============================================================================
def figure_1_tableau_comparatif():
    """Tableau comparatif avec RMSE/R²/Precision/Recall pour tous modèles."""
    print("  Génération figure 1: Tableau comparatif...")

    all_results = {}
    models = ['MLP', 'LSTM', 'PatchTST', 'XGBoost']

    for ds in ['D1', 'D2', 'D3', 'D4']:
        all_results[ds] = {}
        for h in HORIZONS:
            all_results[ds][f'h{h}s'] = {}
            X, y = get_test_data(ds, h)

            for model_name in models:
                if model_name == 'XGBoost':
                    pred = predict_xgb(ds, h, X)
                    if pred is not None:
                        pred_q90 = pred
                        pred_q50 = pred * 0.95
                else:
                    pred = predict_nn(model_name, ds, h, X)
                    if pred is not None:
                        pred_q90 = pred[:, 2]
                        pred_q50 = pred[:, 1]

                if pred is not None:
                    rmse = np.sqrt(mean_squared_error(y, pred_q50))
                    r2 = r2_score(y, pred_q50)

                    true_danger = y >= 0.7
                    pred_danger = pred_q90 >= 0.7
                    recall = np.sum(true_danger & pred_danger) / max(np.sum(true_danger), 1) * 100
                    precision = np.sum(true_danger & pred_danger) / max(np.sum(pred_danger), 1) * 100

                    all_results[ds][f'h{h}s'][model_name] = {
                        'rmse': rmse, 'r2': r2, 'recall': recall, 'precision': precision
                    }

    # Création de la figure
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')

    data = []
    recall_values = []

    for ds in ['D1', 'D2', 'D3', 'D4']:
        for h in HORIZONS:
            hk = f"h{h}s"
            row = [ds, f'{h}s']
            row_recalls = []

            for m in models:
                if m in all_results.get(ds, {}).get(hk, {}):
                    r = all_results[ds][hk][m]
                    cell = f"{r['rmse']:.3f}/{r['r2']:.2f}\n{r['precision']:.0f}%/{r['recall']:.0f}%"
                    row.append(cell)
                    row_recalls.append(r['recall'])
                else:
                    row.append("-")
                    row_recalls.append(0)

            data.append(row)
            recall_values.append(row_recalls)

    headers = ['Config', 'Horizon'] + models

    colors = []
    for row_idx, row in enumerate(data):
        row_colors = ['white', 'white']
        ds = row[0]  # D1, D2, D3, D4
        for col_idx in range(len(models)):
            recall = recall_values[row_idx][col_idx]
            # D1 n'a pas de cas dangereux, couleur neutre
            if ds == 'D1':
                row_colors.append('#f0f0f0')  # Gris neutre
            elif recall >= 80:
                row_colors.append('#a9dfbf')
            elif recall >= 50:
                row_colors.append('#d5f5e3')
            elif recall >= 20:
                row_colors.append('#fef9e7')
            else:
                row_colors.append('#fadbd8')
        colors.append(row_colors)

    table = ax.table(cellText=data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#3498db'] * len(headers), cellColours=colors)

    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=11)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2.5)

    plt.title('Comparaison des modèles\nFormat: RMSE/R² (ligne 1), Precision/Recall Q90 (ligne 2)\n'
              'Couleur selon Recall: vert foncé ≥80%, vert clair ≥50%, jaune ≥20%, rouge <20%',
              fontsize=12, fontweight='bold', pad=20)

    plt.savefig(OUTPUT_DIR / "1_tableau_comparatif.png")
    plt.close()
    print("    -> 1_tableau_comparatif.png")


# =============================================================================
# Figure 2: Matrices de confusion
# =============================================================================
def figure_2_confusion_matrices():
    """Matrices de confusion 4×4 (configs × modèles)."""
    print("  Génération figure 2: Matrices de confusion...")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    bins = [0, 0.7, 0.8, 0.9, 1.5]
    labels = ['<0.7', '0.7-0.8', '0.8-0.9', '≥0.9']
    horizon = 2

    for row, ds in enumerate(['D1', 'D2', 'D3', 'D4']):
        X, y = get_test_data(ds, horizon)

        for col, model_name in enumerate(['MLP', 'LSTM', 'PatchTST', 'XGBoost']):
            ax = axes[row, col]

            if model_name == 'XGBoost':
                pred = predict_xgb(ds, horizon, X)
                pred_val = pred if pred is not None else None
            else:
                pred = predict_nn(model_name, ds, horizon, X)
                pred_val = pred[:, 2] if pred is not None else None

            if pred_val is not None:
                true_cls = np.digitize(y, bins[1:-1])
                pred_cls = np.digitize(pred_val, bins[1:-1])

                cm = confusion_matrix(true_cls, pred_cls, labels=[0, 1, 2, 3])
                cm_pct = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

                sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                            xticklabels=labels, yticklabels=labels,
                            vmin=0, vmax=100, cbar=False, annot_kws={'size': 9})

                true_danger = y >= 0.7
                pred_danger = pred_val >= 0.7
                recall = np.sum(true_danger & pred_danger) / max(np.sum(true_danger), 1) * 100

                ax.set_title(f'{model_name}\nRecall={recall:.0f}%', fontsize=10, color=COLORS[model_name])

            if row == 3:
                ax.set_xlabel('Prédit')
            if col == 0:
                ax.set_ylabel('Réel')

    for i, ds in enumerate(['D1', 'D2', 'D3', 'D4']):
        axes[i, 0].annotate(ds, xy=(-0.4, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', ha='center', va='center')

    fig.suptitle('Matrices de confusion (%) — Horizon 2s — Prédiction Q90', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "2_confusion_matrices.png")
    plt.close()
    print("    -> 2_confusion_matrices.png")


# =============================================================================
# Figure 3: Régression
# =============================================================================
def figure_3_regression():
    """Courbes de régression 4×4."""
    print("  Génération figure 3: Régression...")

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    horizon = 2

    for row, ds in enumerate(['D1', 'D2', 'D3', 'D4']):
        X, y = get_test_data(ds, horizon)

        for col, model_name in enumerate(['MLP', 'LSTM', 'PatchTST', 'XGBoost']):
            ax = axes[row, col]

            if model_name == 'XGBoost':
                pred = predict_xgb(ds, horizon, X)
                pred_val = pred if pred is not None else None
            else:
                pred = predict_nn(model_name, ds, horizon, X)
                pred_val = pred[:, 1] if pred is not None else None

            if pred_val is not None:
                n = min(2000, len(y))
                idx = np.random.choice(len(y), n, replace=False)

                ax.scatter(y[idx], pred_val[idx], alpha=0.3, s=8, c=COLORS[model_name], edgecolors='none')
                ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)

                ax.axvline(0.7, color='red', ls=':', alpha=0.5)
                ax.axhline(0.7, color='red', ls=':', alpha=0.5)

                rmse = np.sqrt(mean_squared_error(y, pred_val))
                r2 = r2_score(y, pred_val)

                ax.text(0.05, 0.95, f'RMSE={rmse:.3f}\nR²={r2:.2f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(model_name, fontsize=11, color=COLORS[model_name], fontweight='bold')

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            if row == 3:
                ax.set_xlabel('LTR réel')
            if col == 0:
                ax.set_ylabel('LTR prédit')

    for i, ds in enumerate(['D1', 'D2', 'D3', 'D4']):
        axes[i, 0].annotate(ds, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', ha='center', va='center')

    fig.suptitle('Régression LTR prédit vs réel — Horizon 2s — Prédiction Q50', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3_regression.png")
    plt.close()
    print("    -> 3_regression.png")


# =============================================================================
# Figure 4: Comparaison recall
# =============================================================================
def figure_4_recall_comparison():
    """Comparaison recall par modèle et configuration."""
    print("  Génération figure 4: Comparaison recall...")

    fig, ax = plt.subplots(figsize=(12, 6))

    models = ['MLP', 'LSTM', 'PatchTST', 'XGBoost']
    configs = ['D2', 'D3', 'D4']
    horizon = 2

    x = np.arange(len(configs))
    width = 0.2

    for i, model_name in enumerate(models):
        recalls = []
        for ds in configs:
            X, y = get_test_data(ds, horizon)

            if model_name == 'XGBoost':
                pred = predict_xgb(ds, horizon, X)
                pred_val = pred if pred is not None else None
            else:
                pred = predict_nn(model_name, ds, horizon, X)
                pred_val = pred[:, 2] if pred is not None else None

            if pred_val is not None:
                true_danger = y >= 0.7
                pred_danger = pred_val >= 0.7
                recall = np.sum(true_danger & pred_danger) / max(np.sum(true_danger), 1) * 100
            else:
                recall = 0
            recalls.append(recall)

        bars = ax.bar(x + i * width, recalls, width, label=model_name, color=COLORS[model_name], alpha=0.85)

        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Recall Q90 (%)')
    ax.set_title('Comparaison du Recall Q90 par modèle — Horizon 2s', fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 110)
    ax.axhline(80, color='green', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "4_recall_comparison.png")
    plt.close()
    print("    -> 4_recall_comparison.png")


# =============================================================================
# Figure 5: Prédictions sur trajectoires
# =============================================================================
def figure_5_predictions_trajectoires():
    """Prédictions sur 3 trajectoires × 4 modèles."""
    print("  Génération figure 5: Prédictions trajectoires...")

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    horizon = 2
    config = 'D4'

    for row, (traj_name, filename) in enumerate(TRAJECTORIES.items()):
        df = load_scenario(DATA_DIR / filename)

        for col, model_name in enumerate(['MLP', 'LSTM', 'PatchTST', 'XGBoost']):
            ax = axes[row, col]

            times, preds, targets = predict_on_trajectory(model_name, config, horizon, df)
            if times is None:
                ax.text(0.5, 0.5, 'Model not found', ha='center', va='center', transform=ax.transAxes)
                continue

            # Style unifié avec figure 6
            ax.plot(times, targets, 'k-', lw=2.5, label='LTR réel')
            ax.fill_between(times, preds[:, 0], preds[:, 2], alpha=0.25,
                            color=COLORS[model_name], label='Intervalle Q10-Q90')
            ax.plot(times, preds[:, 1], '-', lw=2, color=COLORS[model_name], label='Prédiction Q50')

            ax.axhline(0.7, color='red', ls='--', lw=1.5, alpha=0.6, label='Seuil danger')
            ax.fill_between(times, 0.7, 1.0, alpha=0.08, color='red')

            ax.set_ylim(0, 1.05)
            ax.set_xlim(times[0], times[-1])
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(model_name, fontsize=12, fontweight='bold', color=COLORS[model_name])
            if row == 2:
                ax.set_xlabel('Temps (s)')
            if col == 0:
                ax.set_ylabel('LTR')
                ax.annotate(traj_name, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=11, fontweight='bold', ha='center', va='center', rotation=90)

            if row == 0 and col == 3:
                ax.legend(loc='upper left', fontsize=8)

            # Métriques comme figure 6
            rmse = np.sqrt(np.mean((targets - preds[:, 1]) ** 2))
            danger_mask = targets >= 0.7
            if danger_mask.sum() > 0:
                recall = (preds[danger_mask, 2] >= 0.7).mean() * 100
            else:
                recall = np.nan
            recall_str = f'{recall:.0f}%' if not np.isnan(recall) else '-'
            ax.text(0.98, 0.02, f'RMSE={rmse:.3f}\nRecall={recall_str}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle(f'Prédiction LTR sur trajectoires — Horizon {horizon}s — Modèles {config}',
                 fontweight='bold', fontsize=13, y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "5_predictions_trajectoires.png")
    plt.close()
    print("    -> 5_predictions_trajectoires.png")


# =============================================================================
# Figure 6: Détail prédiction
# =============================================================================
def figure_6_prediction_detail():
    """Figure détaillée sur une seule trajectoire."""
    print("  Génération figure 6: Détail prédiction...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    horizon = 2
    config = 'D4'
    filename = 'single_v14_R28_d-1_mu100.csv'

    df = load_scenario(DATA_DIR / filename)

    for idx, model_name in enumerate(['MLP', 'LSTM', 'PatchTST', 'XGBoost']):
        ax = axes[idx]

        times, preds, targets = predict_on_trajectory(model_name, config, horizon, df)
        if times is None:
            ax.text(0.5, 0.5, 'Model not found', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.plot(times, targets, 'k-', lw=2.5, label='LTR réel')
        ax.fill_between(times, preds[:, 0], preds[:, 2], alpha=0.25,
                        color=COLORS[model_name], label='Intervalle Q10-Q90')
        ax.plot(times, preds[:, 1], '-', lw=2, color=COLORS[model_name], label='Prédiction Q50')

        ax.axhline(0.7, color='red', ls='--', lw=1.5, alpha=0.6, label='Seuil danger')
        ax.fill_between(times, 0.7, 1.0, alpha=0.08, color='red')

        ax.set_ylim(0, 1.05)
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('Temps (s)')
        ax.set_ylabel('LTR')
        ax.set_title(model_name, fontsize=12, fontweight='bold', color=COLORS[model_name])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        rmse = np.sqrt(np.mean((targets - preds[:, 1]) ** 2))
        danger_mask = targets >= 0.7
        if danger_mask.sum() > 0:
            recall = (preds[danger_mask, 2] >= 0.7).mean() * 100
        else:
            recall = np.nan

        ax.text(0.98, 0.02, f'RMSE={rmse:.3f}\nRecall Q90={recall:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle(f'Prédiction LTR détaillée — Trajectoire avec oscillations — Horizon {horizon}s — Modèles {config}',
                 fontweight='bold', fontsize=13, y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "6_prediction_detail.png")
    plt.close()
    print("    -> 6_prediction_detail.png")


# =============================================================================
# Programme principal
# =============================================================================
def main():
    print("=" * 60)
    print("Génération des figures pour le rapport LTR prediction")
    print("=" * 60)

    figure_1_tableau_comparatif()
    figure_2_confusion_matrices()
    figure_3_regression()
    figure_4_recall_comparison()
    figure_5_predictions_trajectoires()
    figure_6_prediction_detail()

    print("\n" + "=" * 60)
    print(f"6 figures générées dans {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
