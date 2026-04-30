#!/usr/bin/env python3
"""Génère les figures comparatives NODE (baseline vs LSTM+NODE / PatchTST+NODE)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_node import VehicleDynamicsNet, N_INPUT as NODE_N_INPUT, N_STATE, DT
from train_with_node import (
    LSTM_Physics, PatchTST_Physics, PhysicsProjection,
    FEATURES_8, N_FEATURES, N_INPUT_AUGMENTED,
    SEQ_LEN, load_scenario
)

# =============================================================================
# Configuration — style identique à generate_figures.py
# =============================================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

NODE_DIR = Path(__file__).parent.parent
MAXHORIZON_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/maxhorizon")
DATA_DIR = NODE_DIR / "data"
MODEL_DIR = NODE_DIR / "models"
OUTPUT_DIR = NODE_DIR / "outputs"

HORIZONS = [1, 2, 4]
COLORS = {
    'LSTM': '#3498db',
    'LSTM+NODE': '#1a5276',
    'PatchTST': '#27ae60',
    'PatchTST+NODE': '#0e6027',
}

DATASET_CONFIGS = {
    'D2': {'threshold': 0.7, 'test_above': True},
    'D3': {'threshold': 0.8, 'test_above': True},
    'D4': {'threshold': 0.9, 'test_above': True},
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Modèles baseline (identiques à maxhorizon)
class LSTM_Baseline(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, output_size))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PatchTST_Baseline(nn.Module):
    def __init__(self, seq_len=150, n_features=8, patch_size=15, d_model=64, nhead=4, num_layers=2, output_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)
    def forward(self, x):
        B = x.shape[0]
        x = x[:, :self.n_patches * self.patch_size, :].reshape(B, self.n_patches, -1)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.head(self.transformer(x)[:, 0])


# =============================================================================
# Data & Prediction
# =============================================================================
def get_test_files(config_name):
    config = DATASET_CONFIGS[config_name]
    all_files = sorted(DATA_DIR.glob("*.csv"))
    return [f for f in all_files
            if _file_max_ltr(f) > config['threshold']]

def _file_max_ltr(f):
    try:
        df = pd.read_csv(f)
        return np.max(np.abs(df['LTRmax'].values))
    except:
        return 0

def get_test_data(config_name, horizon):
    """Récupère X, y pour le test set."""
    test_files = get_test_files(config_name)
    horizon_steps = int(horizon / DT)
    X_list, y_list = [], []
    for f in test_files:
        df = load_scenario(f)
        if df is None: continue
        data = df[FEATURES_8].values.astype(np.float32)
        ltr = np.abs(df['LTRmax'].values)
        for i in range(0, len(data) - SEQ_LEN - horizon_steps, 10):
            X_list.append(data[i:i + SEQ_LEN])
            y_list.append(np.max(ltr[i + SEQ_LEN:i + SEQ_LEN + horizon_steps]))
    return np.array(X_list), np.array(y_list)


def predict_baseline(model_type, config_name, horizon, X):
    """Prédictions baseline."""
    hkey = f'h{horizon}s'
    model_path = MAXHORIZON_DIR / "models" / f"{model_type}_{config_name}_{hkey}.pt"
    scaler_path = MAXHORIZON_DIR / "models" / f"scaler_{config_name}_{hkey}.pkl"
    if not model_path.exists(): return None

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    if model_type == 'LSTM':
        model = LSTM_Baseline(N_FEATURES).to(device)
    else:
        model = PatchTST_Baseline(SEQ_LEN, N_FEATURES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    X_s = scaler.transform(X.reshape(-1, N_FEATURES)).reshape(X.shape).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_s)), batch_size=64)
    preds = []
    with torch.no_grad():
        for (batch,) in loader:
            preds.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(preds)


def load_clean_node(config_name=None):
    """Charge le NODE backbone final (unique, 8 features)."""
    node_path = MODEL_DIR / "node_final_8feat.pt"
    scaler_path = MODEL_DIR / "scaler_node_final_8feat.pkl"
    node_model = VehicleDynamicsNet(NODE_N_INPUT, N_STATE, hidden_size=256).to(device)
    node_model.load_state_dict(torch.load(node_path, map_location=device, weights_only=True))
    node_model.eval()
    with open(scaler_path, 'rb') as f:
        scaler_node = pickle.load(f)
    return node_model.shared, scaler_node


def predict_finetuned(model_type, config_name, horizon, X, node_backbone, scaler_node):
    """Prédictions Physics Fine-tuned (clean — NODE entraîné par config)."""
    hkey = f'h{horizon}s'
    ft_path = MODEL_DIR / f"final_{model_type}_{hkey}_{config_name}.pt"
    scaler_path = MAXHORIZON_DIR / "models" / f"scaler_{config_name}_{hkey}.pkl"
    if not ft_path.exists(): return None

    with open(scaler_path, 'rb') as f:
        scaler_8 = pickle.load(f)

    physics_proj = PhysicsProjection(node_backbone, backbone_dim=128).to(device)
    if model_type == 'LSTM':
        model = LSTM_Physics(physics_proj).to(device)
    else:
        model = PatchTST_Physics(physics_proj).to(device)
    model.load_state_dict(torch.load(ft_path, map_location=device, weights_only=True))
    model.eval()

    n = X.shape[0]
    X_8 = scaler_8.transform(X.reshape(-1, N_FEATURES)).reshape(n, SEQ_LEN, N_FEATURES).astype(np.float32)
    X_node = scaler_node.transform(X.reshape(-1, N_FEATURES)).reshape(n, SEQ_LEN, N_FEATURES).astype(np.float32)

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_8), torch.FloatTensor(X_node)), batch_size=64)
    preds = []
    with torch.no_grad():
        for x8, xn in loader:
            preds.append(model(x8.to(device), xn.to(device)).cpu().numpy())
    return np.concatenate(preds)


# =============================================================================
# Figure 1: Tableau comparatif
# =============================================================================
def figure_1_tableau(node_backbone=None, scaler_node=None):
    print("  Figure 1: Tableau comparatif...")
    all_models = ['LSTM', 'LSTM+NODE', 'PatchTST', 'PatchTST+NODE']
    data_rows, recall_values, aucpr_values = [], [], []

    for ds in ['D2', 'D3', 'D4']:
        nb, sn = load_clean_node(ds)
        for h in HORIZONS:
            X, y = get_test_data(ds, h)
            row = [ds, f'{h}s']
            row_recalls = []
            row_aucprs = []
            for model_label in all_models:
                base_type = model_label.replace('+NODE', '')
                if '+NODE' in model_label:
                    pred = predict_finetuned(base_type, ds, h, X, nb, sn)
                else:
                    pred = predict_baseline(base_type, ds, h, X)
                if pred is not None:
                    q50 = pred[:, 1]
                    q90 = pred[:, 2]
                    rmse = np.sqrt(mean_squared_error(y, q50))
                    r2 = r2_score(y, q50)
                    td = y >= 0.7; pd_ = q90 >= 0.7
                    recall = np.sum(td & pd_) / max(np.sum(td), 1) * 100
                    precision = np.sum(td & pd_) / max(np.sum(pd_), 1) * 100
                    auc_pr = float('nan')
                    tb = (y >= 0.7).astype(int)
                    if 0 < tb.sum() < len(tb):
                        auc_pr = average_precision_score(tb, q50)
                    row.append(f"RMSE={rmse:.3f} R²={r2:.2f}\nAUC-PR={auc_pr:.3f}\nPrec={precision:.0f}% Rec={recall:.0f}%")
                    row_recalls.append(recall)
                    row_aucprs.append(auc_pr)
                else:
                    row.append("-"); row_recalls.append(0); row_aucprs.append(0)
            data_rows.append(row)
            recall_values.append(row_recalls)
            aucpr_values.append(row_aucprs)

    fig, ax = plt.subplots(figsize=(20, 14))
    ax.axis('off')
    headers = ['Config', 'Horizon'] + all_models

    # Couleurs douces selon AUC-PR (style maxhorizon)
    def auc_to_color(auc):
        if np.isnan(auc) or auc == 0:
            return '#f0f0f0'
        if auc >= 0.95:
            return '#a9dfbf'  # vert foncé
        elif auc >= 0.85:
            return '#d5f5e3'  # vert clair
        elif auc >= 0.75:
            return '#fef9e7'  # jaune
        else:
            return '#fadbd8'  # rouge clair

    colors = []
    for ri, row in enumerate(data_rows):
        rc = ['white', 'white']
        for ci in range(len(all_models)):
            rc.append(auc_to_color(aucpr_values[ri][ci]))
        colors.append(rc)

    col_colors = [COLORS.get(m, '#3498db') for m in all_models]
    table = ax.table(cellText=data_rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#2c3e50'] * len(headers), cellColours=colors)
    for i in range(len(headers)):
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=10)
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.3, 3.0)
    plt.title('MAX HORIZON — Comparaison baseline vs +NODE\n'
              'Couleur selon AUC-PR : vert foncé $\\geq$ 0.95, vert clair $\\geq$ 0.85, jaune $\\geq$ 0.75, rouge < 0.75',
              fontsize=12, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / "figures" / "node_1_tableau_comparatif.png")
    plt.close()
    print("    -> node_1_tableau_comparatif.png")


# =============================================================================
# Figure 2: Matrices de confusion
# =============================================================================
def figure_2_confusion(node_backbone=None, scaler_node=None):
    print("  Figure 2: Matrices de confusion...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    bins = [0, 0.7, 0.8, 0.9, 1.5]
    labels = ['<0.7', '0.7-0.8', '0.8-0.9', '>=0.9']
    horizon = 2
    all_models = ['LSTM', 'LSTM+NODE', 'PatchTST', 'PatchTST+NODE']

    for row, ds in enumerate(['D2', 'D3', 'D4']):
        nb, sn = load_clean_node(ds)
        X, y = get_test_data(ds, horizon)
        for col, model_label in enumerate(all_models):
            ax = axes[row, col]
            base_type = model_label.replace('+NODE', '')
            if '+NODE' in model_label:
                pred = predict_finetuned(base_type, ds, horizon, X, nb, sn)
            else:
                pred = predict_baseline(base_type, ds, horizon, X)
            if pred is not None:
                pred_val = pred[:, 2]  # Q90
                true_cls = np.digitize(y, bins[1:-1])
                pred_cls = np.digitize(pred_val, bins[1:-1])
                cm = confusion_matrix(true_cls, pred_cls, labels=[0, 1, 2, 3])
                cm_pct = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
                sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                            xticklabels=labels, yticklabels=labels,
                            vmin=0, vmax=100, cbar=False, annot_kws={'size': 9})
                td = y >= 0.7; pd_ = pred_val >= 0.7
                recall = np.sum(td & pd_) / max(np.sum(td), 1) * 100
                ax.set_title(f'{model_label}\nRecall={recall:.0f}%', fontsize=10,
                             color=COLORS[model_label])
            if row == 2: ax.set_xlabel('Prédit')
            if col == 0: ax.set_ylabel('Réel')

    for i, ds in enumerate(['D2', 'D3', 'D4']):
        axes[i, 0].annotate(ds, xy=(-0.4, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', ha='center', va='center')
    fig.suptitle('Physics Fine-tune — Matrices de confusion (%) — h=2s — Q90', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "node_2_confusion_matrices.png")
    plt.close()
    print("    -> node_2_confusion_matrices.png")


# =============================================================================
# Figure 3: Régression
# =============================================================================
def figure_3_regression(node_backbone=None, scaler_node=None):
    print("  Figure 3: Régression...")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    horizon = 2
    all_models = ['LSTM', 'LSTM+NODE', 'PatchTST', 'PatchTST+NODE']

    for row, ds in enumerate(['D2', 'D3', 'D4']):
        nb, sn = load_clean_node(ds)
        X, y = get_test_data(ds, horizon)
        for col, model_label in enumerate(all_models):
            ax = axes[row, col]
            base_type = model_label.replace('+NODE', '')
            if '+NODE' in model_label:
                pred = predict_finetuned(base_type, ds, horizon, X, nb, sn)
            else:
                pred = predict_baseline(base_type, ds, horizon, X)
            if pred is not None:
                pred_val = pred[:, 2]  # Q90
                n = min(2000, len(y))
                idx = np.random.choice(len(y), n, replace=False)
                ax.scatter(y[idx], pred_val[idx], alpha=0.3, s=8, c=COLORS[model_label], edgecolors='none')
                ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
                ax.axvline(0.7, color='red', ls=':', alpha=0.5)
                ax.axhline(0.7, color='red', ls=':', alpha=0.5)
                rmse = np.sqrt(mean_squared_error(y, pred_val))
                r2 = r2_score(y, pred_val)
                ax.text(0.05, 0.95, f'RMSE={rmse:.3f}\nR²={r2:.2f}',
                        transform=ax.transAxes, va='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.set_title(model_label, fontsize=11, color=COLORS[model_label], fontweight='bold')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            if row == 2: ax.set_xlabel('LTR max réel')
            if col == 0: ax.set_ylabel('LTR max prédit Q90')

    for i, ds in enumerate(['D2', 'D3', 'D4']):
        axes[i, 0].annotate(ds, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=14, fontweight='bold', ha='center', va='center')
    fig.suptitle('Physics Fine-tune — Régression Q90 vs réel — h=2s', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "node_3_regression.png")
    plt.close()
    print("    -> node_3_regression.png")


# =============================================================================
# Figure 4: Recall comparison
# =============================================================================
def figure_4_recall(node_backbone=None, scaler_node=None):
    print("  Figure 4: Comparaison recall...")
    fig, ax = plt.subplots(figsize=(12, 6))
    all_models = ['LSTM', 'LSTM+NODE', 'PatchTST', 'PatchTST+NODE']
    configs = ['D2', 'D3', 'D4']
    horizon = 2
    x = np.arange(len(configs))
    width = 0.2

    # Pre-load clean backbones
    clean_nodes = {ds: load_clean_node(ds) for ds in configs}

    for i, model_label in enumerate(all_models):
        recalls = []
        for ds in configs:
            nb, sn = clean_nodes[ds]
            X, y = get_test_data(ds, horizon)
            base_type = model_label.replace('+NODE', '')
            if '+NODE' in model_label:
                pred = predict_finetuned(base_type, ds, horizon, X, nb, sn)
            else:
                pred = predict_baseline(base_type, ds, horizon, X)
            if pred is not None:
                pred_val = pred[:, 2]
                td = y >= 0.7; pd_ = pred_val >= 0.7
                recall = np.sum(td & pd_) / max(np.sum(td), 1) * 100
            else:
                recall = 0
            recalls.append(recall)
        bars = ax.bar(x + i * width, recalls, width, label=model_label,
                      color=COLORS[model_label], alpha=0.85)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Recall Q90 (%)')
    ax.set_title('Physics Fine-tune — Recall Q90 par modèle — h=2s', fontweight='bold')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 110)
    ax.axhline(80, color='green', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "node_4_recall_comparison.png")
    plt.close()
    print("    -> node_4_recall_comparison.png")


# =============================================================================
# Figure 5: Prédictions sur trajectoires
# =============================================================================
def figure_5_trajectoires():
    """Prédictions sur 3 trajectoires × 2 modèles (baseline vs +NODE)."""
    print("  Figure 5: Prédictions sur trajectoires...")

    TRAJECTORIES = {
        'Safe — waypoint S-curve': 'waypoint_scurve_453_v11_a19_mu70.csv',
        'Safe — smooth path': 'smooth_v8_ym26_dm76_a6_n7_mu90_w_s30010.csv',
        'Modéré — circle v14': 'circle_v14_R16_d-1_mu70.csv',
        'Modéré — lemniscate': 'lemniscate_v8_a5_n10_mu75.csv',
        'Critique — smooth v23': 'smooth_v23_ym52_dm42_a3_n12_mu75_nw_s30516.csv',
        'Critique — circle v20': 'circle_v20_R44_d1_mu90.csv',
    }

    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    horizon = 2
    hkey = 'h2s'
    horizon_steps = int(horizon / DT)
    config = 'D4'
    all_models = ['LSTM', 'LSTM+NODE', 'PatchTST', 'PatchTST+NODE']

    nb, sn = load_clean_node(config)

    for row, (traj_name, filename) in enumerate(TRAJECTORIES.items()):
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue
        df = load_scenario(filepath)
        if df is None:
            continue
        data = df[FEATURES_8].values.astype(np.float32)
        ltr = np.abs(df['LTRmax'].values)

        # Load scaler
        scaler_path = MAXHORIZON_DIR / "models" / f"scaler_{config}_{hkey}.pkl"
        with open(scaler_path, 'rb') as f:
            scaler_8 = pickle.load(f)

        for col, model_label in enumerate(all_models):
            ax = axes[row, col]
            base_type = model_label.replace('+NODE', '')

            # Load model ONCE
            if '+NODE' in model_label:
                ft_path = MODEL_DIR / f"final_{base_type}_{hkey}_{config}.pt"
                if not ft_path.exists(): continue
                physics_proj = PhysicsProjection(nb, backbone_dim=128).to(device)
                if base_type == 'LSTM':
                    model = LSTM_Physics(physics_proj).to(device)
                else:
                    model = PatchTST_Physics(physics_proj).to(device)
                model.load_state_dict(torch.load(ft_path, map_location=device, weights_only=True))
            else:
                ta_path = MAXHORIZON_DIR / "models" / f"{base_type}_{config}_{hkey}.pt"
                if not ta_path.exists(): continue
                if base_type == 'LSTM':
                    model = LSTM_Baseline(N_FEATURES).to(device)
                else:
                    model = PatchTST_Baseline(SEQ_LEN, N_FEATURES).to(device)
                model.load_state_dict(torch.load(ta_path, map_location=device, weights_only=True))
            model.eval()

            times, preds_list, targets_list, ltr_now_list = [], [], [], []

            for i in range(SEQ_LEN, len(data) - horizon_steps, 5):
                t = (i) * DT
                target = np.max(ltr[i:i + horizon_steps])
                ltr_current = ltr[i]

                x8 = scaler_8.transform(data[i - SEQ_LEN:i]).astype(np.float32)
                x8_t = torch.FloatTensor(x8).unsqueeze(0).to(device)

                with torch.no_grad():
                    if '+NODE' in model_label:
                        xn = sn.transform(data[i - SEQ_LEN:i]).astype(np.float32)
                        xn_t = torch.FloatTensor(xn).unsqueeze(0).to(device)
                        pred = model(x8_t, xn_t).cpu().numpy()[0]
                    else:
                        pred = model(x8_t).cpu().numpy()[0]

                times.append(t)
                preds_list.append(pred)
                targets_list.append(target)
                ltr_now_list.append(ltr_current)

            if not times:
                continue

            times = np.array(times)
            preds_arr = np.array(preds_list)
            targets_arr = np.array(targets_list)
            ltr_now_arr = np.array(ltr_now_list)

            # Plot
            ax.plot(times, ltr_now_arr, 'gray', lw=1.5, alpha=0.6, label='LTR instantané')
            ax.plot(times, targets_arr, 'k-', lw=2, label=f'Max réel (h={horizon}s)')
            ax.fill_between(times, preds_arr[:, 0], preds_arr[:, 2], alpha=0.2,
                            color=COLORS[model_label])
            ax.plot(times, preds_arr[:, 2], '-', lw=2, color=COLORS[model_label],
                    label='Prédiction Q90')
            ax.axhline(0.7, color='red', ls='--', lw=1.5, alpha=0.6)
            ax.fill_between(times, 0.7, 1.0, alpha=0.05, color='red')

            ax.set_ylim(0, 1.05)
            ax.set_xlim(times[0], times[-1])
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(model_label, fontsize=12, fontweight='bold',
                             color=COLORS[model_label])
            if row == 2:
                ax.set_xlabel('Temps (s)')
            if col == 0:
                ax.set_ylabel('LTR')
                ax.annotate(traj_name, xy=(-0.35, 0.5), xycoords='axes fraction',
                            fontsize=10, fontweight='bold', ha='center', va='center', rotation=90)

            if row == 0 and col == 3:
                ax.legend(loc='upper left', fontsize=7)

            # Recall
            danger_mask = targets_arr >= 0.7
            if danger_mask.sum() > 0:
                recall = (preds_arr[danger_mask, 2] >= 0.7).mean() * 100
            else:
                recall = np.nan
            recall_str = f'{recall:.0f}%' if not np.isnan(recall) else '-'
            ax.text(0.98, 0.02, f'Recall={recall_str}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle(f'MAX HORIZON — Anticipation du danger (h={horizon}s, {config})\n'
                 f'Gris=LTR actuel, Noir=Max futur réel, Couleur=Prédiction Q90',
                 fontweight='bold', fontsize=12, y=0.99)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "node_5_predictions_trajectoires.png")
    plt.close()
    print("    -> node_5_predictions_trajectoires.png")


# =============================================================================
# Figure 6: Profils LTR variés — Max Horizon vs Exact Horizon (LSTM+NODE)
# =============================================================================
def figure_6_profils_varies():
    """7 profils LTR variés × 2 colonnes (max horizon, exact horizon) — LSTM+NODE Q90."""
    print("  Figure 6: Profils LTR variés (max vs exact)...")

    HEXACT_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/horizon_exact")

    TRAJECTORIES = {
        'Croissant\n(single turn)': 'single_v14_R20_d-1_mu80.csv',
        'Décroissant\n(smooth)': 'smooth_v19_ym56_dm32_a4_n10_mu75_nw_s31015.csv',
        'Constant haut\n(circle)': 'circle_v22_R52_d-1_mu85.csv',
        'Sinusoïdal\n(slalom)': 'slalom_v10_L150_A20_n3_mu80.csv',
        'Spike transitoire\n(évitement)': 'waypoint_evitement_252_v15_a20_mu75.csv',
        'Rampe + plateau\n(single turn)': 'single_v16_R40_d1_mu65.csv',
        'Oscillant haut\n(lemniscate)': 'lemniscate_v8_a3_n10_mu80.csv',
    }

    fig, axes = plt.subplots(7, 2, figsize=(16, 28))
    horizon = 2
    hkey = 'h2s'
    horizon_steps = int(horizon / DT)
    config = 'D4'

    nb, sn = load_clean_node(config)

    # Load max horizon model (LSTM+NODE)
    ft_path_max = MODEL_DIR / f"final_LSTM_{hkey}_{config}.pt"
    scaler_path_max = MAXHORIZON_DIR / "models" / f"scaler_{config}_{hkey}.pkl"
    with open(scaler_path_max, 'rb') as f:
        scaler_max = pickle.load(f)
    physics_proj_max = PhysicsProjection(nb, backbone_dim=128).to(device)
    model_max = LSTM_Physics(physics_proj_max).to(device)
    model_max.load_state_dict(torch.load(ft_path_max, map_location=device, weights_only=True))
    model_max.eval()

    # Load exact horizon model (LSTM+NODE)
    ft_path_exact = MODEL_DIR / f"exact_LSTM_{hkey}_{config}.pt"
    scaler_path_exact = HEXACT_DIR / "models" / f"scaler_{config}_{hkey}.pkl"
    with open(scaler_path_exact, 'rb') as f:
        scaler_exact = pickle.load(f)
    physics_proj_exact = PhysicsProjection(nb, backbone_dim=128).to(device)
    model_exact = LSTM_Physics(physics_proj_exact).to(device)
    model_exact.load_state_dict(torch.load(ft_path_exact, map_location=device, weights_only=True))
    model_exact.eval()

    col_titles = ['LSTM+NODE — Max Horizon (h=2s)', 'LSTM+NODE — Exact Horizon (h=2s)']
    col_colors = ['#1a5276', '#8e44ad']

    for row, (traj_name, filename) in enumerate(TRAJECTORIES.items()):
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"    SKIP {filename}")
            continue
        df = load_scenario(filepath)
        if df is None:
            continue
        data = df[FEATURES_8].values.astype(np.float32)
        ltr = np.abs(df['LTRmax'].values)

        for col, (model, scaler, target_mode) in enumerate([
            (model_max, scaler_max, 'max'),
            (model_exact, scaler_exact, 'exact'),
        ]):
            ax = axes[row, col]
            times, preds_list, targets_list, ltr_now_list = [], [], [], []

            for i in range(SEQ_LEN, len(data) - horizon_steps, 5):
                t = i * DT
                if target_mode == 'max':
                    target = np.max(ltr[i:i + horizon_steps])
                else:
                    target = ltr[i + horizon_steps]
                ltr_current = ltr[i]

                x8 = scaler.transform(data[i - SEQ_LEN:i]).astype(np.float32)
                xn = sn.transform(data[i - SEQ_LEN:i]).astype(np.float32)
                with torch.no_grad():
                    pred = model(
                        torch.FloatTensor(x8).unsqueeze(0).to(device),
                        torch.FloatTensor(xn).unsqueeze(0).to(device)
                    ).cpu().numpy()[0]

                times.append(t)
                preds_list.append(pred)
                targets_list.append(target)
                ltr_now_list.append(ltr_current)

            if not times:
                continue

            times = np.array(times)
            preds_arr = np.array(preds_list)
            targets_arr = np.array(targets_list)
            ltr_now_arr = np.array(ltr_now_list)

            c = col_colors[col]
            target_label = 'Max réel (h=2s)' if target_mode == 'max' else 'LTR réel (t+2s)'

            ax.plot(times, ltr_now_arr, 'gray', lw=1, alpha=0.5, label='LTR instantané')
            ax.plot(times, targets_arr, 'k-', lw=2, label=target_label)
            ax.fill_between(times, preds_arr[:, 0], preds_arr[:, 2], alpha=0.15, color=c)
            ax.plot(times, preds_arr[:, 2], '-', lw=2, color=c, label='Prédiction Q90')
            ax.axhline(0.7, color='red', ls='--', lw=1, alpha=0.5)
            ax.fill_between(times, 0.7, 1.0, alpha=0.03, color='red')

            ax.set_ylim(0, 1.05)
            ax.set_xlim(times[0], times[-1])
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight='bold', color=c)
            if row == 6:
                ax.set_xlabel('Temps (s)')
            if col == 0:
                ax.set_ylabel('LTR')
                ax.annotate(traj_name, xy=(-0.22, 0.5), xycoords='axes fraction',
                            fontsize=9, fontweight='bold', ha='center', va='center', rotation=90)

            if row == 0 and col == 1:
                ax.legend(loc='upper right', fontsize=7)

            # Recall + RMSE
            danger_mask = targets_arr >= 0.7
            if danger_mask.sum() > 0:
                recall = (preds_arr[danger_mask, 2] >= 0.7).mean() * 100
            else:
                recall = np.nan
            rmse = np.sqrt(np.mean((targets_arr - preds_arr[:, 1])**2))
            recall_str = f'{recall:.0f}%' if not np.isnan(recall) else '-'
            ax.text(0.98, 0.02, f'Recall={recall_str}  RMSE={rmse:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle('Profils LTR variés — LSTM+NODE Q90 — Max Horizon vs Exact Horizon (h=2s, D4)\n'
                 'Gris=LTR actuel, Noir=Cible réelle, Couleur=Prédiction Q90',
                 fontweight='bold', fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "node_6_profils_varies.png")
    plt.close()
    print("    -> node_6_profils_varies.png")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("NODE — Génération des figures (clean, pas de leakage)")
    print("=" * 60)

    figure_1_tableau()
    figure_2_confusion()
    figure_3_regression()
    figure_4_recall()
    figure_5_trajectoires()
    figure_6_profils_varies()

    print(f"\n6 figures générées dans {OUTPUT_DIR}")
    print("TERMINÉ!")


if __name__ == '__main__':
    main()
