#!/usr/bin/env python3
"""
Generation des matrices de confusion pour les classes de risque LTR.

Classes de risque:
- Classe 1: LTR < 0.7
- Classe 2: 0.7 <= LTR < 0.8
- Classe 3: 0.8 <= LTR < 0.9
- Classe 4: 0.9 <= LTR < 1.0

Matrices generees pour h=4s et h=8s comme demande dans le protocole d'evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

# ============== Configuration ==============

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path("/Users/zak/Documents/vdsim/zak/ltr_prediction/data_new")
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DT = 0.01
SEQUENCE_LENGTH = 150
FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = len(FEATURES)

# Horizons pour les matrices de confusion (en secondes)
HORIZONS_CONFUSION = [1, 2, 4, 6, 8]
HORIZONS_STEPS = {h: int(h / DT) for h in HORIZONS_CONFUSION}

# Classes de risque
CLASS_NAMES = ['Classe 1\n(<0.7)', 'Classe 2\n(0.7-0.8)', 'Classe 3\n(0.8-0.9)', 'Classe 4\n(≥0.9)']

# Configurations des datasets
DATASET_CONFIGS = {
    'D1': {'train_threshold': 0.7, 'test_above': False},
    'D4': {'train_threshold': 0.9, 'test_above': True},
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ============== Architectures des modeles ==============

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
        return self.network(x.view(x.size(0), -1))


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


def ltr_to_class(ltr_values):
    """Convertit les valeurs LTR en classes de risque (0-3)."""
    classes = np.zeros_like(ltr_values, dtype=int)
    classes[ltr_values >= 0.7] = 1
    classes[ltr_values >= 0.8] = 2
    classes[ltr_values >= 0.9] = 3
    return classes


def load_scenario(filepath):
    """Charge un fichier CSV de scenario et calcule les features derivees."""
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def create_sequences_exact_horizon(df, horizon_steps):
    """Cree des sequences pour un horizon exact."""
    data = df[FEATURES].values
    ltr = np.abs(df['LTRmax'].values)

    X, y = [], []
    max_start = len(data) - SEQUENCE_LENGTH - horizon_steps

    for i in range(0, max_start, 10):  # Stride de 10
        X.append(data[i:i + SEQUENCE_LENGTH])
        y.append(ltr[i + SEQUENCE_LENGTH + horizon_steps - 1])

    return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])


def prepare_test_data(dataset_name, horizon_steps):
    """Prepare les donnees de test pour un dataset et un horizon donnes."""
    config = DATASET_CONFIGS[dataset_name]

    all_files = sorted(DATA_DIR.glob("*.csv"))

    # Classifier les scenarios par LTR max
    scenarios_by_ltr = {'low': [], 'high': []}
    for f in all_files:
        df = pd.read_csv(f)
        if 'LTRmax' not in df.columns:
            continue
        ltr_max = np.max(np.abs(df['LTRmax'].values))
        if ltr_max <= config['train_threshold']:
            scenarios_by_ltr['low'].append(f)
        else:
            scenarios_by_ltr['high'].append(f)

    print(f"  Scenarios LTR <= {config['train_threshold']}: {len(scenarios_by_ltr['low'])}")
    print(f"  Scenarios LTR > {config['train_threshold']}: {len(scenarios_by_ltr['high'])}")

    X_test_list, y_test_list = [], []

    if config['test_above']:
        # Test sur high (OOD)
        for f in scenarios_by_ltr['high']:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)
    else:
        # Test sur 20% des scenarios low (D1)
        np.random.seed(42)
        low_files = scenarios_by_ltr['low'].copy()
        np.random.shuffle(low_files)
        split_idx = int(0.8 * len(low_files))

        for f in low_files[split_idx:]:
            df = load_scenario(f)
            X, y = create_sequences_exact_horizon(df, horizon_steps)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)

    if not X_test_list:
        return None, None

    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)

    return X_test, y_test


def plot_confusion_matrix(cm, title, output_path, accuracy):
    """Trace une matrice de confusion."""
    plt.figure(figsize=(8, 6))

    # Normaliser par ligne
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100

    # Annotations
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)'

    sns.heatmap(cm_normalized, annot=annot, fmt='', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=100, cbar_kws={'label': 'Pourcentage (%)'})

    plt.xlabel('Prediction', fontsize=12)
    plt.ylabel('Verite', fontsize=12)
    plt.title(f'{title}\nAccuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("GENERATION DES MATRICES DE CONFUSION")
    print("=" * 60)

    models_names = ['MLP', 'LSTM', 'PatchTST']
    results_confusion = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\n{'='*40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*40}")

        results_confusion[dataset_name] = {}

        for horizon in HORIZONS_CONFUSION:
            horizon_key = f'h{horizon}s'
            horizon_steps = HORIZONS_STEPS[horizon]
            print(f"\n--- Horizon {horizon}s ---")

            results_confusion[dataset_name][horizon_key] = {}

            # Charger les donnees de test
            X_test, y_test = prepare_test_data(dataset_name, horizon_steps)
            if X_test is None or len(X_test) == 0:
                print(f"  Pas de donnees de test!")
                continue

            print(f"  Echantillons de test: {len(X_test)}")

            # Distribution des classes reelles
            y_classes = ltr_to_class(y_test)
            print(f"  Distribution des classes reelles:")
            for i in range(4):
                count = np.sum(y_classes == i)
                print(f"    Classe {i+1}: {count} ({100*count/len(y_classes):.1f}%)")

            # Charger le scaler (un scaler par dataset+horizon)
            scaler_path = MODEL_DIR / f"scaler_{dataset_name}_{horizon_key}.pkl"

            if not scaler_path.exists():
                print(f"  Scaler non trouve: {scaler_path}")
                continue

            with open(scaler_path, 'rb') as f:
                scaler_X = pickle.load(f)

            # Normaliser X
            X_test_scaled = scaler_X.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)
            X_tensor = torch.FloatTensor(X_test_scaled).to(device)

            for model_name in models_names:
                model_path = MODEL_DIR / f"{model_name}_{dataset_name}_{horizon_key}.pt"

                if not model_path.exists():
                    print(f"  {model_name}: Modele non trouve")
                    continue

                # Creer et charger le modele
                if model_name == 'MLP':
                    model = MLP(SEQUENCE_LENGTH * N_FEATURES).to(device)
                elif model_name == 'LSTM':
                    model = LSTM(N_FEATURES).to(device)
                else:
                    model = PatchTST(SEQUENCE_LENGTH, N_FEATURES).to(device)

                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
                model.eval()

                # Predictions par batch
                batch_size = 1024
                all_preds = []
                with torch.no_grad():
                    for i in range(0, len(X_tensor), batch_size):
                        batch = X_tensor[i:i+batch_size]
                        preds = model(batch).cpu().numpy()
                        all_preds.append(preds)

                preds = np.concatenate(all_preds)
                y_true_classes = ltr_to_class(y_test)

                # Evaluer Q50 et Q90
                for q_name, q_idx in [('Q50', 1), ('Q90', 2)]:
                    y_pred = np.clip(preds[:, q_idx], 0, 1)
                    y_pred_classes = ltr_to_class(y_pred)

                    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=[0, 1, 2, 3])
                    accuracy = accuracy_score(y_true_classes, y_pred_classes) * 100

                    key = f"{model_name}_{q_name}"
                    results_confusion[dataset_name][horizon_key][key] = {
                        'confusion_matrix': cm.tolist(),
                        'accuracy': accuracy
                    }

                print(f"  {model_name}: Q50={results_confusion[dataset_name][horizon_key][f'{model_name}_Q50']['accuracy']:.1f}% | Q90={results_confusion[dataset_name][horizon_key][f'{model_name}_Q90']['accuracy']:.1f}%")

    # Sauvegarder les resultats
    with open(OUTPUT_DIR / "confusion_results.json", 'w') as f:
        json.dump(results_confusion, f, indent=2)

    # ============== Figure recapitulative ==============
    print("\n" + "="*60)
    print("Generation des figures recapitulatives...")

    # Generer une figure pour chaque quantile (Q50 et Q90)
    for q_name in ['Q50', 'Q90']:
        n_horizons = len(HORIZONS_CONFUSION)
        fig, axes = plt.subplots(6, n_horizons, figsize=(4*n_horizons, 24))
        fig.suptitle(f'Matrices de Confusion ({q_name}) - Classes de Risque LTR\n(Classe 1: <0.7, Classe 2: 0.7-0.8, Classe 3: 0.8-0.9, Classe 4: ≥0.9)',
                     fontsize=16, fontweight='bold', y=0.995)

        row_idx = 0
        for dataset_name in ['D1', 'D4']:
            for model_name in models_names:
                for col_idx, horizon in enumerate(HORIZONS_CONFUSION):
                    horizon_key = f'h{horizon}s'
                    ax = axes[row_idx, col_idx]
                    key = f"{model_name}_{q_name}"

                    if dataset_name in results_confusion and horizon_key in results_confusion[dataset_name]:
                        if key in results_confusion[dataset_name][horizon_key]:
                            data = results_confusion[dataset_name][horizon_key][key]
                            cm = np.array(data['confusion_matrix'])
                            accuracy = data['accuracy']

                            # Normaliser par ligne
                            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100

                            sns.heatmap(cm_norm, ax=ax, annot=True, fmt='.0f', cmap='Blues',
                                       xticklabels=['1', '2', '3', '4'],
                                       yticklabels=['1', '2', '3', '4'],
                                       vmin=0, vmax=100, cbar=False,
                                       annot_kws={'size': 9})
                            ax.set_title(f'{model_name} - {dataset_name}\nh={horizon}s | Acc: {accuracy:.1f}%',
                                        fontsize=10, fontweight='bold')
                            ax.set_xlabel('Pred' if row_idx == 5 else '')
                            ax.set_ylabel('Vrai' if col_idx == 0 else '')
                        else:
                            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                            ax.set_title(f'{model_name} - {dataset_name}\nh={horizon}s', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
                        ax.set_title(f'{model_name} - {dataset_name}\nh={horizon}s', fontsize=10)

                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(OUTPUT_DIR / f"confusion_all_{q_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - confusion_all_{q_name}.png")

    print(f"\nFigures sauvegardees dans {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
