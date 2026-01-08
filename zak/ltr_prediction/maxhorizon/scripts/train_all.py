#!/usr/bin/env python3
"""
Entrainement des modeles MLP, LSTM et PatchTST pour la prediction du LTR.

Tache: Prediction du maximum de |LTR| sur un horizon de 2 secondes.
Cette approche est conservative car elle predit le pire cas futur.

Architecture des modeles:
- MLP: Reseau fully-connected avec BatchNorm et Dropout
- LSTM: Reseau recurrent simple (1 couche, 64 unites)
- PatchTST: Transformer avec tokenisation par patches temporels

Configuration:
- Entree: 8 features (vx, vy, psi, psi_dot, phi, theta, delta_f, delta_f_dot)
- Sequence: 1.5s (150 pas de temps a 100Hz)
- Horizon de prediction: 2s (200 pas de temps)
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
DATA_DIR = Path("/Users/zak/Documents/vdsim/zak/ltr_prediction/data_new")  # 484 scenarios
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============== Hyperparametres ==============

DT = 0.01                    # Pas de temps (10ms = 100Hz)
SEQUENCE_LENGTH = 150        # Lookback: 1.5s
PREDICTION_HORIZON = 200     # Horizon de prediction: 2s

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
        'desc': 'Distribution normale (LTR <= 0.7)'
    },
    'D4': {
        'train_threshold': 0.9,
        'test_above': True,
        'desc': 'Hors-distribution critique (test > 0.9)'
    },
}

# Selection automatique du device (MPS pour Mac M1/M2, sinon CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Features ({N_FEATURES}): {FEATURES}")
print(f"Sequence: {SEQUENCE_LENGTH} pas ({SEQUENCE_LENGTH*DT:.1f}s)")
print(f"Horizon: {PREDICTION_HORIZON} pas ({PREDICTION_HORIZON*DT:.1f}s)")


# ============== Architectures des modeles ==============

class MLP(nn.Module):
    """
    Perceptron multicouche pour la prediction du LTR.

    Architecture: 1200 -> 512 -> 256 -> 128 -> 3
    Avec BatchNorm et Dropout pour la regularisation.
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
        # Applatissement de la sequence: (batch, 150, 8) -> (batch, 1200)
        return self.network(x.view(x.size(0), -1))


class LSTM(nn.Module):
    """
    LSTM simple pour la prediction du LTR.

    Architecture: LSTM(8 -> 64) puis FC(64 -> 32 -> 3)
    Utilise uniquement le dernier etat cache pour la prediction.
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
        # Prediction basee sur le dernier pas de temps
        return self.fc(out[:, -1, :])


class PatchTST(nn.Module):
    """
    Transformer avec tokenisation par patches pour series temporelles.

    Architecture:
    - Division en patches de 15 pas (0.15s chacun)
    - Embedding lineaire + token CLS
    - Transformer Encoder (2 couches, 4 tetes d'attention)
    - Tete de prediction sur le token CLS

    Reference: Nie et al. (2023) "A Time Series is Worth 64 Words"
    """

    def __init__(self, n_features, seq_len, patch_size=15, d_model=64, output_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (seq_len - patch_size) // patch_size + 1  # = 10 patches

        # Projection des patches vers l'espace d'embedding
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)

        # Token CLS pour l'agregation globale
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encodage positionnel appris
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)

        # Transformer encoder
        encoder = nn.TransformerEncoderLayer(d_model, 4, d_model*4, 0.2, 'gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, 2)

        # Tete de prediction
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(32, output_size)
        )

    def forward(self, x):
        B = x.size(0)

        # Creation des patches: (batch, 150, 8) -> (batch, 10, 120)
        patches = [x[:, i:i+self.patch_size, :].reshape(B, -1)
                   for i in range(0, x.size(1) - self.patch_size + 1, self.patch_size)]
        x = torch.stack(patches, dim=1)

        # Projection et ajout du token CLS
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # Encodage positionnel + Transformer
        x = x + self.pos_enc[:, :x.size(1)]
        x = self.transformer(x)

        # Prediction sur le token CLS
        return self.head(x[:, 0])


# ============== Fonction de perte ==============

def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    """
    Perte quantile pour regression multi-quantile.

    Permet d'estimer Q10, Q50 (mediane) et Q90 simultanement.
    Q90 est particulierement utile pour la detection conservative
    des situations critiques (sous-estimation dangereuse evitee).

    Args:
        pred: Predictions (batch, 3) pour Q10, Q50, Q90
        target: Valeurs cibles (batch,)
        quantiles: Liste des quantiles a estimer
    """
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        # Penalisation asymetrique selon le quantile
        losses.append(torch.mean(torch.max(q * err, (q - 1) * err)))
    return sum(losses)


# ============== Chargement des donnees ==============

def load_scenario(filepath):
    """
    Charge un scenario CSV et calcule les features derivees.

    Ajoute:
    - delta_f_dot: derivee de l'angle de braquage
    - psi_dot: derivee du lacet (si absent)
    """
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    return df


def create_sequences(df):
    """
    Cree des sequences pour l'entrainement.

    Pour chaque position i:
    - X: features de i a i+150 (lookback 1.5s)
    - y: max(|LTR|) de i+150 a i+350 (horizon 2s)

    Le stride=1 maximise les donnees mais cree de la correlation.
    """
    features = df[FEATURES].values
    ltr = np.abs(df['LTRmax'].values)
    X, y = [], []

    for i in range(len(df) - SEQUENCE_LENGTH - PREDICTION_HORIZON):
        X.append(features[i:i+SEQUENCE_LENGTH])
        # Target = maximum du LTR sur l'horizon de prediction
        y.append(np.max(ltr[i+SEQUENCE_LENGTH:i+SEQUENCE_LENGTH+PREDICTION_HORIZON]))

    return np.array(X), np.array(y)


def load_data(dataset_name):
    """
    Charge et prepare les donnees pour un jeu de donnees specifique.

    Strategie de split:
    - D1: Split aleatoire 80/20 sur scenarios avec LTR <= 0.7
    - D4: Train sur LTR <= 0.9, Test sur LTR > 0.9 (OOD)

    Normalisation: StandardScaler par feature sur l'ensemble d'entrainement.
    """
    config = DATASET_CONFIGS[dataset_name]
    files = sorted(DATA_DIR.glob("*.csv"))

    # Chargement de tous les scenarios
    scenarios = []
    for f in files:
        df = load_scenario(f)
        scenarios.append({'df': df, 'max_ltr': np.max(np.abs(df['LTRmax'].values))})

    # Separation train/test selon la configuration
    if config['test_above']:
        # OOD: test sur scenarios critiques
        train_sc = [s for s in scenarios if s['max_ltr'] <= config['train_threshold']]
        test_sc = [s for s in scenarios if s['max_ltr'] > config['train_threshold']]
    else:
        # Split aleatoire sur meme distribution
        eligible = [s for s in scenarios if s['max_ltr'] <= config['train_threshold']]
        np.random.seed(42)
        np.random.shuffle(eligible)
        split = int(0.8 * len(eligible))
        train_sc, test_sc = eligible[:split], eligible[split:]

    # Creation des sequences
    X_train = np.concatenate([create_sequences(s['df'])[0] for s in train_sc])
    y_train = np.concatenate([create_sequences(s['df'])[1] for s in train_sc])
    X_test = np.concatenate([create_sequences(s['df'])[0] for s in test_sc])
    y_test = np.concatenate([create_sequences(s['df'])[1] for s in test_sc])

    # Normalisation par feature
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape)

    return X_train, y_train, X_test, y_test, scaler, len(train_sc), len(test_sc)


# ============== Entrainement ==============

def train_model(model, train_loader, val_loader):
    """
    Boucle d'entrainement avec early stopping.

    - Optimiseur: Adam avec lr=0.001
    - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
    - Gradient clipping: max_norm=1.0
    - Early stopping: patience=15 epochs
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(EPOCHS):
        # Phase d'entrainement
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = quantile_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Phase de validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(quantile_loss(model(X.to(device)), y.to(device)).item()
                          for X, y in val_loader) / len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss, best_state, no_improve = val_loss, model.state_dict().copy(), 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: val_loss={val_loss:.4f}", flush=True)

        if no_improve >= PATIENCE:
            print(f"      Arret premature a l'epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test):
    """
    Evaluation du modele sur le jeu de test.

    Metriques:
    - R2: Coefficient de determination (sur Q50)
    - RMSE: Erreur quadratique moyenne (sur Q50)
    - Critical: % de detection des situations critiques
      (prediction Q90 >= 0.7 quand LTR reel >= 0.9)
    """
    model.eval()
    with torch.no_grad():
        preds = np.concatenate([
            model(torch.FloatTensor(X_test[i:i+512]).to(device)).cpu().numpy()
            for i in range(0, len(X_test), 512)
        ])

    # Predictions Q50 (mediane) et Q90 (conservative)
    pred_med = np.clip(preds[:, 1], 0, 1.5)   # Q50
    pred_high = np.clip(preds[:, 2], 0, 1.5)  # Q90

    r2 = r2_score(y_test, pred_med)
    rmse = np.sqrt(mean_squared_error(y_test, pred_med))

    # Detection des situations critiques
    # On considere qu'une situation est bien detectee si Q90 >= 0.7
    # quand le LTR reel est critique (>= 0.9)
    critical_mask = y_test >= 0.9
    if critical_mask.sum() > 0:
        critical_det = 100 * np.mean(pred_high[critical_mask] >= 0.7)
    else:
        critical_det = 0

    return {
        'R2': r2,
        'RMSE': rmse,
        'Critical': critical_det,
        'pred': pred_med,
        'pred_high': pred_high
    }


# ============== Programme principal ==============

def main():
    """Point d'entree principal pour l'entrainement."""
    print("\n" + "="*70)
    print("ENTRAINEMENT MLP, LSTM, PatchTST")
    print("Tache: max(|LTR|) sur horizon de 2s")
    print("="*70)

    all_results = {}

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Jeu de donnees: {ds} - {DATASET_CONFIGS[ds]['desc']}")
        print(f"{'='*60}")

        # Chargement des donnees
        X_train, y_train, X_test, y_test, scaler, n_train, n_test = load_data(ds)
        print(f"  Train: {n_train} scenarios, {len(X_train)} echantillons")
        print(f"  Test: {n_test} scenarios, {len(X_test)} echantillons")
        print(f"  Plage LTR test: [{y_test.min():.2f}, {y_test.max():.2f}]")
        print(f"  Echantillons critiques (>0.9): {100*np.mean(y_test >= 0.9):.1f}%")

        # Sauvegarde du scaler pour inference future
        with open(MODEL_DIR / f"scaler_{ds}.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
            batch_size=BATCH_SIZE
        )

        results = {}

        # Entrainement de chaque modele
        for name in ['MLP', 'LSTM', 'PatchTST']:
            print(f"\n    {name}...", flush=True)

            # Instanciation du modele
            if name == 'MLP':
                model = MLP(SEQUENCE_LENGTH * N_FEATURES)
            elif name == 'LSTM':
                model = LSTM(N_FEATURES)
            else:
                model = PatchTST(N_FEATURES, SEQUENCE_LENGTH)

            # Entrainement
            model = train_model(model, train_loader, val_loader)

            # Evaluation
            metrics = evaluate(model, X_test, y_test)

            # Sauvegarde du modele
            torch.save(model.state_dict(), MODEL_DIR / f"{name}_{ds}.pt")
            print(f"      R²={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, Critical={metrics['Critical']:.1f}%")

            # Stockage des resultats
            results[name] = {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in metrics.items() if k in ['R2', 'RMSE', 'Critical']
            }
            results[name]['y_test'] = y_test
            results[name]['pred'] = metrics['pred']

        all_results[ds] = results

    # Sauvegarde des resultats en JSON
    save_results = {
        ds: {m: {k: v for k, v in r.items() if k in ['R2', 'RMSE', 'Critical']}
             for m, r in res.items()}
        for ds, res in all_results.items()
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(save_results, f, indent=2)

    # Generation du graphique de comparaison
    print("\n" + "="*60)
    print("Generation des graphiques...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparaison MLP vs LSTM vs PatchTST\n'
                 f'Tache: max(|LTR|) sur horizon 2s | Entree: {N_FEATURES} features, {SEQUENCE_LENGTH*DT:.1f}s',
                 fontsize=14)

    for row, ds in enumerate(DATASETS):
        for col, name in enumerate(['MLP', 'LSTM', 'PatchTST']):
            ax = axes[row, col]
            r = all_results[ds][name]

            # Sous-echantillonnage pour visualisation (1 point sur 10)
            step = 10
            y_sub = r['y_test'][::step]
            pred_sub = r['pred'][::step]

            ax.scatter(y_sub, pred_sub, alpha=0.3, s=2, c='#1f77b4')
            ax.plot([0, 1], [0, 1], 'r--', lw=1.5)
            ax.axvline(0.9, color='red', ls=':', alpha=0.5)
            ax.axhline(0.9, color='red', ls=':', alpha=0.5)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            ax.set_title(f"{ds}: {name}\nR²={r['R2']:.3f}, RMSE={r['RMSE']:.3f}, Crit={r['Critical']:.0f}%")
            if col == 0:
                ax.set_ylabel('LTR Predit')
            if row == 1:
                ax.set_xlabel('LTR Reel')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparaison_modeles.png", dpi=150)
    print(f"  Sauvegarde: {OUTPUT_DIR / 'comparaison_modeles.png'}")

    # Resume final
    print("\n" + "="*70)
    print("RESUME DES RESULTATS")
    print("="*70)
    print(f"{'Dataset':<6} {'Modele':<10} {'R²':<8} {'RMSE':<8} {'Critical%':<10}")
    print("-"*50)
    for ds in DATASETS:
        for m in ['MLP', 'LSTM', 'PatchTST']:
            r = save_results[ds][m]
            print(f"{ds:<6} {m:<10} {r['R2']:<8.3f} {r['RMSE']:<8.3f} {r['Critical']:<10.1f}")

    print("\nTERMINE!")


if __name__ == '__main__':
    main()
