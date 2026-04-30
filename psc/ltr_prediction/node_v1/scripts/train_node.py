#!/usr/bin/env python3
"""Entraîne le Neural ODE backbone sur la dynamique véhicule (5 états + 2 commandes)."""

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
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DT = 0.01

# État — prédit par le modèle
STATE_FEATURES = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta']
# Commande — input exogène, PAS prédit
COMMAND_FEATURES = ['delta_f', 'delta_f_dot']
# Toutes les features d'entrée du modèle
ALL_INPUT_FEATURES = STATE_FEATURES + COMMAND_FEATURES

N_STATE = len(STATE_FEATURES)       # 6
N_COMMAND = len(COMMAND_FEATURES)    # 2
N_INPUT = N_STATE + N_COMMAND        # 8
N_OUTPUT = N_STATE + 1               # 7 (6 dx_state + 1 ltr_pred)

# Horizons de prédiction (en secondes)
HORIZONS = [1, 2, 4]

# Training
BATCH_SIZE = 256
EPOCHS_PHASE1 = 200
EPOCHS_PHASE2 = 80
LEARNING_RATE = 1e-3
PATIENCE = 15
LAMBDA_LTR = 0.5       # Poids de la loss LTR dans Phase 1
LAMBDA_LTR_P2 = 1.0   # Poids de la loss LTR dans Phase 2 (priorité LTR)
ASYM_ALPHA = 3.0       # Pénalité x3 pour sous-estimation du LTR

# Curriculum: horizons progressifs (en steps) pour Phase 2
CURRICULUM_HORIZONS = [5, 10, 20, 50, 100, 200, 400]
CURRICULUM_EPOCHS = [10, 10, 10, 15, 15, 15, 20]  # Epochs par palier

# Scheduled sampling: probabilité d'utiliser ground truth
SS_START = 1.0   # Début: 100% teacher forcing
SS_END = 0.0     # Fin: 0% teacher forcing

# Input noise injection (Phase 1) — robustesse aux erreurs de propagation
NOISE_STD = 0.1  # Écart-type du bruit en espace normalisé

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ============== Loss ==============

def asymmetric_mse_loss(pred, target, alpha=ASYM_ALPHA):
    """
    MSE asymétrique: pénalise la sous-estimation du LTR alpha fois plus.
    Sous-prédire un danger est bien plus grave que sur-prédire.
    """
    error = pred - target
    weight = torch.where(error < 0, alpha, 1.0)  # sous-estimation → poids alpha
    return (weight * error ** 2).mean()


# ============== Model ==============

class VehicleDynamicsNet(nn.Module):
    """
    Apprend: (dx_state, ltr_pred) = f_θ(state_t, command_t)

    dx_state: résidu d'état (5 dim)
    ltr_pred: prédiction directe du LTR (1 dim, sigmoid → [0, 1.5])
    """
    def __init__(self, n_input=N_INPUT, n_state=N_STATE, hidden_size=256):
        super().__init__()
        self.n_state = n_state
        mid_size = hidden_size // 2  # 128
        self.shared = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, mid_size),
            nn.SiLU(),
        )
        self.head_dx = nn.Linear(mid_size, n_state)
        self.head_ltr = nn.Sequential(
            nn.Linear(mid_size, 1),
            nn.Sigmoid(),  # Output in [0, 1], scaled to [0, 1.5]
        )

    def forward(self, x):
        """
        x: (batch, n_input) — [state, command]
        Returns: dx (batch, n_state), ltr (batch,)
        """
        h = self.shared(x)
        dx = self.head_dx(h)
        ltr = self.head_ltr(h).squeeze(-1) * 1.5  # Scale to [0, 1.5]
        return dx, ltr


# ============== Data ==============

def load_scenario(filepath):
    df = pd.read_csv(filepath)
    df['delta_f_dot'] = np.gradient(df['delta_f'].values, DT)
    if 'psi_dot' not in df.columns:
        df['psi_dot'] = np.gradient(df['psi'].values, DT)
    for col in ALL_INPUT_FEATURES:
        df[col] = df[col].clip(-1e6, 1e6)
    return df


def create_pairs(df, stride=1):
    """
    Crée des paires (x_t, x_{t+1}) + commande à t+1 pour scheduled sampling.

    Returns:
      X:     (N, N_INPUT) — [state_t, command_t]
      Y:     (N, N_STATE) — state_{t+1}
      L:     (N,) — LTRmax réel à t+1
      CMD1:  (N, N_COMMAND) — commande à t+1 (pour scheduled sampling)
    """
    input_data = df[ALL_INPUT_FEATURES].values   # (T, 7)
    state_data = df[STATE_FEATURES].values        # (T, 5)
    cmd_data = df[COMMAND_FEATURES].values        # (T, 2)
    ltr = np.abs(df['LTRmax'].values)

    X = input_data[:-1:stride]           # [state_t, command_t]
    Y = state_data[1::stride]            # state_{t+1}
    L = ltr[1::stride]                   # LTR réel à t+1
    CMD1 = cmd_data[1::stride]           # commande à t+1

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    L = L.astype(np.float32)
    CMD1 = CMD1.astype(np.float32)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1) & np.isfinite(L)
    return X[valid], Y[valid], L[valid], CMD1[valid]


def create_multistep_sequences(df, max_steps, stride=10):
    """
    Crée des séquences pour l'entraînement multi-step.

    Returns:
      X_init:   (N, N_INPUT) — état initial + commande initiale
      Y_seq:    (N, max_steps, N_STATE) — séquence d'états ground truth
      L_seq:    (N, max_steps) — séquence de LTR ground truth
      CMD_seq:  (N, max_steps, N_COMMAND) — séquence de commandes futures
    """
    input_data = df[ALL_INPUT_FEATURES].values
    state_only = df[STATE_FEATURES].values
    cmd_data = df[COMMAND_FEATURES].values
    ltr = np.abs(df['LTRmax'].values)

    X_list, Y_list, L_list, C_list = [], [], [], []

    for i in range(0, len(df) - max_steps - 1, stride):
        X_list.append(input_data[i])
        Y_list.append(state_only[i + 1:i + 1 + max_steps])
        L_list.append(ltr[i + 1:i + 1 + max_steps])
        C_list.append(cmd_data[i + 1:i + 1 + max_steps])

    if not X_list:
        return None, None, None, None

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            np.array(L_list, dtype=np.float32),
            np.array(C_list, dtype=np.float32))


def load_all_data():
    """Charge tous les CSV et split train/test."""
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Fichiers: {len(files)}")

    np.random.seed(42)
    indices = np.random.permutation(len(files))
    split = int(0.8 * len(files))
    train_files = [files[i] for i in indices[:split]]
    test_files = [files[i] for i in indices[split:]]

    return train_files, test_files


def build_1step_dataset(files, stride=1):
    """Dataset de paires 1-step."""
    X_all, Y_all, L_all, C_all = [], [], [], []
    for f in files:
        df = load_scenario(f)
        X, Y, L, CMD1 = create_pairs(df, stride=stride)
        X_all.append(X)
        Y_all.append(Y)
        L_all.append(L)
        C_all.append(CMD1)

    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(L_all), np.concatenate(C_all))


def build_multistep_dataset(files, max_steps=100, stride=10):
    """Dataset de séquences multi-step."""
    X_all, Y_all, L_all, C_all = [], [], [], []
    for f in files:
        df = load_scenario(f)
        X, Y, L, CMD = create_multistep_sequences(df, max_steps, stride=stride)
        if X is not None:
            X_all.append(X)
            Y_all.append(Y)
            L_all.append(L)
            C_all.append(CMD)

    return (np.concatenate(X_all), np.concatenate(Y_all),
            np.concatenate(L_all), np.concatenate(C_all))


# ============== Training ==============

def train_1step(model, X_train, Y_train, L_train, X_test, Y_test, L_test, scaler):
    """
    Phase 1: entraînement 1-step.

    Le modèle opère en espace normalisé.
    Loss = MSE(état) + LAMBDA_LTR * MSE(LTR direct)
    """
    print("\n--- Phase 1: Entraînement 1-step ---")

    scaler_mean_state = torch.FloatTensor(scaler.mean_[:N_STATE]).to(device)
    scaler_scale_state = torch.FloatTensor(scaler.scale_[:N_STATE]).to(device)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Normalize state targets
    Y_train_s = (Y_train - scaler.mean_[:N_STATE]) / scaler.scale_[:N_STATE]
    Y_test_s = (Y_test - scaler.mean_[:N_STATE]) / scaler.scale_[:N_STATE]

    train_ds = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.FloatTensor(Y_train_s),
        torch.FloatTensor(L_train)
    )
    test_ds = TensorDataset(
        torch.FloatTensor(X_test_s),
        torch.FloatTensor(Y_test_s),
        torch.FloatTensor(L_test)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, best_state, no_improve = float('inf'), None, 0

    for epoch in range(EPOCHS_PHASE1):
        model.train()
        train_loss_sum = 0
        for X, Y_norm, L in train_loader:
            X, Y_norm, L = X.to(device), Y_norm.to(device), L.to(device)
            optimizer.zero_grad()

            # Input noise injection: rend le modèle robuste aux états imparfaits
            X_noisy = X.clone()
            X_noisy[:, :N_STATE] += NOISE_STD * torch.randn(X.shape[0], N_STATE, device=device)

            dx, ltr_pred = model(X_noisy)
            # Prédiction basée sur l'input bruité, target = état propre
            Y_pred_norm = X_noisy[:, :N_STATE] + dx * DT

            loss_state = nn.MSELoss()(Y_pred_norm, Y_norm)
            loss_ltr = asymmetric_mse_loss(ltr_pred, L)

            loss = loss_state + LAMBDA_LTR * loss_ltr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum, val_ltr_sum = 0, 0
        with torch.no_grad():
            for X, Y_norm, L in test_loader:
                X, Y_norm, L = X.to(device), Y_norm.to(device), L.to(device)
                dx, ltr_pred = model(X)
                Y_pred_norm = X[:, :N_STATE] + dx * DT
                loss_state = nn.MSELoss()(Y_pred_norm, Y_norm)
                loss_ltr = asymmetric_mse_loss(ltr_pred, L)
                val_loss_sum += (loss_state + LAMBDA_LTR * loss_ltr).item()
                val_ltr_sum += loss_ltr.item()

        val_loss = val_loss_sum / len(test_loader)
        val_ltr = val_ltr_sum / len(test_loader)
        train_loss = train_loss_sum / len(train_loader)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss, best_state, no_improve = val_loss, model.state_dict().copy(), 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve >= PATIENCE:
            print(f"  Epoch {epoch:3d}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"ltr_rmse={np.sqrt(val_ltr):.4f} (best={best_loss:.6f}, no_improve={no_improve})")

        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("  WARNING: No valid epoch in Phase 1.")
    return model


def _propagate_with_commands(model, x_init_norm, cmd_seq_norm, n_steps,
                             scaler_mean, scaler_scale,
                             gt_state_norm=None, teacher_prob=0.0):
    """
    Propagation en espace normalisé avec injection des vraies commandes.

    Args:
      x_init_norm: (batch, N_INPUT) — état+commande normalisés à t=0
      cmd_seq_norm: (batch, n_steps, N_COMMAND) — commandes normalisées futures
      n_steps: nombre de steps à propager
      scaler_mean, scaler_scale: pour dénormalisation
      gt_state_norm: (batch, n_steps, N_STATE) — états GT normalisés (pour scheduled sampling)
      teacher_prob: probabilité d'utiliser le GT au lieu de la prédiction

    Returns:
      states_raw: (batch, n_steps, N_STATE) — trajectoire prédite en espace brut
      ltrs_pred: (batch, n_steps) — LTR prédits à chaque step
    """
    batch = x_init_norm.shape[0]
    states_raw = torch.zeros(batch, n_steps, N_STATE, device=x_init_norm.device)
    ltrs_pred = torch.zeros(batch, n_steps, device=x_init_norm.device)

    current_state_norm = x_init_norm[:, :N_STATE]

    for t in range(n_steps):
        # Construire l'input: [state_norm, command_norm]
        cmd_t = cmd_seq_norm[:, t, :]  # (batch, N_COMMAND)
        inp = torch.cat([current_state_norm, cmd_t], dim=1)  # (batch, N_INPUT)

        dx, ltr = model(inp)
        next_state_norm = current_state_norm + dx * DT

        # Dénormaliser pour la sortie
        next_state_raw = next_state_norm * scaler_scale[:N_STATE] + scaler_mean[:N_STATE]
        states_raw[:, t] = next_state_raw
        ltrs_pred[:, t] = ltr

        # Scheduled sampling: utiliser GT ou prédiction pour le prochain step
        if gt_state_norm is not None and teacher_prob > 0 and t < n_steps - 1:
            use_gt = torch.rand(batch, 1, device=x_init_norm.device) < teacher_prob
            current_state_norm = torch.where(use_gt, gt_state_norm[:, t], next_state_norm)
        else:
            current_state_norm = next_state_norm

    return states_raw, ltrs_pred


def train_multistep(model, train_files, test_files, scaler):
    """
    Phase 2: curriculum multi-step avec scheduled sampling.

    Curriculum: horizon 5 → 10 → 20 → 50 → 100 steps
    Scheduled sampling: teacher_prob décroît de 1.0 → 0.0
    """
    print(f"\n--- Phase 2: Curriculum multi-step avec scheduled sampling ---")

    max_steps = CURRICULUM_HORIZONS[-1]

    # Build dataset with max horizon
    X_train, Y_train, L_train, CMD_train = build_multistep_dataset(
        train_files, max_steps, stride=20)
    X_test, Y_test, L_test, CMD_test = build_multistep_dataset(
        test_files, max_steps, stride=20)

    print(f"  Train: {len(X_train)} sequences, Test: {len(X_test)} sequences")

    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Normalize state targets
    Y_train_s = ((Y_train - scaler.mean_[:N_STATE]) / scaler.scale_[:N_STATE]).astype(np.float32)
    Y_test_s = ((Y_test - scaler.mean_[:N_STATE]) / scaler.scale_[:N_STATE]).astype(np.float32)

    # Normalize command sequences
    cmd_mean = scaler.mean_[N_STATE:N_STATE + N_COMMAND]
    cmd_scale = scaler.scale_[N_STATE:N_STATE + N_COMMAND]
    CMD_train_s = ((CMD_train - cmd_mean) / cmd_scale).astype(np.float32)
    CMD_test_s = ((CMD_test - cmd_mean) / cmd_scale).astype(np.float32)

    scaler_mean = torch.FloatTensor(scaler.mean_).to(device)
    scaler_scale = torch.FloatTensor(scaler.scale_).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, best_state = float('inf'), None
    total_epochs = sum(CURRICULUM_EPOCHS)
    global_epoch = 0

    for stage_idx, (horizon, n_epochs) in enumerate(zip(CURRICULUM_HORIZONS, CURRICULUM_EPOCHS)):
        print(f"\n  Stage {stage_idx + 1}/{len(CURRICULUM_HORIZONS)}: horizon={horizon} steps ({horizon * DT:.2f}s)")

        no_improve = 0

        # Datasets with current horizon length (slice from max)
        train_ds = TensorDataset(
            torch.FloatTensor(X_train_s),
            torch.FloatTensor(Y_train_s[:, :horizon]),
            torch.FloatTensor(L_train[:, :horizon]),
            torch.FloatTensor(CMD_train_s[:, :horizon]),
        )
        test_ds = TensorDataset(
            torch.FloatTensor(X_test_s),
            torch.FloatTensor(Y_test_s[:, :horizon]),
            torch.FloatTensor(L_test[:, :horizon]),
            torch.FloatTensor(CMD_test_s[:, :horizon]),
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64)

        for epoch in range(n_epochs):
            # Scheduled sampling probability: linear decay over all stages
            ss_progress = global_epoch / max(total_epochs - 1, 1)
            teacher_prob = SS_START + (SS_END - SS_START) * ss_progress

            model.train()
            train_loss_sum = 0
            n_batches = 0

            for X_init, Y_seq_norm, L_seq, CMD_seq in train_loader:
                X_init = X_init.to(device)
                Y_seq_norm = Y_seq_norm.to(device)
                L_seq = L_seq.to(device)
                CMD_seq = CMD_seq.to(device)

                optimizer.zero_grad()

                states_raw, ltrs_pred = _propagate_with_commands(
                    model, X_init, CMD_seq, horizon,
                    scaler_mean, scaler_scale,
                    gt_state_norm=Y_seq_norm, teacher_prob=teacher_prob
                )

                # Dénormaliser les targets pour la loss en espace brut
                Y_seq_raw = Y_seq_norm * scaler_scale[:N_STATE] + scaler_mean[:N_STATE]

                # State loss: Huber (robuste aux outliers de propagation divergente)
                loss_state = nn.HuberLoss(delta=1.0)(states_raw, Y_seq_raw)

                # LTR loss: asymétrique sur tous les steps
                loss_ltr = asymmetric_mse_loss(ltrs_pred, L_seq)

                loss = loss_state + LAMBDA_LTR_P2 * loss_ltr

                if torch.isnan(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item()
                n_batches += 1

            # Validation (full autoregressive, no teacher forcing)
            model.eval()
            val_loss_sum = 0
            val_ltr_sum = 0
            n_val = 0
            with torch.no_grad():
                for X_init, Y_seq_norm, L_seq, CMD_seq in test_loader:
                    X_init = X_init.to(device)
                    Y_seq_norm = Y_seq_norm.to(device)
                    L_seq = L_seq.to(device)
                    CMD_seq = CMD_seq.to(device)

                    states_raw, ltrs_pred = _propagate_with_commands(
                        model, X_init, CMD_seq, horizon,
                        scaler_mean, scaler_scale,
                        gt_state_norm=None, teacher_prob=0.0
                    )

                    Y_seq_raw = Y_seq_norm * scaler_scale[:N_STATE] + scaler_mean[:N_STATE]
                    loss_state = nn.MSELoss()(states_raw, Y_seq_raw)
                    loss_ltr = asymmetric_mse_loss(ltrs_pred, L_seq)
                    val_loss_sum += (loss_state + LAMBDA_LTR_P2 * loss_ltr).item()
                    val_ltr_sum += loss_ltr.item()
                    n_val += 1

            val_loss = val_loss_sum / max(n_val, 1)
            train_loss = train_loss_sum / max(n_batches, 1)
            scheduler.step(val_loss)

            if not np.isnan(val_loss) and val_loss < best_loss:
                best_loss, best_state, no_improve = val_loss, model.state_dict().copy(), 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                ltr_rmse = np.sqrt(val_ltr_sum / max(n_val, 1))
                print(f"    Epoch {epoch:2d}: train={train_loss:.4f} val={val_loss:.4f} "
                      f"ltr_rmse={ltr_rmse:.4f} teacher_p={teacher_prob:.2f}")

            global_epoch += 1

            if no_improve >= 10:
                print(f"    Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("  WARNING: No valid epoch in Phase 2. Keeping Phase 1 weights.")
    return model


# ============== Command Prediction ==============

# History length for command predictor (must match train_command_predictor.py)
CMD_HISTORY_LEN = 50

def _get_command_sequence(cmd_data, i, n_steps, cmd_mean, cmd_scale_np,
                          cmd_predictor=None, scaler_cmd=None):
    """
    Retourne la séquence de commandes futures normalisées.

    Si cmd_predictor est fourni : prédit les commandes depuis l'historique.
    Sinon : retourne les ground truth (comportement legacy pour ablation).

    Args:
        cmd_data: (T, 2) — commandes brutes [delta_f, delta_f_dot] du scénario
        i: index courant dans le scénario
        n_steps: nombre de steps futurs
        cmd_mean: (2,) — moyenne des commandes (du scaler NODE)
        cmd_scale_np: (2,) — scale des commandes (du scaler NODE)
        cmd_predictor: CommandPredictor model (optional)
        scaler_cmd: StandardScaler du CommandPredictor (optional, requis si cmd_predictor)

    Returns:
        cmd_seq: torch.Tensor (1, n_steps, 2) — commandes normalisées (scaler NODE)
    """
    if cmd_predictor is not None and scaler_cmd is not None:
        # Prédire les commandes futures depuis l'historique
        history_start = max(0, i + 1 - CMD_HISTORY_LEN)
        history_raw = cmd_data[history_start:i + 1]  # (up to CMD_HISTORY_LEN, 2)

        # Pad si l'historique est trop court
        if len(history_raw) < CMD_HISTORY_LEN:
            pad = np.tile(history_raw[0:1], (CMD_HISTORY_LEN - len(history_raw), 1))
            history_raw = np.concatenate([pad, history_raw], axis=0)

        # Normaliser avec le scaler du CommandPredictor
        history_norm = scaler_cmd.transform(history_raw).astype(np.float32)

        from train_command_predictor import predict_commands_autoreg
        cmd_pred_norm_cp = predict_commands_autoreg(
            cmd_predictor, history_norm, n_steps, scaler_cmd,
            device_=next(cmd_predictor.parameters()).device
        )
        # cmd_pred_norm_cp est normalisé avec scaler_cmd
        # On doit le convertir en normalisation scaler NODE
        mean_f_cp, scale_f_cp = scaler_cmd.mean_[0], scaler_cmd.scale_[0]
        mean_fdot_cp, scale_fdot_cp = scaler_cmd.mean_[1], scaler_cmd.scale_[1]

        # Dénormaliser (scaler_cmd) → renormaliser (scaler NODE)
        delta_f_raw = cmd_pred_norm_cp[:, 0] * scale_f_cp + mean_f_cp
        delta_fdot_raw = cmd_pred_norm_cp[:, 1] * scale_fdot_cp + mean_fdot_cp

        cmd_future_raw = np.stack([delta_f_raw, delta_fdot_raw], axis=1)
        cmd_future_s = ((cmd_future_raw - cmd_mean) / cmd_scale_np).astype(np.float32)
    else:
        # Legacy: ground truth commands
        cmd_future = cmd_data[i + 1:i + 1 + n_steps]
        cmd_future_s = ((cmd_future - cmd_mean) / cmd_scale_np).astype(np.float32)

    cmd_seq = torch.FloatTensor(cmd_future_s).unsqueeze(0).to(device)  # (1, n_steps, 2)
    return cmd_seq


# ============== Evaluation ==============

def evaluate_max_ltr(model, test_files, scaler, horizons=[1, 2, 4],
                     cmd_predictor=None, scaler_cmd=None):
    """
    Évalue la prédiction de max(LTR) par propagation.

    Si cmd_predictor est fourni, les commandes futures sont prédites (pas de leakage).
    Sinon, les vraies commandes futures sont injectées (legacy, pour ablation).
    """
    scaler_mean = torch.FloatTensor(scaler.mean_).to(device)
    scaler_scale = torch.FloatTensor(scaler.scale_).to(device)
    cmd_mean = scaler.mean_[N_STATE:N_STATE + N_COMMAND]
    cmd_scale_np = scaler.scale_[N_STATE:N_STATE + N_COMMAND]

    mode = "predicted" if cmd_predictor is not None else "ground truth"
    print(f"  [Commands: {mode}]")

    results = {}

    for horizon in horizons:
        n_steps = int(horizon / DT)
        true_maxltr_all, pred_maxltr_all = [], []

        for f in test_files:
            df = load_scenario(f)
            input_data = df[ALL_INPUT_FEATURES].values.astype(np.float32)
            cmd_data = df[COMMAND_FEATURES].values.astype(np.float32)
            ltr = np.abs(df['LTRmax'].values)

            for i in range(0, len(df) - 150 - n_steps, 50):
                true_max = np.max(ltr[i + 1:i + 1 + n_steps])

                # Initial state+command
                x_init = torch.FloatTensor(scaler.transform(input_data[i:i+1])).to(device)

                # Future commands (predicted or ground truth)
                cmd_seq = _get_command_sequence(
                    cmd_data, i, n_steps, cmd_mean, cmd_scale_np,
                    cmd_predictor=cmd_predictor, scaler_cmd=scaler_cmd
                )

                model.eval()
                with torch.no_grad():
                    _, ltrs_pred = _propagate_with_commands(
                        model, x_init, cmd_seq, n_steps,
                        scaler_mean, scaler_scale
                    )
                    max_ltr_pred = ltrs_pred[0].max().item()

                true_maxltr_all.append(true_max)
                pred_maxltr_all.append(max_ltr_pred)

        true_arr = np.array(true_maxltr_all)
        pred_arr = np.array(pred_maxltr_all)

        if len(true_arr) == 0:
            continue

        rmse = np.sqrt(mean_squared_error(true_arr, pred_arr))
        r2 = r2_score(true_arr, pred_arr)

        true_danger = true_arr >= 0.7
        pred_danger = pred_arr >= 0.7
        recall = np.sum(true_danger & pred_danger) / max(np.sum(true_danger), 1)
        precision = np.sum(true_danger & pred_danger) / max(np.sum(pred_danger), 1)

        results[f'h{horizon}s'] = {
            'rmse': float(rmse),
            'r2': float(r2),
            'recall_0.7': float(recall),
            'precision_0.7': float(precision),
            'n_samples': len(true_arr),
        }

        print(f"  h={horizon}s: RMSE={rmse:.3f} R²={r2:.3f} "
              f"Recall@0.7={recall:.1%} Precision@0.7={precision:.1%} (n={len(true_arr)})")

    return results


def evaluate_1step_accuracy(model, X_test, Y_test, L_test, scaler):
    """Vérifie la précision 1-step (état + LTR)."""
    model.eval()
    X_test_s = torch.FloatTensor(scaler.transform(X_test)).to(device)
    scaler_mean_state = torch.FloatTensor(scaler.mean_[:N_STATE]).to(device)
    scaler_scale_state = torch.FloatTensor(scaler.scale_[:N_STATE]).to(device)

    preds_state, preds_ltr = [], []
    with torch.no_grad():
        for i in range(0, len(X_test_s), 1024):
            batch = X_test_s[i:i + 1024]
            dx, ltr = model(batch)
            Y_pred_norm = batch[:, :N_STATE] + dx * DT
            Y_pred_raw = Y_pred_norm * scaler_scale_state + scaler_mean_state
            preds_state.append(Y_pred_raw.cpu().numpy())
            preds_ltr.append(ltr.cpu().numpy())

    Y_pred = np.concatenate(preds_state)
    ltr_pred = np.concatenate(preds_ltr)

    print("\n  Erreur 1-step par feature:")
    for i, feat in enumerate(STATE_FEATURES):
        rmse = np.sqrt(np.mean((Y_pred[:, i] - Y_test[:, i]) ** 2))
        feat_range = np.ptp(Y_test[:, i])
        nrmse_pct = (rmse / feat_range * 100) if feat_range > 1e-8 else float('nan')
        print(f"    {feat:12s}: RMSE={rmse:.6f}  NRMSE={nrmse_pct:.2f}%  (range={feat_range:.4f})")

    overall_rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
    ltr_rmse = np.sqrt(np.mean((ltr_pred - L_test) ** 2))
    ltr_r2 = r2_score(L_test, ltr_pred)
    print(f"  State RMSE: {overall_rmse:.6f}")
    print(f"  LTR 1-step: RMSE={ltr_rmse:.4f}  R²={ltr_r2:.3f}")
    return overall_rmse


# ============== Main ==============

def main():
    print("=" * 70)
    print("NEURAL ODE v2 — PRÉDICTION LTR (état/commande séparés)")
    print(f"État: {STATE_FEATURES}")
    print(f"Commande: {COMMAND_FEATURES}")
    print(f"Device: {device}")
    print("=" * 70)

    if not DATA_DIR.exists() or len(list(DATA_DIR.glob("*.csv"))) == 0:
        print(f"\nATTENTION: {DATA_DIR} est vide.")
        print(f"Créer un symlink: ln -s ../maxhorizon_v2/data {DATA_DIR}")
        return

    train_files, test_files = load_all_data()
    print(f"Train: {len(train_files)} fichiers, Test: {len(test_files)} fichiers")

    # Phase 1: 1-step training
    print("\n" + "=" * 60)
    print("PHASE 1: Entraînement 1-step")
    print("=" * 60)

    t0 = time.time()
    X_train, Y_train, L_train, _ = build_1step_dataset(train_files, stride=5)
    X_test, Y_test, L_test, _ = build_1step_dataset(test_files, stride=5)
    print(f"1-step dataset: train={len(X_train)}, test={len(X_test)} ({time.time()-t0:.1f}s)")

    scaler = StandardScaler()
    scaler.fit(X_train)

    with open(MODEL_DIR / "scaler_node.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    model = VehicleDynamicsNet(N_INPUT, N_STATE, hidden_size=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    model = train_1step(model, X_train, Y_train, L_train, X_test, Y_test, L_test, scaler)

    evaluate_1step_accuracy(model, X_test, Y_test, L_test, scaler)

    torch.save(model.state_dict(), MODEL_DIR / "node_1step.pt")
    print(f"\nModèle 1-step sauvegardé: {MODEL_DIR / 'node_1step.pt'}")

    # Phase 2: Curriculum multi-step
    print("\n" + "=" * 60)
    print("PHASE 2: Curriculum multi-step + scheduled sampling")
    print("=" * 60)

    model = train_multistep(model, train_files, test_files, scaler)
    torch.save(model.state_dict(), MODEL_DIR / "node_multistep.pt")
    print(f"Modèle multi-step sauvegardé: {MODEL_DIR / 'node_multistep.pt'}")

    # Phase 3: Évaluation max(LTR)
    print("\n" + "=" * 60)
    print("PHASE 3: Évaluation max(LTR) par propagation")
    print("=" * 60)

    results = evaluate_max_ltr(model, test_files, scaler, horizons=HORIZONS)

    with open(OUTPUT_DIR / "results_node.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nRésultats sauvegardés: {OUTPUT_DIR / 'results_node.json'}")
    print("TERMINÉ!")


if __name__ == '__main__':
    main()
