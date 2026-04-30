#!/usr/bin/env python3
"""
HPO Round 2 — LSTM+NODE, architecture fixee du trial 37.
Warm-start depuis hpo_LSTM_h2s_D2.pt.

Explore : activation, weight_decay, optimizer, seq_len, proj_depth, scheduler.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import average_precision_score
import pickle
import json
import time
import gc

import sys
SCRIPT_DIR = Path(__file__).parent
NODE_DIR = SCRIPT_DIR.parent
SCRIPTS_DIR = NODE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train_node import VehicleDynamicsNet, load_scenario, N_INPUT, N_STATE, DT, device
from train_with_node import get_split, build_dataset, N_FEATURES, FEATURES_8

MODEL_DIR = NODE_DIR / "models"
OUTPUT_DIR = NODE_DIR / "outputs"
MAXHORIZON_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/maxhorizon")

CONFIG = 'D2'
HORIZON_KEY = 'h2s'
HORIZON_STEPS = 200
EPOCHS = 80
PATIENCE = 15
N_TRIALS = 50
DANGER_THRESHOLD = 0.7

# Fixed architecture from trial 37
HIDDEN_SIZE = 128
NUM_LAYERS = 1
FC_HIDDEN = 64
PHYSICS_PROJ_DIM = 16


# ============== Activation registry ==============

ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
}


# ============== Models ==============

class PhysicsProjection_R2(nn.Module):
    """Physics projection with configurable depth."""
    def __init__(self, node_backbone, backbone_dim=128, out_dim=16, depth=1, activation='silu'):
        super().__init__()
        self.node_backbone = node_backbone
        act_fn = ACTIVATIONS[activation]
        if depth == 1:
            self.proj = nn.Sequential(
                nn.Linear(backbone_dim, out_dim),
                act_fn(),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(backbone_dim, backbone_dim // 2),
                act_fn(),
                nn.Linear(backbone_dim // 2, out_dim),
                act_fn(),
            )

    def forward(self, x_norm):
        B, T, _ = x_norm.shape
        physics = self.node_backbone(x_norm.reshape(B * T, -1))
        physics = physics.reshape(B, T, -1)
        return self.proj(physics)


class LSTM_Physics_R2(nn.Module):
    """LSTM+NODE with configurable activation."""
    def __init__(self, physics_proj, input_size, hidden_size=128,
                 fc_hidden=64, dropout=0.2, activation='relu'):
        super().__init__()
        self.physics_proj = physics_proj
        act_fn = ACTIVATIONS[activation]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden), act_fn(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, 3),
        )

    def forward(self, x_8feat, x_8feat_node_norm):
        physics = self.physics_proj(x_8feat_node_norm)
        x_aug = torch.cat([x_8feat, physics], dim=2)
        out, _ = self.lstm(x_aug)
        return self.fc(out[:, -1, :])


# ============== Loss ==============

def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.mean(torch.max(q * err, (q - 1) * err)))
    return sum(losses)


# ============== Data ==============

def load_shared_resources():
    print("Loading shared resources...")

    node_model = VehicleDynamicsNet(N_INPUT, N_STATE, hidden_size=256).to(device)
    node_model.load_state_dict(
        torch.load(MODEL_DIR / "node_final_8feat.pt", map_location=device, weights_only=True)
    )
    node_model.eval()
    for p in node_model.parameters():
        p.requires_grad = False
    backbone = node_model.shared

    with torch.no_grad():
        backbone_dim = backbone(torch.randn(1, N_INPUT, device=device)).shape[1]
    print(f"  NODE backbone dim: {backbone_dim}, frozen")

    with open(MODEL_DIR / "scaler_node_final_8feat.pkl", 'rb') as f:
        scaler_node = pickle.load(f)
    with open(MAXHORIZON_DIR / "models" / f"scaler_{CONFIG}_{HORIZON_KEY}.pkl", 'rb') as f:
        scaler_8 = pickle.load(f)

    train_files, test_files = get_split(CONFIG)
    print(f"  {CONFIG} split: {len(train_files)} train, {len(test_files)} test files")

    # Build datasets for multiple seq_lens
    datasets = {}
    for seq_len in [100, 150, 200]:
        X_train, Y_train = build_dataset(train_files, HORIZON_STEPS, stride=10)
        X_test, Y_test = build_dataset(test_files, HORIZON_STEPS, stride=10)

        n_train, n_test = len(X_train), len(X_test)

        # For seq_len < 150, trim. For seq_len > 150, skip (build_dataset uses SEQ_LEN=150)
        if seq_len <= 150:
            offset = 150 - seq_len
            X_tr_8 = scaler_8.transform(X_train.reshape(-1, N_FEATURES)).reshape(
                n_train, 150, N_FEATURES)[:, offset:, :].astype(np.float32)
            X_te_8 = scaler_8.transform(X_test.reshape(-1, N_FEATURES)).reshape(
                n_test, 150, N_FEATURES)[:, offset:, :].astype(np.float32)
            X_tr_node = scaler_node.transform(X_train.reshape(-1, N_FEATURES)).reshape(
                n_train, 150, N_FEATURES)[:, offset:, :].astype(np.float32)
            X_te_node = scaler_node.transform(X_test.reshape(-1, N_FEATURES)).reshape(
                n_test, 150, N_FEATURES)[:, offset:, :].astype(np.float32)
        else:
            # Can't extend beyond what build_dataset gives (SEQ_LEN=150)
            continue

        Y_test_binary = (Y_test >= DANGER_THRESHOLD).astype(np.int32)

        datasets[seq_len] = {
            'X_tr_8': torch.FloatTensor(X_tr_8),
            'X_tr_node': torch.FloatTensor(X_tr_node),
            'Y_train': torch.FloatTensor(Y_train),
            'X_te_8': torch.FloatTensor(X_te_8),
            'X_te_node': torch.FloatTensor(X_te_node),
            'Y_test': torch.FloatTensor(Y_test),
            'Y_test_np': Y_test,
            'Y_test_binary': Y_test_binary,
        }
        n_pos = Y_test_binary.sum()
        print(f"  seq_len={seq_len}: {n_train} train, {n_test} test, "
              f"{n_pos}/{len(Y_test)} danger ({n_pos/len(Y_test)*100:.1f}%)")

    shared = {
        'backbone': backbone,
        'backbone_dim': backbone_dim,
        'datasets': datasets,
    }
    print("  Done.\n")
    return shared


# ============== Warm start ==============

def try_warm_start(model, saved_path):
    """Load weights where shapes match, skip the rest."""
    if not saved_path.exists():
        return model
    saved = torch.load(saved_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    loaded = 0
    for key in saved:
        if key in model_state and model_state[key].shape == saved[key].shape:
            model_state[key] = saved[key]
            loaded += 1
    model.load_state_dict(model_state)
    print(f"    Warm-start: {loaded}/{len(model_state)} params loaded")
    return model


# ============== Metric ==============

def compute_auc_pr(model, test_loader, y_binary):
    model.eval()
    preds_q50 = []
    with torch.no_grad():
        for X8, Xn, Y in test_loader:
            X8, Xn = X8.to(device), Xn.to(device)
            pred = model(X8, Xn)
            preds_q50.append(pred[:, 1].cpu().numpy())
    p50 = np.clip(np.concatenate(preds_q50), 0.0, 1.5)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.0
    return float(average_precision_score(y_binary, p50))


# ============== Objective ==============

def objective(trial, shared):
    # New HPs to explore
    activation = trial.suggest_categorical('activation', ['relu', 'silu', 'gelu'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    lr_projection = trial.suggest_float('lr_projection', 1e-4, 3e-3, log=True)
    lr_other = trial.suggest_float('lr_other', 1e-4, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    seq_len = trial.suggest_categorical('seq_len', [100, 150])
    proj_depth = trial.suggest_categorical('proj_depth', [1, 2])
    sched_patience = trial.suggest_categorical('sched_patience', [3, 5, 8])
    use_adamw = trial.suggest_categorical('use_adamw', [True, False])

    data = shared['datasets'].get(seq_len)
    if data is None:
        return -1.0

    n_features_aug = N_FEATURES + PHYSICS_PROJ_DIM

    try:
        physics_proj = PhysicsProjection_R2(
            shared['backbone'], shared['backbone_dim'],
            out_dim=PHYSICS_PROJ_DIM, depth=proj_depth, activation=activation,
        ).to(device)
        for p in physics_proj.node_backbone.parameters():
            p.requires_grad = False

        model = LSTM_Physics_R2(
            physics_proj, input_size=n_features_aug,
            hidden_size=HIDDEN_SIZE, fc_hidden=FC_HIDDEN,
            dropout=dropout, activation=activation,
        ).to(device)

        # Warm-start from round 1 best (where shapes match)
        model = try_warm_start(model, MODEL_DIR / "hpo_LSTM_h2s_D2.pt")

    except Exception as e:
        print(f"  Trial {trial.number}: failed: {e}")
        return -1.0

    train_ds = TensorDataset(data['X_tr_8'], data['X_tr_node'], data['Y_train'])
    test_ds = TensorDataset(data['X_te_8'], data['X_te_node'], data['Y_test'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    proj_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'physics_proj.proj' in name:
            proj_params.append(p)
        else:
            other_params.append(p)

    OptClass = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptClass([
        {'params': proj_params, 'lr': lr_projection, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=sched_patience, factor=0.5)

    best_auc = -1.0
    no_improve = 0

    try:
        for epoch in range(EPOCHS):
            model.train()
            for X8, Xn, Y in train_loader:
                X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                optimizer.zero_grad()
                pred = model(X8, Xn)
                loss = quantile_loss(pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            v_loss = 0
            with torch.no_grad():
                for X8, Xn, Y in test_loader:
                    X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                    v_loss += quantile_loss(model(X8, Xn), Y).item()
            v_loss /= len(test_loader)
            scheduler.step(v_loss)

            auc = compute_auc_pr(model, test_loader, data['Y_test_binary'])

            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if auc > best_auc:
                best_auc = auc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break

    except optuna.TrialPruned:
        raise
    except RuntimeError as e:
        if 'out of memory' in str(e).lower() or 'mps' in str(e).lower():
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            return -1.0
        raise

    return best_auc


# ============== Retrain ==============

def retrain_best(best_params, shared):
    print(f"\n  Retraining best round 2 config...")

    seq_len = best_params['seq_len']
    data = shared['datasets'][seq_len]
    n_features_aug = N_FEATURES + PHYSICS_PROJ_DIM

    physics_proj = PhysicsProjection_R2(
        shared['backbone'], shared['backbone_dim'],
        out_dim=PHYSICS_PROJ_DIM,
        depth=best_params['proj_depth'],
        activation=best_params['activation'],
    ).to(device)
    for p in physics_proj.node_backbone.parameters():
        p.requires_grad = False

    model = LSTM_Physics_R2(
        physics_proj, input_size=n_features_aug,
        hidden_size=HIDDEN_SIZE, fc_hidden=FC_HIDDEN,
        dropout=best_params['dropout'],
        activation=best_params['activation'],
    ).to(device)
    model = try_warm_start(model, MODEL_DIR / "hpo_LSTM_h2s_D2.pt")

    batch_size = best_params['batch_size']
    train_ds = TensorDataset(data['X_tr_8'], data['X_tr_node'], data['Y_train'])
    test_ds = TensorDataset(data['X_te_8'], data['X_te_node'], data['Y_test'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    proj_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'physics_proj.proj' in name:
            proj_params.append(p)
        else:
            other_params.append(p)

    OptClass = torch.optim.AdamW if best_params['use_adamw'] else torch.optim.Adam
    optimizer = OptClass([
        {'params': proj_params, 'lr': best_params['lr_projection'],
         'weight_decay': best_params['weight_decay']},
        {'params': other_params, 'lr': best_params['lr_other'],
         'weight_decay': best_params['weight_decay']},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=best_params['sched_patience'], factor=0.5)

    best_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for X8, Xn, Y in train_loader:
            X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X8, Xn)
            loss = quantile_loss(pred, Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for X8, Xn, Y in test_loader:
                X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                v_loss += quantile_loss(model(X8, Xn), Y).item()
        t_loss /= len(train_loader)
        v_loss /= len(test_loader)
        scheduler.step(v_loss)

        auc = compute_auc_pr(model, test_loader, data['Y_test_binary'])

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve >= PATIENCE:
            print(f"    Ep {epoch:3d}: t={t_loss:.5f} v={v_loss:.5f} "
                  f"AUC-PR={auc:.4f} ni={no_improve}")
        if no_improve >= PATIENCE:
            print(f"    Early stop ep {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


# ============== Main ==============

def main():
    t_start = time.time()
    print("=" * 70)
    print("HPO Round 2 — LSTM+NODE (fixed arch, new params)")
    print(f"Device: {device}")
    print(f"Exploring: activation, weight_decay, optimizer, seq_len, proj_depth")
    print(f"Warm-start from: hpo_LSTM_h2s_D2.pt")
    print("=" * 70)

    shared = load_shared_resources()

    db_path = OUTPUT_DIR / "hpo_max_horizon.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="hpo_LSTM_NODE_D2_h2s_round2",
        storage=storage,
        direction='maximize',
        sampler=TPESampler(seed=123),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    n_remaining = max(0, N_TRIALS - n_existing)
    if n_existing > 0:
        print(f"  Resuming: {n_existing} trials done, {n_remaining} remaining")

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, shared),
            n_trials=n_remaining,
            show_progress_bar=True,
        )

    best = study.best_trial
    print(f"\n  Best trial #{best.number}: AUC-PR = {best.value:.4f}")
    print(f"  Params: {best.params}")

    model, final_auc = retrain_best(best.params, shared)
    save_path = MODEL_DIR / f"hpo_LSTM_{HORIZON_KEY}_{CONFIG}_r2.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path.name} (AUC-PR={final_auc:.4f})")

    results = {
        'best_trial': best.number,
        'best_auc_pr_search': float(best.value),
        'best_auc_pr_retrain': float(final_auc),
        'best_params': best.params,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]),
        'round1_best_auc': 0.887,
    }

    results_path = OUTPUT_DIR / "results_hpo_lstm_round2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nRound 2 done in {elapsed/60:.1f} min")
    print(f"Round 1 best: 0.887 → Round 2 best: {final_auc:.4f}")
    print("DONE!")


if __name__ == '__main__':
    main()
