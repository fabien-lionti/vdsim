#!/usr/bin/env python3
"""
HPO — Optimisation hyperparametres LSTM+NODE & PatchTST+NODE (max horizon).

Config : D2, h2s (2 secondes), max horizon.
Metrique d'optimisation : AUC-PR (danger >= 0.7) sur le test set D2.
Strategie : NODE backbone frozen pendant le HPO.

Dossier autonome — ne modifie rien dans scripts/.

Usage:
    python hpo/hpo_max_horizon.py
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
from train_with_node import get_split, build_dataset, SEQ_LEN, N_FEATURES, FEATURES_8

MODEL_DIR = NODE_DIR / "models"
OUTPUT_DIR = NODE_DIR / "outputs"
HPO_DIR = SCRIPT_DIR  # hpo/ directory

MAXHORIZON_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/maxhorizon")

# HPO constants
CONFIG = 'D2'
HORIZON_KEY = 'h2s'
HORIZON_STEPS = 200  # 2s / 0.01 = 200
EPOCHS = 80
PATIENCE = 15
N_TRIALS = 50
DANGER_THRESHOLD = 0.7


# ============== Model definitions (self-contained, configurable) ==============

class PhysicsProjection(nn.Module):
    """Projects NODE backbone features to physics embedding."""
    def __init__(self, node_backbone, backbone_dim=128, out_dim=32):
        super().__init__()
        self.node_backbone = node_backbone
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, x_norm):
        B, T, _ = x_norm.shape
        physics = self.node_backbone(x_norm.reshape(B * T, -1))
        physics = physics.reshape(B, T, -1)
        return self.proj(physics)


class LSTM_Physics_HPO(nn.Module):
    """LSTM+NODE with configurable hyperparameters."""
    def __init__(self, physics_proj, input_size, hidden_size=64,
                 num_layers=1, fc_hidden=32, dropout=0.2, output_size=3):
        super().__init__()
        self.physics_proj = physics_proj
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, output_size),
        )

    def forward(self, x_8feat, x_8feat_node_norm):
        physics = self.physics_proj(x_8feat_node_norm)
        x_aug = torch.cat([x_8feat, physics], dim=2)
        out, _ = self.lstm(x_aug)
        return self.fc(out[:, -1, :])


class PatchTST_Physics_HPO(nn.Module):
    """PatchTST+NODE with configurable hyperparameters."""
    def __init__(self, physics_proj, seq_len, n_features, patch_size=15,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                 output_size=3):
        super().__init__()
        self.physics_proj = physics_proj
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, x_8feat, x_8feat_node_norm):
        physics = self.physics_proj(x_8feat_node_norm)
        x_aug = torch.cat([x_8feat, physics], dim=2)
        B = x_aug.shape[0]
        x = x_aug[:, :self.n_patches * self.patch_size, :].reshape(
            B, self.n_patches, -1)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.head(self.transformer(x)[:, 0])


# ============== Loss ==============

def quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9]):
    losses = []
    for i, q in enumerate(quantiles):
        err = target - pred[:, i]
        losses.append(torch.mean(torch.max(q * err, (q - 1) * err)))
    return sum(losses)


# ============== Shared resources ==============

def load_shared_resources():
    """Load NODE backbone (frozen), scalers, and pre-build datasets. Done once."""
    print("Loading shared resources...")

    # NODE backbone
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

    # Scalers
    with open(MODEL_DIR / "scaler_node_final_8feat.pkl", 'rb') as f:
        scaler_node = pickle.load(f)
    with open(MAXHORIZON_DIR / "models" / f"scaler_{CONFIG}_{HORIZON_KEY}.pkl", 'rb') as f:
        scaler_8 = pickle.load(f)

    # Dataset
    train_files, test_files = get_split(CONFIG)
    print(f"  {CONFIG} split: {len(train_files)} train, {len(test_files)} test files")

    X_train, Y_train = build_dataset(train_files, HORIZON_STEPS, stride=10)
    X_test, Y_test = build_dataset(test_files, HORIZON_STEPS, stride=10)
    print(f"  Dataset: {len(X_train)} train, {len(X_test)} test samples")

    n_train, n_test = len(X_train), len(X_test)

    # Pre-normalize
    X_tr_8 = scaler_8.transform(X_train.reshape(-1, N_FEATURES)).reshape(
        n_train, SEQ_LEN, N_FEATURES).astype(np.float32)
    X_te_8 = scaler_8.transform(X_test.reshape(-1, N_FEATURES)).reshape(
        n_test, SEQ_LEN, N_FEATURES).astype(np.float32)

    X_tr_node = scaler_node.transform(X_train.reshape(-1, N_FEATURES)).reshape(
        n_train, SEQ_LEN, N_FEATURES).astype(np.float32)
    X_te_node = scaler_node.transform(X_test.reshape(-1, N_FEATURES)).reshape(
        n_test, SEQ_LEN, N_FEATURES).astype(np.float32)

    # Binary labels for AUC-PR
    Y_test_binary = (Y_test >= DANGER_THRESHOLD).astype(np.int32)
    n_pos = Y_test_binary.sum()
    print(f"  Test danger samples: {n_pos}/{len(Y_test)} ({n_pos/len(Y_test)*100:.1f}%)")

    shared = {
        'backbone': backbone,
        'backbone_dim': backbone_dim,
        'X_tr_8': torch.FloatTensor(X_tr_8),
        'X_tr_node': torch.FloatTensor(X_tr_node),
        'Y_train': torch.FloatTensor(Y_train),
        'X_te_8': torch.FloatTensor(X_te_8),
        'X_te_node': torch.FloatTensor(X_te_node),
        'Y_test': torch.FloatTensor(Y_test),
        'Y_test_np': Y_test,
        'Y_test_binary': Y_test_binary,
    }
    print("  Done.\n")
    return shared


# ============== Model creation ==============

def create_model(arch, params, shared):
    """Create model from params dict. Returns model on device."""
    physics_proj_dim = params['physics_proj_dim']
    n_features_aug = N_FEATURES + physics_proj_dim

    physics_proj = PhysicsProjection(
        shared['backbone'], shared['backbone_dim'], out_dim=physics_proj_dim
    ).to(device)
    # Backbone already frozen, but ensure
    for p in physics_proj.node_backbone.parameters():
        p.requires_grad = False

    if arch == 'LSTM':
        model = LSTM_Physics_HPO(
            physics_proj, input_size=n_features_aug,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            fc_hidden=params['fc_hidden'],
            dropout=params['dropout'],
        ).to(device)

    elif arch == 'PatchTST':
        model = PatchTST_Physics_HPO(
            physics_proj, seq_len=SEQ_LEN,
            n_features=n_features_aug,
            patch_size=params['patch_size'],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dim_feedforward=params['d_model'] * params['dim_ff_mult'],
        ).to(device)

    return model


def sample_params(trial, arch):
    """Sample hyperparameters for a trial."""
    params = {
        'physics_proj_dim': trial.suggest_categorical('physics_proj_dim', [16, 32, 48, 64]),
        'lr_projection': trial.suggest_float('lr_projection', 3e-4, 3e-3, log=True),
        'lr_other': trial.suggest_float('lr_other', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    }

    if arch == 'LSTM':
        params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2])
        params['fc_hidden'] = trial.suggest_categorical('fc_hidden', [32, 64])
        params['dropout'] = trial.suggest_float('dropout', 0.05, 0.4)

    elif arch == 'PatchTST':
        params['patch_size'] = trial.suggest_categorical('patch_size', [10, 15, 25, 30])
        params['d_model'] = trial.suggest_categorical('d_model', [32, 64, 128])
        # nhead must divide d_model
        valid_nheads = [h for h in [1, 2, 4, 8] if params['d_model'] % h == 0]
        params['nhead'] = trial.suggest_categorical('nhead', valid_nheads)
        params['num_layers'] = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
        params['dim_ff_mult'] = trial.suggest_categorical('dim_ff_mult', [2, 4])

    return params


def compute_auc_pr(model, test_loader, y_binary):
    """Compute AUC-PR using Q50 predictions as scores."""
    model.eval()
    preds_q50 = []
    with torch.no_grad():
        for X8, Xn, Y in test_loader:
            X8, Xn = X8.to(device), Xn.to(device)
            pred = model(X8, Xn)
            preds_q50.append(pred[:, 1].cpu().numpy())  # Q50

    p50 = np.concatenate(preds_q50)
    # Clamp to avoid edge cases
    p50 = np.clip(p50, 0.0, 1.5)

    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return 0.0  # degenerate case

    return float(average_precision_score(y_binary, p50))


# ============== Objective ==============

def objective(trial, arch, shared):
    """Optuna objective: train model and return best AUC-PR."""
    try:
        params = sample_params(trial, arch)
        model = create_model(arch, params, shared)
    except Exception as e:
        print(f"  Trial {trial.number}: model creation failed: {e}")
        return -1.0

    batch_size = params['batch_size']

    train_ds = TensorDataset(shared['X_tr_8'], shared['X_tr_node'], shared['Y_train'])
    test_ds = TensorDataset(shared['X_te_8'], shared['X_te_node'], shared['Y_test'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Differential LR: projection vs other (backbone frozen)
    proj_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'physics_proj.proj' in name:
            proj_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': proj_params, 'lr': params['lr_projection']},
        {'params': other_params, 'lr': params['lr_other']},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_auc = -1.0
    no_improve = 0

    try:
        for epoch in range(EPOCHS):
            # Train
            model.train()
            for X8, Xn, Y in train_loader:
                X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                optimizer.zero_grad()
                pred = model(X8, Xn)
                loss = quantile_loss(pred, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Eval: val loss for scheduler + AUC-PR
            model.eval()
            v_loss = 0
            with torch.no_grad():
                for X8, Xn, Y in test_loader:
                    X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                    v_loss += quantile_loss(model(X8, Xn), Y).item()
            v_loss /= len(test_loader)
            scheduler.step(v_loss)

            auc = compute_auc_pr(model, test_loader, shared['Y_test_binary'])

            # Report to Optuna for pruning
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
            print(f"  Trial {trial.number}: OOM/MPS error")
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            return -1.0
        raise

    return best_auc


# ============== Retrain best ==============

def retrain_best(arch, best_params, shared):
    """Retrain best config with full epochs, return (model, auc_pr)."""
    print(f"\n  Retraining best {arch} config (full {EPOCHS} epochs)...")

    model = create_model(arch, best_params, shared)

    batch_size = best_params['batch_size']
    train_ds = TensorDataset(shared['X_tr_8'], shared['X_tr_node'], shared['Y_train'])
    test_ds = TensorDataset(shared['X_te_8'], shared['X_te_node'], shared['Y_test'])
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

    optimizer = torch.optim.Adam([
        {'params': proj_params, 'lr': best_params['lr_projection']},
        {'params': other_params, 'lr': best_params['lr_other']},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

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

        auc = compute_auc_pr(model, test_loader, shared['Y_test_binary'])

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
    print(f"HPO — LSTM+NODE & PatchTST+NODE (max horizon, {CONFIG}, {HORIZON_KEY})")
    print(f"Device: {device}")
    print(f"Metric: AUC-PR (danger >= {DANGER_THRESHOLD})")
    print(f"Trials per arch: {N_TRIALS}")
    print("=" * 70)

    shared = load_shared_resources()

    db_path = OUTPUT_DIR / "hpo_max_horizon.db"
    storage = f"sqlite:///{db_path}"

    results = {}

    for arch in ['LSTM', 'PatchTST']:
        print(f"\n{'='*60}")
        print(f"  HPO: {arch}+NODE")
        print(f"{'='*60}")

        study_name = f"hpo_{arch}_NODE_{CONFIG}_{HORIZON_KEY}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=15),
            load_if_exists=True,
        )

        n_existing = len(study.trials)
        n_remaining = max(0, N_TRIALS - n_existing)
        if n_existing > 0:
            print(f"  Resuming: {n_existing} trials done, {n_remaining} remaining")

        if n_remaining > 0:
            study.optimize(
                lambda trial: objective(trial, arch, shared),
                n_trials=n_remaining,
                show_progress_bar=True,
            )

        # Best trial
        best = study.best_trial
        print(f"\n  Best trial #{best.number}: AUC-PR = {best.value:.4f}")
        print(f"  Params: {best.params}")

        # Retrain best
        model, final_auc = retrain_best(arch, best.params, shared)
        save_path = MODEL_DIR / f"hpo_{arch}_{HORIZON_KEY}_{CONFIG}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"  Saved: {save_path.name} (AUC-PR={final_auc:.4f})")

        results[arch] = {
            'best_trial': best.number,
            'best_auc_pr_search': float(best.value),
            'best_auc_pr_retrain': float(final_auc),
            'best_params': best.params,
            'n_trials': len(study.trials),
            'n_pruned': len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.PRUNED]),
        }

    # Save results
    results_path = OUTPUT_DIR / "results_hpo_max_horizon.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print("HPO SUMMARY")
    print(f"{'='*70}")
    print(f"{'Arch':<12} {'Search AUC-PR':<16} {'Retrain AUC-PR':<16} "
          f"{'Trials':<8} {'Pruned':<8}")
    print("-" * 60)

    for arch in ['LSTM', 'PatchTST']:
        r = results[arch]
        print(f"{arch:<12} {r['best_auc_pr_search']:<16.4f} "
              f"{r['best_auc_pr_retrain']:<16.4f} "
              f"{r['n_trials']:<8} {r['n_pruned']:<8}")

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results: {results_path}")
    print(f"DB: {db_path}")
    print(f"Models: {MODEL_DIR}/hpo_*_{HORIZON_KEY}_{CONFIG}.pt")
    print("DONE!")


if __name__ == '__main__':
    main()
