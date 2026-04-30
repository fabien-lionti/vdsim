#!/usr/bin/env python3
"""
HPO PatchTST — informed by LSTM best params.

From LSTM rounds: seq_len=200-250, gelu, proj_depth=2, weight_decay~0, Adam.
Explores PatchTST-specific: patch_size, d_model, nhead, num_layers, dim_ff_mult.
Also explores seq_len [150, 200, 250, 300].
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
from train_with_node import get_split, N_FEATURES, FEATURES_8

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

# From LSTM best: proj_depth=2, gelu
PHYSICS_PROJ_DIM = 16  # LSTM best used 16

SEQ_LENS = [150, 200, 250, 300]

ACTIVATIONS = {
    'relu': nn.ReLU,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}


# ============== Variable-length build_dataset ==============

def build_dataset_variable(files, horizon_steps, seq_len, stride=10):
    X_all, Y_all = [], []
    for f in files:
        try:
            df = load_scenario(f)
            if df is None or len(df) < seq_len + horizon_steps + 10:
                continue
            data = df[FEATURES_8].values.astype(np.float32)
            ltr = np.abs(df['LTRmax'].values).astype(np.float32)
            for i in range(seq_len, len(df) - horizon_steps, stride):
                X_all.append(data[i - seq_len:i])
                Y_all.append(np.max(ltr[i:i + horizon_steps]))
        except:
            continue
    return np.array(X_all), np.array(Y_all, dtype=np.float32)


# ============== Models ==============

class PhysicsProjection_Deep(nn.Module):
    """Depth-2 physics projection with gelu (from LSTM best)."""
    def __init__(self, node_backbone, backbone_dim=128, out_dim=16):
        super().__init__()
        self.node_backbone = node_backbone
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.GELU(),
            nn.Linear(backbone_dim // 2, out_dim),
            nn.GELU(),
        )

    def forward(self, x_norm):
        B, T, _ = x_norm.shape
        physics = self.node_backbone(x_norm.reshape(B * T, -1))
        physics = physics.reshape(B, T, -1)
        return self.proj(physics)


class PatchTST_Physics(nn.Module):
    def __init__(self, physics_proj, seq_len, n_features, patch_size=10,
                 d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
                 dropout=0.1, output_size=3):
        super().__init__()
        self.physics_proj = physics_proj
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

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

    datasets = {}
    for seq_len in SEQ_LENS:
        X_train, Y_train = build_dataset_variable(train_files, HORIZON_STEPS, seq_len, stride=10)
        X_test, Y_test = build_dataset_variable(test_files, HORIZON_STEPS, seq_len, stride=10)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"  seq_len={seq_len}: SKIPPED")
            continue

        n_train, n_test = len(X_train), len(X_test)

        X_tr_8 = scaler_8.transform(X_train.reshape(-1, N_FEATURES)).reshape(
            n_train, seq_len, N_FEATURES).astype(np.float32)
        X_te_8 = scaler_8.transform(X_test.reshape(-1, N_FEATURES)).reshape(
            n_test, seq_len, N_FEATURES).astype(np.float32)
        X_tr_node = scaler_node.transform(X_train.reshape(-1, N_FEATURES)).reshape(
            n_train, seq_len, N_FEATURES).astype(np.float32)
        X_te_node = scaler_node.transform(X_test.reshape(-1, N_FEATURES)).reshape(
            n_test, seq_len, N_FEATURES).astype(np.float32)

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
    seq_len = trial.suggest_categorical('seq_len', [s for s in SEQ_LENS if s in shared['datasets']])

    # Fixed choices that divide ALL seq_lens (150,200,250,300): 5,10,25,50
    patch_size = trial.suggest_categorical('patch_size', [5, 10, 25, 50])

    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    # 1,2,4,8 all divide 64,128,256
    nhead = trial.suggest_categorical('nhead', [1, 2, 4, 8])
    num_layers = trial.suggest_categorical('num_layers', [2, 3, 4, 6])
    dim_ff_mult = trial.suggest_categorical('dim_ff_mult', [2, 4])
    dropout = trial.suggest_float('dropout', 0.05, 0.4)

    # Training params (informed by LSTM best)
    lr_projection = trial.suggest_float('lr_projection', 3e-4, 3e-3, log=True)
    lr_other = trial.suggest_float('lr_other', 5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    # Also explore physics_proj_dim
    physics_proj_dim = trial.suggest_categorical('physics_proj_dim', [16, 32, 48])

    data = shared['datasets'].get(seq_len)
    if data is None:
        return -1.0

    n_features_aug = N_FEATURES + physics_proj_dim

    try:
        physics_proj = PhysicsProjection_Deep(
            shared['backbone'], shared['backbone_dim'], out_dim=physics_proj_dim
        ).to(device)
        for p in physics_proj.node_backbone.parameters():
            p.requires_grad = False

        model = PatchTST_Physics(
            physics_proj, seq_len=seq_len, n_features=n_features_aug,
            patch_size=patch_size, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dim_feedforward=d_model * dim_ff_mult,
            dropout=dropout,
        ).to(device)

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

    optimizer = torch.optim.Adam([
        {'params': proj_params, 'lr': lr_projection, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': lr_other, 'weight_decay': weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

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
    seq_len = best_params['seq_len']
    print(f"\n  Retraining best PatchTST (seq_len={seq_len})...")

    data = shared['datasets'][seq_len]
    physics_proj_dim = best_params['physics_proj_dim']
    n_features_aug = N_FEATURES + physics_proj_dim

    physics_proj = PhysicsProjection_Deep(
        shared['backbone'], shared['backbone_dim'], out_dim=physics_proj_dim
    ).to(device)
    for p in physics_proj.node_backbone.parameters():
        p.requires_grad = False

    model = PatchTST_Physics(
        physics_proj, seq_len=seq_len, n_features=n_features_aug,
        patch_size=best_params['patch_size'],
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers'],
        dim_feedforward=best_params['d_model'] * best_params['dim_ff_mult'],
        dropout=best_params['dropout'],
    ).to(device)

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

    optimizer = torch.optim.Adam([
        {'params': proj_params, 'lr': best_params['lr_projection'],
         'weight_decay': best_params['weight_decay']},
        {'params': other_params, 'lr': best_params['lr_other'],
         'weight_decay': best_params['weight_decay']},
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
    print("HPO PatchTST — informed by LSTM best params")
    print(f"Device: {device}")
    print(f"Seq lens: {SEQ_LENS}")
    print(f"From LSTM: gelu, proj_depth=2, Adam, weight_decay~0")
    print(f"Exploring: seq_len, patch_size, d_model, nhead, layers, dropout, lr")
    print("=" * 70)

    shared = load_shared_resources()

    db_path = OUTPUT_DIR / "hpo_max_horizon.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="hpo_PatchTST_NODE_D2_h2s_final_v2",
        storage=storage,
        direction='maximize',
        sampler=TPESampler(seed=2026),
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
    save_path = MODEL_DIR / f"hpo_PatchTST_{HORIZON_KEY}_{CONFIG}_final.pt"
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
        'lstm_best_auc': 0.892,
    }

    results_path = OUTPUT_DIR / "results_hpo_patchtst_final.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t_start
    print(f"\nPatchTST HPO done in {elapsed/60:.1f} min")
    print(f"LSTM best: 0.892 → PatchTST best: {final_auc:.4f}")
    print("DONE!")


if __name__ == '__main__':
    main()
