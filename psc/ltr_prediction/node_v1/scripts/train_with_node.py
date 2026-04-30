#!/usr/bin/env python3
"""Pipeline LSTM/PatchTST avec features NODE en entrée (warm-start + fine-tune)."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, average_precision_score
import pickle
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_node import (
    VehicleDynamicsNet, load_scenario,
    train_1step, train_multistep,
    build_1step_dataset,
    N_INPUT, N_STATE, N_COMMAND, ALL_INPUT_FEATURES,
    DT, device
)

# ============== Configuration ==============

SCRIPT_DIR = Path(__file__).parent
NODE_DIR = SCRIPT_DIR.parent
MAXHORIZON_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/maxhorizon")
DATA_DIR = NODE_DIR / "data"
MODEL_DIR = NODE_DIR / "models"
OUTPUT_DIR = NODE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES_8 = ['vx', 'vy', 'psi', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
N_FEATURES = 8
N_PHYSICS_PROJ = 32
N_INPUT_AUGMENTED = N_FEATURES + N_PHYSICS_PROJ  # 8 + 32 = 40

SEQ_LEN = 150
HORIZONS = [1, 2, 4]
HORIZON_KEYS = [f'h{h}s' for h in HORIZONS]

BATCH_SIZE = 64
EPOCHS = 80
LR_FINETUNE = 3e-4
LR_BACKBONE = 1e-5  # très faible pour le backbone dégelé
PATIENCE = 15
DANGER_THRESHOLD = 0.7

DATASET_CONFIGS = {
    'D2': {'threshold': 0.7},
    'D3': {'threshold': 0.8},
    'D4': {'threshold': 0.9},
}


# ============== Model Definitions ==============

class PhysicsProjection(nn.Module):
    """Projette les features NODE (backbone output) vers N_PHYSICS_PROJ dims."""
    def __init__(self, node_backbone, backbone_dim=128, out_dim=N_PHYSICS_PROJ):
        super().__init__()
        self.node_backbone = node_backbone
        # backbone NON frozen — sera entraîné avec lr très faible
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, out_dim),
            nn.SiLU(),
        )

    def forward(self, x_norm):
        """x_norm: (batch, seq_len, 8) — features NODE normalisées"""
        B, T, _ = x_norm.shape
        physics = self.node_backbone(x_norm.reshape(B * T, -1))
        physics = physics.reshape(B, T, -1)
        return self.proj(physics)


# Modèles baseline (sans NODE)
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
    def __init__(self, seq_len=150, n_features=8, patch_size=15, d_model=64,
                 nhead=4, num_layers=2, output_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)
    def forward(self, x):
        B = x.shape[0]
        x = x[:, :self.n_patches * self.patch_size, :].reshape(B, self.n_patches, -1)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        return self.head(self.transformer(x)[:, 0])


# Physics-augmented models
class LSTM_Physics(nn.Module):
    def __init__(self, physics_proj, input_size=N_INPUT_AUGMENTED,
                 hidden_size=64, output_size=3):
        super().__init__()
        self.physics_proj = physics_proj
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, output_size))
    def forward(self, x_8feat, x_8feat_node_norm):
        physics = self.physics_proj(x_8feat_node_norm)
        x_aug = torch.cat([x_8feat, physics], dim=2)
        out, _ = self.lstm(x_aug)
        return self.fc(out[:, -1, :])


class PatchTST_Physics(nn.Module):
    def __init__(self, physics_proj, seq_len=SEQ_LEN,
                 n_features=N_INPUT_AUGMENTED, patch_size=15,
                 d_model=64, nhead=4, num_layers=2, output_size=3):
        super().__init__()
        self.physics_proj = physics_proj
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size * n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_size)
    def forward(self, x_8feat, x_8feat_node_norm):
        physics = self.physics_proj(x_8feat_node_norm)
        x_aug = torch.cat([x_8feat, physics], dim=2)
        B = x_aug.shape[0]
        x = x_aug[:, :self.n_patches * self.patch_size, :].reshape(B, self.n_patches, -1)
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

def get_split(config_name):
    threshold = DATASET_CONFIGS[config_name]['threshold']
    all_files = sorted(DATA_DIR.glob("*.csv"))
    train, test = [], []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            ltr = np.max(np.abs(df['LTRmax'].values))
            (train if ltr <= threshold else test).append(f)
        except:
            pass
    return train, test


def build_dataset(files, horizon_steps, stride=10):
    X_all, Y_all = [], []
    for f in files:
        try:
            df = load_scenario(f)
            if df is None or len(df) < SEQ_LEN + horizon_steps + 10:
                continue
            data = df[FEATURES_8].values.astype(np.float32)
            ltr = np.abs(df['LTRmax'].values).astype(np.float32)
            for i in range(SEQ_LEN, len(df) - horizon_steps, stride):
                X_all.append(data[i - SEQ_LEN:i])
                Y_all.append(np.max(ltr[i:i + horizon_steps]))
        except:
            continue
    return np.array(X_all), np.array(Y_all, dtype=np.float32)


# ============== Warm-start ==============

def warm_start_lstm(model, pretrained_path):
    old_state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    new_state = model.state_dict()
    for key in old_state:
        if key.startswith('lstm.') and 'weight_ih' in key:
            old_w = old_state[key]
            new_w = new_state[key]
            new_w.zero_()
            new_w[:, :old_w.shape[1]] = old_w
            new_state[key] = new_w
        elif key.startswith('lstm.') or key.startswith('fc.'):
            if new_state[key].shape == old_state[key].shape:
                new_state[key] = old_state[key]
    model.load_state_dict(new_state)
    return model


def warm_start_patchtst(model, pretrained_path):
    old_state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
    new_state = model.state_dict()
    for key in old_state:
        if key == 'patch_embed.weight':
            old_w = old_state[key]
            new_w = new_state[key]
            new_w.zero_()
            patch_size = model.patch_size
            for t in range(patch_size):
                old_start = t * N_FEATURES
                old_end = old_start + N_FEATURES
                new_start = t * N_INPUT_AUGMENTED
                new_end = new_start + N_FEATURES
                new_w[:, new_start:new_end] = old_w[:, old_start:old_end]
            new_state[key] = new_w
        elif key in new_state and new_state[key].shape == old_state[key].shape:
            new_state[key] = old_state[key]
    model.load_state_dict(new_state)
    return model


# ============== Training ==============

def train_finetune(model, X_train, Y_train, X_test, Y_test,
                   scaler_8, scaler_node, name=""):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable: {trainable:,}")

    n_train, n_test = len(X_train), len(X_test)

    # Normaliser avec scaler baseline (8 features)
    X_tr_8 = scaler_8.transform(X_train.reshape(-1, N_FEATURES)).reshape(
        n_train, SEQ_LEN, N_FEATURES).astype(np.float32)
    X_te_8 = scaler_8.transform(X_test.reshape(-1, N_FEATURES)).reshape(
        n_test, SEQ_LEN, N_FEATURES).astype(np.float32)

    # Normaliser avec scaler NODE (maintenant aussi 8 features)
    X_tr_node = scaler_node.transform(X_train.reshape(-1, N_FEATURES)).reshape(
        n_train, SEQ_LEN, N_FEATURES).astype(np.float32)
    X_te_node = scaler_node.transform(X_test.reshape(-1, N_FEATURES)).reshape(
        n_test, SEQ_LEN, N_FEATURES).astype(np.float32)

    train_ds = TensorDataset(torch.FloatTensor(X_tr_8), torch.FloatTensor(X_tr_node),
                             torch.FloatTensor(Y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_te_8), torch.FloatTensor(X_te_node),
                            torch.FloatTensor(Y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Differential LR: backbone très faible, le reste normal
    backbone_params, proj_params, other_params = [], [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'node_backbone' in pname:
            backbone_params.append(p)
        elif 'physics_proj.proj' in pname:
            proj_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},      # 1e-5
        {'params': proj_params, 'lr': LR_FINETUNE * 3},       # 9e-4
        {'params': other_params, 'lr': LR_FINETUNE},           # 3e-4
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss, best_state, no_improve = float('inf'), None, 0

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
        preds_q50, targets = [], []
        with torch.no_grad():
            for X8, Xn, Y in test_loader:
                X8, Xn, Y = X8.to(device), Xn.to(device), Y.to(device)
                pred = model(X8, Xn)
                v_loss += quantile_loss(pred, Y).item()
                preds_q50.append(pred[:, 1].cpu().numpy())
                targets.append(Y.cpu().numpy())

        t_loss /= len(train_loader)
        v_loss /= len(test_loader)
        scheduler.step(v_loss)

        if v_loss < best_loss:
            best_loss, best_state, no_improve = v_loss, model.state_dict().copy(), 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or no_improve >= PATIENCE:
            p50 = np.concatenate(preds_q50)
            tgt = np.concatenate(targets)
            r2 = r2_score(tgt, p50)
            print(f"      Ep {epoch:3d}: t={t_loss:.5f} v={v_loss:.5f} R²={r2:.3f} ni={no_improve}")

        if no_improve >= PATIENCE:
            print(f"      Early stop ep {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ============== Evaluation ==============

def evaluate_model(model, test_files, scaler_8, scaler_node, horizon_steps):
    model.eval()
    true_all, pred_q50, pred_q90 = [], [], []

    for f in test_files:
        try:
            df = load_scenario(f)
            if df is None:
                continue
            data = df[FEATURES_8].values.astype(np.float32)
            ltr = np.abs(df['LTRmax'].values)
        except:
            continue

        for i in range(SEQ_LEN, len(df) - horizon_steps, 50):
            true_all.append(np.max(ltr[i:i + horizon_steps]))
            x8 = scaler_8.transform(data[i - SEQ_LEN:i]).astype(np.float32)
            xn = scaler_node.transform(data[i - SEQ_LEN:i]).astype(np.float32)
            x8_t = torch.FloatTensor(x8).unsqueeze(0).to(device)
            xn_t = torch.FloatTensor(xn).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x8_t, xn_t).cpu().numpy()[0]
            pred_q50.append(pred[1])
            pred_q90.append(pred[2])

    t, p50, p90 = np.array(true_all), np.array(pred_q50), np.array(pred_q90)
    tb = (t >= DANGER_THRESHOLD).astype(int)
    pb = (p50 >= DANGER_THRESHOLD).astype(int)
    tp = np.sum(tb & pb)

    auc = float('nan')
    if 0 < tb.sum() < len(tb):
        auc = float(average_precision_score(tb, p50))

    return {
        'rmse': float(np.sqrt(mean_squared_error(t, p50))),
        'r2': float(r2_score(t, p50)),
        'auc_pr': auc,
        'recall_0.7': float(tp / max(np.sum(tb), 1)),
        'precision_0.7': float(tp / max(np.sum(pb), 1)),
        'coverage_90': float(np.mean(t <= p90)),
        'n_samples': len(t),
    }


def evaluate_baseline(model_type, pretrained_path, test_files, scaler_8, horizon_steps):
    if model_type == 'LSTM':
        model = LSTM_Baseline(N_FEATURES).to(device)
    else:
        model = PatchTST_Baseline(SEQ_LEN, N_FEATURES).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    model.eval()

    true_all, pred_q50 = [], []
    for f in test_files:
        try:
            df = load_scenario(f)
            if df is None:
                continue
            data = df[FEATURES_8].values.astype(np.float32)
            ltr = np.abs(df['LTRmax'].values)
        except:
            continue
        for i in range(SEQ_LEN, len(df) - horizon_steps, 50):
            true_all.append(np.max(ltr[i:i + horizon_steps]))
            x = scaler_8.transform(data[i - SEQ_LEN:i]).astype(np.float32)
            with torch.no_grad():
                pred_q50.append(
                    model(torch.FloatTensor(x).unsqueeze(0).to(device)).cpu().numpy()[0, 1])

    t, p = np.array(true_all), np.array(pred_q50)
    tb = (t >= DANGER_THRESHOLD).astype(int)
    pb = (p >= DANGER_THRESHOLD).astype(int)
    tp = np.sum(tb & pb)
    auc = float('nan')
    if 0 < tb.sum() < len(tb):
        auc = float(average_precision_score(tb, p))
    return {
        'rmse': float(np.sqrt(mean_squared_error(t, p))),
        'r2': float(r2_score(t, p)),
        'auc_pr': auc,
        'recall_0.7': float(tp / max(np.sum(tb), 1)),
        'precision_0.7': float(tp / max(np.sum(pb), 1)),
        'n_samples': len(t),
    }


# ============== Main ==============

def main():
    print("=" * 70)
    print("PIPELINE FINAL — NODE 8-features, unique, backbone dégelé")
    print(f"Device: {device}")
    print("=" * 70)

    # ── Step 1: Entraîner NODE 8-features sur D2 train (≤0.7) ──
    print("\n" + "=" * 60)
    print("  STEP 1: Entraîner NODE 8-features sur fichiers ≤0.7")
    print("=" * 60)

    train_d2, test_d2 = get_split('D2')
    print(f"  Train: {len(train_d2)} fichiers (≤0.7)")

    # Vérification
    assert N_INPUT == 8, f"NODE devrait avoir 8 features, a {N_INPUT}"
    print(f"  NODE features: {ALL_INPUT_FEATURES} ({N_INPUT} dims)")

    # Build 1-step dataset
    t0 = time.time()
    X_train, Y_train, L_train, _ = build_1step_dataset(train_d2, stride=5)

    # Split interne 80/20
    np.random.seed(42)
    n = len(X_train)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    scaler_node = StandardScaler()
    scaler_node.fit(X_train[idx[:split]])
    print(f"  1-step: {n} samples ({time.time()-t0:.1f}s)")

    model = VehicleDynamicsNet(N_INPUT, N_STATE, hidden_size=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    model = train_1step(model, X_train[idx[:split]], Y_train[idx[:split]],
                        L_train[idx[:split]], X_train[idx[split:]],
                        Y_train[idx[split:]], L_train[idx[split:]], scaler_node)

    # Phase 2: multi-step
    np.random.seed(42)
    file_idx = np.random.permutation(len(train_d2))
    file_split = int(0.8 * len(train_d2))
    model = train_multistep(model,
                            [train_d2[i] for i in file_idx[:file_split]],
                            [train_d2[i] for i in file_idx[file_split:]],
                            scaler_node)

    # Save
    torch.save(model.state_dict(), MODEL_DIR / "node_final_8feat.pt")
    with open(MODEL_DIR / "scaler_node_final_8feat.pkl", 'wb') as f:
        pickle.dump(scaler_node, f)
    print(f"  Saved: node_final_8feat.pt")

    # Verify backbone dim
    with torch.no_grad():
        feat = model.shared(torch.randn(1, N_INPUT).to(device))
    backbone_dim = feat.shape[1]
    print(f"  Backbone output dim: {backbone_dim}")

    # ── Step 2: Physics finetune pour chaque config ──
    all_results = {}

    for config_name in ['D2', 'D3', 'D4']:
        print(f"\n{'='*60}")
        print(f"  STEP 2: Fine-tune {config_name}")
        print(f"{'='*60}")

        train_files, test_files = get_split(config_name)
        print(f"  {len(train_files)} train, {len(test_files)} test")

        # Vérification no leakage
        node_train_set = set(f.name for f in train_d2)
        test_set = set(f.name for f in test_files)
        leaked = node_train_set & test_set
        print(f"  NODE leakage check: {len(leaked)} fichiers test vus par NODE")
        assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} fichiers!"

        config_results = {}

        for model_type in ['LSTM', 'PatchTST']:
            print(f"\n  --- {model_type} ---")

            for horizon, hkey in zip(HORIZONS, HORIZON_KEYS):
                horizon_steps = int(horizon / DT)

                ta_path = MAXHORIZON_DIR / "models" / f"{model_type}_{config_name}_{hkey}.pt"
                scaler_path = MAXHORIZON_DIR / "models" / f"scaler_{config_name}_{hkey}.pkl"

                if not ta_path.exists():
                    print(f"    {hkey}: SKIP")
                    continue

                with open(scaler_path, 'rb') as f:
                    scaler_8 = pickle.load(f)

                X_train_ft, Y_train_ft = build_dataset(train_files, horizon_steps, stride=10)
                X_test_ft, Y_test_ft = build_dataset(test_files, horizon_steps, stride=10)
                print(f"    {hkey}: {len(X_train_ft)} train, {len(X_test_ft)} test")

                # Baseline
                res_base = evaluate_baseline(model_type, ta_path, test_files,
                                             scaler_8, horizon_steps)
                print(f"      Base: R²={res_base['r2']:.3f}  AUC-PR={res_base['auc_pr']:.4f}")

                # Create model with UNFROZEN backbone
                physics_proj = PhysicsProjection(model.shared, backbone_dim).to(device)
                if model_type == 'LSTM':
                    ft_model = LSTM_Physics(physics_proj).to(device)
                    ft_model = warm_start_lstm(ft_model, ta_path)
                else:
                    ft_model = PatchTST_Physics(physics_proj).to(device)
                    ft_model = warm_start_patchtst(ft_model, ta_path)

                ft_model = train_finetune(ft_model, X_train_ft, Y_train_ft,
                                          X_test_ft, Y_test_ft,
                                          scaler_8, scaler_node,
                                          name=f"{model_type} {config_name} {hkey}")

                # Save
                torch.save(ft_model.state_dict(),
                           MODEL_DIR / f"final_{model_type}_{hkey}_{config_name}.pt")

                # Evaluate
                res_ft = evaluate_model(ft_model, test_files, scaler_8,
                                        scaler_node, horizon_steps)
                delta = res_ft['auc_pr'] - res_base['auc_pr']
                print(f"      Fine: R²={res_ft['r2']:.3f}  AUC-PR={res_ft['auc_pr']:.4f}  "
                      f"Delta={delta:+.4f}")

                config_results[f"{model_type}_{hkey}"] = {
                    'baseline': res_base,
                    'finetuned': res_ft,
                    'delta_auc_pr': delta,
                }

        all_results[config_name] = config_results

    # ── Final table ──
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX — NODE 8-feat, unique, backbone dégelé")
    print("=" * 70)

    print(f"\n{'Config':<8} {'Model':<18} {'AUC base':<12} {'AUC ft':<12} {'Delta':<10} "
          f"{'R² base':<10} {'R² ft':<10}")
    print("-" * 80)

    for config_name in ['D2', 'D3', 'D4']:
        for key in sorted(all_results.get(config_name, {}).keys()):
            r = all_results[config_name][key]
            b, f_ = r['baseline'], r['finetuned']
            print(f"{config_name:<8} {key:<18} {b['auc_pr']:<12.4f} {f_['auc_pr']:<12.4f} "
                  f"{r['delta_auc_pr']:+10.4f} {b['r2']:<10.3f} {f_['r2']:<10.3f}")

    with open(OUTPUT_DIR / "results_final.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSauvegardé: {OUTPUT_DIR / 'results_final.json'}")
    print("TERMINÉ!")


if __name__ == '__main__':
    main()
