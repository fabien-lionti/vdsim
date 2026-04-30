#!/usr/bin/env python3
"""Entraîne LSTM + features NODE (backbone figé) sur données réelles."""

import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    DT, N_FEATURES, FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LR, PATIENCE, create_sequences_max, load_all_scenarios,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)

# --- NODE config ---
NODE_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/node_v1")
NODE_MODEL_PATH = NODE_DIR / "models" / "node_D4_clean.pt"
NODE_SCALER_PATH = NODE_DIR / "models" / "scaler_node_D4_clean.pkl"
# NODE a été entraîné sur 7 features (sans psi, sans LTR)
NODE_FEATURES_7 = ['vx', 'vy', 'psi_dot', 'phi', 'theta', 'delta_f', 'delta_f_dot']
# Ces 7 features correspondent aux indices [0,1,2,3,4,5,6] dans nos FEATURES à 9
# (FEATURES = [vx, vy, psi_dot, phi, theta, delta_f, delta_f_dot, LTR_current, dLTR_dt])
NODE_FEAT_INDICES = [FEATURES.index(f) for f in NODE_FEATURES_7]  # [0,1,2,3,4,5,6]

NODE_HIDDEN = 256   # hidden_size de VehicleDynamicsNet
NODE_MID = 128      # mid_size (output de shared)
PHYSICS_PROJ_DIM = 16

# --- Training config ---
SEEDS = [0, 1, 2]
CONFIG = ('D4', 2)
SEQ_LEN = 150   # on garde celui de notre baseline pour comparaison directe

OUTPUT_FILE = OUTPUT_DIR / "lstm_node_real_results.json"


# ============== NODE shared backbone (for feature extraction) ==============

class NodeSharedBackbone(nn.Module):
    """Reproduit VehicleDynamicsNet.shared (128-dim output)."""
    def __init__(self, n_input=8, hidden_size=NODE_HIDDEN, mid_size=NODE_MID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, mid_size), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


def load_node_backbone():
    """Charge la `shared` du NODE et retourne le backbone figé + le scaler."""
    ckpt = torch.load(NODE_MODEL_PATH, map_location=device, weights_only=False)
    # Le checkpoint contient state_dict du VehicleDynamicsNet entier
    # On extrait les poids de `shared.*`
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    # Détection du n_input à partir du premier Linear
    first_key = next(k for k in sd if 'shared' in k and 'weight' in k and '0.weight' in k)
    n_input = sd[first_key].shape[1]
    print(f"  NODE shared : n_input={n_input}, backbone_dim={NODE_MID}")

    backbone = NodeSharedBackbone(n_input=n_input)
    # Copie des poids de shared.*
    new_sd = {}
    for k, v in sd.items():
        if k.startswith('shared.'):
            new_sd[k.replace('shared.', 'net.')] = v
    missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
    print(f"  Loaded shared: missing={len(missing)}, unexpected={len(unexpected)}")
    backbone = backbone.to(device)

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    # Scaler NODE (à appliquer aux 7 features avant le backbone)
    with open(NODE_SCALER_PATH, 'rb') as f:
        node_scaler = pickle.load(f)
    return backbone, node_scaler, n_input


# ============== Model : LSTM + NODE-features ==============

class LSTMWithNode(nn.Module):
    def __init__(self, input_size_total, hidden_size=128, fc_hidden=64, dropout=0.48):
        super().__init__()
        self.lstm = nn.LSTM(input_size_total, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x_aug):
        out, _ = self.lstm(x_aug)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMPhysicsWrapper(nn.Module):
    """Pipeline complet : prépare features + augmente NODE + LSTM."""
    def __init__(self, node_backbone, node_n_input, physics_proj_dim=PHYSICS_PROJ_DIM,
                 base_feat_dim=N_FEATURES, dropout=0.48):
        super().__init__()
        self.node_backbone = node_backbone
        self.node_n_input = node_n_input
        self.proj = nn.Sequential(
            nn.Linear(NODE_MID, NODE_MID // 2),
            nn.GELU(),
            nn.Linear(NODE_MID // 2, physics_proj_dim),
            nn.GELU(),
        )
        total_feat = base_feat_dim + physics_proj_dim
        self.lstm = LSTMWithNode(input_size_total=total_feat, dropout=dropout)

    def forward(self, x_base, x_node_raw):
        # x_base : (B, T, 9) déjà normalisé (pour LSTM)
        # x_node_raw : (B, T, NODE_n_input) normalisé pour NODE
        B, T, _ = x_node_raw.shape
        with torch.no_grad():
            h = self.node_backbone(x_node_raw.reshape(B * T, -1))
        phys = self.proj(h).reshape(B, T, -1)
        x_aug = torch.cat([x_base, phys], dim=2)
        return self.lstm(x_aug)


# ============== Helpers ==============

def build_sequences(scenarios, horizon_steps):
    Xl, yl = [], []
    for s in scenarios:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if X is not None:
            Xl.append(X); yl.append(y)
    if not Xl:
        return None, None
    return np.concatenate(Xl), np.concatenate(yl)


def normalize_for_node(X_raw, node_scaler, node_n_input):
    """X_raw (seqs, T, 9) → normalise les premières node_n_input features pour le backbone NODE."""
    # NODE_FEAT_INDICES = [0..6] → on prend 7 features
    X_sub = X_raw[:, :, NODE_FEAT_INDICES[:node_n_input]]
    orig_shape = X_sub.shape
    X_sub_flat = X_sub.reshape(-1, node_n_input)
    X_sub_norm = node_scaler.transform(X_sub_flat).reshape(orig_shape)
    return X_sub_norm.astype(np.float32)


def train_physics_model(model, X_base_train, X_node_train, y_train,
                        X_base_val, X_node_val, y_val,
                        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE):
    """Training loop avec 2 inputs (base + node)."""
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    X_bt = torch.FloatTensor(X_base_train.astype(np.float32))
    X_nt = torch.FloatTensor(X_node_train)
    y_tr = torch.FloatTensor(y_train.astype(np.float32))

    X_bv = torch.FloatTensor(X_base_val.astype(np.float32))
    X_nv = torch.FloatTensor(X_node_val)
    y_va = torch.FloatTensor(y_val.astype(np.float32))

    n = len(X_bt)
    best_val = float('inf')
    best_sd = None
    no_improve = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_bt[idx].to(device)
            xn = X_nt[idx].to(device)
            yb = y_tr[idx].to(device)
            opt.zero_grad()
            pred = model(xb, xn)
            loss = weighted_mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            total += loss.item(); nb += 1
        train_loss = total / nb

        # Val
        model.eval()
        with torch.no_grad():
            preds_val = []
            for i in range(0, len(X_bv), batch_size):
                xb = X_bv[i:i+batch_size].to(device)
                xn = X_nv[i:i+batch_size].to(device)
                preds_val.append(model(xb, xn).cpu().numpy())
            preds_val = np.concatenate(preds_val)
        preds_val = np.clip(preds_val, 0, 1.5)
        weights = 1.0 + 9.0 * y_val
        val_loss = float(np.mean(weights * (preds_val - y_val) ** 2))

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"    ep {ep+1:2d}  train {train_loss:.4f}  val {val_loss:.4f}"
              f"{'  *' if no_improve==0 else ''}", flush=True)
        if no_improve >= patience:
            print(f"    Early stop at ep {ep+1}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model, {'best_val_loss': best_val, 'n_epochs': ep + 1}


def evaluate_physics(model, X_base_test, X_node_test, y_test):
    """Evaluate : apply model to test, compute metrics."""
    from sklearn.metrics import average_precision_score, r2_score, mean_squared_error
    model.eval()
    X_bt = torch.FloatTensor(X_base_test.astype(np.float32))
    X_nt = torch.FloatTensor(X_node_test)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_bt), 512):
            xb = X_bt[i:i+512].to(device)
            xn = X_nt[i:i+512].to(device)
            preds.append(model(xb, xn).cpu().numpy())
    preds = np.concatenate(preds)
    preds = np.clip(preds, 0, 1.5)

    thr = 0.6
    y_bin = (y_test > thr).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        auc_pr = float('nan')
    else:
        auc_pr = float(average_precision_score(y_bin, preds))

    pred_bin = (preds > thr).astype(int)
    tp = int(((pred_bin == 1) & (y_bin == 1)).sum())
    fn = int(((pred_bin == 0) & (y_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (y_bin == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    return {
        'rmse': float(np.sqrt(mean_squared_error(y_test, preds))),
        'r2': float(r2_score(y_test, preds)),
        'auc_pr': auc_pr,
        'recall': float(recall),
        'precision': float(precision),
        'threshold': thr,
        'n_test': int(len(y_test)),
    }


def clean_for_json(met):
    out = {}
    for k, v in met.items():
        if isinstance(v, (np.floating,)):
            fv = float(v); out[k] = None if np.isnan(fv) else fv
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


# ============== Main ==============

def main():
    print("=" * 70)
    print(f"LSTM + NODE features sur réel D4 h2s (real-only, pas de pretrain sim LSTM)")
    print(f"Seeds: {SEEDS}")
    print("=" * 70)

    horizon_steps = int(2 / DT)

    print(f"\n[1/3] Load NODE backbone pretrained on sim V2 D4...")
    node_backbone, node_scaler, node_n_input = load_node_backbone()

    print(f"\n[2/3] Load real scenarios...")
    scs, _ = load_all_scenarios(use_cache=True)
    ds, h = CONFIG
    cfg = DATASET_CONFIGS[ds]
    train_full = [s for s in scs if s['max_ltr'] <= cfg['threshold']]
    test_sc = [s for s in scs if s['max_ltr'] > cfg['threshold']]
    train_pool, val_sc = split_train_val(train_full, val_fraction=0.2, seed=42)
    print(f"  Train: {len(train_pool)}, Val: {len(val_sc)}, Test: {len(test_sc)}")

    X_train_raw, y_train = build_sequences(train_pool, horizon_steps)
    X_val_raw, y_val = build_sequences(val_sc, horizon_steps)
    X_test_raw, y_test = build_sequences(test_sc, horizon_steps)

    print(f"  Sequences: train {X_train_raw.shape}, val {X_val_raw.shape}, test {X_test_raw.shape}")

    # Base LSTM scaler (9 features, fit on train)
    base_scaler = StandardScaler()
    X_train_base = base_scaler.fit_transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val_base = base_scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_test_base = base_scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)

    # NODE scaler (déjà chargé depuis pickle) pour les 7 features NODE
    X_train_node = normalize_for_node(X_train_raw, node_scaler, node_n_input)
    X_val_node = normalize_for_node(X_val_raw, node_scaler, node_n_input)
    X_test_node = normalize_for_node(X_test_raw, node_scaler, node_n_input)

    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)

    cfg_key = f"D4_h2s"
    if cfg_key not in results:
        results[cfg_key] = []
    done_seeds = {r['seed'] for r in results[cfg_key]}

    print(f"\n[3/3] Training LSTM+NODE, 3 seeds...")
    for seed in SEEDS:
        if seed in done_seeds:
            print(f"  seed={seed}: skip"); continue
        t0 = time.time()
        print(f"  seed={seed}...", flush=True)

        torch.manual_seed(seed); np.random.seed(seed)
        model = LSTMPhysicsWrapper(
            node_backbone=node_backbone,
            node_n_input=node_n_input,
            physics_proj_dim=PHYSICS_PROJ_DIM,
            base_feat_dim=N_FEATURES,
            dropout=0.48,
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params: {n_params:,}")

        model, hist = train_physics_model(
            model, X_train_base, X_train_node, y_train,
            X_val_base, X_val_node, y_val,
            epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
        )

        met = evaluate_physics(model, X_test_base, X_test_node, y_test)
        met['seed'] = int(seed)
        met['finetune_epochs'] = int(hist['n_epochs'])
        met['best_val_loss'] = float(hist['best_val_loss'])

        dt = time.time() - t0
        print(f"       → R²={met['r2']:.3f} AUC-PR={met['auc_pr']:.3f} "
              f"Recall={met['recall']:.0%} ({dt:.0f}s)")
        results[cfg_key].append(clean_for_json(met))
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n[Résumé] D4 h2s, LSTM + NODE (sim-pretrained)")
    r2s = [r['r2'] for r in results[cfg_key]]
    aucs = [r['auc_pr'] for r in results[cfg_key]]
    print(f"  R²     : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}  (baseline real-only: 0.737)")
    print(f"  AUC-PR : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  (baseline real-only: 0.916)")


if __name__ == "__main__":
    main()
