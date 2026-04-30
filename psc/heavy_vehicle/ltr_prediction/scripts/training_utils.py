#!/usr/bin/env python3
"""Utilitaires d'entraînement avec protocole train/val/test rigoureux (early stopping sur val)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, average_precision_score


DEFAULT_EPOCHS = 80
DEFAULT_PATIENCE = 15
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 64
DEFAULT_DANGER = 0.7


def split_train_val(scenarios, val_fraction=0.2, seed=42):
    """
    Split une liste de scénarios en train'/val par scénario.

    Args:
        scenarios: list of dicts with 'df' and 'max_ltr' (as produced by
                   load_all_scenarios in train_real_data.py)
        val_fraction: fraction of scenarios to hold out for validation
        seed: random seed for reproducibility

    Returns:
        (train_sc, val_sc): two disjoint lists of scenarios
    """
    rng = np.random.default_rng(seed)
    n = len(scenarios)
    n_val = max(1, int(val_fraction * n))
    perm = rng.permutation(n)
    val_idx = set(perm[:n_val].tolist())
    train_sc = [scenarios[i] for i in range(n) if i not in val_idx]
    val_sc = [scenarios[i] for i in range(n) if i in val_idx]
    return train_sc, val_sc


def weighted_mse_loss(pred, target, alpha=9.0):
    """Weighted MSE: (1 + alpha*y) * (pred - y)²."""
    weight = 1.0 + alpha * target
    return (weight * (pred - target) ** 2).mean()


def train_model_clean(
    model,
    X_train, y_train,
    X_val, y_val,
    device,
    loss_fn=None,
    epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_BATCH,
    weight_decay=1e-5,
    grad_clip=1.0,
    verbose=False,
):
    """
    Entraîne un modèle avec early stopping sur le VAL set (pas le test).

    Le meilleur état (best_state) est celui qui minimise la val loss.
    Le test set n'est JAMAIS touché ici.

    Args:
        model: nn.Module à entraîner
        X_train, y_train: données d'entraînement (numpy arrays normalisés)
        X_val, y_val: données de validation (numpy arrays normalisés)
        device: torch device
        loss_fn: fonction de perte. Si None, utilise weighted_mse_loss.
        epochs, patience, lr, batch_size, weight_decay, grad_clip: hyperparams
        verbose: afficher la val_loss à chaque époque

    Returns:
        (model, history): modèle avec best_state chargé + dict des losses
    """
    if loss_fn is None:
        loss_fn = weighted_mse_loss

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_tr = torch.FloatTensor(X_train.astype(np.float32))
    # NOTE: PAS de unsqueeze(1) sur y_tr. Le modèle sort en [batch] après squeeze(-1).
    # Si y_tr est [batch,1], le broadcast donne [batch,batch] et la loss est diluée.
    # Ce bug existait dans train_real_data.py (ancienne version) et cassait les
    # résultats (R²=-1). colab_train.py était correct sans unsqueeze.
    y_tr = torch.FloatTensor(y_train.astype(np.float32))
    X_va_np = X_val.astype(np.float32)
    y_va_np = y_val.astype(np.float32)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss_avg = train_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            preds_val = np.concatenate([
                model(torch.FloatTensor(X_va_np[i:i + 512]).to(device)).cpu().numpy()
                for i in range(0, len(X_va_np), 512)
            ])
        # Val loss = même forme que la training loss (weighted MSE).
        # Sinon le critère d'arrêt est incohérent avec la fonction de coût
        # et l'early stopping arrête avant que le modèle ne se spécialise
        # sur les LTR élevés (car MSE pure est minimisée en prédisant la moyenne).
        preds_val_clip = np.clip(preds_val, 0, 1.5)
        weights_val = 1.0 + 9.0 * y_va_np
        val_loss = float(np.mean(weights_val * (preds_val_clip - y_va_np) ** 2))

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss)

        if verbose:
            print(f"    epoch {epoch+1:3d}: train={train_loss_avg:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history['best_val_loss'] = best_val_loss
    history['n_epochs'] = len(history['val_loss'])
    return model, history


def predict(model, X, device, clip=(0, 1.5), batch_size=512):
    """Inférence batchée + clip."""
    model.eval()
    X_np = X.astype(np.float32)
    with torch.no_grad():
        preds = np.concatenate([
            model(torch.FloatTensor(X_np[i:i + batch_size]).to(device)).cpu().numpy()
            for i in range(0, len(X_np), batch_size)
        ])
    return np.clip(preds, clip[0], clip[1])


def select_best_threshold(preds, y, danger=DEFAULT_DANGER, grid=None):
    """
    Balaye un seuil sur (preds, y) et renvoie celui qui maximise F1.

    À appeler UNIQUEMENT sur le jeu de validation, jamais sur le test.
    Ensuite passer le seuil retourné à evaluate(..., threshold=...).
    """
    if grid is None:
        grid = np.arange(0.2, 0.8, 0.05)
    y_bin = (y >= danger).astype(int)
    n_danger = int(y_bin.sum())
    best_f1, best_thresh = 0.0, 0.5
    for t in grid:
        det = preds >= t
        tp = int(np.sum(det & (y_bin == 1)))
        rec = tp / max(n_danger, 1)
        prec = tp / max(int(np.sum(det)), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)
    return best_thresh, float(best_f1)


def evaluate(model, X_test, y_test, device, danger=DEFAULT_DANGER, clip=(0, 1.5),
             threshold=None):
    """
    Évalue un modèle sur le test set. À appeler UNE SEULE FOIS.

    Args:
        threshold: seuil de décision figé (idéalement issu de select_best_threshold
                   appelé sur le val set). Si None, un sweep de seuil est fait sur
                   les prédictions test — DEPRECATED (biais optimiste), conservé
                   pour compatibilité descendante avec les anciens scripts.

    Returns:
        dict with rmse, r2, auc_pr, recall, precision, f1, threshold, n_test, n_danger
    """
    preds = predict(model, X_test, device, clip=clip)
    y = y_test
    r2 = float(r2_score(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    y_bin = (y >= danger).astype(int)
    n_danger = int(y_bin.sum())
    auc_pr = float('nan')
    if 0 < n_danger < len(y_bin):
        auc_pr = float(average_precision_score(y_bin, preds))

    if threshold is None:
        # DEPRECATED path — sweep sur test, biais optimiste. Utiliser
        # select_best_threshold(preds_val, y_val) puis passer threshold=...
        best_thresh, best_f1 = select_best_threshold(preds, y, danger=danger)
    else:
        best_thresh = float(threshold)
        det = preds >= best_thresh
        tp = int(np.sum(det & (y_bin == 1)))
        rec = tp / max(n_danger, 1)
        prec = tp / max(int(np.sum(det)), 1)
        best_f1 = 2 * prec * rec / max(prec + rec, 1e-10)

    det = preds >= best_thresh
    tp = int(np.sum(det & (y_bin == 1)))
    recall = tp / max(n_danger, 1)
    precision = tp / max(int(np.sum(det)), 1)

    return {
        'rmse': rmse, 'r2': r2, 'auc_pr': auc_pr,
        'recall': float(recall), 'precision': float(precision),
        'f1': float(best_f1), 'threshold': float(best_thresh),
        'n_test': int(len(y)), 'n_danger': n_danger,
    }


def build_sequences_from_scenarios(scenarios, horizon_steps, seq_len, features, target_col,
                                    stride=10):
    """
    Concatène les séquences X/y de plusieurs scénarios.

    Args:
        scenarios: liste de dicts {'df': pd.DataFrame, ...}
        horizon_steps: nombre de pas d'horizon de prédiction
        seq_len: longueur de la fenêtre d'observation
        features: liste des noms de colonnes à utiliser comme features
        target_col: nom de la colonne cible (e.g. 'LTRmax')
        stride: pas entre deux fenêtres consécutives

    Returns:
        (X, y) concaténés, ou (None, None) si vide
    """
    X_list, y_list = [], []
    for s in scenarios:
        df = s['df']
        feats = df[features].values
        tgt = df[target_col].values
        for i in range(0, len(df) - seq_len - horizon_steps, stride):
            X_list.append(feats[i:i + seq_len])
            y_list.append(np.max(tgt[i + seq_len:i + seq_len + horizon_steps]))
    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list)


def save_scenarios_cache(scenarios, cache_path):
    """Sauvegarde les scénarios en pickle pour accélérer les re-runs."""
    import pickle
    with open(cache_path, 'wb') as f:
        pickle.dump(scenarios, f)


def load_scenarios_cache(cache_path):
    """Charge un cache de scénarios. Returns None if not found."""
    import pickle
    from pathlib import Path
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None
