#!/usr/bin/env python3
"""
Ablation: impact de la quantité de données réelles sur les performances.

Protocole (rigoureux):
  - Split scénarios selon OOD (D2 ou D4): train_full vs test
  - Split train_full en train_pool + val_fixed (val fixe, jamais sampled)
  - Pour chaque fraction f ∈ {25%, 50%, 75%}:
      - Stratégie 1 (uniform): random choice dans train_pool, 3 seeds
      - Stratégie 2 (progressive_ltr): les N scénarios avec le plus faible LTR_max
          (reflète un plan d'essais DGA par ordre croissant de risque), 3 seeds
                                      sur init du training
  - Early stopping sur val_fixed (JAMAIS sur test)
  - Test évalué une seule fois après training

Points 0% (sim-only) et 100% (full real) sont dans le rapport existant.

Configs: D2 h2s + D4 h2s → 3 × 2 × 3 × 2 = 36 runs.
Sauvegarde incrémentale (resume possible).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import time as time_module

sys.path.insert(0, str(Path(__file__).parent))
from train_real_data import (
    LSTM, create_sequences_max, load_all_scenarios,
    DT, N_FEATURES, DATASET_CONFIGS, device, OUTPUT_DIR,
    BATCH_SIZE, EPOCHS, LR, PATIENCE,
)
from training_utils import (
    train_model_clean, evaluate, split_train_val, weighted_mse_loss,
)

FRACTIONS = [0.25, 0.50, 0.75]
SEEDS = [0, 1, 2]
STRATEGIES = ['uniform', 'progressive_ltr']
CONFIGS = [('D2', 2), ('D4', 2)]

VAL_FRACTION = 0.2
VAL_SEED = 42

OUTPUT_FILE = OUTPUT_DIR / "data_ablation_results.json"


def build_sequences(scenarios, horizon_steps):
    Xl, yl = [], []
    for s in scenarios:
        X, y = create_sequences_max(s['df'], horizon_steps)
        if X is not None:
            Xl.append(X)
            yl.append(y)
    if not Xl:
        return None, None
    return np.concatenate(Xl), np.concatenate(yl)


def build_test_split(scenarios, dataset_name, horizon_steps):
    """Split scenarios -> (train_pool, val_sc, X_test_raw, y_test).

    val_sc est fixe pour toutes les runs de cette config (seed=VAL_SEED).
    train_pool est ce dans quoi on va sampler selon la fraction.
    X_test, y_test sont les séquences du test set (non normalisées).
    """
    cfg = DATASET_CONFIGS[dataset_name]
    if cfg['ood']:
        train_full = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
        test_sc = [s for s in scenarios if s['max_ltr'] > cfg['threshold']]
    else:
        eligible = [s for s in scenarios if s['max_ltr'] <= cfg['threshold']]
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(eligible))
        split = int(0.8 * len(eligible))
        train_full = [eligible[i] for i in perm[:split]]
        test_sc = [eligible[i] for i in perm[split:]]

    if not train_full or not test_sc:
        return None, None, None, None

    train_pool, val_sc = split_train_val(train_full, val_fraction=VAL_FRACTION, seed=VAL_SEED)
    if not train_pool or not val_sc:
        return None, None, None, None

    X_test_raw, y_test = build_sequences(test_sc, horizon_steps)
    return train_pool, val_sc, X_test_raw, y_test


def sample_scenarios(pool, n, strategy, seed):
    """Sample n scénarios depuis pool selon la stratégie."""
    n = min(max(1, n), len(pool))
    if strategy == 'uniform':
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), n, replace=False)
        return [pool[i] for i in idx]
    elif strategy == 'progressive_ltr':
        # N scénarios avec le plus faible LTR_max (plan d'essais par risque croissant)
        sorted_pool = sorted(pool, key=lambda s: s['max_ltr'])
        return sorted_pool[:n]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


PREDS_DIR = OUTPUT_DIR / "preds_ablation"
PREDS_DIR.mkdir(parents=True, exist_ok=True)


def run_ablation_point(train_pool, val_sc, X_test_raw, y_test, horizon_steps,
                       fraction, strategy, seed, save_preds_key=None):
    """Un point d'ablation: sample, train, evaluate.

    Si save_preds_key est donné (seed=0 typiquement), sauvegarde preds + y_test
    en .npz pour tracer les courbes de régression a posteriori.
    """
    n_sample = max(1, int(fraction * len(train_pool)))
    train_sampled = sample_scenarios(train_pool, n_sample, strategy, seed)

    X_train_raw, y_train = build_sequences(train_sampled, horizon_steps)
    X_val_raw, y_val = build_sequences(val_sc, horizon_steps)

    if X_train_raw is None or X_val_raw is None or X_test_raw is None:
        return None

    # Normalize: scaler fit on sampled train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(X_train_raw.shape)
    X_val   = scaler.transform(X_val_raw.reshape(-1, N_FEATURES)).reshape(X_val_raw.shape)
    X_test  = scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(X_test_raw.shape)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTM(N_FEATURES)
    model, hist = train_model_clean(
        model, X_train, y_train, X_val, y_val,
        device=device, loss_fn=weighted_mse_loss,
        epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    )

    # Get predictions for saving
    import torch as _torch
    model.eval()
    with _torch.no_grad():
        preds_test = np.concatenate([
            model(_torch.FloatTensor(X_test[i:i + 512].astype(np.float32)).to(device)).cpu().numpy()
            for i in range(0, len(X_test), 512)
        ])
    preds_test = np.clip(preds_test, 0, 1.5)

    met = evaluate(model, X_test, y_test, device)
    met['n_train_scenarios'] = int(len(train_sampled))
    met['n_train_samples'] = int(len(y_train))
    met['n_val_scenarios'] = int(len(val_sc))
    met['n_val_samples'] = int(len(y_val))
    met['fraction'] = float(fraction)
    met['strategy'] = strategy
    met['seed'] = int(seed)
    met['best_val_loss'] = float(hist.get('best_val_loss', float('nan')))
    met['n_epochs'] = int(hist.get('n_epochs', 0))

    # Save preds/targets for regression plots (seed=0 only to save disk)
    if save_preds_key is not None:
        preds_file = PREDS_DIR / f"{save_preds_key}.npz"
        np.savez_compressed(
            preds_file,
            preds=preds_test.astype(np.float32),
            targets=y_test.astype(np.float32),
            fraction=float(fraction),
            strategy=strategy,
            seed=int(seed),
        )

    return met


def clean_for_json(met):
    out = {}
    for k, v in met.items():
        if isinstance(v, (np.floating,)):
            fv = float(v)
            out[k] = None if np.isnan(fv) else fv
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


def main():
    print("=" * 70)
    print("ABLATION: quantité de données réelles (protocole propre)")
    print(f"Device: {device}")
    print(f"Fractions: {FRACTIONS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {CONFIGS}")
    print(f"Val set: {int(VAL_FRACTION*100)}% du train full, seed fixe={VAL_SEED}")
    print("=" * 70)

    print("\n[1/3] Chargement des fichiers DXD (avec cache)...")
    scenarios, skipped = load_all_scenarios(use_cache=True)
    print(f"  → {len(scenarios)} scénarios valides ({skipped} skipped)")

    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        total_done = sum(
            len(runs)
            for cfg in results.values()
            for frac in cfg.values()
            for strat in frac.values()
            for runs in [strat]
        ) if results else 0
        print(f"  → {total_done} runs déjà sauvegardés, reprise activée")

    total_runs = len(CONFIGS) * len(FRACTIONS) * len(STRATEGIES) * len(SEEDS)
    run_idx = 0

    for ds_name, h in CONFIGS:
        horizon_steps = int(h / DT)
        hkey = f"h{h}s"
        cfg_key = f"{ds_name}_{hkey}"
        print(f"\n[2/3] Config: {cfg_key}")

        train_pool, val_sc, X_test_raw, y_test = build_test_split(scenarios, ds_name, horizon_steps)
        if train_pool is None:
            print(f"  Impossible de construire le split pour {cfg_key}, skip")
            continue

        print(f"  Train pool: {len(train_pool)} scénarios")
        print(f"  Val fixe: {len(val_sc)} scénarios")
        print(f"  Test: {y_test.shape[0] if y_test is not None else 0} échantillons "
              f"(LTR > {DATASET_CONFIGS[ds_name]['threshold']})")

        if cfg_key not in results:
            results[cfg_key] = {}

        for frac in FRACTIONS:
            frac_key = f"{int(frac * 100)}pct"
            if frac_key not in results[cfg_key]:
                results[cfg_key][frac_key] = {}

            for strategy in STRATEGIES:
                if strategy not in results[cfg_key][frac_key]:
                    results[cfg_key][frac_key][strategy] = []
                done_seeds = {r['seed'] for r in results[cfg_key][frac_key][strategy]}

                for seed in SEEDS:
                    run_idx += 1
                    if seed in done_seeds:
                        print(f"  [{run_idx}/{total_runs}] {frac_key} {strategy} seed={seed}: "
                              f"déjà fait, skip")
                        continue

                    n_sample = int(frac * len(train_pool))
                    t0 = time_module.time()
                    print(f"  [{run_idx}/{total_runs}] {frac_key} {strategy} seed={seed} "
                          f"(n_train={n_sample} scénarios)...", flush=True)

                    # Save preds only for seed=0 (enough for regression plots)
                    save_key = (f"{cfg_key}_{frac_key}_{strategy}_seed{seed}"
                                if seed == 0 else None)
                    met = run_ablation_point(
                        train_pool, val_sc, X_test_raw, y_test, horizon_steps,
                        frac, strategy, seed,
                        save_preds_key=save_key,
                    )
                    dt = time_module.time() - t0

                    if met is None:
                        print(f"       → ÉCHEC (séquences vides)")
                        continue

                    auc = met.get('auc_pr')
                    auc_s = f"{auc:.3f}" if auc is not None and not (isinstance(auc, float) and np.isnan(auc)) else "N/A"
                    print(f"       → R²={met['r2']:.3f} AUC-PR={auc_s} "
                          f"Recall={met['recall']:.0%} ({dt:.0f}s, {met['n_epochs']} epochs)")

                    results[cfg_key][frac_key][strategy].append(clean_for_json(met))

                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(results, f, indent=2)

    print(f"\n[3/3] Terminé. Résultats: {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    for cfg_key in results:
        print(f"\n  {cfg_key}:")
        for frac_key in sorted(results[cfg_key].keys(), key=lambda k: int(k.replace('pct', ''))):
            for strategy in STRATEGIES:
                runs = results[cfg_key][frac_key].get(strategy, [])
                if not runs:
                    continue
                r2s = [r['r2'] for r in runs]
                aucs = [r['auc_pr'] for r in runs if r.get('auc_pr') is not None]
                recalls = [r['recall'] for r in runs]
                auc_str = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}" if aucs else "N/A"
                print(f"    {frac_key:>5} {strategy:>15}: R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}  "
                      f"AUC-PR={auc_str}  Rec={np.mean(recalls):.0%} ({len(runs)} seeds)")


if __name__ == "__main__":
    main()
