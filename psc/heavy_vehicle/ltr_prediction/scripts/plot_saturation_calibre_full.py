#!/usr/bin/env python3
"""
Figure de saturation : pseudo-replay calibré × 5 fractions × D2/D4 × h1/h2/h4.
Grille 2×3 (datasets × horizons). Ligne real-only full en pointillés.
"""
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_FILE = OUTPUT_DIR / "ablation_calibre_full_results.json"
REAL_ONLY_FULL = OUTPUT_DIR / "real_only_baseline_full_results.json"

# Real-only baselines 100% : existants + nouveaux. Source : train_real_data,
# pseudoreplay_finetune json, et real_only_baseline_full_results.json.
REAL_ONLY_100 = {
    'D2_h1s': {'auc_pr': 0.854, 'r2': 0.74},   # nouveau
    'D2_h2s': {'auc_pr': 0.836, 'r2': 0.60},   # existant
    'D2_h4s': {'auc_pr': 0.801, 'r2': 0.37},   # nouveau
    'D4_h1s': {'auc_pr': 0.932, 'r2': 0.84},   # existant
    'D4_h2s': {'auc_pr': 0.916, 'r2': 0.74},   # existant
    'D4_h4s': {'auc_pr': 0.893, 'r2': 0.55},   # nouveau
}

DATASETS = ['D2', 'D4']
HORIZONS = [1, 2, 4]

COLOR_PSEUDO = '#059669'
COLOR_REAL = '#7e22ce'


def agg(runs, metric):
    vals = [r[metric] for r in runs if r.get(metric) is not None]
    return (np.mean(vals), np.std(vals)) if vals else (None, None)


def plot_metric(metric_key, metric_label, save_name):
    with open(ABLATION_FILE) as f:
        data = json.load(f)

    fig, axes = plt.subplots(len(DATASETS), len(HORIZONS),
                              figsize=(4.5 * len(HORIZONS), 3.8 * len(DATASETS)),
                              squeeze=False)

    for i, ds in enumerate(DATASETS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j]
            key = f"{ds}_h{h}s"
            cfg = data.get(key, {})

            xs, means, stds = [], [], []
            for fk in sorted(cfg.keys(), key=lambda k: int(k.split('pct')[0])):
                m, s = agg(cfg[fk], metric_key)
                if m is not None:
                    xs.append(int(fk.split('pct')[0])); means.append(m); stds.append(s)
            xs, means, stds = np.array(xs), np.array(means), np.array(stds)

            if len(xs):
                ax.plot(xs, means, marker='^', color=COLOR_PSEUDO,
                        linewidth=2, markersize=8,
                        label='Pseudo-replay calibré + finetune')
                ax.fill_between(xs, means - stds, means + stds,
                                alpha=0.18, color=COLOR_PSEUDO)

            ref = REAL_ONLY_100.get(key, {}).get(metric_key)
            if ref is not None:
                ax.axhline(ref, color=COLOR_REAL, linestyle='--', linewidth=1.5,
                           alpha=0.8, label=f'Real-only 100% ({ref:.2f})')

            ax.set_title(f"{ds} — $h={h}$s", fontsize=11)
            ax.set_xlim(-5, 105)
            ax.grid(True, alpha=0.25)
            if i == len(DATASETS) - 1:
                ax.set_xlabel('% données réelles', fontsize=10)
            if j == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=8)

            # ylim adapté
            if metric_key == 'auc_pr':
                ax.set_ylim(0.4, 1.0)
            else:
                ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        out = FIG_DIR / f"{save_name}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plt.rcParams['text.usetex'] = False
    plot_metric('auc_pr', 'AUC-PR', 'saturation_calibre_full_auc')
    plot_metric('r2', r'$R^2$', 'saturation_calibre_full_r2')
