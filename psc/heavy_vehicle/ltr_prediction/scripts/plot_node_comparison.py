#!/usr/bin/env python3
"""Plot comparatif real-only vs pseudo-replay+finetune vs NODE+pseudo-replay+finetune."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

CONFIGS = ['D4_h2s']

FILES = {
    'pretrain_pseudoreplay': OUTPUT_DIR / "data_ablation_pseudoreplay_finetune_results.json",
    'pretrain_replay': OUTPUT_DIR / "data_ablation_finetune_results.json",
    'real_only': OUTPUT_DIR / "data_ablation_realonly_results.json",
    'node_pseudoreplay': OUTPUT_DIR / "lstm_node_pseudoreplay_ablation_results.json",
}
STYLE = {
    'real_only':            {'color': '#7e22ce', 'marker': 's', 'label': 'Real-only'},
    'pretrain_replay':      {'color': '#2563eb', 'marker': 'o', 'label': 'Replay DXD + finetune'},
    'pretrain_pseudoreplay':{'color': '#059669', 'marker': '^', 'label': 'Pré-entraînement synthétique + finetune'},
    'node_pseudoreplay':    {'color': '#ea580c', 'marker': 'D', 'label': 'NODE + pré-entraînement synthétique + finetune'},
}
FULL_REAL = {'D4_h2s': {'r2': 0.737, 'auc_pr': 0.916}}
NODE_REAL_ONLY = {'D4_h2s': {'auc_pr': 0.944, 'r2': 0.735}}


def agg(cfg_data, metric):
    xs, means, stds = [], [], []
    for frac_key in sorted(cfg_data.keys(), key=lambda k: int(k.split('pct')[0])):
        runs = cfg_data[frac_key]
        if isinstance(runs, dict):
            runs = runs.get('uniform', [])
        vals = [r[metric] for r in runs if r.get(metric) is not None]
        if not vals:
            continue
        xs.append(int(frac_key.split('pct')[0]))
        means.append(np.mean(vals)); stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


def plot(metric, label, fname):
    data = {k: json.loads(p.read_text()) if p.exists() else None for k, p in FILES.items()}
    fig, ax = plt.subplots(figsize=(9, 5))
    for cfg in CONFIGS:
        for src, st in STYLE.items():
            d = data.get(src)
            if not d: continue
            cd = d.get(cfg, {})
            if not cd: continue
            xs, m, s = agg(cd, metric)
            # Ajoute le point 100% pour real-only depuis results_real_data.json
            if src == 'real_only':
                full = FULL_REAL.get(cfg, {}).get(metric)
                if full is not None:
                    xs = np.append(xs, 100)
                    m = np.append(m, full)
                    s = np.append(s, 0.0)
            if len(xs) == 0: continue
            ax.plot(xs, m, marker=st['marker'], color=st['color'], label=st['label'],
                    linewidth=2, markersize=8)
            ax.fill_between(xs, m - s, m + s, alpha=0.15, color=st['color'])
        node_ro = NODE_REAL_ONLY.get(cfg, {}).get(metric)
        if node_ro is not None:
            ax.axhline(node_ro, color='#ea580c', linestyle=':', alpha=0.6,
                       label=f'LSTM+NODE real-only ({node_ro:.2f})')

    ax.set_xlabel('Pourcentage de données réelles utilisées', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'D4 h2s — Comparaison des 4 pipelines', fontsize=13)
    ax.set_xlim(-5, 105); ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        out = FIG_DIR / f"{fname}.{ext}"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot('auc_pr', 'AUC-PR', 'pipeline_comparison_with_node_auc')
    plot('r2', 'R²', 'pipeline_comparison_with_node_r2')
