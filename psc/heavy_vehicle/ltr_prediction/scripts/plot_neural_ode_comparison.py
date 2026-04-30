#!/usr/bin/env python3
"""Barplot comparant LSTM, PatchTST, LSTM+NODE et PatchTST+NODE sur D2/D4 aux 3 horizons."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# AUC-PR (baseline vs finetuned NODE) extraits de results_physics_finetune_all.json
DATA = {
    'D2 h1s': {'LSTM': 0.866, 'PatchTST': 0.691, 'LSTM+NODE': 0.884, 'PatchTST+NODE': 0.878},
    'D2 h2s': {'LSTM': 0.848, 'PatchTST': 0.748, 'LSTM+NODE': 0.874, 'PatchTST+NODE': 0.875},
    'D2 h4s': {'LSTM': 0.859, 'PatchTST': 0.607, 'LSTM+NODE': 0.876, 'PatchTST+NODE': 0.847},
    'D4 h1s': {'LSTM': 0.987, 'PatchTST': 0.988, 'LSTM+NODE': 0.991, 'PatchTST+NODE': 0.995},
    'D4 h2s': {'LSTM': 0.948, 'PatchTST': 0.963, 'LSTM+NODE': 0.991, 'PatchTST+NODE': 0.994},
    'D4 h4s': {'LSTM': 0.947, 'PatchTST': 0.965, 'LSTM+NODE': 0.985, 'PatchTST+NODE': 0.984},
}

CONFIGS = list(DATA.keys())
SERIES = ['LSTM', 'PatchTST', 'LSTM+NODE', 'PatchTST+NODE']
COLORS = {
    'LSTM':          '#60a5fa',
    'PatchTST':      '#34d399',
    'LSTM+NODE':     '#a78bfa',
    'PatchTST+NODE': '#c084fc',
}
LABELS = {
    'LSTM':          'LSTM',
    'PatchTST':      'PatchTST',
    'LSTM+NODE':     'LSTM + NODE',
    'PatchTST+NODE': 'PatchTST + NODE',
}


def fmt_pct(gain):
    return f'+{int(round(gain * 100))}%'


def main():
    fig, ax = plt.subplots(figsize=(13, 6))
    n = len(SERIES)
    width = 0.8 / n
    xs = np.arange(len(CONFIGS))
    for i, s in enumerate(SERIES):
        vals = [DATA[c][s] for c in CONFIGS]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(xs + offset, vals, width, color=COLORS[s], label=LABELS[s], edgecolor='none')

        # Annotation du gain pour les variantes NODE
        if s.endswith('+NODE'):
            base = s.replace('+NODE', '').strip()
            for j, c in enumerate(CONFIGS):
                gain = DATA[c][s] - DATA[c][base]
                if gain >= 0.02:  # seuil d'affichage
                    ax.text(xs[j] + offset, DATA[c][s] + 0.003, fmt_pct(gain),
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color=COLORS[s])

    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace(' ', '\n') for c in CONFIGS], fontsize=11)
    ax.set_ylabel('AUC-PR', fontsize=12)
    ax.set_ylim(0.55, 1.02)
    ax.set_title('Impact du Neural ODE comme extracteur de features physiques', fontsize=13)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_dir = Path(__file__).parent.parent.parent.parent / 'rapport_final' / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = out_dir / f'neural_ode_comparison.{ext}'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Saved: {out}')


if __name__ == '__main__':
    main()
