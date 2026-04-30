#!/usr/bin/env python3
"""
Visualisation des trajectoires du dataset avec coloration selon le LTR.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = Path("/Users/zak/Documents/vdsim/psc/ltr_prediction/data_new")
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def get_colored_line(x, y, c, cmap='RdYlGn_r', vmin=0, vmax=1, linewidth=2.5):
    """Cree un LineCollection colore selon c."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c[:-1])
    lc.set_linewidth(linewidth)
    return lc


def main():
    print("Chargement des trajectoires...")

    # Charger quelques exemples de chaque type
    examples = {}

    for traj_type in ['circle', 'single', 'lemniscate', 'slalom', 'dlc', 'waypoint']:
        files = sorted(DATA_DIR.glob(f"{traj_type}_*.csv"))[:30]
        if not files:
            continue

        # Trouver un exemple avec LTR eleve (plus interessant visuellement)
        best = None
        best_ltr = 0
        for f in files:
            df = pd.read_csv(f)
            ltr_max = np.max(np.abs(df['LTRmax'].values))
            if 0.7 < ltr_max < 0.95 and ltr_max > best_ltr:
                best = (f.stem, df, ltr_max)
                best_ltr = ltr_max

        if best is None and files:
            df = pd.read_csv(files[0])
            best = (files[0].stem, df, np.max(np.abs(df['LTRmax'].values)))

        if best:
            examples[traj_type] = best

    print(f"Exemples charges: {list(examples.keys())}")

    # === FIGURE PRINCIPALE: 2 lignes x 3 colonnes ===
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Trajectoires du Dataset - Coloration selon LTR\n'
                 '(Vert = Safe, Jaune = Moyen, Rouge = Critique)', fontsize=14, fontweight='bold')

    # Ligne 1: Trajectoires 2D (circle, single, lemniscate)
    traj_2d = ['circle', 'single', 'lemniscate']
    for col, traj_type in enumerate(traj_2d):
        ax = fig.add_subplot(2, 3, col + 1)

        if traj_type in examples:
            name, df, ltr_max = examples[traj_type]
            x = df['x'].values
            y = df['y'].values
            ltr = np.abs(df['LTRmax'].values)

            lc = get_colored_line(x, y, ltr)
            ax.add_collection(lc)

            # Limites adaptees avec marge
            x_range = x.max() - x.min()
            y_range = y.max() - y.min()
            margin = max(x_range, y_range) * 0.1
            ax.set_xlim(x.min() - margin, x.max() + margin)
            ax.set_ylim(y.min() - margin, y.max() + margin)
            ax.set_aspect('equal')

            ax.set_title(f"{traj_type.upper()}\nLTR max: {ltr_max:.2f}", fontsize=11, fontweight='bold')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)

    # Ligne 2: Profils y(x) pour trajectoires lineaires (slalom, dlc, waypoint)
    traj_linear = ['slalom', 'dlc', 'waypoint']
    for col, traj_type in enumerate(traj_linear):
        ax = fig.add_subplot(2, 3, col + 4)

        if traj_type in examples:
            name, df, ltr_max = examples[traj_type]
            x = df['x'].values
            y = df['y'].values
            ltr = np.abs(df['LTRmax'].values)

            lc = get_colored_line(x, y, ltr)
            ax.add_collection(lc)

            # Limites: x complet, y zoome sur les deviations
            ax.set_xlim(x.min() - 5, x.max() + 5)
            y_center = (y.min() + y.max()) / 2
            y_range = max(y.max() - y.min(), 5) * 1.3  # Au moins 5m de range
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)

            ax.set_title(f"{traj_type.upper()}\nLTR max: {ltr_max:.2f}", fontsize=11, fontweight='bold')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.3)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('LTR (Load Transfer Ratio)', fontsize=11)
    cbar.ax.axhline(0.7, color='black', linewidth=1, linestyle='--')
    cbar.ax.axhline(0.9, color='black', linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    output_path = OUTPUT_DIR / "trajectoires_types.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegarde: {output_path}")

    # === FIGURE 2: Evolution temporelle du LTR et Braquage ===
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    fig.suptitle('Evolution du LTR et du Braquage par type de trajectoire', fontsize=14, fontweight='bold')

    all_types = ['circle', 'single', 'lemniscate', 'slalom', 'dlc', 'waypoint']
    for idx, traj_type in enumerate(all_types):
        # LTR (ligne du haut)
        ax_ltr = axes[0, idx]
        # Braquage (ligne du bas)
        ax_delta = axes[1, idx]

        if traj_type in examples:
            name, df, ltr_max = examples[traj_type]
            t = df['time'].values if 'time' in df.columns else np.arange(len(df)) * 0.01
            ltr = np.abs(df['LTRmax'].values)
            delta_f = np.degrees(df['delta_f'].values)  # Convertir en degres

            # Plot LTR
            lc = get_colored_line(t, ltr, ltr, linewidth=2)
            ax_ltr.add_collection(lc)
            ax_ltr.set_xlim(t.min(), t.max())
            ax_ltr.set_ylim(0, 1.05)
            ax_ltr.axhline(0.7, color='orange', linestyle='--', alpha=0.7)
            ax_ltr.axhline(0.9, color='red', linestyle='--', alpha=0.7)
            ax_ltr.fill_between([t.min(), t.max()], 0.9, 1.05, color='red', alpha=0.1)
            ax_ltr.set_title(f"{traj_type.upper()}\nLTR max: {ltr_max:.2f}", fontsize=10, fontweight='bold')

            # Plot Braquage avec coloration LTR
            lc_delta = get_colored_line(t, delta_f, ltr, linewidth=2)
            ax_delta.add_collection(lc_delta)
            ax_delta.set_xlim(t.min(), t.max())
            delta_max = max(abs(delta_f.min()), abs(delta_f.max()))
            ax_delta.set_ylim(-delta_max * 1.1, delta_max * 1.1)
            ax_delta.axhline(0, color='gray', linestyle='-', alpha=0.3)

        ax_ltr.set_ylabel('LTR' if idx == 0 else '')
        ax_ltr.grid(True, alpha=0.3)
        ax_ltr.set_xticklabels([])

        ax_delta.set_xlabel('Temps (s)')
        ax_delta.set_ylabel('Braquage (°)' if idx == 0 else '')
        ax_delta.grid(True, alpha=0.3)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('LTR', fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    output_path = OUTPUT_DIR / "ltr_et_braquage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sauvegarde: {output_path}")

    # === FIGURE 3: Distribution LTR ===
    print("\nAnalyse de la distribution LTR...")
    all_ltr_max = []
    type_ltr = {t: [] for t in all_types}

    for f in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        ltr_max = np.max(np.abs(df['LTRmax'].values))
        all_ltr_max.append(ltr_max)

        for t in all_types:
            if f.name.startswith(t):
                type_ltr[t].append(ltr_max)
                break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribution du LTR dans le Dataset', fontsize=14, fontweight='bold')

    # Histogramme
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    n, bins_edges, patches = ax.hist(all_ltr_max, bins=bins, edgecolor='black', alpha=0.7)

    # Colorer les barres selon le niveau de danger
    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center < 0.7:
            patch.set_facecolor('#2ecc71')  # Vert
        elif bin_center < 0.9:
            patch.set_facecolor('#f39c12')  # Orange
        else:
            patch.set_facecolor('#e74c3c')  # Rouge

    ax.axvline(0.7, color='orange', linestyle='--', linewidth=2, label='Seuil D1/D2')
    ax.axvline(0.9, color='red', linestyle='--', linewidth=2, label='Seuil critique')
    ax.set_xlabel('LTR max du scenario', fontsize=11)
    ax.set_ylabel('Nombre de scenarios', fontsize=11)
    ax.set_title(f'Distribution globale - {len(all_ltr_max)} scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Boxplot par type
    ax = axes[1]
    data = [type_ltr[t] for t in all_types if type_ltr[t]]
    labels = [t for t in all_types if type_ltr[t]]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    colors = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#e74c3c', '#f1c40f']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(0.9, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.fill_between([-0.5, len(labels)+0.5], 0.9, 1.0, color='red', alpha=0.1)
    ax.set_ylabel('LTR max', fontsize=11)
    ax.set_title('Distribution par type de trajectoire')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0.5, len(labels) + 0.5)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "distribution_ltr.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Sauvegarde: {output_path}")

    print("\nTermine!")


if __name__ == '__main__':
    main()
