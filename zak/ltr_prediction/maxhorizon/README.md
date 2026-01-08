# Prédiction LTR - Approche Max Horizon

## Contexte

Prédiction du risque de renversement véhicule via le LTR (Load Transfer Ratio).

**Attention** : Cette approche prédit le **maximum du LTR sur un horizon de 2s**, pas la valeur ponctuelle LTR(t+h). La tâche officielle (prédire LTR à t+1s, t+2s, etc.) sera implémentée séparément.

## Tâche

- **Entrée** : Séquence de 150 pas (1.5s) de 8 variables d'état
- **Sortie** : max(|LTR|) sur les 200 pas suivants (2.0s)
- **Méthode** : Régression quantile (q10, q50, q90)

Variables d'entrée : `vx, vy, psi, psi_dot, phi, theta, delta_f, delta_f_dot`

## Datasets

| Config | Train | Test | Type |
|--------|-------|------|------|
| D1 | LTR ≤ 0.7 | LTR ≤ 0.7 | In-distribution |
| D4 | LTR ≤ 0.9 | LTR > 0.9 | Out-of-distribution |

484 scénarios au total (circle, single, lemniscate, slalom, dlc, waypoint).

## Modèles

- **MLP** : Baseline, entrée aplatie
- **LSTM** : 2 couches, hidden=128
- **PatchTST** : Transformer adapté aux séries temporelles

## Structure

```
maxhorizon/
├── models/          # Modèles entraînés (.pt) et scalers (.pkl)
├── outputs/         # Figures et résultats
└── scripts/
    ├── train_all.py          # Entraînement complet
    ├── generate_scenarios.py # Génération des données
    └── plot_trajectories.py  # Visualisation
```

## Utilisation

```bash
# Entraîner tous les modèles
python scripts/train_all.py

# Visualiser les trajectoires
python scripts/plot_trajectories.py
```

## Résultats

Voir `outputs/comparaison_modeles.png` et `outputs/results.json`.

## TODO

Implémenter la tâche officielle : prédire LTR(t+h) pour h ∈ {1, 2, 4, 6, 8}s avec les 4 configurations de dataset (D1-D4).
