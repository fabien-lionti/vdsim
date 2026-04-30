# PSC INF06 — Prédiction du risque de retournement par IA

Projet Scientifique Collectif, École polytechnique, promotion X2024.
Encadrement : Sébastien Aubin (DGA), Fabien Lionti (vdsim).

Ce dossier contient le code développé en complément du simulateur
[`vdsim`](https://github.com/fabien-lionti/vdsim) pour le rapport final
(avril 2026).

## Structure

```
psc/
├── ltr_prediction/         # Section 3-4 du rapport (sim non calibré)
│   ├── horizon_exact/      # Cible ponctuelle LTR(t+h)
│   ├── maxhorizon/         # Cible max-horizon (retenue pour la suite)
│   └── node_v1/            # Neural ODE comme feature extractor
└── heavy_vehicle/          # Section 5-7 du rapport (réel + transfer sim→réel)
    ├── params/             # Calibration véhicule sur données DXD
    ├── scenario_generation/# Pseudo-replay et replay calibré
    ├── ltr_prediction/     # Pipelines LSTM, ablations, hyperparams
    └── verification/       # Sim vs réel, distributions
```

## Pipeline de reproduction (sections du rapport final)

1. **Données simulées initiales (484 scénarios)** — `ltr_prediction/horizon_exact/`
   et `ltr_prediction/maxhorizon/`. Génération via `scripts/generate_scenarios.py`,
   entraînement via `scripts/train_all.py`, figures via `scripts/generate_figures.py`.

2. **Neural ODE (section 4)** — `ltr_prediction/node_v1/`. Pré-entraînement
   du backbone (`scripts/train_node.py`), pipeline LSTM/PatchTST avec
   features NODE en entrée (`scripts/train_with_node.py`), figures
   (`scripts/plot.py`).

3. **Calibration DOF10 (section 6.1)** — `heavy_vehicle/params/vehicle_params.py`
   définit les paramètres calibrés (m, T, L, H, ks, ds, By) extraits des
   mesures DXD. La fonction `make_heavy_vehicle()` instancie le simulateur
   `vdsim` standard avec ces paramètres. **Pas d'extension physique au-delà
   de la dataclass `vdsim`** (pas de tire relaxation, pas de Kamm, pas
   d'ARB, pas de bump stop : restitution Lionti pure).

4. **Real-only baseline (section 5)** — `heavy_vehicle/ltr_prediction/scripts/`
   - `lstm_node_real.py`, `train_real_data.py`, `real_only_baseline_full.py`
   - Loss : `weighted MSE` (cf. `training_utils.py`).

5. **Pseudo-replay open-loop (section 6.3)** —
   `heavy_vehicle/scenario_generation/generate_pseudo_replay_v1calib.py`
   (signaux Ornstein-Uhlenbeck calibrés sur DXD, simulateur DOF10 calibré).

6. **Ablation pretrain synthétique + finetune X% réel (section 6.5)** —
   `heavy_vehicle/ltr_prediction/scripts/ablation_calibre_full.py`
   (pseudo-replay seul) et `ablation_calibre_node.py` (avec features NODE).
   Plots : `plot_saturation_calibre_full.py`, `plot_node_comparison.py`.

7. **Hyperparamètres Optuna (section 7)** —
   `heavy_vehicle/ltr_prediction/scripts/hyperparams_optuna_config.py`,
   `hyperparams_pseudoreplay.py`.

## Données

- **Datasets simulés** : `ltr_prediction/data_v3_684/` (684 scénarios,
  paramétriques + trajectoires aléatoires lisses).
- **Données réelles** : 488 fichiers DXD fournis par la DGA TT (Angers, 2025),
  392 exploitables après filtrage. Hors du dépôt (confidentielles).
- Caches `.npz`, modèles entraînés `.pt/.pkl` et résultats prédictions `preds_*/`
  régénérables depuis les scripts, exclus via `.gitignore`.

## Équipe

| Membre | Contributions principales |
|--------|--------------------------|
| Louis VOLLAND | Métriques d'évaluation, calibration physique du simulateur, génération de bruit réaliste |
| Adam EL KHARRAZE | Architecture MLP, stabilité numérique 10-DOF |
| Saad CHAIRI | Génération de trajectoires aléatoires |
| Zaccharie TARDY | Architectures LSTM, PatchTST, Neural ODE et ablations |
| Mohamed Taha RAMLAOUI | Approches PIML |

## Rapport

`rapport_final/` (sur Overleaf, hors dépôt).
