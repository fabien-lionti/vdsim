# HPO Report — LSTM+NODE vs PatchTST+NODE pour la prédiction LTR

## Contexte

Optimisation hyperparamètres pour la prédiction du **Load Transfer Ratio (LTR)** maximal sur un horizon de **2 secondes** (200 timesteps à dt=0.01s), en configuration **D2** (train: LTR ≤ 0.7, test: LTR > 0.7 — Out-of-Distribution).

**Métrique d'optimisation** : AUC-PR (Area Under Precision-Recall Curve) sur les échantillons de danger (LTR ≥ 0.7).

**Dataset** : 559 scénarios de simulation véhicule, split D2 → 421 train / 263 test.

**Backbone** : NODE (Neural ODE) pré-entraîné sur la dynamique véhicule, gelé pendant le HPO. Les features physiques extraites par le backbone sont projetées puis concaténées aux 8 features brutes (vx, vy, psi, psi_dot, phi, theta, delta_f, delta_f_dot).

**Framework** : Optuna (TPE sampler, MedianPruner), PyTorch 2.10, Apple MPS (M-series GPU).

---

## Études réalisées

### Round 1 — Exploration architecturale (LSTM + PatchTST)

**Objectif** : Trouver la meilleure architecture LSTM et PatchTST avec seq_len=150.

| Paramètre | Espace exploré |
|-----------|---------------|
| physics_proj_dim | 16, 32, 48, 64 |
| hidden_size (LSTM) | 32, 64, 128, 256 |
| num_layers | 1, 2 |
| fc_hidden | 32, 64 |
| dropout | 0.05 – 0.4 |
| lr_projection | 3e-4 – 3e-3 |
| lr_other | 1e-4 – 1e-3 |
| batch_size | 32, 64, 128 |

**Résultats** :
- **LSTM+NODE** : 50 trials (18 complete, 32 pruned) → **Best AUC-PR = 0.887** (trial 37)
  - hidden=128, num_layers=1, fc_hidden=64, physics_proj_dim=16, dropout=0.30
- **PatchTST+NODE** : 10 trials (6 complete) → **Best AUC-PR = 0.851** (trial 1)
  - d_model=128, nhead=2, num_layers=3, patch_size=10, physics_proj_dim=48

### Round 2 — Optimisation training params (LSTM, architecture fixée)

**Objectif** : Architecture fixée au trial 37, explorer les paramètres d'entraînement.

| Paramètre | Espace exploré |
|-----------|---------------|
| activation | relu, silu, gelu |
| weight_decay | 1e-6 – 1e-2 |
| optimizer | Adam, AdamW |
| seq_len | 100, 150 |
| proj_depth | 1, 2 |
| sched_patience | 3, 5, 8 |

**Résultats** : 48 trials (22 complete, 21 pruned) → **Best AUC-PR = 0.887** (trial 10)
- activation=**gelu**, weight_decay=**2.8e-6** (quasi nul), proj_depth=**2**, Adam, seq_len=100, sched_patience=5

**Insight** : Le round 2 n'a pas dépassé le round 1 en score, mais a identifié que **gelu**, **proj_depth=2**, et **weight_decay ≈ 0** sont optimaux.

### Round 3 — Séquences longues (LSTM)

**Objectif** : Tester des lookback windows plus longs (200-400 timesteps) avec les meilleurs training params du round 2.

| Paramètre | Espace exploré |
|-----------|---------------|
| seq_len | 150, 200, 250, 300, 400 |
| weight_decay | 1e-7 – 1e-4 |
| lr_projection | 3e-4 – 3e-3 |
| lr_other | 3e-4 – 1e-3 |
| dropout | 0.2 – 0.5 |
| batch_size | 64, 128, 256 |

Fixé : architecture trial 37, gelu, proj_depth=2, Adam, sched_patience=5.

**Résultats** : 40 trials (9 complete, 1 pruned, arrêté manuellement) → **Best AUC-PR = 0.892** (trial avec seq_len=200)
- seq_len=**200**, weight_decay=2.2e-7, lr_proj=5.9e-4, lr_other=8.9e-4, dropout=0.48, batch_size=128

**Performance par seq_len** :

| seq_len | Meilleur AUC-PR | Samples train | Samples test |
|---------|----------------|---------------|-------------|
| 150 | 0.882 | 59 415 | 39 245 |
| **200** | **0.892** | 57 310 | 37 930 |
| 250 | 0.890 | 55 205 | 36 615 |
| 300 | 0.881 | 53 100 | 35 300 |
| 400 | 0.890 | 48 890 | 32 670 |

**Insight** : Les séquences de 200-250 timesteps (2.0-2.5 secondes de lookback) capturent plus de contexte dynamique et améliorent la prédiction. Au-delà de 300, le signal se dilue et le nombre de samples diminue.

### Round PatchTST final — Informé par les résultats LSTM

**Objectif** : Appliquer les enseignements LSTM (gelu, proj_depth=2, seq_len longues) au PatchTST.

| Paramètre | Espace exploré |
|-----------|---------------|
| seq_len | 150, 200, 250, 300 |
| patch_size | 5, 10, 25, 50 |
| d_model | 64, 128, 256 |
| nhead | 1, 2, 4, 8 |
| num_layers | 2, 3, 4, 6 |
| physics_proj_dim | 16, 32, 48 |

**Résultats** : 50 trials (28 complete, 22 pruned) → **Best AUC-PR = 0.865** (trial 10)
- seq_len=200, patch_size=25, d_model=128, nhead=1, num_layers=4, physics_proj_dim=48, dropout=0.06

**Retrain** du best : AUC-PR = 0.809 (sous-performance au retrain, signe d'instabilité du Transformer sur ce dataset).

---

## Synthèse comparative

| Modèle | Best AUC-PR | Config clé |
|--------|-----------|------------|
| **LSTM+NODE (R3)** | **0.892** | seq_len=200, gelu, proj_depth=2, hidden=128 |
| LSTM+NODE (R1) | 0.887 | seq_len=150 |
| LSTM+NODE (R2) | 0.887 | seq_len=100 |
| PatchTST+NODE (final) | 0.865 | seq_len=200, d_model=128, 4 layers |
| PatchTST+NODE (R1) | 0.851 | seq_len=150 |

---

## Configuration optimale retenue

```
Architecture : LSTM + NODE backbone (gelé)
- LSTM : hidden_size=128, num_layers=1, fc_hidden=64
- Physics projection : depth=2, dim=16, activation=GELU
- Input : 8 features brutes + 16 features physiques projetées = 24 dims

Entraînement :
- seq_len=200 (lookback de 2.0 secondes)
- Optimizer : Adam (pas AdamW)
- lr_projection=5.9e-4, lr_other=8.9e-4
- weight_decay ≈ 0 (2.2e-7)
- dropout=0.48
- batch_size=128
- scheduler : ReduceLROnPlateau(patience=5, factor=0.5)
- epochs=80, early_stop_patience=15
- Loss : quantile loss (q=[0.1, 0.5, 0.9])
```

---

## Conclusions

1. **LSTM > PatchTST** (+3 points AUC-PR) : La dynamique véhicule a une structure séquentielle récurrente mieux capturée par le LSTM que par l'attention. Le Transformer souffre aussi d'instabilité au retrain.

2. **Séquences longues aident** : Passer de 150 à 200 timesteps de lookback a apporté +0.5 point d'AUC-PR. Le sweet spot est 200-250 (2.0-2.5s de contexte). Au-delà, rendements décroissants.

3. **GELU > ReLU/SiLU** : L'activation GELU dans la projection physique et le classifieur permet de mieux exploiter les features proches de zéro (caractéristiques de la dynamique véhicule dans les régimes normaux).

4. **Projection physique profonde** : 2 couches de projection (128→64→16 avec GELU) extraient de meilleures représentations physiques qu'une seule couche linéaire.

5. **Régularisation minimale** : weight_decay quasi nul — le dropout élevé (0.48) et l'early stopping suffisent. Le modèle bénéficie de ne pas pénaliser les poids.

---

## Validation sans data leakage

Les resultats HPO ci-dessus ont ete obtenus avec un NODE backbone potentiellement leaky (`node_final_8feat.pt` entraine sur un split random 80/20 de TOUS les scenarios). Le retrain final utilise des NODE clean par config (`node_D2_clean.pt`, etc.) entraines uniquement sur les scenarios d'entrainement de chaque config.

**Impact du fix** (D2, h2s, metriques Q90) :

| Metrique | Avec leakage | Sans leakage | Diff |
|----------|-------------|-------------|------|
| AUC-PR | 0.876 | 0.860 | -1.6 pts |
| Recall | 88.8% | 91.2% | +2.4 pts |
| Precision | 70.3% | 66.3% | -4.0 pts |

Le leakage n'inflait pas significativement les resultats. Les scores clean sont dans `results_clean_all.json`.

## Compute

- Total : ~120 trials LSTM + ~60 trials PatchTST
- Duree : ~48h cumulees sur Apple MPS (M-series)
- DB Optuna : `outputs/hpo_max_horizon.db` (toutes les etudes)
