# Prédiction LTR - Horizon Exact

## Contexte

Prédiction du risque de renversement véhicule via le LTR (Load Transfer Ratio).

Cette approche prédit **LTR(t+h)** à l'instant exact t+h, pour h ∈ {1, 2, 4, 6, 8} secondes.

## Tâche

- **Entrée** : Séquence de 150 pas (1.5s) de 8 variables d'état
- **Sortie** : LTR à l'instant t+h (3 quantiles: Q10, Q50, Q90)
- **Horizons** : 1s, 2s, 4s, 6s, 8s
- **Méthode** : Régression quantile (pinball loss)

Variables d'entrée : `vx, vy, psi, psi_dot, phi, theta, delta_f, delta_f_dot`

## Datasets

| Config | Train | Test | Type |
|--------|-------|------|------|
| D1 | LTR ≤ 0.7 | LTR ≤ 0.7 | In-distribution |
| D4 | LTR ≤ 0.9 | LTR > 0.9 | Out-of-distribution |

484 scénarios au total (circle, single, lemniscate, slalom, dlc, waypoint).

## Modèles

- **MLP** : Perceptron multicouche (1200 → 512 → 256 → 128 → 3)
- **LSTM** : 1 couche, hidden=64
- **PatchTST** : Transformer adapté aux séries temporelles

## Structure

```
horizon_exact/
├── models/          # 30 modèles (.pt) + scalers (.pkl)
├── outputs/         # Figures et résultats
├── scripts/
│   ├── train_all.py
│   ├── confusion_matrices.py
│   ├── generate_scenarios.py
│   └── plot_trajectories.py
└── README.md
```

## Utilisation

```bash
# Entraîner tous les modèles (5 horizons × 3 modèles × 2 datasets = 30 modèles)
python scripts/train_all.py
```

---

## Métriques d'évaluation

### Régression quantile

Tous les modèles sortent 3 valeurs :
- **Q10** : 10ème percentile (borne basse)
- **Q50** : médiane (prédiction centrale)
- **Q90** : 90ème percentile (borne haute, conservative)

L'intervalle [Q10, Q90] forme une bande de confiance à 80%.

### Détection de danger (Precision / Recall)

Seuil de danger : **LTR > 0.7**

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Precision** | TP / (TP + FP) | Parmi les alertes levées, combien sont justifiées ? |
| **Recall** | TP / (TP + FN) | Parmi les situations dangereuses, combien sont détectées ? |

Où :
- **TP** (True Positive) : LTR prédit > 0.7 ET LTR réel > 0.7
- **FP** (False Positive) : LTR prédit > 0.7 ET LTR réel ≤ 0.7 (fausse alarme)
- **FN** (False Negative) : LTR prédit ≤ 0.7 ET LTR réel > 0.7 (danger manqué)

### Q50 vs Q90 : Quel quantile utiliser ?

| Aspect | Q50 (médiane) | Q90 (conservative) |
|--------|---------------|-------------------|
| **Philosophie** | Prédiction centrée | Prédiction prudente |
| **RMSE** | Plus bas | Plus élevé |
| **Precision** | ~90-99% | ~73-88% |
| **Recall** | ~60-93% | **~91-100%** |
| **Usage** | Estimation précise | **Sécurité** (recommandé) |

**Recommandation** : Pour une application sécurité, utiliser **Q90** car manquer un danger (FN) est plus grave qu'une fausse alarme (FP).

---

## Résultats

### Fichiers de données
- `results.json` : Métriques avec évaluation Q50
- `results_q90.json` : Métriques avec évaluation Q50 et Q90
- `confusion_results.json` : Matrices de confusion par classe de risque

### Visualisations

| Fichier | Description |
|---------|-------------|
| `comparaison_rmse.png` | RMSE par horizon et modèle |
| `comparaison_r2.png` | R² par horizon et modèle |
| `comparaison_q50_vs_q90.png` | Comparaison Q50 vs Q90 |
| `regression_all_models.png` | Toutes les courbes de régression (30 graphiques) |
| `confusion_all_Q50.png` | Matrices de confusion Q50 (6×5 grille) |
| `confusion_all_Q90.png` | Matrices de confusion Q90 (6×5 grille) |
| `predictions_horizons.png` | Prédictions sur trajectoires types |
| `trajectoires_types.png` | Visualisation des trajectoires |
| `ltr_et_braquage.png` | LTR et braquage par type |
| `distribution_ltr.png` | Distribution du LTR |

### Résultats clés (D4 - hors-distribution)

**Avec Q50** (prédiction médiane) :
| Horizon | Meilleur modèle | RMSE | R² | Precision | Recall |
|---------|-----------------|------|-----|-----------|--------|
| 1s | PatchTST | 0.093 | 0.69 | 89% | 93% |
| 2s | PatchTST | 0.092 | 0.49 | 96% | 80% |
| 4s | LSTM | 0.096 | 0.36 | 94% | 78% |
| 6s | PatchTST | 0.108 | 0.21 | 93% | 71% |
| 8s | PatchTST | 0.118 | 0.20 | 94% | 73% |

**Avec Q90** (prédiction conservative) :
| Horizon | Meilleur modèle | RMSE | Precision | Recall |
|---------|-----------------|------|-----------|--------|
| 1s | PatchTST | 0.120 | 82% | **98%** |
| 2s | LSTM | 0.125 | 81% | **97%** |
| 4s | LSTM | 0.127 | 81% | **98%** |
| 6s | MLP | 0.179 | 73% | **100%** |
| 8s | MLP | 0.192 | 73% | **100%** |

**Conclusion** : Q90 détecte quasi tous les dangers (recall ~98-100%) au prix de plus de fausses alarmes.
