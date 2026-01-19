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
| D2 | LTR ≤ 0.7 | LTR > 0.7 | Out-of-distribution |
| D3 | LTR ≤ 0.8 | LTR > 0.8 | Out-of-distribution |
| D4 | LTR ≤ 0.9 | LTR > 0.9 | Out-of-distribution |

484 scénarios au total (circle, single, lemniscate, slalom, dlc, waypoint).

## Modèles

- **MLP** : Perceptron multicouche (1200 → 512 → 256 → 128 → 3)
- **LSTM** : 1 couche, hidden=64
- **PatchTST** : Transformer adapté aux séries temporelles
- **XGBoost** : Gradient boosting avec régression quantile Q90

## Structure

```
horizon_exact/
├── models/          # 60 modèles + 20 scalers
│   ├── MLP_*.pt     # Modèles MLP (D1-D4, h1-h8)
│   ├── LSTM_*.pt    # Modèles LSTM (D1-D4, h1-h8)
│   ├── PatchTST_*.pt  # Modèles PatchTST (D1-D4, h1-h8)
│   ├── XGBoost_*.json # Modèles XGBoost (D1-D4, h1-h8)
│   └── scaler_*.pkl   # Scalers par config
├── outputs/         # Figures et résultats JSON
├── scripts/
│   ├── train_all.py        # Entraînement MLP/LSTM/PatchTST
│   ├── train_xgboost_all.py  # Entraînement XGBoost
│   ├── generate_figures.py   # Génération des 6 figures
│   └── generate_scenarios.py # Génération des scénarios
└── README.md
```

## Utilisation

```bash
# Entraîner les modèles neuronaux (5 horizons × 3 modèles × 4 datasets)
python scripts/train_all.py

# Entraîner les modèles XGBoost
python scripts/train_xgboost_all.py

# Générer les figures
python scripts/generate_figures.py
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

**Recommandation** : Pour une application sécurité, utiliser **Q90** car manquer un danger (FN) est plus grave qu'une fausse alarme (FP)

---

## Résultats

### Fichiers de données
- `results.json` : Métriques MLP/LSTM/PatchTST (D1-D4)
- `results_xgboost.json` : Métriques XGBoost (D1-D4)

### Visualisations

| Fichier | Description |
|---------|-------------|
| `1_tableau_comparatif.png` | Tableau comparatif de tous les modèles |
| `2_confusion_matrices.png` | Matrices de confusion Q90 (4 classes de risque) |
| `3_regression.png` | Courbes de régression prédit vs réel |
| `4_recall_comparison.png` | Comparaison du recall par horizon |
| `5_predictions_trajectoires.png` | Prédictions sur 3 trajectoires types |
| `6_prediction_detail.png` | Détail d'une prédiction avec intervalles Q10-Q90 |

### Résultats clés

**D4 (OOD extreme : train ≤ 0.9, test > 0.9) - Recall Q90** :
| Horizon | MLP | LSTM | PatchTST | XGBoost |
|---------|-----|------|----------|---------|
| 1s | 98% | 97% | 98% | 95% |
| 2s | 97% | 97% | 97% | 94% |
| 4s | 98% | 98% | 97% | 92% |
| 6s | 100% | 99% | 99% | 90% |
| 8s | 100% | 99% | 99% | 88% |

**Conclusion** : Les modèles neuronaux (MLP, LSTM, PatchTST) avec Q90 détectent quasi tous les dangers (recall ~97-100%). XGBoost reste performant mais légèrement en dessous.
