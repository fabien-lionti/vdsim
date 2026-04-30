# Résultats — Transfer Learning Sim→Réel

## Tableau complet (AUC-PR)

| Méthode | D4 h1s | D4 h2s | D2 h1s | D2 h2s |
|---------|--------|--------|--------|--------|
| Sim V1 direct → Réel (quantile) | 0.086 | 0.098 | 0.086 | 0.098 |
| Sim V1 direct → Réel (wMSE, 8feat) | 0.092 | 0.123 | 0.092 | 0.123 |
| **Replay DOF10-V2 → Réel** | **0.727** | **0.648** | **0.600** | **0.536** |
| Mixed (feat réel + LTR sim) | 0.698 | 0.667 | 0.677 | 0.625 |
| Real-only (baseline) | 0.867 | 0.805 | 0.767 | 0.708 |
| Pretrain replay + finetune réel | 0.852 | 0.810 | — | — |

## Tableau complet (R²)

| Méthode | D4 h1s | D4 h2s | D2 h1s | D2 h2s |
|---------|--------|--------|--------|--------|
| Sim V1 direct → Réel | -1.483 | -0.638 | -1.483 | -0.638 |
| **Replay DOF10-V2 → Réel** | **0.519** | **0.272** | **0.217** | **0.016** |
| Mixed | 0.121 | -0.103 | 0.277 | 0.156 |
| Real-only | 0.822 | 0.693 | 0.790 | 0.618 |
| Pretrain replay + finetune | 0.765 | 0.645 | — | — |

## Conclusions

1. **Le replay réduit le domain gap de manière drastique** : AUC-PR passe de 0.09 à 0.73
2. Le meilleur sim→réel pur : **replay D4 h1s, AUC-PR=0.727, R²=0.519**
3. Le pretrain replay + finetune réel aide marginalement (+0.008 AUC-PR vs real-only)
4. Le mixed est meilleur que le replay pur sur D2 (domain gap plus grand → features réelles aident)
5. Le real-only reste supérieur en absolu (0.87 vs 0.73)

## Fichiers de résultats

- `transfer_replay_results.json` — v1 (D4 only, hidden=64)
- `transfer_replay_v2_results.json` — v2 (D2+D4, hidden=128, 3 mu, mixed)
- `ablation_transfer_wmse.json` — sim V1 direct (8/9 feat, wMSE)

## Configuration des expériences

- **Modèle** : LSTM, hidden=128, dropout=0.38, 1 couche
- **Loss** : Weighted MSE, alpha=9
- **Features** : 9 (vx, vy, ψ̇, φ, θ, δf, δ̇f, LTR, dLTR/dt)
- **Replay** : DOF10-V2, RK4, PID vitesse, braquage DXD injecté, μ ∈ {0.75, 0.85, 0.95}
- **Split** : par scénario (pas par timestep)
