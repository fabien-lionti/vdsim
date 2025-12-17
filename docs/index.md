# Objectif

VDSim a pour objectif d’offrir une base open-source, commune et extensible pour la dynamique du véhicule, en intégrant des méthodes **état de l’art** sur l’ensemble de la chaîne de traitement :

- **Nettoyage des données** (préparation, filtrage, mise en forme)
- **Fiabilisation / validation** (qualité des signaux, détection d’anomalies)
- **Identification paramétrique** (modèles pneus, véhicule, frictions…)
- **Contrôle et simulation** (lois de commande, scénarios, tests)
- **Analyse et exploitation** (évaluation de performances, data-driven models)

> VDSim se veut une plateforme de référence open-source pour l’étude, la simulation et l’expérimentation en dynamique du véhicule, à la frontière entre **modélisation physique** et **méthodes basées sur les données**.

## 💡 Pourquoi VDSim ?

La dynamique du véhicule est un domaine riche mais souvent fragmenté : chaque laboratoire, école ou entreprise développe ses propres outils, scripts, modèles de pneus ou méthodes d’identification, sans base commune. Cela engendre plusieurs difficultés :

- Outils incompatibles entre eux
- Difficile de comparer les performances
- Solutions propriétaires fermées et coûteuses
- Reproduire l’état de l’art demande trop de temps
- Peu de ressources open-source fiables et didactiques

VDSim répond à ce besoin en proposant une plateforme ouverte, modulaire et documentée, conçue pour :

- l’enseignement
- la recherche
- le prototypage industriel
- les approches hybrides physique + data-driven

> *L’objectif est de fournir un socle commun pour développer, tester, comparer et partager des méthodes état de l’art en dynamique du véhicule.*

## Fonctionnalités principales

- Simulation de dynamique véhicule : modèles 7DOF et 10DOF, architecture modulaire
- Modèles de pneumatiques : linéaire et Simplified Pacejka (extensible)
- Identification : estimation de paramètres (pneus, véhicule, adhérence…)
- Contrôle et estimation : PID, MPC, observateurs, scénarios d’essai
- Approche data-driven : nettoyage, validation, analyse et apprentissage sur données
- Open-source et extensible : dataclasses, registry, documentation automatique

## Installation

### Prérequis
- Python 3.8 ou plus récent
- pip installé

### Installation depuis le dépôt

```bash
git clone <votre_lien_git>
cd vdsim
pip install -r requirements.txt
```

## Exemple d’utilisation rapide

Après installation, vous pouvez créer un modèle de pneu et calculer une force simple :

## Exemple complet : slalom en boucle fermée (10 DOF + Pacejka)

```python
import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireParams, SimplifiedPacejkaTireModel
from controllers import SpeedPIDController, StanleyController
from trajectories import DoubleLaneChangeTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner

# ---------------------------------------------------------
# 1) Paramètres véhicule (10 DOF)
# ---------------------------------------------------------
vehicle_params = VehiclePhysicalParams10DOF(
    g=9.81,
    m=1500.0,
    ms=1300.0,
    lf=1.6,
    lr=1.6,
    h=0.55,
    L1=0.75,
    L2=0.75,
    r=0.3,

    # Inerties
    ix=400.0,
    iy=1200.0,
    iz=2500.0,
    ir=1.2,

    # Forces résistives
    ra=12.0,
    s=2.2,
    cx=0.32,

    # Suspensions (raideurs)
    ks1=30000.0,
    ks2=30000.0,
    ks3=30000.0,
    ks4=30000.0,

    # Suspensions (amortisseurs)
    ds1=3500.0,
    ds2=3500.0,
    ds3=3500.0,
    ds4=3500.0,
)

# ---------------------------------------------------------
# 2) Paramètres pneus (Simplified Pacejka)
# ---------------------------------------------------------
tire_params = SimplifiedPacejkaTireParams(
    # Lateral
    By=10.0,
    Cy=1.3,
    Dy=3500.0,
    Ey=-1.6,
    # Longitudinal
    Bx=12.0,
    Cx=1.4,
    Dx=3000.0,
    Ex=-1.2,
)

config = VehicleConfig10DOF(
    vehicle=vehicle_params,
    tire1=tire_params,
    tire2=tire_params,
    tire3=tire_params,
    tire4=tire_params,
)

# Modèle véhicule 10 DOF
model = DOF10(config)

# État initial (exemple)
x0 = np.array([
    0.0, 20.0,     # x, vx
    0.0, 0.0,      # y, vy
    0.55, 0.0,     # zs, dzs
    0.0, 0.0,      # roll, droll
    0.0, 0.0,      # pitch, dpitch
    0.0, 0.0,      # yaw, dyaw
    0.0, 0.0, 0.0, 0.0,  # wheel speeds
])

# ---------------------------------------------------------
# 3) Contrôleurs
# ---------------------------------------------------------
speed_ctrl = SpeedPIDController(kp=1000.0, ki=10.0, kd=0.0)
steer_ctrl = StanleyController(k=0.2)

# ---------------------------------------------------------
# 4) Trajectoire de référence (double changement de voie)
# ---------------------------------------------------------
traj = DoubleLaneChangeTrajectory(v_ref=20.0)

# ---------------------------------------------------------
# 5) Simulation boucle fermée
# ---------------------------------------------------------
runner = ClosedLoopRunner(
    vehicle_model=model,
    speed_controller=speed_ctrl,
    steering_controller=steer_ctrl,
    trajectory=traj,
)

T = 10.0
dt = 0.001
time_array = np.linspace(0.0, T, int(T/dt) + 1)

result = runner.run(x0, time_array, method="euler")

# Échantillonnage de la trajectoire de référence
ref_x, ref_y, ref_v = [], [], []
for t in time_array:
    ref = traj.sample(t)
    ref_x.append(ref.x)
    ref_y.append(ref.y)
    ref_v.append(ref.v)

ref_x = np.array(ref_x)
ref_y = np.array(ref_y)
ref_v = np.array(ref_v)

# ---------------------------------------------------------
# 6) Visualisation des résultats
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# Trajectoire XY
plt.subplot(1, 2, 1)
plt.plot(ref_x, ref_y, "k--", label="Trajectoire de référence")
plt.plot(result.vehicle.x[:, 0], result.vehicle.x[:, 2], label="Trajectoire véhicule")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Double changement de voie - boucle fermée")
plt.grid(True)
plt.axis("equal")
plt.legend()

# Vitesse longitudinale
plt.subplot(1, 2, 2)
plt.plot(time_array, ref_v, "k--", label="Vitesse de référence")
plt.plot(time_array, result.vehicle.x[:, 1], label="Vitesse véhicule")
plt.xlabel("Temps [s]")
plt.ylabel("Vitesse [m/s]")
plt.title("Suivi de vitesse")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```