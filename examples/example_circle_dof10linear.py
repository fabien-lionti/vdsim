# examples/slalom_closed_loop_linear_tire.py

import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import LinearTireParams  # <- pneu linéaire (comme tu avais envoyé)

from controllers import SpeedPIDController, StanleyController
from trajectories import CircleTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner


# ---- Vehicle params for DOF10 ----
params = VehiclePhysicalParams10DOF(
    g=9.81,
    m=1500.0,
    ms=1300.0,
    lf=1.6,
    lr=1.6,
    h=0.55,
    L1=0.75,
    L2=0.75,
    r=0.3,

    ix=400.0,
    iy=1200.0,
    iz=2500.0,
    ir=1.2,

    ra=12.0,
    s=2.2,
    cx=0.32,

    ks1=30000.0,
    ks2=30000.0,
    ks3=30000.0,
    ks4=30000.0,

    ds1=3500.0,
    ds2=3500.0,
    ds3=3500.0,
    ds4=3500.0
)

# ---------------------------------------------------------
# ---- Linear tire params (remplace Pacejka) ----
# ---------------------------------------------------------
tire_params = LinearTireParams(
    Cx=80000.0,  # N (par slip ratio)
    Cy=80000.0,  # N/rad
)

# ---- Build vehicle configuration ----
config = VehicleConfig10DOF(
    vehicle=params,
    tire1=tire_params,
    tire2=tire_params,
    tire3=tire_params,
    tire4=tire_params
)

# ---- Instantiate DOF10 model ----
model = DOF10(config)

sim_pos = []
pos = []

# ---------------------------------------------------------
# 2) Setup controllers
# ---------------------------------------------------------
speed_ctrl = SpeedPIDController(kp=1000, ki=10, kd=0.0)
steer_ctrl = StanleyController(k=0.001)

T = 15.0
dt = 0.0005
time_array = np.linspace(0, T, int(T/dt)+1)

for vref in range(5, 25, 5):

    # ---------------------------------------------------------
    # 3) Create trajectory
    # ---------------------------------------------------------
    R = 50
    traj = CircleTrajectory(v_ref=vref, R=R)

    x0 = np.array([
        traj.sample(0).x, vref,
        traj.sample(0).y, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        traj.sample(0).psi, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    # ---------------------------------------------------------
    # 4) Closed-loop runner
    # ---------------------------------------------------------
    runner = ClosedLoopRunner(
        vehicle_model=model,
        speed_controller=speed_ctrl,
        steering_controller=steer_ctrl,
        trajectory=traj,
    )

    result = runner.run(x0, time_array, method="euler")

    sim_pos.append(result.vehicle.x[:, [model.state_keys[key] for key in ['x', 'y']]])
    pos.append(traj.sample_array(time_array)[:, 1:3])


# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
labels = ["FL", "FR", "RL", "RR"]

# (1) Normalisé Fx/Fz, Fy/Fz (pas un "cercle µ" exact en linéaire, mais utile)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    fx = result.tires.Fx[:, i] / (result.tires.Fz[:, i] + 1e-9)
    fy = result.tires.Fy[:, i] / (result.tires.Fz[:, i] + 1e-9)

    ax.plot(fx, fy, linewidth=1.5)
    ax.set_title(labels[i])
    ax.set_xlabel("Fx/Fz [-]")
    ax.set_ylabel("Fy/Fz [-]")
    ax.axis("equal")
    ax.grid(True)

fig.suptitle("Forces normalisées (linéaire)", fontsize=14)
plt.tight_layout()
plt.show()

# (2) Glissements combinés
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    alpha = np.rad2deg(result.tires.alpha[:, i])
    kappa = result.tires.kappa[:, i]

    ax.plot(kappa, alpha, linewidth=1.5)
    ax.set_title(labels[i])
    ax.set_xlabel("κ [-]")
    ax.set_ylabel("α [deg]")
    ax.grid(True)

fig.suptitle("Glissements combinés (α, κ)", fontsize=14)
plt.tight_layout()
plt.show()

# (3) Trajectoire XY
plt.figure(figsize=(10, 10))
x, y = pos[-1][:, 0], pos[-1][:, 1]
sim_x, sim_y = sim_pos[-1][:, 0], sim_pos[-1][:, 1]
plt.plot(sim_x, sim_y, label='pred')
plt.plot(x, y, label='truth')
plt.xlim(-R, R)
plt.ylim(-R, R)
plt.axis("equal")
plt.legend()
plt.show()
