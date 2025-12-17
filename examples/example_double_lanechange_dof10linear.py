# examples/slalom_closed_loop.py

import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF, LinearTireParams

from controllers import SpeedPIDController, StanleyController
from trajectories import SlalomTrajectory, StraightTrajectory, DoubleLaneChangeTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner

from analysis.rollover import compute_rollover_metrics


# ---- Vehicle params for DOF10 ----
params = VehiclePhysicalParams10DOF(
    g=9.81,              # Gravité
    m=1500.0,            # Masse totale [kg]
    ms=1300.0,           # Masse suspendue [kg]
    lf=1.6,              # Distance CG → essieu avant [m]
    lr=1.6,              # Distance CG → essieu arrière [m]
    h=0.55,              # Hauteur CG [m]
    L1=0.75,             # Demi-voie avant [m]
    L2=0.75,             # Demi-voie arrière [m]
    r=0.3,               # Rayon effectif roue [m]

    # ---- Inerties ----
    ix=400.0,            # Inertie en roulis [kg·m²]
    iy=1200.0,           # Inertie en tangage [kg·m²]
    iz=2500.0,           # Inertie en lacet [kg·m²]
    ir=1.2,              # Inertie roue [kg·m²]

    # ---- Forces résistives ----
    ra=12.0,             # Résistance au roulement [N·s/m]
    s=2.2,               # Surface frontale [m²]
    cx=0.32,             # Coefficient traînée [-]

    # ---- Raideurs suspension ----
    ks1=30000.0,         # FL
    ks2=30000.0,         # FR
    ks3=30000.0,         # RL
    ks4=30000.0,         # RR

    # ---- Amortisseurs suspension ----
    ds1=3500.0,          # FL
    ds2=3500.0,          # FR
    ds3=3500.0,          # RL
    ds4=3500.0           # RR
)

# ---- Tire model parameters ----
# tire_params = {"model": "linear", "Cx": 80000.0, "Cy": 80000.0}
tire_params = LinearTireParams(
        Cx=80000.0,
        Cy=80000.0,
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
steer_ctrl = StanleyController(k=0.2)

T = 15.0
dt = 0.0005
time_array = np.linspace(0, T, int(T/dt)+1)

for vref in range(10, 25, 5):

    x0 = np.array([
        0.0, vref,
        0.0, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    # ---------------------------------------------------------
    # 3) Create trajectory
    # ---------------------------------------------------------
    traj = DoubleLaneChangeTrajectory(v_ref=vref)

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

# plt.figure(figsize = (10, 10))
# for i in range(len(sim_pos)):
#     sim_x, sim_y = sim_pos[i][:, 0], sim_pos[i][:, 1]
#     plt.plot(sim_x, sim_y)
# plt.legend()
# plt.show()

# plt.figure(figsize = (10, 10))
# for i in range(len(sim_pos)):
#     x, y = pos[i][:, 0], pos[i][:, 1]
#     plt.plot(x, y)
# plt.legend()
# plt.show()

plt.figure(figsize = (10, 10))
x, y = pos[2][:, 0], pos[2][:, 1]
sim_x, sim_y = sim_pos[2][:, 0], sim_pos[2][:, 1]
plt.plot(sim_x, sim_y, label='pred')
plt.plot(x, y, label='truth')
plt.legend()
plt.show()