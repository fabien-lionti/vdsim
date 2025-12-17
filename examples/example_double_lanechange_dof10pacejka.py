# examples/slalom_closed_loop.py

import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireModel, SimplifiedPacejkaTireParams

from controllers import SpeedPIDController, StanleyController
from trajectories import DoubleLaneChangeTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner

from analysis.rollover import compute_rollover_metrics


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
# ---- Pacejka simplifié : "asphalte sec" (μ ~ 1.05) ----
# ---------------------------------------------------------
mu = 1.05  # asphalte sec typique (pneu route)

# Charges nominales statiques AV/AR
L = params.lf + params.lr
Fzf0 = params.m * params.g * (params.lr / L)  # essieu avant
Fzr0 = params.m * params.g * (params.lf / L)  # essieu arrière

# Par roue
fz0_fl = 0.5 * Fzf0
fz0_fr = 0.5 * Fzf0
fz0_rl = 0.5 * Fzr0
fz0_rr = 0.5 * Fzr0

def make_pacejka_params(fz0_wheel: float) -> SimplifiedPacejkaTireParams:
    # Pics de force à la charge nominale (Dx/Dy)
    Dx0 = mu * fz0_wheel
    Dy0 = mu * fz0_wheel
    return SimplifiedPacejkaTireParams(
        # Lateral
        By=10.0,   # raideur latérale
        Cy=1.3,    # shape
        Dy=Dy0,    # pic latéral à fz0
        Ey=-1.0,   # courbure

        # Longitudinal
        Bx=12.0,   # raideur longitudinale
        Cx=1.6,    # shape
        Dx=Dx0,    # pic longitudinal à fz0
        Ex=-0.5,   # courbure
    )

# Params Pacejka par roue
tire1 = make_pacejka_params(fz0_fl)  # FL
tire2 = make_pacejka_params(fz0_fr)  # FR
tire3 = make_pacejka_params(fz0_rl)  # RL
tire4 = make_pacejka_params(fz0_rr)  # RR

# ---- Build vehicle configuration ----
config = VehicleConfig10DOF(
    vehicle=params,
    tire1=tire1,
    tire2=tire2,
    tire3=tire3,
    tire4=tire4
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


plt.figure(figsize = (10, 10))
x, y = pos[2][:, 0], pos[2][:, 1]
sim_x, sim_y = sim_pos[2][:, 0], sim_pos[2][:, 1]
plt.plot(sim_x, sim_y, label='pred')
plt.plot(x, y, label='truth')
plt.legend()
plt.show()