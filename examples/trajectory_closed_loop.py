# examples/slalom_closed_loop.py

import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF7, VehiclePhysicalParams7DOF, VehicleConfig7DOF
from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import LinearTireParams, LinearTireModel, SimplifiedPacejkaTireParams, SimplifiedPacejkaTireModel

from controllers import SpeedPIDController, StanleyController
from trajectories import SlalomTrajectory, StraightTrajectory, DoubleLaneChangeTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner


model_name = "dof10"

# ---------------------------------------------------------
# 1) Setup vehicle model
# ---------------------------------------------------------
if model_name == "dof7":
    params = VehiclePhysicalParams7DOF(
        g=9.81, m=1500, lf=1.2, lr=1.6,
        h=0.5, L1=1.0, L2=1.6, r=0.3,
        iz=2500.0, ir=1.2, ra=0.015, s=2.2, cx=0.32
    )

    tire_params = {"model": "linear", "Cx": 80000.0, "Cy": 80000.0}

    config = VehicleConfig7DOF(
        vehicle=params, tire1=tire_params, tire2=tire_params,
        tire3=tire_params, tire4=tire_params
    )

    model = DOF7(config)

    # initial vehicle state
    x0 = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])

elif model_name == "dof10":
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
    tire_params = LinearTireParams(
        Cx=80000.0,
        Cy=80000.0,
    )

    # Pacejka coefficients example for a normal car tire
    tire_params = SimplifiedPacejkaTireParams(
        # Lateral
        By=10.0,       # stiffness factor
        Cy=1.3,        # shape factor
        Dy=3500.0,     # peak lateral force [N]
        Ey=-1.6,       # curvature factor

        # Longitudinal
        Bx=12.0,       # stiffness factor
        Cx=1.4,        # shape factor
        Dx=3000.0,     # peak longitudinal force [N]
        Ex=-1.2        # curvature factor
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

    x0 = np.array([
        0.0, 20.0,
        0.0, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

# ---------------------------------------------------------
# 2) Setup controllers
# ---------------------------------------------------------
speed_ctrl = SpeedPIDController(kp=1000, ki=10, kd=0.0)
steer_ctrl = StanleyController(k=0.2)

# ---------------------------------------------------------
# 3) Create trajectory
# ---------------------------------------------------------
traj = SlalomTrajectory(v_ref=20.0, A=3.0, omega=0.3)
# traj = StraightTrajectory(x0=0.0, y0=0.0, psi=0.0, v_ref=20.0)
traj = DoubleLaneChangeTrajectory(v_ref=20)

# v_ref: float = 20.0,
#         A: float = 3.5,
#         T_total: float = 6.0,
#         x0: float = 0.0,
#         y0: float = 0.0,
#     )

# ---------------------------------------------------------
# 4) Closed-loop runner
# ---------------------------------------------------------
runner = ClosedLoopRunner(
    vehicle_model=model,
    speed_controller=speed_ctrl,
    steering_controller=steer_ctrl,
    trajectory=traj,
)

T = 10.0
dt = 0.0001
time_array = np.linspace(0, T, int(T/dt)+1)

# run simulation for 10 seconds
result = runner.run(x0, time_array, method="euler")

# --- Sample reference trajectory on same time grid ---
ref_x = []
ref_y = []
ref_v = []

for t in time_array:  # `time` = time_array from your simulation
    ref = traj.sample(t)
    ref_x.append(ref.x)
    ref_y.append(ref.y)
    ref_v.append(ref.v)

ref_x = np.array(ref_x)
ref_y = np.array(ref_y)
ref_v = np.array(ref_v)

plt.figure(figsize=(12, 5))

# ---- XY plot ----
plt.subplot(1, 2, 1)
plt.plot(ref_x, ref_y, "k--", label="Reference path")
plt.plot(result.vehicle.x[:,0], result.vehicle.x[:,2], label="Vehicle path")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Closed-loop Slalom trajectory")
plt.grid(True)
plt.axis('equal')
plt.legend()

# ---- Speed plot ----
plt.subplot(1, 2, 2)
plt.plot(time_array, ref_v, "k--", label="Reference speed")
plt.plot(time_array, result.vehicle.x[:,1], label="Vehicle speed")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.title("Closed-loop speed tracking")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ---- Speed plot ----
plt.subplot(1, 2, 2)
plt.plot(time_array, result.vehicle.x[:,6], label="Vehicle speed")
plt.xlabel("Time [s]")
plt.ylabel("Roulis [m/s]")
plt.title("Closed-loop speed tracking")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5) Plot results
# ---------------------------------------------------------
x = result.vehicle.x[:, 0]
y = result.vehicle.x[:, 2]
vx = result.vehicle.x[:, 1]
time = result.vehicle.time

plt.figure(figsize=(12, 5))

# XY plot
plt.subplot(1, 2, 1)
plt.plot(x, y, label="Vehicle path")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Closed-loop Slalom trajectory")
plt.grid(True)
plt.axis('equal')
plt.legend()

# Speed plot
plt.subplot(1, 2, 2)
plt.plot(time, vx, label="Vx [m/s]")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.title("Closed-loop speed tracking")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
