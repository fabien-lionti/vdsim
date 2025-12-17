# examples/slalom_closed_loop.py

import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF
from models.tires import SimplifiedPacejkaTireModel, SimplifiedPacejkaTireParams

from controllers import SpeedPIDController, StanleyController
from trajectories import CircleTrajectory
from simulation.closed_loop_runner import ClosedLoopRunner

def plot_traction_efficiency(
    time,
    result,
    mu=1.05,
    labels=("FL","FR","RL","RR"),
    smooth=0
):
    """
    Trace l'efficacité de traction par roue.

    Affiche :
    - efficacité basée sur le slip (eta_slip)
    - efficacité d'utilisation de l'adhérence (eta_mu)

    Args:
        time (array): temps [s]
        result: résultat de simulation
        mu (float): coefficient d'adhérence
        labels (tuple): labels roues
        smooth (int): fenêtre de moyenne glissante (0 = pas de lissage)
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i, label in enumerate(labels):
        kappa = np.abs(result.tires.kappa[:, i])
        Fx = result.tires.Fx[:, i]
        Fz = result.tires.Fz[:, i]

        eta_slip = 1.0 / (1.0 + kappa)
        eta_mu = np.abs(Fx) / (mu * Fz + 1e-9)

        # clipping pour lisibilité
        eta_slip = np.clip(eta_slip, 0.0, 1.05)
        eta_mu = np.clip(eta_mu, 0.0, 1.2)

        # lissage optionnel
        if smooth and smooth > 1:
            kernel = np.ones(smooth) / smooth
            eta_slip = np.convolve(eta_slip, kernel, mode="same")
            eta_mu = np.convolve(eta_mu, kernel, mode="same")

        axes[0].plot(time, eta_slip, label=label)
        axes[1].plot(time, eta_mu, label=label)

    axes[0].set_ylabel("η_slip [-]")
    axes[0].set_title("Efficacité de traction (basée sur le slip)")
    axes[0].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylabel("η_μ [-]")
    axes[1].set_xlabel("Temps [s]")
    axes[1].set_title("Utilisation de l’adhérence longitudinale")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_speed(time, result, model, vref):
    vx = result.vehicle.x[:, model.state_keys["vx"]]
    np.ones(vx.shape)
    plt.figure(figsize=(8, 4))
    plt.plot(time, vx, label="vx")
    plt.plot(time, np.ones(vx.shape)*vref, "--", label="vref")
    plt.xlabel("Temps [s]")
    plt.ylabel("Vitesse [m/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_steering(time, result, front_idx=4, rear_idx=5):
    """
    Trace les angles de braquage avant et arrière.

    Args:
        time (array): temps [s]
        result: résultat de la simulation
        front_idx (int): index commande braquage avant
        rear_idx (int): index commande braquage arrière
    """
    delta_f = result.vehicle.u[:, front_idx]
    delta_r = result.vehicle.u[:, rear_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(time, np.rad2deg(delta_f), label="Front steering [deg]")
    plt.plot(time, np.rad2deg(delta_r), label="Rear steering [deg]")
    plt.xlabel("Temps [s]")
    plt.ylabel("Angle [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_wheel_loads(time, result, labels=("FL","FR","RL","RR")):
    """
    Trace les charges verticales Fz aux quatre roues.
    """
    plt.figure(figsize=(8, 4))
    for i in range(4):
        plt.plot(time, result.tires.Fz[:, i], label=labels[i])

    plt.xlabel("Temps [s]")
    plt.ylabel("Fz [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_slip_mosaic(result, alpha_max=7.0, kappa_max=0.15, labels=("FL","FR","RL","RR")):
    """
    Mosaïque 2x2 des glissements combinés (kappa, alpha) avec ellipse de zone admissible.

    Args:
        result: objet résultat (doit avoir result.tires.alpha, result.tires.kappa)
        alpha_max (float): limite alpha en degrés
        kappa_max (float): limite kappa (sans dimension)
        labels (tuple/list): labels roues
    """
    theta = np.linspace(0, 2*np.pi, 300)
    ellipse_kappa = kappa_max * np.cos(theta)
    ellipse_alpha = alpha_max * np.sin(theta)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        alpha = np.rad2deg(result.tires.alpha[:, i])
        kappa = result.tires.kappa[:, i]

        ax.plot(kappa, alpha, linewidth=1.5, label="traj")
        ax.plot(ellipse_kappa, ellipse_alpha, "k--", alpha=0.6, label="zone nominale")

        ax.set_title(labels[i])
        ax.set_xlabel("κ [-]")
        ax.set_ylabel("α [deg]")
        ax.set_xlim(-1.2*kappa_max, 1.2*kappa_max)
        ax.set_ylim(-1.2*alpha_max, 1.2*alpha_max)
        ax.grid(True)

    fig.suptitle(f"Glissements combinés (α, κ) — alpha_max={alpha_max}°, kappa_max={kappa_max}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_trajectory_xy(sim_pos, pos, R=None, idx=-1, labels=("pred", "truth")):
    """
    Trace la trajectoire simulée vs référence en XY.

    Args:
        sim_pos (list): liste de tableaux Nx2 (simulation)
        pos (list): liste de tableaux Nx2 (référence)
        R (float or None): rayon pour fixer xlim/ylim (optionnel)
        idx (int): index de la trajectoire à afficher (défaut: dernière)
        labels (tuple): labels (sim, ref)
    """
    sim_x, sim_y = sim_pos[idx][:, 0], sim_pos[idx][:, 1]
    x, y = pos[idx][:, 0], pos[idx][:, 1]

    plt.figure(figsize=(8, 8))
    plt.plot(sim_x, sim_y, label=labels[0])
    plt.plot(x, y, label=labels[1])

    if R is not None:
        plt.xlim(-R, R)
        plt.ylim(-R, R)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_under_oversteer_from_slip(time, result, use_abs=True, deg=True, smooth=0):
    """
    Indicateur sur/sous-virage basé sur les slip angles avant/arrière.

    Args:
        time: array (N,)
        result: doit contenir result.tires.alpha (N,4) ordre FL,FR,RL,RR
        use_abs (bool): compare |alpha| (recommandé) plutôt que alpha signé
        deg (bool): affiche en degrés
        smooth (int): fenêtre de moyenne glissante (0 = pas de lissage)
    """
    alpha = result.tires.alpha  # (N,4) en rad

    if use_abs:
        alpha_used = np.abs(alpha)
    else:
        alpha_used = alpha

    # moyen AV / AR
    alpha_f = 0.5 * (alpha_used[:, 0] + alpha_used[:, 1])
    alpha_r = 0.5 * (alpha_used[:, 2] + alpha_used[:, 3])

    I = alpha_f - alpha_r  # >0 sous-virage, <0 sur-virage

    # lissage optionnel
    if smooth and smooth > 1:
        kernel = np.ones(smooth) / smooth
        alpha_f = np.convolve(alpha_f, kernel, mode="same")
        alpha_r = np.convolve(alpha_r, kernel, mode="same")
        I = np.convolve(I, kernel, mode="same")

    if deg:
        alpha_f = np.rad2deg(alpha_f)
        alpha_r = np.rad2deg(alpha_r)
        I = np.rad2deg(I)
        ylab = "Δα = α_f - α_r [deg]"
    else:
        ylab = "Δα = α_f - α_r [rad]"

    plt.figure(figsize=(10, 5))
    plt.plot(time, I, linewidth=2, label="Δα (AV - AR)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)

    # zones visuelles
    plt.fill_between(time, 0, I, where=(I >= 0), alpha=0.15, label="Sous-virage")
    plt.fill_between(time, 0, I, where=(I < 0), alpha=0.15, label="Sur-virage")

    plt.xlabel("Temps [s]")
    plt.ylabel(ylab)
    plt.title("Indicateur sur/sous-virage basé sur les slip angles AV/AR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # optionnel: plot alpha_f et alpha_r
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, alpha_f, label="α_f (moy AV)", linewidth=1.8)
    # plt.plot(time, alpha_r, label="α_r (moy AR)", linewidth=1.8)
    # plt.xlabel("Temps [s]")
    # plt.ylabel("α [deg]" if deg else "α [rad]")
    # plt.title("Slip angles moyens AV vs AR")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

def plot_wheel_torques(time, result, labels=("FL","FR","RL","RR"), idx=(0,1,2,3)):
    """
    Trace les couples aux roues.

    Args:
        time (array): temps [s]
        result: résultat de simulation
        labels (tuple): labels roues
        idx (tuple): indices dans result.vehicle.u
    """
    plt.figure(figsize=(8, 4))
    for i, label in zip(idx, labels):
        plt.plot(time, result.vehicle.u[:, i], label=f"{label} torque")

    plt.xlabel("Temps [s]")
    plt.ylabel("Couple roue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def _moving_average(x, w: int):
    if w is None or w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")

def plot_lateral_force_balance_rho(time, result, labels=("FL","FR","RL","RR"), smooth=0):
    """
    Ratio de demande latérale ρ = |Fy_f| / (|Fy_f| + |Fy_r|)
    avec Fy_f = Fy_FL + Fy_FR et Fy_r = Fy_RL + Fy_RR.
    """
    if not hasattr(result, "tires") or not hasattr(result.tires, "Fy"):
        raise AttributeError("result.tires.Fy est requis pour tracer ρ (forces latérales pneus).")

    Fy = result.tires.Fy  # (N,4)
    Fy_f = Fy[:, 0] + Fy[:, 1]
    Fy_r = Fy[:, 2] + Fy[:, 3]

    rho = np.abs(Fy_f) / (np.abs(Fy_f) + np.abs(Fy_r) + 1e-12)
    rho = _moving_average(rho, smooth)

    plt.figure(figsize=(10, 4))
    plt.plot(time, rho, linewidth=2, label="ρ = |Fy_f| / (|Fy_f|+|Fy_r|)")
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1, label="neutre (0.5)")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Temps [s]")
    plt.ylabel("ρ [-]")
    plt.title("Répartition de la demande latérale AV/AR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_yaw_moment_from_tires(time, result, lf, lr, smooth=0):
    """
    Moment de lacet (approx) généré par les pneus :
    Mz ≈ lf*(Fy_FL+Fy_FR) - lr*(Fy_RL+Fy_RR)
    """
    if not hasattr(result, "tires") or not hasattr(result.tires, "Fy"):
        raise AttributeError("result.tires.Fy est requis pour tracer Mz (forces latérales pneus).")

    Fy = result.tires.Fy
    Fy_f = Fy[:, 0] + Fy[:, 1]
    Fy_r = Fy[:, 2] + Fy[:, 3]

    Mz = lf * Fy_f - lr * Fy_r
    Mz = _moving_average(Mz, smooth)

    plt.figure(figsize=(10, 4))
    plt.plot(time, Mz, linewidth=2, label="Mz pneus (approx)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Temps [s]")
    plt.ylabel("Mz [N·m]")
    plt.title("Moment de lacet généré par les pneus (approx.)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lateral_load_transfer(time, result, smooth=0, labels=("Front","Rear")):
    """
    ΔFz latéral :
      Avant : Fz_FR - Fz_FL
      Arrière : Fz_RR - Fz_RL
    """
    if not hasattr(result, "tires") or not hasattr(result.tires, "Fz"):
        raise AttributeError("result.tires.Fz est requis pour tracer ΔFz.")

    Fz = result.tires.Fz  # (N,4)
    dFz_f = Fz[:, 1] - Fz[:, 0]
    dFz_r = Fz[:, 3] - Fz[:, 2]

    dFz_f = _moving_average(dFz_f, smooth)
    dFz_r = _moving_average(dFz_r, smooth)

    plt.figure(figsize=(10, 4))
    plt.plot(time, dFz_f, label=f"ΔFz {labels[0]} (FR-FL)")
    plt.plot(time, dFz_r, label=f"ΔFz {labels[1]} (RR-RL)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Temps [s]")
    plt.ylabel("ΔFz [N]")
    plt.title("Transfert de charge latéral (ΔFz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_load_transfer_distribution_lambda(time, result, smooth=0):
    """
    Répartition de transfert :
      λ = (Fz_FR - Fz_FL) / (Fz_RR - Fz_RL)
    Interprétation (souvent) :
      λ > 1 : transfert surtout à l'avant (tendance sous-vireuse)
      λ < 1 : transfert surtout à l'arrière (tendance sur-vireuse)
    """
    if not hasattr(result, "tires") or not hasattr(result.tires, "Fz"):
        raise AttributeError("result.tires.Fz est requis pour tracer λ.")

    Fz = result.tires.Fz
    dFz_f = Fz[:, 1] - Fz[:, 0]
    dFz_r = Fz[:, 3] - Fz[:, 2]

    lam = dFz_f / (dFz_r + 1e-12)
    lam = _moving_average(lam, smooth)

    plt.figure(figsize=(10, 4))
    plt.plot(time, lam, linewidth=2, label="λ = (FR-FL)/(RR-RL)")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=1, label="λ=1")
    plt.xlabel("Temps [s]")
    plt.ylabel("λ [-]")
    plt.title("Répartition du transfert de charge AV/AR (λ)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roll_pitch(time, result, model, phi_key="phi", theta_key="theta",
                    dphi_key=None, dtheta_key=None, deg=True, smooth=0):
    """
    Trace φ(t), θ(t) (+ vitesses si disponibles).
    Les clés doivent exister dans model.state_keys.
    """
    keys = model.state_keys

    if phi_key not in keys or theta_key not in keys:
        raise KeyError(f"États requis introuvables dans model.state_keys: '{phi_key}' et/ou '{theta_key}'.")

    phi = result.vehicle.x[:, keys[phi_key]]
    theta = result.vehicle.x[:, keys[theta_key]]

    # vitesses (si pas fournies, on les approxime par dérivée numérique)
    if dphi_key is not None and dphi_key in keys:
        dphi = result.vehicle.x[:, keys[dphi_key]]
    else:
        dphi = np.gradient(phi, time)

    if dtheta_key is not None and dtheta_key in keys:
        dtheta = result.vehicle.x[:, keys[dtheta_key]]
    else:
        dtheta = np.gradient(theta, time)

    phi = _moving_average(phi, smooth)
    theta = _moving_average(theta, smooth)
    dphi = _moving_average(dphi, smooth)
    dtheta = _moving_average(dtheta, smooth)

    if deg:
        phi_plot = np.rad2deg(phi)
        theta_plot = np.rad2deg(theta)
        dphi_plot = np.rad2deg(dphi)
        dtheta_plot = np.rad2deg(dtheta)
        y1, y2 = "Angle [deg]", "Vitesse angulaire [deg/s]"
    else:
        phi_plot, theta_plot, dphi_plot, dtheta_plot = phi, theta, dphi, dtheta
        y1, y2 = "Angle [rad]", "Vitesse angulaire [rad/s]"

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(time, phi_plot, label="φ (roulis)")
    ax[0].plot(time, theta_plot, label="θ (tangage)")
    ax[0].set_ylabel(y1)
    ax[0].set_title("Roulis / tangage")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(time, dphi_plot, label="φ̇")
    ax[1].plot(time, dtheta_plot, label="θ̇")
    ax[1].set_xlabel("Temps [s]")
    ax[1].set_ylabel(y2)
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_wheel_power(time, result, model=None, torque_u_idx=(0,1,2,3),
                     omega_state_keys=None, labels=("FL","FR","RL","RR"), smooth=0):
    """
    Puissance aux roues : Pi = Ti * ωi.
    - Couples : result.vehicle.u[:, idx] (comme ton plot_wheel_torques)
    - ωi :
        * soit via omega_state_keys (tuple de 4 clés dans model.state_keys)
        * sinon essaie result.vehicle.omega (N,4) si présent
    """
    T = np.vstack([result.vehicle.u[:, i] for i in torque_u_idx]).T  # (N,4)

    if omega_state_keys is not None:
        if model is None:
            raise ValueError("model est requis si tu passes omega_state_keys.")
        for k in omega_state_keys:
            if k not in model.state_keys:
                raise KeyError(f"Clé ω '{k}' introuvable dans model.state_keys.")
        omega = np.vstack([result.vehicle.x[:, model.state_keys[k]] for k in omega_state_keys]).T
    elif hasattr(result.vehicle, "omega"):
        omega = result.vehicle.omega
    else:
        raise AttributeError(
            "Impossible de trouver ω. Fournis omega_state_keys=(...), "
            "ou expose result.vehicle.omega (N,4)."
        )

    P = T * omega  # (N,4)
    if smooth and smooth > 1:
        for i in range(4):
            P[:, i] = _moving_average(P[:, i], smooth)

    plt.figure(figsize=(10, 4))
    for i, lab in enumerate(labels):
        plt.plot(time, P[:, i], label=f"P {lab}")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Temps [s]")
    plt.ylabel("Puissance [W]")
    plt.title("Puissance aux roues : P = T · ω")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_wheel_work(time, result, model=None, torque_u_idx=(0,1,2,3),
                    omega_state_keys=None, labels=("FL","FR","RL","RR"), smooth_power=0):
    """
    Travail cumulé Wi(t) = ∫ Pi dt, avec Pi = Ti*ωi.
    """
    # réutilise la logique puissance
    T = np.vstack([result.vehicle.u[:, i] for i in torque_u_idx]).T

    if omega_state_keys is not None:
        if model is None:
            raise ValueError("model est requis si tu passes omega_state_keys.")
        omega = np.vstack([result.vehicle.x[:, model.state_keys[k]] for k in omega_state_keys]).T
    elif hasattr(result.vehicle, "omega"):
        omega = result.vehicle.omega
    else:
        raise AttributeError(
            "Impossible de trouver ω. Fournis omega_state_keys=(...), "
            "ou expose result.vehicle.omega (N,4)."
        )

    P = T * omega
    if smooth_power and smooth_power > 1:
        for i in range(4):
            P[:, i] = _moving_average(P[:, i], smooth_power)

    # intégration trapèzes
    W = np.zeros_like(P)
    dt = np.diff(time)
    for i in range(4):
        # W[k] = W[k-1] + 0.5*(P[k]+P[k-1])*dt
        W[1:, i] = np.cumsum(0.5 * (P[1:, i] + P[:-1, i]) * dt)

    plt.figure(figsize=(10, 4))
    for i, lab in enumerate(labels):
        plt.plot(time, W[:, i], label=f"W {lab}")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Temps [s]")
    plt.ylabel("Travail [J]")
    plt.title("Travail cumulé aux roues : W = ∫(T·ω) dt")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



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
steer_ctrl = StanleyController(k=0.1)

T = 15.0
dt = 0.0001
time_array = np.linspace(0, T, int(T/dt)+1)

for vref in range(10, 25, 5):

    # ---------------------------------------------------------
    # 3) Create trajectory
    # ---------------------------------------------------------
    R = 50
    traj = CircleTrajectory(v_ref=vref, R=R)

    traj.sample(0)
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


# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------

vref_arr = np.full_like(time_array, float(vref))  # (30001,)


plot_traction_efficiency(
    time_array,
    result,
    mu=1.05,
    labels=("FL","FR","RL","RR"),
    smooth=0
)

plot_wheel_torques(time_array, result)

plot_speed(time_array, result, model, vref)

plot_steering(time_array, result, front_idx=4, rear_idx=5)

plot_wheel_loads(time_array, result, labels=("FL","FR","RL","RR"))

plot_under_oversteer_from_slip(time_array, result, use_abs=True, deg=True, smooth=0)

plot_slip_mosaic(result, alpha_max=7.0, kappa_max=0.15, labels=("FL","FR","RL","RR"))

plot_trajectory_xy(sim_pos, pos, R=None, idx=-1, labels=("pred", "truth"))

plot_lateral_force_balance_rho(time_array, result, smooth=50)

plot_yaw_moment_from_tires(time_array, result, lf=params.lf, lr=params.lr, smooth=50)

plot_lateral_load_transfer(time_array, result, smooth=50)

plot_load_transfer_distribution_lambda(time_array, result, smooth=50)