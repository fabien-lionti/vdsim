from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class TireLog:
    """
    Logged tire-level variables over the simulation horizon.

    Attributes:
        Fx (Optional[np.ndarray]): Longitudinal tire forces [N], shape (T, n_tires).
        Fy (np.ndarray): Lateral tire forces [N], shape (T, n_tires).
        Fz (Optional[np.ndarray]): Normal tire loads [N], shape (T, n_tires).
        kappa (Optional[np.ndarray]): Longitudinal slips [-], shape (T, n_tires).
        alpha (Optional[np.ndarray]): Lateral slip angles [rad], shape (T, n_tires).
        omega (Optional[np.ndarray]): Wheel angular speeds [rad/s], shape (T, n_tires).
    """
    Fx: Optional[np.ndarray] = None
    Fy: Optional[np.ndarray] = None
    Fz: Optional[np.ndarray] = None
    kappa: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    omega: Optional[np.ndarray] = None

@dataclass
class VehicleLog:
    """
    Logged vehicle-level variables over the simulation horizon.

    Mandatory fields are state trajectory, its time derivative, inputs,
    and the associated timestamp. All other fields are optional and depend
    on the specific vehicle model capabilities.

    Attributes:
        time (np.ndarray): Time vector [s], shape (T,).
        x (np.ndarray): State trajectory, shape (T, n_states).
        dx (np.ndarray): State derivatives `dx/dt`, shape (T, n_states).
        u (np.ndarray): Control inputs applied, shape (T, n_inputs).

        ax (Optional[np.ndarray]): Longitudinal acceleration at CG [m/s²], shape (T,).
        ay (Optional[np.ndarray]): Lateral acceleration at CG [m/s²], shape (T,).
        az (Optional[np.ndarray]): Vertical acceleration at CG [m/s²], shape (T,).

        yaw (Optional[np.ndarray]): Yaw angle ψ [rad], shape (T,).
        yaw_rate (Optional[np.ndarray]): Yaw rate ψ̇ [rad/s], shape (T,).
        yaw_acc (Optional[np.ndarray]): Yaw angular acceleration ψ̈ [rad/s²], shape (T,).

        roll (Optional[np.ndarray]): Roll angle φ [rad], shape (T,).
        roll_rate (Optional[np.ndarray]): Roll rate φ̇ [rad/s], shape (T,).
        roll_acc (Optional[np.ndarray]): Roll angular acceleration φ̈ [rad/s²], shape (T,).

        pitch (Optional[np.ndarray]): Pitch angle θ [rad], shape (T,).
        pitch_rate (Optional[np.ndarray]): Pitch rate θ̇ [rad/s], shape (T,).
        pitch_acc (Optional[np.ndarray]): Pitch angular acceleration θ̈ [rad/s²], shape (T,).

        beta (Optional[np.ndarray]): Vehicle sideslip angle β [rad], shape (T,).

        Fx_aero (Optional[np.ndarray]): Aerodynamic longitudinal force (drag) [N], shape (T,).
        Fy_aero (Optional[np.ndarray]): Aerodynamic lateral force (side force) [N], shape (T,).
        Fz_aero (Optional[np.ndarray]): Aerodynamic vertical force (lift/downforce) [N], shape (T,).
    """
    time: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    u: np.ndarray

    ax: Optional[np.ndarray] = None
    ay: Optional[np.ndarray] = None
    az: Optional[np.ndarray] = None

    yaw: Optional[np.ndarray] = None
    yaw_rate: Optional[np.ndarray] = None
    yaw_acc: Optional[np.ndarray] = None

    roll: Optional[np.ndarray] = None
    roll_rate: Optional[np.ndarray] = None
    roll_acc: Optional[np.ndarray] = None

    pitch: Optional[np.ndarray] = None
    pitch_rate: Optional[np.ndarray] = None
    pitch_acc: Optional[np.ndarray] = None

    beta: Optional[np.ndarray] = None

    Fx_aero: Optional[np.ndarray] = None
    Fy_aero: Optional[np.ndarray] = None
    Fz_aero: Optional[np.ndarray] = None

@dataclass
class SimulationResult:
    """
    Full simulation result, including vehicle-level states and tire-level
    forces/slips. Also exposes raw logged variables for extended analysis.

    Attributes:
        vehicle (VehicleLog): Vehicle-level trajectories and signals.
        tires (TireLog): Tire-level forces and slip variables.
    """
    vehicle: VehicleLog
    tires: TireLog