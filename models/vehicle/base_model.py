from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Tuple


class BaseVehicleModel(ABC):
    """
        Abstract base class for vehicle dynamic models.

        Subclasses must implement `get_dx__dt`, which defines the state-space
        dynamics of the model. Time integration (Euler, RK4, etc.) is handled
        externally in the `simulation.integrators` module.
    """
    def __init__(self, params):
        """
        Args:
            params: Model-specific parameter container (dataclass, dict, etc.).
        """
        self.params = params

    @staticmethod
    def get_slipping_angle(vx: float, vy: float, delta: float) -> float:
        """
        Compute the lateral slip angle for a wheel.

        Args:
            vx (float): Longitudinal velocity of the contact point [m/s]
            vy (float): Lateral velocity of the contact point [m/s]
            delta (float): Steering angle (wheel angle) [rad]

        Returns:
            float: Lateral slip angle [rad]
                - Positive means the tire is sliding outward
                - Zero means pure rolling in the steering direction
        """
        if vx == 0:
            return 0.0
        return delta - np.arctan(vy / vx)

    @staticmethod
    def get_slipping_rate(vxp: float, w: float, r: float) -> float:
        """
        Compute the longitudinal slip ratio (sigma) for a wheel.

        The slip ratio quantifies the difference between the rotational speed of the wheel 
        (converted to linear speed via the radius) and the actual longitudinal velocity 
        of the contact point on the ground. It's used to detect whether the wheel is in 
        traction, braking, or rolling conditions.

        Formula (simplified cases):
            - Perfect rolling:    sigma = 0
            - Traction (drive):   clamped to max 1
            - Braking (drag):     clamped to min -1

        Args:
            vxp (float): Longitudinal velocity of the contact point [m/s]
            w   (float): Wheel angular velocity [rad/s]
            r   (float): Effective wheel radius [m]

        Returns:
            float: Longitudinal slip ratio in range [-1, 1]
                - Positive - traction
                - Negative - braking
                - Zero - pure rolling
        """
        
        eps=1e-3
        wr = w * r  # vitesse périphérique de la roue
        v_ref = max(abs(vxp), eps)  # évite la division par zéro
        sigma = (wr - vxp) / v_ref
        return max(min(sigma, 1.0), -1.0)
        
    def get_dx__dt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_dx__dt(
        self,
        x: np.ndarray,
        u: np.ndarray,
        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute state derivatives and all physical outputs at the current step.

        Args:
            x (np.ndarray): Current state vector, shape (n_states,).
            u (np.ndarray): Current control input vector, shape (n_inputs,).

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray]]:
                - dx (np.ndarray): State derivatives dx/dt, shape (n_states,).
                - outputs (Dict[str, np.ndarray]): Physical variables to be logged,
                  such as tire forces, slips, accelerations, aero forces, angles, etc.
        """
        raise NotImplementedError