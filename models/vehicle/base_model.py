from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union

class BaseVehicleModel(ABC):
    def __init__(self, params):
        self.params = params

    # @abstractmethod
    # def get_state_derivative(
    #     self,
    #     state: Union[np.ndarray, list],
    #     control: Union[np.ndarray, list]
    # ) -> np.ndarray:
    #     """
    #     Compute the time derivative of the vehicle state vector 
    #     given the current state and control inputs.

    #     Args:
    #         state (Union[np.ndarray, list]): Current state vector, shape (n_states,)
    #         control (Union[np.ndarray, list]): Control input vector, shape (n_inputs,)

    #     Returns:
    #         np.ndarray: Time derivative of the state vector, shape (n_states,)
    #     """
    #     pass

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
            - Traction (drive):   sigma = (r * w - vxp) / (r * |w|), clamped to max 1
            - Braking (drag):     sigma = (r * w - vxp) / |vxp|, clamped to min -1

        Args:
            vxp (float): Longitudinal velocity of the contact point [m/s]
            w   (float): Wheel angular velocity [rad/s]
            r   (float): Effective wheel radius [m]

        Returns:
            float: Longitudinal slip ratio in range [-1, 1]
                - Positive → traction
                - Negative → braking
                - Zero → pure rolling
        """
        wr = w * r
        # Case 1: Perfect rolling — the wheel speed matches ground speed exactly
        if np.isclose(wr, vxp):
            return 0.0
        # Case 2: Traction phase — wheel is spinning faster than the ground speed (acceleration)
        elif wr > vxp:
            # Edge case: wheel is stopped, but car is moving, infinite slip, return maximum (1.0)
            if np.isclose(w, 0.0):
                return 1.0
            sigma = (wr - vxp) / (r * abs(w))
            # Clamp the result to avoid exceeding physical bounds (e.g., bad data or extreme torque)
            return min(sigma, 1.0)
        # Case 3: Braking phase — the wheel is rotating slower than the ground speed
        else:
            # Edge case: ground speed is zero, but wheel is turning → infinite braking slip
            if np.isclose(vxp, 0.0):
                return -1.0
            sigma = (wr - vxp) / abs(vxp)
            return max(sigma, -1.0)