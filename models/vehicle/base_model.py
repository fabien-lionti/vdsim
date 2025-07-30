from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Union

class BaseVehicleModel(ABC):
    def __init__(self, params):
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
        
    def get_dx__dt(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        pass
        
    def euler_integration(self, x0, u: np.ndarray, time_array: np.ndarray) -> np.ndarray:
        """
        Perform Euler integration with time-varying control inputs.

        Args:
            x0: Initial state (np.ndarray)
            u: Control input array of shape (T, n_inputs)
            time_array: Time vector of shape (T,)

        Returns:
            Trajectory: np.ndarray of shape (T, n_states)
        """
        dt = time_array[1] - time_array[0]
        x = np.array(x0, dtype=float)
        traj = [x.copy()]

        for i in range(1, len(time_array)):
            control_i = u[i - 1]
            dx = self.get_dx__dt(x, control_i)
            x += dt * np.asarray(dx)
            traj.append(x.copy())

        return np.stack(traj)


    def rk4_integration(self, x0, u, time_array):
        dt = time_array[1] - time_array[0]
        x = np.array(x0)
        traj = [x.copy()]
        for _ in time_array[1:]:
            k1 = self.get_state_derivative(x, u)
            k2 = self.get_state_derivative(x + 0.5 * dt * k1, u)
            k3 = self.get_state_derivative(x + 0.5 * dt * k2, u)
            k4 = self.get_state_derivative(x + dt * k3, u)
            x += (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(x.copy())
        return np.stack(traj)