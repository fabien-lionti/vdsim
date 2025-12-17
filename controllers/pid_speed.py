# controllers/pid_speed.py
import numpy as np
from .base import BaseController

class SpeedPIDController(BaseController):
    """
    Basic PID controller for longitudinal speed control.
    Generates symmetric torque on all wheels.
    """

    def __init__(
        self, 
        kp=300.0, ki=10.0, kd=0.0,
        torque_limits=(-4000, 4000)
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.torque_limits = torque_limits
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, state, reference, dt=0.01):
        """
        Args:
            state: dict containing at least {"vx"}
            reference: dict containing at least {"v"}
            dt: sampling period [s]

        Returns:
            dict: control torques {t1,t2,t3,t4}
        """
        v = state["vx"]
        v_ref = reference["v"]
        error = v_ref - v

        # PID terms
        self.integral += error * dt
        derivative = (error - self.prev_error) / max(dt, 1e-6)
        self.prev_error = error

        torque = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # saturation
        torque = np.clip(torque, *self.torque_limits)
        # print(torque)
        return {
            "t1": torque,
            "t2": torque,
            "t3": torque,
            "t4": torque,
        }
