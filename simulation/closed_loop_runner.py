# simulation/closed_loop_runner.py

import numpy as np
from typing import Optional, Dict, Any

from controllers.base import BaseController
from simulation.results import SimulationResult
from simulation.integrators import euler_integration, rk4_integration
from simulation.integrators.step import integrate_one_step

class ClosedLoopRunner:
    """
    Closed-loop simulation runner.

    Orchestration of:
      - vehicle dynamic model (DOF7, DOF10, ...)
      - trajectory provider (BaseTrajectory)
      - speed controller (PID)
      - steering controller (Stanley, MPC, ...)
      - integrator (Euler or RK4)

    Example usage:
        runner = ClosedLoopRunner(model, speed_ctrl, steer_ctrl, trajectory)
        result = runner.run(x0, T=10.0, dt=0.001, method="euler")
    """

    def __init__(
        self,
        vehicle_model,
        speed_controller: BaseController,
        steering_controller: BaseController,
        trajectory,
    ):
        self.vehicle_model = vehicle_model
        self.speed_controller = speed_controller
        self.steering_controller = steering_controller
        self.trajectory = trajectory

    def reset(self, vehicle_model=None):
        """Optional reset of the runner or model."""
        if vehicle_model is not None:
            self.vehicle_model = vehicle_model

    def _build_controller_state(self, x: np.ndarray) -> dict:
        """
        Build a minimal state dict for controllers from the full state vector x.

        Works for both DOF7 and DOF10 as long as the model exposes `state_keys`.
        """
        sk = getattr(self.vehicle_model, "state_keys", {})

        def get(name: str, default: float = 0.0) -> float:
            idx = sk.get(name, None)
            if idx is None:
                return default
            return float(x[idx])

        return {
            "x":   get("x"),
            "vx":  get("vx"),
            "y":   get("y"),
            "vy":  get("vy"),
            "psi": get("psi"),
        }


    # ------------------------------------------------------------------
    # Single closed-loop step
    # ------------------------------------------------------------------
    def compute_controls(self, state: Dict[str, float], reference: Dict[str, float]) -> Dict[str, float]:
        """
        Compute the combined control command from speed + steering controllers.

        The two outputs MUST be merged into a single dict:
            {
                "t1","t2","t3","t4",
                "df","dr"
            }
        """
        u_speed = self.speed_controller.compute(state, reference)
        u_steer = self.steering_controller.compute(state, reference)

        # merge (speed has priority on wheel torque keys, steering on angles)
        return {**u_speed, **u_steer}

    def run(self, x0: np.ndarray, time_array: np.ndarray, method: str = "euler"):
        dt = time_array[1] - time_array[0]
        u_array = np.zeros((len(time_array), 6))

        x = x0.copy()  # current state

        for k, t in enumerate(time_array):

            traj_point = self.trajectory.sample(t)
            reference = traj_point.to_dict()

            # Build controller state (adapt to your model indexing)
            # state = dict(
            #     x=x[0],
            #     vx=x[1],
            #     y=x[2],
            #     vy=x[3],
            #     psi=x[4],
            # )
            state = self._build_controller_state(x)

            u_cmd = self.compute_controls(state, reference)
            u_array[k] = np.array([
                u_cmd.get("t1", 0.0), u_cmd.get("t2", 0.0),
                u_cmd.get("t3", 0.0), u_cmd.get("t4", 0.0),
                u_cmd.get("df", 0.0), u_cmd.get("dr", 0.0),
            ])

            # propagate state immediately
            x, _ = integrate_one_step(self.vehicle_model, x, u_array[k], dt, method)

        # Now use your existing logging integrators (don’t touch them)
        if method.lower() == "euler":
            result = euler_integration(self.vehicle_model, x0, u_array, time_array)
        else:
            result = rk4_integration(self.vehicle_model, x0, u_array, time_array)

        return result

