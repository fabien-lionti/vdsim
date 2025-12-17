# controllers/stanley.py
import numpy as np
from .base import BaseController

class StanleyController(BaseController):
    """
    Stanley lateral controller for trajectory tracking.
    Produces a front steering angle (rear steer = 0 by default).
    """

    def __init__(self, k=1.0, max_steer=np.deg2rad(35)):
        self.k = k               # gain for cross track error
        self.max_steer = max_steer

    def compute(self, state, reference):
        """
        Args:
            state: must contain {x, y, psi, vx}
            reference: must contain {x, y, psi, v}

        Returns:
            dict: {"df", "dr"}
        """

        # 🔹 Extract data
        x, y, psi, vx = state["x"], state["y"], state["psi"], state["vx"]
        xr, yr, psi_ref = reference["x"], reference["y"], reference["psi"]

        # Cross-track error
        dx = xr - x
        dy = yr - y
        error_ct = np.sin(psi) * dx - np.cos(psi) * dy

        # Heading error
        error_heading = psi_ref - psi
        error_heading = np.arctan2(np.sin(error_heading), np.cos(error_heading))

        # Stanley control law
        steer_correction = np.arctan2(self.k * error_ct, vx + 1e-3)

        delta = error_heading + steer_correction

        # saturation
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        return {
            "df": delta,
            "dr": 0.0  # NO rear steering for now
        }
