# trajectories/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class TrajectoryPoint:
    t: float
    x: float
    y: float
    psi: float      # heading [rad]
    v: float        # ref speed [m/s]
    kappa: float  # curvature [1/m], optionnel

    def to_dict(self) -> dict:
        return dict(t=self.t, x=self.x, y=self.y, psi=self.psi, v=self.v, kappa=self.kappa)

class BaseTrajectory(ABC):
    """Abstract base class for reference trajectories."""

    @abstractmethod
    def sample(self, t: float) -> TrajectoryPoint:
        """Return reference at time t."""
        pass

    def sample_array(self, time_array: np.ndarray) -> np.ndarray:
        """
        Retourne un tableau numpy de forme (N, 6):
        [t, x, y, psi, v, kappa]
        """
        return np.array([
            [
                tp.t,
                tp.x,
                tp.y,
                tp.psi,
                tp.v,
                tp.kappa,
            ]
            for tp in (self.sample(float(t)) for t in time_array)
        ])
