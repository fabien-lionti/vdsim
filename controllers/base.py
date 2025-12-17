# controllers/base.py

from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for all vehicle controllers.

    A controller receives the current state and a reference (trajectory,
    desired velocity, posture, etc.) and returns a dict of control outputs
    matching the vehicle's actuator interface.
    
        Example required keys for your models:
            {
                "t1", "t2", "t3", "t4",   # wheel torques
                "df", "dr"               # front / rear steering angles
            }

    Concrete implementations (PID, Stanley, MPC, etc.)
    should override the `compute` method.
    """

    @abstractmethod
    def compute(self, state: dict, reference: dict, **kwargs) -> dict:
        """
        Compute control commands given the current vehicle state and a reference.

        Args:
            state (dict): Must include needed model states, e.g.:
                {
                    "x", "y", "vx", "vy", "psi", ...
                }
            reference (dict): Must include trajectory/target signals, e.g.:
                {
                    "x", "y", "psi", "v", "kappa", ...
                }
            **kwargs: Optional context (dt, gains override, debug flags, etc.)

        Returns:
            dict: Keys must match the actuator convention, e.g.
                {
                    "t1", "t2", "t3", "t4",
                    "df", "dr"
                }
        """
        pass
