from abc import ABC, abstractmethod

class BaseTireModel(ABC):
    """
    Abstract base class defining the interface for tire force models.

    All tire models (e.g. linear, simplified Pacejka, etc.) must inherit
    from this class and implement the minimal functions required to compute
    longitudinal and lateral forces.

    Args:
        params (dict): Model parameters. Their structure depends on the
            specific implementation (e.g. stiffness values, Pacejka
            coefficients, etc.).
    """
    def __init__(self, params):
        self.params = params
        # pass

    @abstractmethod
    def get_fx0(
        
            self, 
            fz0: float,
            fz: float, 
            sigma: float
        ) -> float:
        """
        Compute the longitudinal tire force at zero slip angle.

        Args:
            fz (float): Normal load applied on the tire (in Newtons).
            sigma (float): Longitudinal slip ratio (dimensionless).

        Returns:
            float: Generated longitudinal force (in Newtons).
        """
        pass

    @abstractmethod
    def get_fy0(
        self, 
        fz0: float,
        fz: float, 
        alpha: float
        ) -> float:
        """
        Compute the lateral tire force at zero longitudinal slip.

        Args:
            fz (float): Normal load applied on the tire (in Newtons).
            alpha (float): Slip angle (in radians).

        Returns:
            float: Generated lateral force (in Newtons).
        """
        pass
