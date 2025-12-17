from dataclasses import dataclass
from .BaseTireModel import BaseTireModel
import numpy as np

@dataclass
class SimplifiedPacejkaTireParams:
    """
    Parameter container for the Simplified Pacejka tire model.

    Attributes:
        By (float): Lateral stiffness factor.
        Cy (float): Lateral shape factor.
        Dy (float): Lateral peak factor [N].
        Ey (float): Lateral curvature factor.

        Bx (float): Longitudinal stiffness factor.
        Cx (float): Longitudinal shape factor.
        Dx (float): Longitudinal peak factor [N].
        Ex (float): Longitudinal curvature factor.
    """

    # Lateral coefficients
    By: float
    Cy: float
    Dy: float
    Ey: float

    # Longitudinal coefficients
    Bx: float
    Cx: float
    Dx: float
    Ex: float

class SimplifiedPacejkaTireModel(BaseTireModel):
    """
    Simplified Pacejka tire force model (Magic Formula).

    This version implements a reduced Pacejka formulation suitable for
    real-time simulations and small computation budgets. It captures
    non-linear tire behavior more accurately than linear models, while
    remaining lighter than full Pacejka implementations.

    Args:
        params (SimplifiedPacejkaTireParams): Tire model parameters
            containing the Magic Formula coefficients.

    Attributes:
        By, Cy, Dy, Ey: Lateral coefficients.
        Bx, Cx, Dx, Ex: Longitudinal coefficients.
    """
    def __init__(self, params: SimplifiedPacejkaTireParams):
        super().__init__(params)
        self.params = params

        # Lateral
        self.By: float = params.By
        self.Cy: float = params.Cy
        self.Dy: float = params.Dy
        self.Ey: float = params.Ey

        # Longitudinal
        self.Bx: float = params.Bx
        self.Cx: float = params.Cx
        self.Dx: float = params.Dx
        self.Ex: float = params.Ex

    def get_fx0(self, fz0, fz, sigma):
        """
        Compute the longitudinal force using the Magic Formula.

        Fx = Dx * sin( Cx * atan( Bx * sigma - Ex * (Bx * sigma - atan(Bx * sigma)) ) ) * (fz / fz0)

        Args:
            fz (float): Normal load on the tire [N].
            sigma (float): Longitudinal slip ratio [-].

        Returns:
            float: Generated longitudinal force [N].
        """
        dfz = fz / fz0
        phi = self.Bx * sigma
        return self.Dx * np.sin(self.Cx * np.arctan(phi - self.Ex * (phi - np.arctan(phi)))) * dfz

    def get_fy0(self, fz0, fz, alpha):
        """
        Compute the lateral force using the Magic Formula.

        Fy = Dy * sin( Cy * atan( By * alpha - Ey * (By * alpha - atan(By * alpha)) ) ) * (fz / fz0)

        Args:
            fz (float): Normal load on the tire [N].
            alpha (float): Slip angle [rad].

        Returns:
            float: Generated lateral force [N].
        """
        dfz = fz / fz0
        phi = self.By * alpha
        return self.Dy * np.sin(self.Cy * np.arctan(phi - self.Ey * (phi - np.arctan(phi)))) * dfz