from dataclasses import dataclass
# from models.tires import BaseTireModel
import numpy as np

@dataclass
class SimplifiedPacejkaTireParams:
    fz0: float

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

class SimplifiedPacejkaTireModel(): # BaseTireModel
    def __init__(self, params: SimplifiedPacejkaTireParams):
        super().__init__(fz0=params.fz0)

        # Lateral
        self.By = params.By
        self.Cy = params.Cy
        self.Dy = params.Dy
        self.Ey = params.Ey

        # Longitudinal
        self.Bx = params.Bx
        self.Cx = params.Cx
        self.Dx = params.Dx
        self.Ex = params.Ex

        self.fz0 = params.fz0

    def get_fx0(self, fz, sigma):
        dfz = fz / self.fz0
        phi = self.Bx * sigma
        return self.Dx * np.sin(self.Cx * np.arctan(phi - self.Ex * (phi - np.arctan(phi)))) * dfz

    def get_fy0(self, fz, alpha):
        dfz = fz / self.fz0
        phi = self.By * alpha
        return self.Dy * np.sin(self.Cy * np.arctan(phi - self.Ey * (phi - np.arctan(phi)))) * dfz
    
class LinearTireModel(): # BaseTireModel
    def __init__(self, params):
        super().__init__() # fz0=params.fz0
        self.Cx = params['Cx']
        self.Cy = params['Cy']
        # self.fz0 = params.fz0

    def get_fx0(
            self,
            fz0: float,
            fz: float, 
            sigma: float
            ) -> float:
        dfz = fz / fz0
        return dfz * sigma * self.Cx

    def get_fy0(
            self,
            fz0: float, 
            fz: float, 
            alpha: float
            ) -> float:
        dfz = fz / fz0
        return dfz * alpha * self.Cy

TIRE_MODEL_REGISTRY = {
    "linear": LinearTireModel,
    "pacejka": SimplifiedPacejkaTireModel,
}