from dataclasses import dataclass
from models.tires import BaseTireModel

class LinearTireModel(): # BaseTireModel
    def __init__(self, params):
        super().__init__() # fz0=params.fz0
        self.Cx = params.Cx
        self.Cy = params.Cy
        self.fz0 = params.fz0

    def get_fx0(
            self, 
            fz: float, 
            sigma: float
            ) -> float:
        dfz = fz / self.fz0
        return dfz * sigma * self.Cx

    def get_fy0(
            self, 
            fz: float, 
            alpha: float
            ) -> float:
        dfz = fz / self.fz0
        return dfz * alpha * self.Cy
