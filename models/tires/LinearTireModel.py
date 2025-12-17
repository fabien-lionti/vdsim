from dataclasses import dataclass
from .BaseTireModel import BaseTireModel

@dataclass
class LinearTireParams:
    """
    Parameter container for a linear tire model.

    Attributes:
        Cx (float): Linear longitudinal stiffness coefficient.
        Cy (float): Linear lateral stiffness coefficient.
    """
    Cx: float
    Cy: float

class LinearTireModel(BaseTireModel): # BaseTireModel
    def __init__(self, params: LinearTireParams):
        super().__init__(params)
        self.params = params
        self.Cx: float = params.Cx
        self.Cy: float = params.Cy

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