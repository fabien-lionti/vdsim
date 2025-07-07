from abc import ABC, abstractmethod

class BaseTireModel(ABC):
    def __init__(self, params):
        pass
    @abstractmethod
    def get_fx0(
            self, 
            fz: float, 
            sigma: float
        ) -> float:
        pass

    @abstractmethod
    def get_fy0(
        self, 
        fz: float, 
        alpha: float
        ) -> float:
        pass
