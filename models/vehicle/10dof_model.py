import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from models.vehicle.base_model import BaseVehicleModel
from models.tires.registry import TIRE_MODEL_REGISTRY
from dataclasses import asdict

@dataclass
class VehiclePhysicalParams10DOF:
    g: float
    m: float
    lf: float
    lr: float
    h: float
    L1: float
    L2: float
    r: float
    iz: float
    ir: float
    ra: float
    s: float
    cx: float

    @property
    def l(self):
        return self.lf + self.lr

    @property
    def L(self):
        return self.L1 + self.L2

    @property
    def fz012(self):
        return (self.m * self.lr * self.g) / (2 * self.l)

    @property
    def fz034(self):
        return (self.m * self.lf * self.g) / (2 * self.l)

    @property
    def m_inv(self):
        return 1 / self.m

    @property
    def iz_inv(self):
        return 1 / self.iz

    @property
    def ir_inv(self):
        return 1 / self.ir

@dataclass
class VehicleConfig10DOF:
    vehicle: VehiclePhysicalParams10DOF
    tire1: Dict[str, Any]
    tire2: Dict[str, Any]
    tire3: Dict[str, Any]
    tire4: Dict[str, Any]

    def build_tire_models(self) -> Dict[str, Any]:
        tire_models = {}
        for name, cfg in {
            "tire1": self.tire1,
            "tire2": self.tire2,
            "tire3": self.tire3,
            "tire4": self.tire4
        }.items():
            cfg = cfg.copy()  # avoid mutating the original dict
            model_name = cfg.pop("model")
            model_class = TIRE_MODEL_REGISTRY.get(model_name)
            if not model_class:
                raise ValueError(f"Unknown tire model: {model_name}")
            tire_models[name] = model_class(cfg)
        return tire_models