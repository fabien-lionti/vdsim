import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from models.vehicle.base_model import BaseVehicleModel
from models.tires.registry import TIRE_MODEL_REGISTRY
from dataclasses import asdict
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class VehiclePhysicalParams7DOF:
    """
    Physical parameters for a 7-DOF vehicle model.
    """

    g: float  # [m/s²] Gravity acceleration
    m: float  # [kg] Vehicle mass
    lf: float  # [m] Distance from CG to front axle
    lr: float  # [m] Distance from CG to rear axle
    h: float  # [m] Height of the center of gravity
    L1: float  # [m] Distance from CG to front wheel center (longitudinal)
    L2: float  # [m] Distance from CG to rear wheel center (longitudinal)
    r: float  # [m] Wheel radius
    iz: float  # [kg·m²] Yaw moment of inertia
    ir: float  # [kg·m²] Wheel rotational inertia
    ra: float  # [N·s/m] Aerodynamic drag coefficient
    s: float  # [m²] Frontal area
    cx: float  # [-] Air drag coefficient

    @property
    def l(self) -> float:
        """[m] Total wheelbase (lf + lr)"""
        return self.lf + self.lr

    @property
    def L(self) -> float:
        """[m] Total wheelbase (L1 + L2) — for alternative reference frame"""
        return self.L1 + self.L2

    @property
    def fz012(self) -> float:
        """
        [N] Static vertical load on front axle (shared by front-left and front-right tires)
        Computed using static weight transfer assumption.
        """
        return (self.m * self.lr * self.g) / (2 * self.l)

    @property
    def fz034(self) -> float:
        """
        [N] Static vertical load on rear axle (shared by rear-left and rear-right tires)
        Computed using static weight transfer assumption.
        """
        return (self.m * self.lf * self.g) / (2 * self.l)

    @property
    def m_inv(self) -> float:
        """[1/kg] Inverse of vehicle mass"""
        return 1 / self.m

    @property
    def iz_inv(self) -> float:
        """[1/kg·m²] Inverse of yaw inertia"""
        return 1 / self.iz

    @property
    def ir_inv(self) -> float:
        """[1/kg·m²] Inverse of wheel inertia"""
        return 1 / self.ir

@dataclass
class VehicleConfig7DOF:
    vehicle: VehiclePhysicalParams7DOF
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

class DOF7(BaseVehicleModel):
    """
    7 Degrees-of-Freedom Vehicle Model with Linear Tire Forces.

    State vector (10):
        x       - Global x-position [m]
        vx      - Longitudinal velocity [m/s]
        y       - Global y-position [m]
        vy      - Lateral velocity [m/s]
        psi     - Yaw angle [rad]
        psidt   - Yaw rate [rad/s]
        omega1  - Wheel angular velocity FL [rad/s]
        omega2  - Wheel angular velocity FR [rad/s]
        omega3  - Wheel angular velocity RL [rad/s]
        omega4  - Wheel angular velocity RR [rad/s]

    Input vector (6):
        tf1     - Drive torque front-left [Nm]
        tf2     - Drive torque front-right [Nm]
        tr1     - Drive torque rear-left [Nm]
        tr2     - Drive torque rear-right [Nm]
        df      - Front steering angle [rad]
        dr      - Rear steering angle [rad]

    Parameters:
        A dictionary containing:
            m, Iz      - mass and inertia
            lf, lr     - distances to axles
            Cf, Cr     - cornering stiffness
            R          - wheel radius
            J          - wheel inertia
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.n_states: int = 10
        self.n_inputs: int = 6
        self.params = params

        for key, value in asdict(params.vehicle).items():
            setattr(self, key, value)
        self.fz012 = params.vehicle.fz012
        self.fz034 = params.vehicle.fz034
        self.l = params.vehicle.l
        self.L = params.vehicle.L
        self.m_inv = params.vehicle.m_inv
        self.iz_inv = params.vehicle.iz_inv
        self.ir_inv = params.vehicle.ir_inv
        
        # Compute constants
        self.fz012 = (self.m * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.m * self.lf * self.g) / (2 * self.l)
        self.m_inv = 1 / self.m
        self.iz_inv = 1 / self.iz
        self.ir_inv = 1 / self.ir

        # Tires
        tires = params.build_tire_models()
        self.f1 = tires['tire1']
        self.f2 = tires['tire2']
        self.f3 = tires['tire3']
        self.f4 = tires['tire4']
    
    def get_state_derivatives(self, vx, vy, fx1, fx2, fx3, fx4, fy1, fy2, fy3, fy4, fxp1, fxp2, fxp3, fxp4, psi, psidt, t1, t2, t3, t4, faero):

        self.xdt = vx * np.cos(psi) - vy * np.sin(psi)
        self.vxdt = vy * psidt  + self.m_inv * (fx1 + fx2 + fx3 + fx4 - faero) 
        self.ydt = vx * np.sin(psi) + vy * np.cos(psi)
        self.vydt = -vx * psidt + self.m_inv * (fy1 + fy2 + fy3 + fy4)
        
        self.psidt = psidt
        self.psidt2 = self.iz_inv * (self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4))

        self.omega1dt = self.ir_inv * (t1 - fxp1 * self.r)
        self.omega2dt = self.ir_inv * (t2 - fxp2 * self.r)
        self.omega3dt = self.ir_inv * (t3 - fxp3 * self.r)
        self.omega4dt = self.ir_inv * (t4 - fxp4 * self.r)

    def get_faero(self, vx):
        return 0.5 * self.ra * self.s * self.cx * vx**2

    def get_dx__dt(self, state, control):

        x, vx, y, vy, psi, psidt, w1, w2, w3, w4 = state

        t1, t2, t3, t4, d1, d2 = control

        # Project on tire frame
        vxp1 = (vx - self.L1 * psidt) * np.cos(d1) + (vy + self.lf * psidt) * np.sin(d1)
        vxp2 = (vx + self.L1 * psidt) * np.cos(d1) + (vy + self.lf * psidt) * np.sin(d1)
        vxp3 = (vx - self.L1 * psidt) * np.cos(d2) + (vy - self.lf * psidt) * np.sin(d2)
        vxp4 = (vx + self.L1 * psidt) * np.cos(d2) + (vy - self.lf * psidt) * np.sin(d2)

        vyp1 = (vy + self.lf * psidt) * np.cos(d1) - (vx - self.L1 * psidt) * np.sin(d1)
        vyp2 = (vy + self.lf * psidt) * np.cos(d1) - (vx + self.L1 * psidt) * np.sin(d1)
        vyp3 = (vy - self.lf * psidt) * np.cos(d2) - (vx - self.L1 * psidt) * np.sin(d2)
        vyp4 = (vy - self.lf * psidt) * np.cos(d2) - (vx + self.L1 * psidt) * np.sin(d2)

        # Compute lateral slip angles
        self.alpha1 = self.get_slipping_angle(vxp1, vyp1, d1)
        self.alpha2 = self.get_slipping_angle(vxp2, vyp2, d1)
        self.alpha3 = self.get_slipping_angle(vxp3, vyp3, d2)
        self.alpha4 = self.get_slipping_angle(vxp4, vyp4, d2)

        # Compute longitudinal slip
        self.sigma1 = self.get_slipping_rate(vxp1, w1, self.r)
        self.sigma2 = self.get_slipping_rate(vxp2, w2, self.r)
        self.sigma3 = self.get_slipping_rate(vxp3, w3, self.r)
        self.sigma4 = self.get_slipping_rate(vxp4, w4, self.r)
        
        # Compute lateral tire forces
        fyp1 = self.f1.get_fy0(self.fz012, self.fz012, self.alpha1)
        fyp2 = self.f2.get_fy0(self.fz012, self.fz012, self.alpha2)
        fyp3 = self.f3.get_fy0(self.fz034, self.fz034, self.alpha3)
        fyp4 = self.f4.get_fy0(self.fz034, self.fz034, self.alpha4)

        fxp1 = self.f1.get_fx0(self.fz012, self.fz012, self.sigma1)
        fxp2 = self.f2.get_fx0(self.fz012, self.fz012, self.sigma2)
        fxp3 = self.f3.get_fx0(self.fz034, self.fz034, self.sigma3)
        fxp4 = self.f4.get_fx0(self.fz034, self.fz034, self.sigma4)

        # Project on carbody frame
        self.fx1 = fxp1 * np.cos(d1) - fyp1 * np.sin(d1)
        self.fx2 = fxp2 * np.cos(d1) - fyp2 * np.sin(d1)
        self.fx3 = fxp3 * np.cos(d2) - fyp3 * np.sin(d2)
        self.fx4 = fxp4 * np.cos(d2) - fyp4 * np.sin(d2)

        self.fy1 = fyp1 * np.cos(d1) + fxp1 * np.sin(d1)
        self.fy2 = fyp2 * np.cos(d1) + fxp2 * np.sin(d1)
        self.fy3 = fyp3 * np.cos(d2) + fxp3 * np.sin(d2)
        self.fy4 = fyp4 * np.cos(d2) + fxp4 * np.sin(d2)

        faero = self.get_faero(vx)

        self.get_state_derivatives(vx, vy, self.fx1, self.fx2, self.fx3, self.fx4, self.fy1, self.fy2, self.fy3, self.fy4, fxp1, fxp2, fxp3, fxp4, psi, psidt, t1, t2, t3, t4, faero)

        return [self.xdt, self.vxdt, self.ydt, self.vydt, self.psidt, self.psidt2, self.omega1dt, self.omega2dt, self.omega3dt, self.omega4dt]
if __name__ == "__main__":

    vehicle_params = VehiclePhysicalParams7DOF(
        g=9.81, 
        m=1500, 
        lf=1.2, 
        lr=1.6, 
        h=0.5,
        L1=1.0, 
        L2=1.6, 
        r=0.3, 
        iz=2500.0, 
        ir=1.2,
        ra=0.015, 
        s=0.01, 
        cx=0.3
    )
    
    tire_params = {
        "model": "linear",
        "Cx": 80000.0,
        "Cy": 80000.0
    }

    config = VehicleConfig7DOF(
        vehicle=vehicle_params,
        tire1=tire_params,
        tire2=tire_params,
        tire3=tire_params,
        tire4=tire_params
    )
    
    # Instantiate the model
    model = DOF7(config)

    initial_state = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])   # 10D state vector
    dt = 0.0005
    sim_time = 10.0
    time_array = np.arange(0, sim_time, dt)

    # Control vector for a single time step
    tf = 0 # Drive torque (front) [Nm]
    tr = 0 # Drive torque (rear) [Nm]
    df = 0.01 # Front steering angle δf [rad]
    dr = 0
    control_input = np.array([tf, tf, tr, tr, df, dr])

    # Expand to match time steps
    u_array = np.tile(control_input, (len(time_array), 1))  # shape: (T, 6)

    trajectory = model.euler_integration(initial_state, u_array, time_array)

    x = trajectory[:, 0]
    vx = trajectory[:, 1]
    y = trajectory[:, 2]
    vy = trajectory[:, 3]
    psi = trajectory[:, 4]
    psidt = trajectory[:, 5]
    w1 = trajectory[:, 6]
    w2 = trajectory[:, 7]
    w3 = trajectory[:, 8]
    w4 = trajectory[:, 9]

    plt.figure(figsize=(12, 8))
    plt.plot(time_array, psidt, label='Vy')
    plt.show()