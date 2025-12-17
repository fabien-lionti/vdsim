import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from models.vehicle.base_model import BaseVehicleModel
from models.tires.LinearTireModel import LinearTireParams
from models.tires.registry import TIRE_MODEL_REGISTRY
from dataclasses import asdict
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simulation.integrators import euler_integration

@dataclass
class VehiclePhysicalParams10DOF:
    """
    Physical parameters for a 10 Degrees-of-Freedom vehicle model,
    with suspension properties indexed by wheel position (0:FL, 1:FR, 2:RL, 3:RR).

    Attributes:
        g (float): Gravitational acceleration [m/s²]
        m (float): Total vehicle mass [kg]
        ms (float): Suspended mass of the vehicle [kg]
        lf (float): Distance from CG to front axle [m]
        lr (float): Distance from CG to rear axle [m]
        h (float): Height of CG above ground [m]
        L1 (float): Half of front track width [m]
        L2 (float): Half of rear track width [m]
        r (float): Effective tire rolling radius [m]
        ix (float): Roll moment of inertia (around x-axis) [kg·m²]
        iy (float): Pitch moment of inertia (around y-axis) [kg·m²]
        iz (float): Yaw moment of inertia (around z-axis) [kg·m²]
        ir (float): Rotational inertia of one wheel [kg·m²]
        ra (float): Rolling resistance coefficient [N·s/m]
        s (float): Frontal surface area [m²]
        cx (float): Aerodynamic drag coefficient [-]

        Suspension stiffness [N/m]:
        ks1: Front-left
        ks2: Front-right
        ks3: Rear-left
        ks4: Rear-right

        Suspension damping [N·s/m]:
        ds1: Front-left
        ds2: Front-right
        ds3: Rear-left
        ds4: Rear-right
    """

    # Vehicle geometry & dynamics
    g: float
    m: float
    ms: float
    lf: float
    lr: float
    h: float
    L1: float
    L2: float
    r: float
    ix: float
    iy: float
    iz: float
    ir: float
    ra: float
    s: float
    cx: float

    # Suspension stiffness [N/m]
    ks1: float  # Front-left
    ks2: float  # Front-right
    ks3: float  # Rear-left
    ks4: float  # Rear-right

    # Suspension damping [N·s/m]
    ds1: float  # Front-left
    ds2: float  # Front-right
    ds3: float  # Rear-left
    ds4: float  # Rear-right

    @property
    def l(self):
        """Total wheelbase [m]"""
        return self.lf + self.lr

    @property
    def L(self):
        """Total track width [m]"""
        return self.L1 + self.L2

    @property
    def fz012(self):
        """Static vertical load on each front wheel [N] (approximate)"""
        return (self.m * self.lr * self.g) / (2 * self.l)

    @property
    def fz034(self):
        """Static vertical load on each rear wheel [N] (approximate)"""
        return (self.m * self.lf * self.g) / (2 * self.l)

    @property
    def m_inv(self):
        """Inverse of vehicle mass [1/kg]"""
        return 1 / self.m
    
    @property
    def ms_inv(self):
        """Inverse of vehicle mass [1/kg]"""
        return 1 / self.ms
    
    @property
    def ix_inv(self):
        """Inverse of yaw moment of inertia [1/(kg·m²)]"""
        return 1 / self.ix
    
    @property
    def iy_inv(self):
        """Inverse of yaw moment of inertia [1/(kg·m²)]"""
        return 1 / self.iy

    @property
    def iz_inv(self):
        """Inverse of yaw moment of inertia [1/(kg·m²)]"""
        return 1 / self.iz

    @property
    def ir_inv(self):
        """Inverse of wheel inertia [1/(kg·m²)]"""
        return 1 / self.ir

# @dataclass
# class VehicleConfig10DOF:
#     vehicle: VehiclePhysicalParams10DOF
#     tire1: Dict[str, Any]
#     tire2: Dict[str, Any]
#     tire3: Dict[str, Any]
#     tire4: Dict[str, Any]

#     def build_tire_models(self) -> Dict[str, Any]:
#         tire_models = {}
#         for name, cfg in {
#             "tire1": self.tire1,
#             "tire2": self.tire2,
#             "tire3": self.tire3,
#             "tire4": self.tire4
#         }.items():
#             cfg = cfg.copy()  # avoid mutating the original dict
#             model_name = cfg.pop("model")
#             model_class = TIRE_MODEL_REGISTRY.get(model_name)
#             if not model_class:
#                 raise ValueError(f"Unknown tire model: {model_name}")
#             tire_models[name] = model_class(cfg)
#         return tire_models

@dataclass
class VehicleConfig10DOF:
    vehicle: VehiclePhysicalParams10DOF
    tire1: Any  # ou Union[LinearTireParams, SimplifiedPacejkaTireParams]
    tire2: Any
    tire3: Any
    tire4: Any

    def build_tire_models(self) -> Dict[str, Any]:
        tire_models: Dict[str, Any] = {}
        for name, params in {
            "tire1": self.tire1,
            "tire2": self.tire2,
            "tire3": self.tire3,
            "tire4": self.tire4,
        }.items():
            model_class = TIRE_MODEL_REGISTRY.get(type(params))
            if model_class is None:
                raise ValueError(f"Unknown tire params type: {type(params)!r}")
            tire_models[name] = model_class(params)
        return tire_models
    
class DOF10(BaseVehicleModel):
    """
    10 Degrees-of-Freedom Vehicle Model with Linear Tire Forces.

    State vector (16):
        x       - Global x-position [m]
        vx      - Longitudinal velocity [m/s]
        y       - Global y-position [m]
        vy      - Lateral velocity [m/s]
        z       - Global z-position [m]
        vz      - Vertical velocity [m/s]
        phi     - Roll angle [rad]
        phidt   - Roll rate [rad/s]
        theta   - Pitch angle [rad]
        thetadt - Pitch rate [rad/s]
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
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.n_states: int = 16
        self.n_inputs: int = 6
        self.params = params
        self.state_keys = {
            "x": 0,
            "vx": 1,
            "y": 2,
            "vy": 3,
            "z": 4,
            "vz": 5,
            "theta": 6,
            "thetadt": 7,
            "phi": 8,
            "phidt": 9,
            "psi": 10,
            "psidt": 11,
            "omega1": 12,
            "omega2": 13,
            "omega3": 14,
            "omega4": 15,
        }


        for key, value in asdict(params.vehicle).items():
            setattr(self, key, value)
        self.fz012 = params.vehicle.fz012
        self.fz034 = params.vehicle.fz034
        self.l = params.vehicle.l
        self.L = params.vehicle.L
        self.m_inv = params.vehicle.m_inv
        self.ms_inv = params.vehicle.ms_inv
        self.ix_inv = params.vehicle.ix_inv
        self.iy_inv = params.vehicle.iy_inv
        self.iz_inv = params.vehicle.iz_inv
        self.ir_inv = params.vehicle.ir_inv
        
        # Compute constants
        self.fz012 = (self.ms * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.ms * self.lf * self.g) / (2 * self.l)
        # self.m_inv = 1 / self.m
        # self.iz_inv = 1 / self.iz
        # self.ir_inv = 1 / self.ir

        # Tires
        tires = params.build_tire_models()
        self.f1 = tires['tire1']
        self.f2 = tires['tire2']
        self.f3 = tires['tire3']
        self.f4 = tires['tire4']
    
    def get_slipping_rates(self, vxp1, w1, vxp2, w2, vxp3, w3, vxp4, w4, r):

        sigma1 = self.get_slipping_rate(vxp1, w1, r)
        sigma2 = self.get_slipping_rate(vxp2, w2, r)
        sigma3 = self.get_slipping_rate(vxp3, w3, r)
        sigma4 = self.get_slipping_rate(vxp4, w4, r)

        return sigma1, sigma2, sigma3, sigma4
    
    def get_wheel_speed_in_vehicle_frame(self, vx, vy, psidt, idx):
        if idx == 1:
            return vx - self.L1 * psidt, vy + self.lf * psidt
        elif idx == 2:
            return vx + self.L1 * psidt, vy + self.lf * psidt
        elif idx == 3:
            return vx - self.L2 * psidt, vy - self.lr * psidt
        elif idx == 4:
            return vx + self.L2 * psidt, vy - self.lr * psidt

    def get_slipping_angles(self, vx1, vy1, vx2, vy2, vx3, vy3, vx4, vy4, delta1, delta2, delta3, delta4):

        alpha1 = self.get_slipping_angle(vx1, vy1, delta1)
        alpha2 = self.get_slipping_angle(vx2, vy2, delta2)
        alpha3 = self.get_slipping_angle(vx3, vy3, delta3)
        alpha4 = self.get_slipping_angle(vx4, vy4, delta4)

        return alpha1, alpha2, alpha3, alpha4
    
    # def get_suspension_travel(self, theta, phi):

    #     dzg1 = self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
    #     dzg2 = -self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
    #     dzg3 = self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)
    #     dzg4 = -self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)

    #     return dzg1, dzg2, dzg3, dzg4
    
    # def get_suspension_speed_travel(self, theta, thetadt, phi, phidt):

    #     dzg1dt = self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
    #     dzg2dt = -self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
    #     dzg3dt =  self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)
    #     dzg4dt = -self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)

    #     return dzg1dt, dzg2dt, dzg3dt, dzg4dt
    
    # def get_vertical_normal_forces(self, dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt):

    #     fs1 = -self.ks1 * dzg1 - self.ds1 * dzg1dt
    #     fs2 = -self.ks2 * dzg2 - self.ds2 * dzg2dt
    #     fs3 = -self.ks3 * dzg3 - self.ds3 * dzg3dt
    #     fs4 = -self.ks4 * dzg4 - self.ds4 * dzg4dt

    #     fz1 = self.fz012 + fs1
    #     fz2 = self.fz012 + fs2
    #     fz3 = self.fz034 + fs3
    #     fz4 = self.fz034 + fs4

    #     return fz1, fz2, fz3, fz4, fs1, fs2, fs3, fs4

    # def get_suspension_travel(self, z, theta, phi):
    #     """
    #     Compute suspension deflection (sprung mass → wheel center) for each corner.

    #     Args:
    #         z (float): Heave displacement of the sprung mass [m]
    #                 (z = 0 at static equilibrium).
    #         theta (float): Pitch angle [rad]
    #         phi (float): Roll angle [rad]

    #     Returns:
    #         dzg1, dzg2, dzg3, dzg4 (floats): Suspension travel at
    #         front-left, front-right, rear-left, rear-right.
    #     """

    #     # Contribution pitch/roll (ton expression existante)
    #     dzg1_pr = self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
    #     dzg2_pr = -self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
    #     dzg3_pr = self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)
    #     dzg4_pr = -self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)

    #     # Ajout du heave global (même déplacement sur les 4 coins)
    #     dzg1 = z + dzg1_pr
    #     dzg2 = z + dzg2_pr
    #     dzg3 = z + dzg3_pr
    #     dzg4 = z + dzg4_pr

    #     return dzg1, dzg2, dzg3, dzg4

    def get_suspension_travel(self, z, theta, phi):
        """
        Suspension travel at each corner.

        Args:
            z (float): absolute CG height [m]
            theta (float): pitch angle [rad]
            phi (float): roll angle [rad]

        Convention:
            - z = self.h_cg  → static equilibrium (no suspension deflection)
            - z > self.h_cg  → body moves up
            - z < self.h_cg  → body moves down
        """
        # Heave relatif par rapport à la hauteur statique du CG
        z_rel = z - self.h

        # Contribution pitch/roll (tes formules d’origine)
        dzg1_pr = self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg2_pr = -self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg3_pr = self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)
        dzg4_pr = -self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)

        # Ajout de la composante heave relative (même sur les 4 coins)
        dzg1 = z_rel + dzg1_pr
        dzg2 = z_rel + dzg2_pr
        dzg3 = z_rel + dzg3_pr
        dzg4 = z_rel + dzg4_pr

        return dzg1, dzg2, dzg3, dzg4

    def get_suspension_speed_travel(self, z, zdot, theta, thetadt, phi, phidt):
        """
        Compute time derivative of suspension travel for each corner.

        Args:
            z (float): Heave displacement [m]
            zdot (float): Heave velocity [m/s]
            theta (float): Pitch angle [rad]
            thetadt (float): Pitch rate [rad/s]
            phi (float): Roll angle [rad]
            phidt (float): Roll rate [rad/s]

        Returns:
            dzg1dt, dzg2dt, dzg3dt, dzg4dt: Suspension velocity at each corner.
        """

        # Contribution pitch/roll (ton expression existante)
        dzg1dt_pr = (
            self.L1 * thetadt * np.cos(theta)
            + self.lf * thetadt * np.sin(theta) * np.sin(phi)
            - self.lf * phidt * np.cos(theta) * np.cos(phi)
        )
        dzg2dt_pr = (
            -self.L1 * thetadt * np.cos(theta)
            + self.lf * thetadt * np.sin(theta) * np.sin(phi)
            - self.lf * phidt * np.cos(theta) * np.cos(phi)
        )
        dzg3dt_pr = (
            self.L2 * thetadt * np.cos(theta)
            - self.lr * thetadt * np.sin(theta) * np.sin(phi)
            + self.lr * phidt * np.cos(theta) * np.cos(phi)
        )
        dzg4dt_pr = (
            -self.L2 * thetadt * np.cos(theta)
            - self.lr * thetadt * np.sin(theta) * np.sin(phi)
            + self.lr * phidt * np.cos(theta) * np.cos(phi)
        )

        # Ajout du heave (zdot) identique sur les 4 coins
        dzg1dt = zdot + dzg1dt_pr
        dzg2dt = zdot + dzg2dt_pr
        dzg3dt = zdot + dzg3dt_pr
        dzg4dt = zdot + dzg4dt_pr

        return dzg1dt, dzg2dt, dzg3dt, dzg4dt
    
    def get_vertical_normal_forces(self, dzg1, dzg2, dzg3, dzg4,
                               dzg1dt, dzg2dt, dzg3dt, dzg4dt):

        fs1 = -self.ks1 * dzg1 - self.ds1 * dzg1dt
        fs2 = -self.ks2 * dzg2 - self.ds2 * dzg2dt
        fs3 = -self.ks3 * dzg3 - self.ds3 * dzg3dt
        fs4 = -self.ks4 * dzg4 - self.ds4 * dzg4dt

        fz1 = self.fz012 + fs1
        fz2 = self.fz012 + fs2
        fz3 = self.fz034 + fs3
        fz4 = self.fz034 + fs4

        return fz1, fz2, fz3, fz4, fs1, fs2, fs3, fs4


    
    def get_longitudinal_tire_forces(self, sigma1, alpha1, fz1, sigma2, alpha2, fz2, sigma3, alpha3, fz3, sigma4, alpha4, fz4):
        '''Return longitudinal tires forces in the tire frame'''

        # Compute longitudinal tire forces
        fx1 = self.f1.get_fx0(self.fz012, self.fz012, sigma1)
        fx2 = self.f2.get_fx0(self.fz012, self.fz012, sigma2)
        fx3 = self.f3.get_fx0(self.fz034, self.fz034, sigma3)
        fx4 = self.f4.get_fx0(self.fz034, self.fz034, sigma4)

        return fx1, fx2, fx3, fx4
    
    def get_lateral_tire_forces(self, sigma1, alpha1, fz1, sigma2, alpha2, fz2, sigma3, alpha3, fz3, sigma4, alpha4, fz4):
        '''Return lateral tire forces in the tire frame'''

        # Compute lateral tire forces
        fy1 = self.fl_tire_model.get_fy0(fz1, alpha1)
        fy2 = self.fl_tire_model.get_fy0(fz2, alpha2)
        fy3 = self.fl_tire_model.get_fy0(fz3, alpha3)
        fy4 = self.fl_tire_model.get_fy0(fz4, alpha4)

        return fy1, fy2, fy3, fy4
    
    def project_from_tire_to_carbody_frame(self, uxt, uyt, delta, phi, theta, fz):
        uxc = (uxt * np.cos(delta) - uyt * np.sin(delta)) * np.cos(phi) + fz * np.sin(phi)
        uyc = (uxt * np.cos(delta) - uyt * np.sin(delta)) * np.sin(theta) * np.sin(phi) + (uxt * np.sin(delta) + uyt * np.cos(delta)) * np.cos(theta) + fz * np.sin(theta) * np.cos(phi)
        return uxc, uyc
    
    def project_from_carbody_to_tire_frame(self, uxc, uyc, delta):
        uxt = uxc * np.cos(delta) + uyc * np.sin(delta)
        uyt = -uxc * np.sin(delta) + uyc * np.cos(delta) 
        return uxt, uyt
    
    def get_faero(self, vx):
        return 0.5 * self.ra * self.s * self.cx * vx**2
    
    # def get_state_derivatives(self, x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
    #                           fx1, fx2, fx3, fx4, 
    #                           fy1, fy2, fy3, fy4,
    #                           fz1, fz2, fz3, fz4,
    #                           fs1, fs2, fs3, fs4,
    #                           fxp1, fxp2, fxp3, fxp4,
    #                           faero):
        
    #     fz1 = fz1 * np.cos(x9) * np.cos(x7)
    #     fz2 = fz2 * np.cos(x9) * np.cos(x7)
    #     fz3 = fz3 * np.cos(x9) * np.cos(x7)
    #     fz4 = fz4 * np.cos(x9) * np.cos(x7)

    #     self.x1dt = x2 * np.cos(x11) - x4 * np.sin(x11)
    #     self.x2dt = x12 * x4 - x10 * x6 + self.m_inv * (fx1 + fx2 + fx3 + fx4 - faero * np.cos(x9))
    #     self.x3dt = x2 * np.sin(x11) + x4 * np.cos(x11)
    #     self.x4dt = -x12 * x2 + x8 * x6 + self.m_inv * (fy1 + fy2 + fy3 + fy4)
    #     self.x5dt = x6
    #     self.x6dt = self.ms_inv * (fz1 + fz2 + fz3 + fz4) - self.g * np.cos(x7) * np.cos(x9)
        
    #     self.x7dt = x8
    #     self.x8dt = self.ix_inv * ( self.L1 * (fz1 + fz3 - fz2 - fz4) + x5 * (fy1 + fy2 + fy3 + fy4) )
    #     self.x9dt = x10
    #     self.x10dt = self.iy_inv * ( -self.lf * (fz1 + fz2 ) + self.lr * (fz3 + fz4) - x5 * (fx1 + fx2 + fx3 + fx4) )
    #     self.x11dt = x12
    #     self.x12dt = self.iz_inv * ( self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4) + self.L1 * (fx2 + fx4 - fx1 - fx3) )
        
    #     self.x13dt = (u1 - self.r * fxp1) / (self.ir)
    #     self.x14dt = (u2 - self.r * fxp2) / (self.ir)
    #     self.x15dt = (u3 - self.r * fxp3) / (self.ir)
    #     self.x16dt = (u4 - self.r * fxp4) / (self.ir)

    def get_state_derivatives(
        self,
        Vx, Vy,
        z, Vz,
        theta, thetadt,
        phi, phidt,
        psi, psidt,
        t1, t2, t3, t4,
        fx1, fx2, fx3, fx4,
        fy1, fy2, fy3, fy4,
        fz1, fz2, fz3, fz4,
        fs1, fs2, fs3, fs4,
        fxp1, fxp2, fxp3, fxp4,
        faero,
    ):

        # Correction orientation verticale des forces normales
        fz1 *= np.cos(phi) * np.cos(theta)
        fz2 *= np.cos(phi) * np.cos(theta)
        fz3 *= np.cos(phi) * np.cos(theta)
        fz4 *= np.cos(phi) * np.cos(theta)

        # ------------------------------------------------------------
        # Positions globales
        # ------------------------------------------------------------
        self.X_dt = Vx * np.cos(psi) - Vy * np.sin(psi)
        self.Y_dt = Vx * np.sin(psi) + Vy * np.cos(psi)

        # ------------------------------------------------------------
        # Dynamique longitudinale / latérale
        # ------------------------------------------------------------
        self.Vx_dt = psidt * Vy - phidt * Vz + self.m_inv * (
            fx1 + fx2 + fx3 + fx4 - faero * np.cos(phi)
        )

        self.Vy_dt = -psidt * Vx + thetadt * Vz + self.m_inv * (
            fy1 + fy2 + fy3 + fy4
        )

        # ------------------------------------------------------------
        # Heave
        # ------------------------------------------------------------
        
        self.Z_dt = Vz
        self.Vz_dt = self.ms_inv * (fz1 + fz2 + fz3 + fz4) - self.g * np.cos(theta) * np.cos(phi)

        # print(self.ms_inv * (fz1 + fz2 + fz3 + fz4), self.g * np.cos(theta) * np.cos(phi))
        # ------------------------------------------------------------
        # Tangage (pitch)
        # ------------------------------------------------------------
        self.theta_dt = thetadt
        self.thetadt_dt = self.ix_inv * (
            self.L1 * (fz1 + fz3 - fz2 - fz4) + z * (fy1 + fy2 + fy3 + fy4)
        )
        # print(fz1 + fz3, fz2 + fz4)
        # ------------------------------------------------------------
        # Roulis (roll)
        # ------------------------------------------------------------
        self.phi_dt = phidt
        self.phidt_dt = self.iy_inv * (
            -self.lf * (fz1 + fz2) + self.lr * (fz3 + fz4) - z * (fx1 + fx2 + fx3 + fx4)
        )

        # ------------------------------------------------------------
        # Lacet (yaw)
        # ------------------------------------------------------------
        self.psi_dt = psidt
        self.psidt_dt = self.iz_inv * (
            self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4)
            + self.L1 * (fx2 + fx4 - fx1 - fx3)
        )

        # ------------------------------------------------------------
        # Dynamique roue
        # ------------------------------------------------------------
        self.omega1_dt = (t1 - self.r * fxp1) / self.ir
        self.omega2_dt = (t2 - self.r * fxp2) / self.ir
        self.omega3_dt = (t3 - self.r * fxp3) / self.ir
        self.omega4_dt = (t4 - self.r * fxp4) / self.ir


    # def get_dx__dt(self, x, u):
        
        
    #     ### Control input correspondances
    #     # u1 = t1
    #     # u2 = t2
    #     # u3 = t3
    #     # u4 = t4
    #     # u5 = df
    #     # u6 = dr

    #     ### State correspondances
    #     # x1 = x
    #     # x2 = Vx
    #     # x3 = y
    #     # x4 = Vy
    #     # x5 = z
    #     # x6 = Vz
    #     # x7 = theta
    #     # x8 = thetadt
    #     # x9 = phi
    #     # x10 = phidt
    #     # x11 = psi
    #     # x12 = psidt
    #     # x13 = omega1
    #     # x14 = omega2
    #     # x15 = omega3
    #     # x16 = omega4

    #     x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = x

    #     u1, u2, u3, u4, u5, u6 = u
        
    #     # Compute vxp for each wheels in wheel basis

    #     vxp1 = (x2 - self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
    #     vxp2 = (x2 + self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
    #     vxp3 = (x2 - self.L1 * x12) * np.cos(u6) + (x4 - self.lf * x12) * np.sin(u6)
    #     vxp4 = (x2 + self.L1 * x12) * np.cos(u6) + (x4 - self.lf * x12) * np.sin(u6)

    #     vyp1 = (x4 + self.lf * x12) * np.cos(u5) - (x2 - self.L1 * x12) * np.sin(u5)
    #     vyp2 = (x4 + self.lf * x12) * np.cos(u5) - (x2 + self.L1 * x12) * np.sin(u5)
    #     vyp3 = (x4 - self.lf * x12) * np.cos(u6) - (x2 - self.L1 * x12) * np.sin(u6)
    #     vyp4 = (x4 - self.lf * x12) * np.cos(u6) - (x2 + self.L1 * x12) * np.sin(u6)

    #     # Compute lateral slip angles
    #     self.alpha1 = self.get_slipping_angle(vxp1, vyp1, u5)
    #     self.alpha2 = self.get_slipping_angle(vxp2, vyp2, u5)
    #     self.alpha3 = self.get_slipping_angle(vxp3, vyp3, u6)
    #     self.alpha4 = self.get_slipping_angle(vxp4, vyp4, u6)

    #     # Compute longitudinal slip
    #     self.sigma1 = self.get_slipping_rate(vxp1, x13, self.r)
    #     self.sigma2 = self.get_slipping_rate(vxp2, x14, self.r)
    #     self.sigma3 = self.get_slipping_rate(vxp3, x15, self.r)
    #     self.sigma4 = self.get_slipping_rate(vxp4, x16, self.r)

    #     # Compute suspension and normal reaction forces
    #     dzg1, dzg2, dzg3, dzg4 = self.get_suspension_travel(theta = x7, phi = x9)
    #     dzg1dt, dzg2dt, dzg3dt, dzg4dt = self.get_suspension_speed_travel(theta = x7, thetadt = x8, phi = x9, phidt = x10)
    #     self.fz1, self.fz2, self.fz3, self.fz4, self.fs1, self.fs2, self.fs3, self.fs4 = self.get_vertical_normal_forces(dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt)

    #     fxp1 = self.f1.get_fx0(self.fz012, self.fz1, self.sigma1)
    #     fxp2 = self.f2.get_fx0(self.fz012, self.fz2, self.sigma2)
    #     fxp3 = self.f3.get_fx0(self.fz034, self.fz3, self.sigma3)
    #     fxp4 = self.f4.get_fx0(self.fz034, self.fz4, self.sigma4)

    #     fyp1 = self.f1.get_fy0(self.fz012, self.fz1, self.alpha1)
    #     fyp2 = self.f2.get_fy0(self.fz012, self.fz2, self.alpha2)
    #     fyp3 = self.f3.get_fy0(self.fz034, self.fz3, self.alpha3)
    #     fyp4 = self.f4.get_fy0(self.fz034, self.fz4, self.alpha4)

    #     # Get tire forces in vehicle basis

    #     self.fx1, self.fy1 = self.project_from_tire_to_carbody_frame(fxp1, fyp1, u5, x9, x7, self.fz1)
    #     self.fx2, self.fy2 = self.project_from_tire_to_carbody_frame(fxp2, fyp2, u5, x9, x7, self.fz2)
    #     self.fx3, self.fy3 = self.project_from_tire_to_carbody_frame(fxp3, fyp3, u6, x9, x7, self.fz3)
    #     self.fx4, self.fy4 = self.project_from_tire_to_carbody_frame(fxp4, fyp4, u6, x9, x7, self.fz4)

    #     # TODO
    #     faero = self.get_faero(x2)

    #     self.get_state_derivatives(x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
    #                         self.fx1, self.fx2, self.fx3, self.fx4, 
    #                         self.fy1, self.fy2, self.fy3, self.fy4,
    #                         self.fz1, self.fz2, self.fz3, self.fz4,
    #                         self.fs1, self.fs2, self.fs3, self.fs4,
    #                         fxp1, fxp2, fxp3, fxp4,
    #                         faero)
        
    #     dx = np.array(
    #     [
    #         self.x1dt,
    #         self.x2dt,
    #         self.x3dt,
    #         self.x4dt,
    #         self.x5dt,
    #         self.x6dt,
    #         self.x7dt,
    #         self.x8dt,
    #         self.x9dt,
    #         self.x10dt,
    #         self.x11dt,
    #         self.x12dt,
    #         self.x13dt,
    #         self.x14dt,
    #         self.x15dt,
    #         self.x16dt,
    #     ],
    #     dtype=float,
    #     )

    #     # ------------------------------------------------------------------
    #     # Outputs à logger (pas d’états, seulement grandeurs dérivées)
    #     # ------------------------------------------------------------------
    #     Fx = np.array([self.fx1, self.fx2, self.fx3, self.fx4], dtype=float)
    #     Fy = np.array([self.fy1, self.fy2, self.fy3, self.fy4], dtype=float)
    #     Fz = np.array([self.fz1, self.fz2, self.fz3, self.fz4], dtype=float)
    #     kappa = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4], dtype=float)
    #     alpha = np.array([self.alpha1, self.alpha2, self.alpha3, self.alpha4], dtype=float)
    #     Fs = np.array([self.fs1, self.fs2, self.fs3, self.fs4], dtype=float)

    #     # Accélérations CG dans le repère véhicule
    #     ax = self.x2dt       # dvx/dt
    #     ay = self.x4dt       # dvy/dt
    #     az = self.x6dt       # d(zdot)/dt

    #     outputs: Dict[str, np.ndarray] = {
    #         "Fx": Fx,
    #         "Fy": Fy,
    #         "Fz": Fz,
    #         "kappa": kappa,
    #         "alpha": alpha,
    #         "Fs": Fs,  # suspension forces (optionnel, pour analyse)
    #         "Fx_aero": np.array([faero], dtype=float),
    #         "ax": np.array([ax], dtype=float),
    #         "ay": np.array([ay], dtype=float),
    #         "az": np.array([az], dtype=float),
    #     }

    #     return dx, outputs

    def get_dx__dt(self, x: np.ndarray, u: np.ndarray):
        # ------------------------------------------------------------------
        # Unpack state and inputs using physical names
        # ------------------------------------------------------------------
        x_pos, Vx, y_pos, Vy, z, Vz, theta, thetadt, phi, phidt, psi, psidt, omega1, omega2, omega3, omega4 = x
        t1, t2, t3, t4, df, dr = u

        # ------------------------------------------------------------------
        # Compute vxp / vyp for each wheel in wheel basis
        # ------------------------------------------------------------------
        # Front axle uses df, rear axle uses dr
        vxp1 = (Vx - self.L1 * psidt) * np.cos(df) + (Vy + self.lf * psidt) * np.sin(df)
        vxp2 = (Vx + self.L1 * psidt) * np.cos(df) + (Vy + self.lf * psidt) * np.sin(df)
        vxp3 = (Vx - self.L1 * psidt) * np.cos(dr) + (Vy - self.lf * psidt) * np.sin(dr)
        vxp4 = (Vx + self.L1 * psidt) * np.cos(dr) + (Vy - self.lf * psidt) * np.sin(dr)

        vyp1 = (Vy + self.lf * psidt) * np.cos(df) - (Vx - self.L1 * psidt) * np.sin(df)
        vyp2 = (Vy + self.lf * psidt) * np.cos(df) - (Vx + self.L1 * psidt) * np.sin(df)
        vyp3 = (Vy - self.lf * psidt) * np.cos(dr) - (Vx - self.L1 * psidt) * np.sin(dr)
        vyp4 = (Vy - self.lf * psidt) * np.cos(dr) - (Vx + self.L1 * psidt) * np.sin(dr)

        # ------------------------------------------------------------------
        # Lateral slip angles
        # ------------------------------------------------------------------
        self.alpha1 = self.get_slipping_angle(vxp1, vyp1, df)
        self.alpha2 = self.get_slipping_angle(vxp2, vyp2, df)
        self.alpha3 = self.get_slipping_angle(vxp3, vyp3, dr)
        self.alpha4 = self.get_slipping_angle(vxp4, vyp4, dr)

        # ------------------------------------------------------------------
        # Longitudinal slip ratios
        # ------------------------------------------------------------------
        self.sigma1 = self.get_slipping_rate(vxp1, omega1, self.r)
        self.sigma2 = self.get_slipping_rate(vxp2, omega2, self.r)
        self.sigma3 = self.get_slipping_rate(vxp3, omega3, self.r)
        self.sigma4 = self.get_slipping_rate(vxp4, omega4, self.r)

        # ------------------------------------------------------------------
        # Suspension travel and vertical forces
        # ------------------------------------------------------------------
        dzg1, dzg2, dzg3, dzg4 = self.get_suspension_travel(z, theta=theta, phi=phi)
        dzg1dt, dzg2dt, dzg3dt, dzg4dt = self.get_suspension_speed_travel(
            z, Vz, theta=theta, thetadt=thetadt, phi=phi, phidt=phidt
        )

        self.fz1, self.fz2, self.fz3, self.fz4, self.fs1, self.fs2, self.fs3, self.fs4 = \
            self.get_vertical_normal_forces(
                dzg1, dzg2, dzg3, dzg4,
                dzg1dt, dzg2dt, dzg3dt, dzg4dt,
            )

        # ------------------------------------------------------------------
        # Tire forces in tire frame (fxp, fyp)
        # ------------------------------------------------------------------
        fxp1 = self.f1.get_fx0(self.fz012, self.fz1, self.sigma1)
        fxp2 = self.f2.get_fx0(self.fz012, self.fz2, self.sigma2)
        fxp3 = self.f3.get_fx0(self.fz034, self.fz3, self.sigma3)
        fxp4 = self.f4.get_fx0(self.fz034, self.fz4, self.sigma4)

        fyp1 = self.f1.get_fy0(self.fz012, self.fz1, self.alpha1)
        fyp2 = self.f2.get_fy0(self.fz012, self.fz2, self.alpha2)
        fyp3 = self.f3.get_fy0(self.fz034, self.fz3, self.alpha3)
        fyp4 = self.f4.get_fy0(self.fz034, self.fz4, self.alpha4)

        # ------------------------------------------------------------------
        # Projection des forces dans le repère châssis
        # project_from_tire_to_carbody_frame(fxp, fyp, delta, phi, theta, fz)
        # ------------------------------------------------------------------
        self.fx1, self.fy1 = self.project_from_tire_to_carbody_frame(fxp1, fyp1, df, phi, theta, self.fz1)
        self.fx2, self.fy2 = self.project_from_tire_to_carbody_frame(fxp2, fyp2, df, phi, theta, self.fz2)
        self.fx3, self.fy3 = self.project_from_tire_to_carbody_frame(fxp3, fyp3, dr, phi, theta, self.fz3)
        self.fx4, self.fy4 = self.project_from_tire_to_carbody_frame(fxp4, fyp4, dr, phi, theta, self.fz4)

        # ------------------------------------------------------------------
        # Aérodynamique
        # ------------------------------------------------------------------
        faero = self.get_faero(Vx)

        # ------------------------------------------------------------------
        # Dérivées d'état (implémentation existante)
        # get_state_derivatives(Vx, Vy, z, Vz, theta, thetadt, phi, phidt, psi, psidt, t1..t4, forces, ...)
        # ------------------------------------------------------------------
        self.get_state_derivatives(
            Vx, Vy,
            z, Vz,
            theta, thetadt,
            phi, phidt,
            psi, psidt,
            t1, t2, t3, t4,
            self.fx1, self.fx2, self.fx3, self.fx4,
            self.fy1, self.fy2, self.fy3, self.fy4,
            self.fz1, self.fz2, self.fz3, self.fz4,
            self.fs1, self.fs2, self.fs3, self.fs4,
            fxp1, fxp2, fxp3, fxp4,
            faero,
        )

        dx = np.array([
            self.X_dt,
            self.Vx_dt,
            self.Y_dt,
            self.Vy_dt,
            self.Z_dt,
            self.Vz_dt,
            self.theta_dt,
            self.thetadt_dt,
            self.phi_dt,
            self.phidt_dt,
            self.psi_dt,
            self.psidt_dt,
            self.omega1_dt,
            self.omega2_dt,
            self.omega3_dt,
            self.omega4_dt
        ])


        # ------------------------------------------------------------------
        # Outputs à logger (pas d’états, seulement grandeurs dérivées)
        # ------------------------------------------------------------------
        Fx = np.array([self.fx1, self.fx2, self.fx3, self.fx4], dtype=float)
        Fy = np.array([self.fy1, self.fy2, self.fy3, self.fy4], dtype=float)
        Fz = np.array([self.fz1, self.fz2, self.fz3, self.fz4], dtype=float)
        kappa = np.array([self.sigma1, self.sigma2, self.sigma3, self.sigma4], dtype=float)
        alpha = np.array([self.alpha1, self.alpha2, self.alpha3, self.alpha4], dtype=float)
        Fs = np.array([self.fs1, self.fs2, self.fs3, self.fs4], dtype=float)

        # Accélérations CG dans le repère véhicule
        ax = self.Vx_dt   # dVx/dt
        ay = self.Vy_dt   # dVy/dt
        az = self.Vz_dt   # dVz/dt

        outputs: Dict[str, np.ndarray] = {
            "Fx": Fx,
            "Fy": Fy,
            "Fz": Fz,
            "kappa": kappa,
            "alpha": alpha,
            "Fs": Fs,
            "Fx_aero": np.array([faero], dtype=float),
            "ax": np.array([ax], dtype=float),
            "ay": np.array([ay], dtype=float),
            "az": np.array([az], dtype=float),
        }

        return dx, outputs

    
if __name__ == "__main__":

    # vehicle_params = VehiclePhysicalParams10DOF(
    #     g=9.81,              # Gravité
    #     m=1500.0,            # Masse totale [kg]
    #     ms=1300.0,           # Masse suspendue [kg]
    #     lf=1.6,              # Distance CG → essieu avant [m]
    #     lr=1.6,              # Distance CG → essieu arrière [m]
    #     h=0.55,              # Hauteur CG [m]
    #     L1=0.75,             # Demi-voie avant [m]
    #     L2=0.75,             # Demi-voie arrière [m]
    #     r=0.3,               # Rayon effectif roue [m]

    #     # Inerties
    #     ix=400.0,            # Inertie en roulis [kg·m²]
    #     iy=1200.0,           # Inertie en tangage [kg·m²]
    #     iz=2500.0,           # Inertie en lacet [kg·m²]
    #     ir=1.2,              # Inertie roue [kg·m²]

    #     # Forces résistives
    #     ra=12.0,             # Résistance au roulement [N·s/m]
    #     s=2.2,               # Surface frontale [m²]
    #     cx=0.32,             # Coefficient traînée [-]

    #     # Raideurs suspension (symétrique ici) [N/m]
    #     ks1=30000.0,         # FL
    #     ks2=30000.0,         # FR
    #     ks3=30000.0,         # RL
    #     ks4=30000.0,         # RR

    #     # Amortisseurs suspension [N·s/m]
    #     ds1=3500.0,          # FL
    #     ds2=3500.0,          # FR
    #     ds3=3500.0,          # RL
    #     ds4=3500.0           # RR
    # )


    # tire_params = {
    #     "model": "linear",
    #     "Cx": 80000.0,
    #     "Cy": 80000.0
    # }

    # config = VehicleConfig10DOF(
    #     vehicle=vehicle_params,
    #     tire1=tire_params,
    #     tire2=tire_params,
    #     tire3=tire_params,
    #     tire4=tire_params
    # )
    
    # # Instantiate the model
    # model = DOF10(config)

    # # Optional: simulate a single step
    # initial_state = np.array([0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # 10D state vector
    
    # dt = 0.0001
    # sim_time = 10.0
    # time_array = np.arange(0, sim_time, dt)

    # # Control vector for a single time step
    # tf = 0 # Drive torque (front) [Nm]
    # tr = 0 # Drive torque (rear) [Nm]
    # df = 0.001 # Front steering angle δf [rad]
    # dr = 0
    # control_input = np.array([tf, tf, tr, tr, df, dr])

    # # Expand to match time steps
    # u_array = np.tile(control_input, (len(time_array), 1))  # shape: (T, 6)

    # trajectory = model.euler_integration(initial_state, u_array, time_array)
    
    # x = trajectory[:, 0]
    # vx = trajectory[:, 1]
    # y = trajectory[:, 2]
    # vy = trajectory[:, 3]
    # z = trajectory[:, 4]
    # vz = trajectory[:, 5]
    # phi = trajectory[:, 6]
    # phidt = trajectory[:, 7]
    # theta = trajectory[:, 8]
    # thetadt = trajectory[:, 9]
    # psi = trajectory[:, 10]
    # psidt = trajectory[:, 11]
    # w1 = trajectory[:, 12]
    # w2 = trajectory[:, 13]
    # w3 = trajectory[:, 14]
    # w4 = trajectory[:, 15]

    # plt.figure(figsize=(12, 8))
    # plt.plot(time_array, z, label='vx')
    # plt.show()

    vehicle_params = VehiclePhysicalParams10DOF(
    g=9.81,
    m=1500.0,
    ms=1300.0,
    lf=1.6,
    lr=1.6,
    h=0.55,
    L1=0.75,
    L2=0.75,
    r=0.3,

    # Inertias
    ix=400.0,
    iy=1200.0,
    iz=2500.0,
    ir=1.2,

    # Resistive forces
    ra=12.0,
    s=2.2,
    cx=0.32,

    # Suspension stiffness [N/m]
    ks1=30000.0,
    ks2=30000.0,
    ks3=30000.0,
    ks4=30000.0,

    # Suspension damping [N·s/m]
    ds1=3500.0,
    ds2=3500.0,
    ds3=3500.0,
    ds4=3500.0,
    )

    # tire_params = {
    #     "model": "linear",
    #     "Cx": 80000.0,
    #     "Cy": 80000.0,
    # }

    tire_params = LinearTireParams(
        Cx=80000.0,
        Cy=80000.0,
    )

    config = VehicleConfig10DOF(
        vehicle=vehicle_params,
        tire1=tire_params,
        tire2=tire_params,
        tire3=tire_params,
        tire4=tire_params,
    )

    # Instantiate the model
    model = DOF10(config)

    # 16D state vector: [X, vx, Y, vy, z, zdot, theta, thetadot, phi, phidot, psi, psidot, w1..w4]
    initial_state = np.array([
        0.0, 20.0,  # X, vx
        0.0, 0.0,   # Y, vy
        0.5, 0.01,   # z, zdot
        0.0, 0.0,   # theta, thetadot
        0.0, 0.0,   # phi, phidot
        0.0, 0.0,   # psi, psidot
        0.0, 0.0, 0.0, 0.0,  # w1..w4
    ])

    dt = 0.0001
    sim_time = 10.0
    time_array = np.arange(0.0, sim_time, dt)

    # Control: [t1, t2, t3, t4, df, dr]
    tf = 500   # front drive torque [Nm]
    tr = 0.0   # rear drive torque [Nm]
    df = 0.01 # front steering [rad]
    dr = 0.0   # rear steering [rad]
    control_input = np.array([tf, tf, tr, tr, df, dr])

    u_array = np.tile(control_input, (len(time_array), 1))  # shape (T, 6)

    # Nouvelle interface : retourne SimulationResult
    result = euler_integration(model, initial_state, u_array, time_array)

    # State trajectory (T, 16)
    x_traj = result.vehicle.x

    X      = x_traj[:, 0]
    vx     = x_traj[:, 1]
    Y      = x_traj[:, 2]
    vy     = x_traj[:, 3]
    z      = x_traj[:, 4]
    zdot   = x_traj[:, 5]
    theta  = x_traj[:, 6]
    thetad = x_traj[:, 7]
    phi    = x_traj[:, 8]
    phid   = x_traj[:, 9]
    psi    = x_traj[:, 10]
    psid   = x_traj[:, 11]
    w1     = x_traj[:, 12]
    w2     = x_traj[:, 13]
    w3     = x_traj[:, 14]
    w4     = x_traj[:, 15]

    plt.figure(figsize=(12, 8))
    plt.scatter(X, Y)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Heave z [m]")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(time_array, phi, label="heave z")
    plt.xlabel("Time [s]")
    plt.ylabel("Heave z [m]")
    plt.legend()
    plt.show()