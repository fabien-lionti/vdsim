import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from models.vehicle.base_model import BaseVehicleModel
from models.tires.registry import TIRE_MODEL_REGISTRY
from dataclasses import asdict

from dataclasses import dataclass

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
        self.fz012 = (self.m * self.lr * self.g) / (2 * self.l)
        self.fz034 = (self.m * self.lf * self.g) / (2 * self.l)
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
    
    def get_suspension_travel(self, theta, phi):

        dzg1 = self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg2 = -self.L1 * np.sin(theta) - self.lf * np.cos(theta) * np.sin(phi)
        dzg3 = self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)
        dzg4 = -self.L2 * np.sin(theta) + self.lr * np.cos(theta) * np.sin(phi)

        return dzg1, dzg2, dzg3, dzg4
    
    def get_suspension_speed_travel(self, theta, thetadt, phi, phidt):

        dzg1dt = self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
        dzg2dt = -self.L1 * thetadt * np.cos(theta) + self.lf * thetadt * np.sin(theta) * np.sin(phi) - self.lf * phidt * np.cos(theta) * np.cos(phi)
        dzg3dt =  self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)
        dzg4dt = -self.L2 * thetadt * np.cos(theta) - self.lr * thetadt * np.sin(theta) * np.sin(phi) + self.lr * phidt * np.cos(theta) * np.cos(phi)

        return dzg1dt, dzg2dt, dzg3dt, dzg4dt
    
    def get_vertical_normal_forces(self, dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt):

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
    
    def get_state_derivatives(self, x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
                              fx1, fx2, fx3, fx4, 
                              fy1, fy2, fy3, fy4,
                              fz1, fz2, fz3, fz4,
                              fs1, fs2, fs3, fs4,
                              fxp1, fxp2, fxp3, fxp4,
                              faero):
        
        fz1 = fz1 * np.cos(x9) * np.cos(x7)
        fz2 = fz2 * np.cos(x9) * np.cos(x7)
        fz3 = fz3 * np.cos(x9) * np.cos(x7)
        fz4 = fz4 * np.cos(x9) * np.cos(x7)

        self.x1dt = x2 * np.cos(x11) - x4 * np.sin(x11)
        self.x2dt = x12 * x4 - x10 * x6 + self.m_inv * (fx1 + fx2 + fx3 + fx4 - faero * np.cos(x9))
        self.x3dt = x2 * np.sin(x11) + x4 * np.cos(x11)
        self.x4dt = -x12 * x2 + x8 * x6 + self.m_inv * (fy1 + fy2 + fy3 + fy4)
        self.x5dt = x6
        self.x6dt = self.ms_inv * (fz1 + fz2 + fz3 + fz4) - self.g * np.cos(x7) * np.cos(x9)
        
        self.x7dt = x8
        self.x8dt = self.ix_inv * ( self.L1 * (fz1 + fz3 - fz2 - fz4) + x5 * (fy1 + fy2 + fy3 + fy4) )
        self.x9dt = x10
        self.x10dt = self.iy_inv * ( -self.lf * (fz1 + fz2 ) + self.lr * (fz3 + fz4) - x5 * (fx1 + fx2 + fx3 + fx4) )
        self.x11dt = x12
        self.x12dt = self.iz_inv * ( self.lf * (fy1 + fy2) - self.lr * (fy3 + fy4) + self.L1 * (fx2 + fx4 - fx1 - fx3) )
        
        self.x13dt = (u1 - self.r * fxp1) / (self.ir)
        self.x14dt = (u2 - self.r * fxp2) / (self.ir)
        self.x15dt = (u3 - self.r * fxp3) / (self.ir)
        self.x16dt = (u4 - self.r * fxp4) / (self.ir)

    def get_dx__dt(self, x, u):

        u1, u2, u3, u4, u5, u6 = u

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = x
        
        # Compute vxp for each wheels in wheel basis

        vxp1 = (x2 - self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
        vxp2 = (x2 + self.L1 * x12) * np.cos(u5) + (x4 + self.lf * x12) * np.sin(u5)
        vxp3 = (x2 - self.L1 * x12) * np.cos(u6) + (x4 - self.lf * x12) * np.sin(u6)
        vxp4 = (x2 + self.L1 * x12) * np.cos(u6) + (x4 - self.lf * x12) * np.sin(u6)

        vyp1 = (x4 + self.lf * x12) * np.cos(u5) - (x2 - self.L1 * x12) * np.sin(u5)
        vyp2 = (x4 + self.lf * x12) * np.cos(u5) - (x2 + self.L1 * x12) * np.sin(u5)
        vyp3 = (x4 - self.lf * x12) * np.cos(u6) - (x2 - self.L1 * x12) * np.sin(u6)
        vyp4 = (x4 - self.lf * x12) * np.cos(u6) - (x2 + self.L1 * x12) * np.sin(u6)

        # Compute slipping rates and angles
        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = self.get_slipping_angles(vxp1, vyp1, vxp2, vyp2, vxp3, vyp3, vxp4, vyp4, u5, u5, u6, u6)
        self.sigma1, self.sigma2, self.sigma3, self.sigma4 = self.get_slipping_rates(vxp1, x13, vxp2, x14, vxp3, x15, vxp4, x16, self.r)
        
        # Compute suspension and normal reaction forces
        dzg1, dzg2, dzg3, dzg4 = self.get_suspension_travel(theta = x7, phi = x9)
        dzg1dt, dzg2dt, dzg3dt, dzg4dt = self.get_suspension_speed_travel(theta = x7, thetadt = x8, phi = x9, phidt = x10)
        self.fz1, self.fz2, self.fz3, self.fz4, self.fs1, self.fs2, self.fs3, self.fs4 = self.get_vertical_normal_forces(dzg1, dzg2, dzg3, dzg4, dzg1dt, dzg2dt, dzg3dt, dzg4dt)

        # Compute tire forces in tire basis
        # fxp1, fxp2, fxp3, fxp4 = self.get_longitudinal_tire_forces(self.sigma1, self.alpha1, self.fz1, self.sigma2, self.alpha2, self.fz2, self.sigma3, self.alpha3, self.fz3, self.sigma4, self.alpha4, self.fz4)
        # fyp1, fyp2, fyp3, fyp4 = self.get_lateral_tire_forces(self.sigma1, self.alpha1, self.fz1, self.sigma2, self.alpha2, self.fz2, self.sigma3, self.alpha3, self.fz3, self.sigma4, self.alpha4, self.fz4)
        
        fxp1 = self.f1.get_fx0(self.fz012, self.fz012, self.sigma1)
        fxp2 = self.f2.get_fx0(self.fz012, self.fz012, self.sigma2)
        fxp3 = self.f3.get_fx0(self.fz034, self.fz034, self.sigma3)
        fxp4 = self.f4.get_fx0(self.fz034, self.fz034, self.sigma4)

        fyp1 = self.f1.get_fy0(self.fz012, self.fz012, self.alpha1)
        fyp2 = self.f2.get_fy0(self.fz012, self.fz012, self.alpha2)
        fyp3 = self.f3.get_fy0(self.fz034, self.fz034, self.alpha3)
        fyp4 = self.f4.get_fy0(self.fz034, self.fz034, self.alpha4)

        # Get tire forces in vehicle basis

        self.fx1, self.fy1 = self.project_from_tire_to_carbody_frame(fxp1, fyp1, u5, x9, x7, self.fz1)
        self.fx2, self.fy2 = self.project_from_tire_to_carbody_frame(fxp2, fyp2, u5, x9, x7, self.fz2)
        self.fx3, self.fy3 = self.project_from_tire_to_carbody_frame(fxp3, fyp3, u6, x9, x7, self.fz3)
        self.fx4, self.fy4 = self.project_from_tire_to_carbody_frame(fxp4, fyp4, u6, x9, x7, self.fz4)

        # TODO
        faero = self.get_faero(x2)

        self.get_state_derivatives(x2, x4, x5, x6, x7, x8, x9, x10, x11, x12, u1, u2, u3, u4, 
                            self.fx1, self.fx2, self.fx3, self.fx4, 
                            self.fy1, self.fy2, self.fy3, self.fy4,
                            self.fz1, self.fz2, self.fz3, self.fz4,
                            self.fs1, self.fs2, self.fs3, self.fs4,
                            fxp1, fxp2, fxp3, fxp4,
                            faero)
        
        return [self.x1dt, self.x2dt, self.x3dt, self.x4dt, self.x5dt, self.x6dt, self.x7dt, self.x8dt, self.x9dt, self.x10dt, self.x11dt, self.x12dt, self.x13dt, self.x14dt, self.x15dt, self.x16dt]

if __name__ == "__main__":

    vehicle_params = VehiclePhysicalParams10DOF(
        g=9.81,              # Gravité
        m=1500.0,            # Masse totale [kg]
        ms=1300.0,           # Masse suspendue [kg]
        lf=1.2,              # Distance CG → essieu avant [m]
        lr=1.6,              # Distance CG → essieu arrière [m]
        h=0.55,              # Hauteur CG [m]
        L1=0.75,             # Demi-voie avant [m]
        L2=0.75,             # Demi-voie arrière [m]
        r=0.3,               # Rayon effectif roue [m]

        # Inerties
        ix=400.0,            # Inertie en roulis [kg·m²]
        iy=1200.0,           # Inertie en tangage [kg·m²]
        iz=2500.0,           # Inertie en lacet [kg·m²]
        ir=1.2,              # Inertie roue [kg·m²]

        # Forces résistives
        ra=12.0,             # Résistance au roulement [N·s/m]
        s=2.2,               # Surface frontale [m²]
        cx=0.32,             # Coefficient traînée [-]

        # Raideurs suspension (symétrique ici) [N/m]
        ks1=30000.0,         # FL
        ks2=30000.0,         # FR
        ks3=28000.0,         # RL
        ks4=28000.0,         # RR

        # Amortisseurs suspension [N·s/m]
        ds1=3500.0,          # FL
        ds2=3500.0,          # FR
        ds3=3200.0,          # RL
        ds4=3200.0           # RR
    )


    tire_params = {
        "model": "linear",
        "Cx": 80000.0,
        "Cy": 80000.0
    }

    config = VehicleConfig10DOF(
        vehicle=vehicle_params,
        tire1=tire_params,
        tire2=tire_params,
        tire3=tire_params,
        tire4=tire_params
    )
    
    # Instantiate the model
    model = DOF10(config)

    # Optional: simulate a single step
    initial_state = np.array([0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])   # 10D state vector
    
    dt = 0.0001
    sim_time = 10.0
    time_array = np.arange(0, sim_time, dt)

    # Control vector for a single time step
    tf = 0 # Drive torque (front) [Nm]
    tr = 0 # Drive torque (rear) [Nm]
    df = 0.001 # Front steering angle δf [rad]
    dr = 0
    control_input = np.array([tf, tf, tr, tr, df, dr])

    # Expand to match time steps
    u_array = np.tile(control_input, (len(time_array), 1))  # shape: (T, 6)

    import matplotlib.pyplot as plt

    trajectory = model.euler_integration(initial_state, u_array, time_array)
    
    x = trajectory[:, 0]
    vx = trajectory[:, 1]
    y = trajectory[:, 2]
    vy = trajectory[:, 3]
    z = trajectory[:, 4]
    vz = trajectory[:, 5]
    phi = trajectory[:, 6]
    phidt = trajectory[:, 7]
    theta = trajectory[:, 8]
    thetadt = trajectory[:, 9]
    psi = trajectory[:, 10]
    psidt = trajectory[:, 11]
    w1 = trajectory[:, 12]
    w2 = trajectory[:, 13]
    w3 = trajectory[:, 14]
    w4 = trajectory[:, 15]

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, label='vx')
    plt.show()