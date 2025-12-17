import numpy as np
import pytest
from models.vehicle import DOF7, VehiclePhysicalParams7DOF, VehicleConfig7DOF, LinearTireParams
from models.tires.registry import TIRE_MODEL_REGISTRY
from simulation.integrators import euler_integration, rk4_integration
import matplotlib.pyplot as plt

@pytest.fixture
def base_model():
    params = VehiclePhysicalParams7DOF(
        g=9.81, m=1500, lf=1.2, lr=1.6,
        h=0.5, L1=1.0, L2=1.6, r=0.3,
        iz=2500.0, ir=1.2, ra=0.015, s=0.01, cx=0.3
    )

    tire_params = LinearTireParams(
        Cx=80000.0,
        Cy=80000.0,
    )

    config = VehicleConfig7DOF(
        vehicle=params,
        tire1=tire_params,
        tire2=tire_params,
        tire3=tire_params,
        tire4=tire_params,
    )

    return DOF7(config)

@pytest.mark.parametrize("drive_torque, expected_trend", [
    (-500.0, "decreasing"),  # freinage
    ( 500.0, "increasing"),  # traction
])
def test_longitudinal_acceleration(base_model, drive_torque, expected_trend):
    model = base_model
    initial_state = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])
    dt = 0.0005
    time_array = np.arange(0, 3.0, dt)

    tf = drive_torque
    tr = drive_torque
    df = 0.0
    dr = 0.0
    control_input = np.array([tf, tf, tr, tr, df, dr])
    u_array = np.tile(control_input, (len(time_array), 1))

    result = euler_integration(model, initial_state, u_array, time_array)
    vx = result.vehicle.x[:, 1]

    if expected_trend == "increasing":
        assert vx[-1] > vx[0]
    elif expected_trend == "decreasing":
        assert vx[-1] < vx[0]

def test_euler_vs_rk4_consistency(base_model):
    model = base_model
    initial_state = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])
    dt = 0.0005
    time_array = np.arange(0, 10.0, dt)
    u_array = np.zeros((len(time_array), 6))

    res_euler = euler_integration(model, initial_state, u_array, time_array)
    res_rk4 = rk4_integration(model, initial_state, u_array, time_array)

    vx_euler = res_euler.vehicle.x[:, 1]
    vx_rk4 = res_rk4.vehicle.x[:, 1]

    # plt.figure(figsize = (10, 10))
    # plt.plot(vx_euler)
    # plt.plot(vx_rk4)
    # plt.show()

    # les deux solutions doivent être proches
    assert np.allclose(vx_euler, vx_rk4, atol=1e-1)

@pytest.mark.parametrize(
    "df, expected_sign",
    [
        (0.01, +1),   # petit braquage gauche
        (-0.01, -1),  # petit braquage droite
    ],
)
def test_yaw_rate_sign_with_steering(base_model, df, expected_sign):
    model = base_model

    # état initial : vx = 20 m/s, tout le reste à zéro sauf vitesses roues
    initial_state = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])

    dt = 0.0005
    sim_time = 5.0
    time_array = np.arange(0.0, sim_time, dt)

    tf = 0.0
    tr = 0.0
    dr = 0.0
    control_input = np.array([tf, tf, tr, tr, df, dr])
    u_array = np.tile(control_input, (len(time_array), 1))

    result = euler_integration(model, initial_state, u_array, time_array)
    x_traj = result.vehicle.x

    # index 5 = yaw rate (psidt)
    psi_dot = x_traj[:, 5]
    final_yaw_rate = psi_dot[-1]

    # on vérifie déjà qu'il se passe quelque chose
    assert abs(final_yaw_rate) > 1e-5

    # puis que le signe est cohérent avec le signe du braquage
    assert np.sign(final_yaw_rate) == expected_sign


if __name__ == "__main__":

    pytest.main([__file__])

    # vehicle_params = VehiclePhysicalParams7DOF(
    #     g=9.81, 
    #     m=1500, 
    #     lf=1.2, 
    #     lr=1.6, 
    #     h=0.5,
    #     L1=1.0, 
    #     L2=1.6, 
    #     r=0.3, 
    #     iz=2500.0, 
    #     ir=1.2,
    #     ra=0.015, 
    #     s=0.01, 
    #     cx=0.3
    # )
    
    # tire_params = {
    #     "model": "linear",
    #     "Cx": 80000.0,
    #     "Cy": 80000.0
    # }

    # config = VehicleConfig7DOF(
    #     vehicle=vehicle_params,
    #     tire1=tire_params,
    #     tire2=tire_params,
    #     tire3=tire_params,
    #     tire4=tire_params
    # )
    
    # # Instantiate the model
    # model = DOF7(config)

    # initial_state = np.array([0, 20, 0, 0, 0, 0, 10, 10, 10, 10])   # 10D state vector
    # dt = 0.0005
    # sim_time = 10.0
    # time_array = np.arange(0, sim_time, dt)

    # # Control vector for a single time step
    # tf = 0 # Drive torque (front) [Nm]
    # tr = 0 # Drive torque (rear) [Nm]
    # df = -0.01 # Front steering angle δf [rad]
    # dr = 0
    # control_input = np.array([tf, tf, tr, tr, df, dr])

    # # Expand to match time steps
    # u_array = np.tile(control_input, (len(time_array), 1))  # shape: (T, 6)

    # result = euler_integration(model, initial_state, u_array, time_array)

    # # State trajectory: shape (T, n_states)
    # x_traj = result.vehicle.x

    # x   = x_traj[:, 0]
    # vx  = x_traj[:, 1]
    # y   = x_traj[:, 2]
    # vy  = x_traj[:, 3]
    # psi = x_traj[:, 4]
    # psidt = x_traj[:, 5]
    # w1  = x_traj[:, 6]
    # w2  = x_traj[:, 7]
    # w3  = x_traj[:, 8]
    # w4  = x_traj[:, 9]

    # plt.figure(figsize=(12, 8))
    # plt.plot(time_array, vx, label="yaw_rate")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Yaw rate [rad/s]")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(12, 8))
    # plt.scatter(x, y, label="yaw_rate")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Yaw rate [rad/s]")
    # plt.legend()
    # plt.show()