import numpy as np
import pytest

from models.vehicle import DOF10, VehiclePhysicalParams10DOF, VehicleConfig10DOF, LinearTireParams
from simulation.integrators import euler_integration, rk4_integration

@pytest.fixture
def base_model_10dof():
    params = VehiclePhysicalParams10DOF(
        g=9.81,
        m=1500.0,
        ms=1300.0,
        lf=1.6,
        lr=1.6,
        h=0.55,
        L1=0.75,
        L2=0.75,
        r=0.3,

        ix=400.0,
        iy=1200.0,
        iz=2500.0,
        ir=1.2,

        ra=12.0,
        s=2.2,
        cx=0.32,

        ks1=30000.0,
        ks2=30000.0,
        ks3=30000.0,
        ks4=30000.0,

        ds1=3500.0,
        ds2=3500.0,
        ds3=3500.0,
        ds4=3500.0,
    )

    tire_params = LinearTireParams(
        Cx=80000.0,
        Cy=80000.0,
    )

    config = VehicleConfig10DOF(
        vehicle=params,
        tire1=tire_params,
        tire2=tire_params,
        tire3=tire_params,
        tire4=tire_params,
    )

    return DOF10(config)

@pytest.mark.parametrize("drive_torque, expected_trend", [
    (-500.0, "decreasing"),  # freinage
    ( 500.0, "increasing"),  # traction
])
def test_longitudinal_acceleration_10dof(base_model_10dof, drive_torque, expected_trend):
    model = base_model_10dof

    # x = [x, Vx, y, Vy, z, Vz, theta, thetadt, phi, phidt, psi, psidt, w1..w4]
    initial_state = np.array([
        0.0, 20.0,   # x, Vx
        0.0, 0.0,    # y, Vy
        0.55, 0.0,    # z, Vz
        0.0, 0.0,    # theta, thetadt
        0.0, 0.0,    # phi, phidt
        0.0, 0.0,    # psi, psidt
        0.0, 0.0, 0.0, 0.0,  # w1..w4
    ])

    dt = 0.0005
    time_array = np.arange(0.0, 5.0, dt)

    tf = drive_torque
    tr = drive_torque
    df = 0.0
    dr = 0.0
    control_input = np.array([tf, tf, tr, tr, df, dr])
    u_array = np.tile(control_input, (len(time_array), 1))

    result = euler_integration(model, initial_state, u_array, time_array)
    vx = result.vehicle.x[:, 1]  # Vx = état 2 → index 1

    if expected_trend == "increasing":
        assert vx[-1] > vx[0]
    elif expected_trend == "decreasing":
        assert vx[-1] < vx[0]

def test_euler_vs_rk4_consistency_10dof(base_model_10dof):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 20.0,
        0.0, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    dt = 0.0005
    # pour que le test reste rapide en 10DOF, je réduis un peu la durée
    time_array = np.arange(0.0, 3.0, dt)
    u_array = np.zeros((len(time_array), 6))

    res_euler = euler_integration(model, initial_state, u_array, time_array)
    res_rk4 = rk4_integration(model, initial_state, u_array, time_array)

    vx_euler = res_euler.vehicle.x[:, 1]
    vx_rk4 = res_rk4.vehicle.x[:, 1]

    assert np.allclose(vx_euler, vx_rk4, atol=1e-1)

@pytest.mark.parametrize(
    "df, expected_sign",
    [
        (0.01, +1),   # petit braquage gauche
        (-0.01, -1),  # petit braquage droite
    ],
)
def test_yaw_rate_sign_with_steering_10dof(base_model_10dof, df, expected_sign):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 20.0,   # x, Vx
        0.0, 0.0,    # y, Vy
        0.55, 0.0,    # z, Vz
        0.0, 0.0,    # theta, thetadt
        0.0, 0.0,    # phi, phidt
        0.0, 0.0,    # psi, psidt
        0.0, 0.0, 0.0, 0.0,  # w1..w4
    ])

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

    # DOF10: psi = x11, psidt = x12 → index 11
    psi_dot = x_traj[:, 11]
    final_yaw_rate = psi_dot[-1]

    assert abs(final_yaw_rate) > 1e-5
    assert np.sign(final_yaw_rate) == expected_sign

def test_static_vertical_equilibrium_10dof(base_model_10dof):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 0.0,   # x, Vx
        0.0, 0.0,   # y, Vy
        0.55, 0.0,   # z, Vz
        0.0, 0.0,   # theta, thetadt
        0.0, 0.0,   # phi, phidt
        0.0, 0.0,   # psi, psidt
        0.0, 0.0, 0.0, 0.0,  # w1..w4
    ])

    u = np.zeros(6)  # pas de couple, pas de braquage

    dx, outputs = model.get_dx__dt(initial_state, u)

    # Z_dt = Vz, Vz_dt = accélération verticale
    Z_dt   = dx[4]
    Vz_dt  = dx[5]

    assert abs(Z_dt) < 1e-8
    assert abs(Vz_dt) < 1e-6

def test_heave_damped_oscillation_10dof(base_model_10dof):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 0.0,   # x, Vx
        0.0, 0.0,   # y, Vy
        0.5, 0.0,  # z = 5 cm, Vz = 0.01
        0.0, 0.0,   # theta, thetadt
        0.0, 0.0,   # phi, phidt
        0.0, 0.0,   # psi, psidt
        0.0, 0.0, 0.0, 0.0,
    ])

    dt = 0.001
    time_array = np.arange(0.0, 3.0, dt)
    u_array = np.zeros((len(time_array), 6))

    result = euler_integration(model, initial_state, u_array, time_array)
    z = result.vehicle.x[:, 4]

    # amplitude finale plus petite que l’amplitude initiale
    assert abs(model.h-z[-1]) <= abs(model.h-z[0])
    # et globalement, z reste raisonnable :
    assert np.max(np.abs(model.h-z)) < 0.1  # ne part pas en vrille

def test_vertical_force_balance_10dof(base_model_10dof):
    model = base_model_10dof

    state = np.zeros(16)
    state[4] = 0.55
    u = np.zeros(6)

    dx, outputs = model.get_dx__dt(state, u)

    Fz = outputs["Fz"]  # (4,)
    total_Fz = np.sum(Fz)

    # à l’équilibre, total_Fz ≈ m*g (ou ms*g selon comment tu répartis)
    assert np.isclose(total_Fz, model.params.vehicle.ms * model.params.vehicle.g, rtol=1e-2)

@pytest.mark.parametrize("df, expected_side", [
    (0.05, "right"),   # braquage gauche → charge sur roues droites
    (-0.05, "left"),   # braquage droite → charge sur roues gauches
])
def test_lateral_load_transfer_sign_10dof(base_model_10dof, df, expected_side):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 20.0,
        0.0, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    dt = 0.0005
    time_array = np.arange(0.0, 5.0, dt)

    tf = 0.0
    tr = 0.0
    dr = 0.0
    control_input = np.array([tf, tf, tr, tr, df, dr])
    u_array = np.tile(control_input, (len(time_array), 1))

    result = euler_integration(model, initial_state, u_array, time_array)

    # Fz loggés dans TireLog (si tu les as)
    Fz = result.tires.Fz  # (T, 4) : [FL, FR, RL, RR] par exemple
    Fz_final = Fz[-1]

    Fz_FL, Fz_FR, Fz_RL, Fz_RR = Fz_final

    if expected_side == "right":
        assert (Fz_FR + Fz_RR) > (Fz_FL + Fz_RL)
    else:
        assert (Fz_FL + Fz_RL) > (Fz_FR + Fz_RR)

def test_left_right_symmetry_10dof(base_model_10dof):
    model = base_model_10dof

    initial_state = np.array([
        0.0, 20.0,
        0.0, 0.0,
        0.55, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ])

    dt = 0.001
    sim_time = 3.0
    time_array = np.arange(0.0, sim_time, dt)

    def run(df):
        tf = tr = 0.0
        dr = 0.0
        u_input = np.array([tf, tf, tr, tr, df, dr])
        u_array = np.tile(u_input, (len(time_array), 1))
        res = euler_integration(model, initial_state, u_array, time_array)
        x_traj = res.vehicle.x
        y = x_traj[:, 2]
        psi = x_traj[:, 10]
        return y[-1], psi[-1]

    y_left, psi_left = run(0.02)
    y_right, psi_right = run(-0.02)

    assert np.sign(y_left) == -np.sign(y_right)
    assert np.sign(psi_left) == -np.sign(psi_right)
    assert np.isclose(abs(y_left), abs(y_right), rtol=0.2)
    assert np.isclose(abs(psi_left), abs(psi_right), rtol=0.2)

if __name__ == '__main__':
    pytest.main([__file__])

