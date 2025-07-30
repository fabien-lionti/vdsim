import streamlit as st
from models.vehicle.dof7_model import DOF7, VehicleConfig7DOF, VehiclePhysicalParams7DOF
from models.tires.registry import TIRE_MODEL_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# from models.vehicle import VehiclePhysicalParams7DOF

st.title("Vehicle Simulator")

model_choice = st.selectbox(
    "Select a vehicle model:",
    ("7-DOF", "Bicycle", "Kinematic", "Dynamic")
)

# User inputs
if model_choice == "7-DOF":
    # Create UI inputs for each parameter
    g = st.number_input("Gravity [m/s²]", value=9.81, step=0.01)
    m = st.slider("Mass [kg]", min_value=500.0, max_value=3000.0, value=1600.0)
    lf = st.slider("Distance to front axle (lf) [m]", 0.5, 3.0, 1.3)
    lr = st.slider("Distance to rear axle (lr) [m]", 0.5, 3.0, 1.4)
    h = st.slider("CG Height (h) [m]", 0.2, 1.5, 0.5)
    L1 = st.slider("L1 [m]", 0.5, 3.0, 1.1)
    L2 = st.slider("L2 [m]", 0.5, 3.0, 1.1)
    r = st.slider("Wheel radius (r) [m]", 0.2, 0.6, 0.3)
    iz = st.number_input("Yaw inertia (Iz) [kg·m²]", value=2500.0)
    ir = st.number_input("Wheel inertia (Ir) [kg·m²]", value=1.2)
    ra = st.number_input("Air drag coefficient (Ra) [N·s/m]", value=0.3)
    s = st.number_input("Frontal area (S) [m²]", value=2.2)
    cx = st.number_input("Drag coefficient (Cx)", value=0.35)

    vehicle_params = VehiclePhysicalParams7DOF(
        g=g, m=m, lf=lf, lr=lr, h=h,
        L1=L1, L2=L2, r=r, iz=iz, ir=ir,
        ra=ra, s=s, cx=cx
    )

st.subheader("Tire Parameters")

tire_model = st.selectbox("Select tire model", list(TIRE_MODEL_REGISTRY.keys()))

if tire_model == "linear":

    Cx = st.number_input("Cx (longitudinal stiffness)", value=80000.0)
    Cy = st.number_input("Cy (lateral stiffness)", value=80000.0)

    tire_params = {
        "model": "linear",
        "Cx": Cx,
        "Cy": Cy,
    }

elif tire_model == "pacejka":

    Bx = st.number_input("Bx", value=10.0)
    Cx = st.number_input("Cx", value=1.9)
    Dx = st.number_input("Dx", value=1.0)
    Ex = st.number_input("Ex", value=0.97)

    By = st.number_input("By", value=8.0)
    Cy = st.number_input("Cy", value=1.3)
    Dy = st.number_input("Dy", value=0.7)
    Ey = st.number_input("Ey", value=0.97)

    tire_params = {
        "model": "pacejka",
        "Bx": Bx, "Cx": Cx, "Dx": Dx, "Ex": Ex,
        "By": By, "Cy": Cy, "Dy": Dy, "Ey": Ey,
    }

st.subheader("Simulation Parameters")
sim_time = st.number_input("Simulation Time [s]", min_value=0.1, value=5.0, step=0.1)
dt = st.number_input("Time Step [s]", min_value=0.0001, value=0.001, step=0.001)
time_array = np.arange(0, sim_time, dt)
integration_method = st.selectbox("Integration Method", options=["Euler", "RK4"])

st.subheader("Initial Conditions")
if model_choice == "7-DOF":
    x0 = st.number_input("Initial x [m]", value=0.0)
    y0 = st.number_input("Initial y [m]", value=0.0)
    vx0 = st.number_input("Initial longitudinal speed vx [m/s]", value=5.0)
    vy0 = st.number_input("Initial lateral speed vy [m/s]", value=0.0)
    psi0 = st.number_input("Initial yaw angle ψ [rad]", value=0.0)
    psidt0 = st.number_input("Initial yaw rate ψ̇ [rad/s]", value=0.0)

    omega1 = st.number_input("Initial ω1 (FL) [rad/s]", value=0.0)
    omega2 = st.number_input("Initial ω2 (FR) [rad/s]", value=0.0)
    omega3 = st.number_input("Initial ω3 (RL) [rad/s]", value=0.0)
    omega4 = st.number_input("Initial ω4 (RR) [rad/s]", value=0.0)

    initial_state = np.array([x0, vx0, y0, vy0, psi0, psidt0, omega1, omega2, omega3, omega4])

st.subheader("Commands")
if model_choice == "7-DOF":
    tf = st.slider("Drive torque (front) [Nm]", -2000, 2000, 500)
    tr = st.slider("Drive torque (rear) [Nm]", -2000, 2000, 500)
    df = st.slider("Front steering angle δf [rad]", -0.5, 0.5, 0.0, step=0.01)
    dr = st.slider("Rear steering angle δr [rad]", -0.5, 0.5, 0.0, step=0.01)

    # Control vector for a single time step
    control_input = np.array([tf, tf, tr, tr, df, dr])

    # Expand to match time steps
    u_array = np.tile(control_input, (len(time_array), 1))  # shape: (T, 6)

if st.button("Run Simulation"):
    if model_choice == "7-DOF":
        config = VehicleConfig7DOF(
            vehicle=vehicle_params,
            tire1=tire_params,
            tire2=tire_params,
            tire3=tire_params,
            tire4=tire_params
        )
        model = DOF7(config)

    if integration_method == "Euler":
        trajectory = model.euler_integration(initial_state, u_array, time_array)
        # print(trajectory)
        # st.subheader("Done")
        # st.subheader("2D Trajectory Plot")

        # fig, ax = plt.subplots()
        # ax.plot(trajectory[:, 0], trajectory[:, 2], label="Vehicle Path")
        # ax.set_xlabel("X Position [m]")
        # ax.set_ylabel("Y Position [m]")
        # ax.set_title("Trajectory (X-Y)")
        # ax.axis("equal")
        # ax.grid(True)
        # st.pyplot(fig)

        vx = trajectory[:, 1]
        vy = trajectory[:, 3]
        psi = trajectory[:, 4]
        psidt = trajectory[:, 5]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_array, y=vx, name="vx (longitudinal)"))
        fig.add_trace(go.Scatter(x=time_array, y=vy, name="vy (lateral)"))
        fig.add_trace(go.Scatter(x=time_array, y=psidt, name="yaw rate"))
        fig.update_layout(title="State Variables over Time", xaxis_title="Time Step")
        st.plotly_chart(fig)

        # 
        yaw_rate_actual = trajectory[:, 5]
        vx = trajectory[:, 1]
        L = vehicle_params.l  # wheelbase (lf + lr)
        delta_f = control_input[4]  # assumed constant in your current UI

        yaw_rate_ideal = vx / L * delta_f
        understeer_indicator = yaw_rate_actual - yaw_rate_ideal
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_array, y=yaw_rate_actual, name="Actual Yaw Rate"))
        fig.add_trace(go.Scatter(x=time_array, y=yaw_rate_ideal, name="Ideal Yaw Rate"))
        fig.add_trace(go.Scatter(x=time_array, y=understeer_indicator, name="Understeer Indicator", line=dict(dash='dash')))

        fig.update_layout(
            title="Understeer / Oversteer Analysis",
            xaxis_title="Time [s]",
            yaxis_title="Yaw Rate [rad/s]",
            legend=dict(x=0, y=1.1, orientation="h")
        )

        st.plotly_chart(fig)
        threshold = 0.05  # tunable threshold
        status = np.where(
            understeer_indicator > threshold, "Oversteer",
            np.where(understeer_indicator < -threshold, "Understeer", "Neutral")
        )
        

        # Build DataFrame for plotting
        df = pd.DataFrame({
            "Time": time_array,
            "Indicator": understeer_indicator,
            "Status": status
        })

        fig = px.scatter(
            df, x="Time", y="Indicator", color="Status",
            title="Understeer / Oversteer Indicator Over Time",
            labels={"Indicator": "Yaw Rate Error (Actual - Ideal) [rad/s]"},
            color_discrete_map={
                "Understeer": "blue",
                "Oversteer": "red",
                "Neutral": "green"
            }
        )
        st.plotly_chart(fig)

        # Yaw rate (ψ̇)
        yaw_rate = trajectory[:, 5]
        # Front steering angle δf (assumed constant)
        delta_f = control_input[4]

        # Avoid division by zero
        if abs(delta_f) > 1e-6:
            yaw_rate_gain = yaw_rate / delta_f
        else:
            yaw_rate_gain = np.zeros_like(yaw_rate)

        fig_yaw_gain = go.Figure()
        fig_yaw_gain.add_trace(go.Scatter(x=time_array, y=yaw_rate_gain, name="Yaw Rate Gain"))
        fig_yaw_gain.update_layout(
            title="Yaw Rate Gain Over Time",
            xaxis_title="Time [s]",
            yaxis_title="ψ̇ / δf [rad/s per rad]"
        )
        st.plotly_chart(fig_yaw_gain)

        vx = trajectory[:, 1]
        vy = trajectory[:, 3]

        # Avoid division by zero
        vx_safe = np.where(np.abs(vx) < 1e-6, 1e-6, vx)
        beta = np.arctan(vy / vx_safe)

        fig_beta = go.Figure()
        fig_beta.add_trace(go.Scatter(x=time_array, y=beta, name="Sideslip Angle β"))
        fig_beta.update_layout(
            title="Sideslip Angle Over Time",
            xaxis_title="Time [s]",
            yaxis_title="β [rad]"
        )
        st.plotly_chart(fig_beta)


#     elif integration_method == "RK4":
#         trajectory = model.rk4_integration(initial_state, control_input, time_array)
#     else:
#         st.error("Unknown integration method selected.")
#         st.stop()

#     st.success("Simulation complete.")
#     st.line_chart(trajectory[:, :2])  # example: plot x vs y
