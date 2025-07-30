import streamlit as st
import numpy as np

def command_ui(model_choice: str) -> np.ndarray:
    st.subheader("Control Inputs")

    if model_choice == "7-DOF":
        tf = st.slider("Drive torque (front) [Nm]", -2000, 2000, 500)
        tr = st.slider("Drive torque (rear) [Nm]", -2000, 2000, 500)
        df = st.slider("Front steering angle δf [rad]", -0.5, 0.5, 0.0, step=0.01)
        dr = st.slider("Rear steering angle δr [rad]", -0.5, 0.5, 0.0, step=0.01)

        # Create a control vector: [tf1, tf2, tr1, tr2, df, dr]
        control_vector = np.array([tf, tf, tr, tr, df, dr])

    elif model_choice == "Bicycle":
        T = st.slider("Drive torque [Nm]", -2000, 2000, 500)
        delta = st.slider("Steering angle δ [rad]", -0.5, 0.5, 0.0, step=0.01)
        control_vector = np.array([T, delta])

    else:
        st.warning("Unknown model or controls not defined yet.")
        control_vector = np.array([])

    return control_vector