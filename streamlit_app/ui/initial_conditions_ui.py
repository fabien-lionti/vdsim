import streamlit as st
import numpy as np

def get_initial_conditions(model_choice: str) -> np.ndarray:
    """
    Generate Streamlit UI elements to gather initial conditions
    based on the selected vehicle model.

    Parameters:
        model_choice (str): Selected model name (e.g., "7-DOF")

    Returns:
        np.ndarray: Initial state vector
    """
    st.subheader("Initial Conditions")

    x0 = st.number_input("Initial x [m]", value=0.0)
    y0 = st.number_input("Initial y [m]", value=0.0)
    vx0 = st.number_input("Initial longitudinal speed vx [m/s]", value=5.0)
    vy0 = st.number_input("Initial lateral speed vy [m/s]", value=0.0)
    psi0 = st.number_input("Initial yaw angle ψ [rad]", value=0.0)
    psidt0 = st.number_input("Initial yaw rate ψ̇ [rad/s]", value=0.0)

    if model_choice == "7-DOF":
        omega1 = st.number_input("Initial ω1 (FL) [rad/s]", value=0.0)
        omega2 = st.number_input("Initial ω2 (FR) [rad/s]", value=0.0)
        omega3 = st.number_input("Initial ω3 (RL) [rad/s]", value=0.0)
        omega4 = st.number_input("Initial ω4 (RR) [rad/s]", value=0.0)

        initial_state = np.array([
            x0, vx0, y0, vy0, psi0, psidt0,
            omega1, omega2, omega3, omega4
        ])
    else:
        # fallback for simpler models (e.g., bicycle)
        initial_state = np.array([
            x0, vx0, y0, vy0, psi0, psidt0
        ])

    return initial_state
