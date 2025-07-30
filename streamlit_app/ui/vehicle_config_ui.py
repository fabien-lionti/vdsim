import streamlit as st
from models.vehicle.dof7_model import DOF7, VehicleConfig7DOF, VehiclePhysicalParams7DOF

def get_vehicle_params_7dof():
    st.subheader("Vehicle Parameters - 7DOF")
    return VehiclePhysicalParams7DOF(
        g=st.number_input("Gravity [m/s²]", value=9.81),
        m=st.slider("Mass [kg]", 500.0, 3000.0, 1600.0),
        lf=st.slider("lf [m]", 0.5, 3.0, 1.3),
        lr=st.slider("lr [m]", 0.5, 3.0, 1.4),
        h=st.slider("CG height [m]", 0.2, 1.5, 0.5),
        L1=st.slider("L1 [m]", 0.5, 3.0, 1.1),
        L2=st.slider("L2 [m]", 0.5, 3.0, 1.1),
        r=st.slider("Wheel radius [m]", 0.2, 0.6, 0.3),
        iz=st.number_input("Yaw inertia [kg·m²]", value=2500.0),
        ir=st.number_input("Wheel inertia [kg·m²]", value=1.2),
        ra=st.number_input("Air drag coeff (Ra)", value=0.3),
        s=st.number_input("Frontal area (S) [m²]", value=2.2),
        cx=st.number_input("Drag coeff (Cx)", value=0.35)
    )
