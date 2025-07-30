import streamlit as st
from models.vehicle.dof7_model import DOF7, VehicleConfig7DOF, VehiclePhysicalParams7DOF

def get_linear_tire_params():
    Cx = st.number_input("Cx (longitudinal stiffness)", value=80000.0)
    Cy = st.number_input("Cy (lateral stiffness)", value=80000.0)

    return  {
        "model": "linear",
        "Cx": Cx,
        "Cy": Cy,
    }

def get_pacejka_tire_params():
    st.subheader("Pacejka Tire Parameters")

    # Longitudinal coefficients
    Bx = st.number_input("Bx (longitudinal)", value=10.0)
    Cx = st.number_input("Cx (longitudinal)", value=1.9)
    Dx = st.number_input("Dx (longitudinal)", value=1.0)
    Ex = st.number_input("Ex (longitudinal)", value=0.97)

    # Lateral coefficients
    By = st.number_input("By (lateral)", value=8.0)
    Cy = st.number_input("Cy (lateral)", value=1.3)
    Dy = st.number_input("Dy (lateral)", value=0.7)
    Ey = st.number_input("Ey (lateral)", value=0.97)

    return {
        "model": "pacejka",
        "Bx": Bx,
        "Cx": Cx,
        "Dx": Dx,
        "Ex": Ex,
        "By": By,
        "Cy": Cy,
        "Dy": Dy,
        "Ey": Ey
    }