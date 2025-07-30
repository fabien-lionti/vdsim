import streamlit as st

st.title("Step 1: Model Selection")

model = st.selectbox("Choose a vehicle model", ["7-DOF", "Bicycle", "Dynamic"])
st.session_state["selected_model"] = model
