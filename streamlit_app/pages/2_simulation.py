import streamlit as st

st.title("Step 2: Run Simulation")

if "selected_model" not in st.session_state:
    st.warning("Please select a model first on the Model Selection page.")
else:
    st.write(f"Simulating for model: {st.session_state['selected_model']}")
    # Continue with simulation logic
