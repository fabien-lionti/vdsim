# state_plot.py

import plotly.graph_objects as go
import numpy as np

def plot_states(time_array: np.ndarray, trajectory: np.ndarray):
    """Plot vx, vy, and yaw rate over time using Plotly."""
    vx = trajectory[:, 1]
    vy = trajectory[:, 3]
    psidt = trajectory[:, 5]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_array, y=vx, name="vx (longitudinal)"))
    fig.add_trace(go.Scatter(x=time_array, y=vy, name="vy (lateral)"))
    fig.add_trace(go.Scatter(x=time_array, y=psidt, name="Yaw Rate ψ̇"))

    fig.update_layout(
        title="State Variables Over Time",
        xaxis_title="Time [s]",
        yaxis_title="Value",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white"
    )
    return fig
