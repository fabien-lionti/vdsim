# sideslip_plot.py

import plotly.graph_objects as go
import numpy as np

def plot_sideslip(time_array: np.ndarray, trajectory: np.ndarray):
    """Plot sideslip angle β over time."""
    vx = trajectory[:, 1]
    vy = trajectory[:, 3]
    vx_safe = np.where(np.abs(vx) < 1e-6, 1e-6, vx)
    beta = np.arctan(vy / vx_safe)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_array, y=beta, name="Sideslip Angle β"))

    fig.update_layout(
        title="Sideslip Angle Over Time",
        xaxis_title="Time [s]",
        yaxis_title="β [rad]",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white"
    )
    return fig