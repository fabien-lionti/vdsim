# yaw_analysis_plot.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_yaw_comparison(time_array, yaw_rate_actual, yaw_rate_ideal, threshold=0.05):
    understeer_indicator = yaw_rate_actual - yaw_rate_ideal

    # Plot: Yaw rate comparison
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(x=time_array, y=yaw_rate_actual, name="Actual Yaw Rate"))
    fig_comparison.add_trace(go.Scatter(x=time_array, y=yaw_rate_ideal, name="Ideal Yaw Rate"))
    fig_comparison.add_trace(go.Scatter(
        x=time_array, y=understeer_indicator,
        name="Understeer Indicator", line=dict(dash="dash")
    ))

    fig_comparison.update_layout(
        title="Yaw Rate vs Ideal and Understeer Indicator",
        xaxis_title="Time [s]",
        yaxis_title="Yaw Rate [rad/s]",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white"
    )

    # Classify understeer/oversteer/neutral
    status = np.where(
        understeer_indicator > threshold, "Oversteer",
        np.where(understeer_indicator < -threshold, "Understeer", "Neutral")
    )

    df = pd.DataFrame({
        "Time": time_array,
        "Indicator": understeer_indicator,
        "Status": status
    })

    # Plot classification scatter
    fig_status = px.scatter(
        df, x="Time", y="Indicator", color="Status",
        title="Understeer / Oversteer Classification",
        labels={"Indicator": "Yaw Rate Error [rad/s]"},
        color_discrete_map={
            "Understeer": "blue",
            "Oversteer": "red",
            "Neutral": "green"
        },
        template="plotly_white"
    )

    return fig_comparison, fig_status
