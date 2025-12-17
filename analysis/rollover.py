# metrics/rollover.py

from typing import Dict
import numpy as np
from simulation.results import SimulationResult

def compute_rollover_metrics(result: SimulationResult) -> Dict[str, float]:
    """
    Compute rollover-related metrics from a simulation log.

    This function extracts normal loads at the tires and computes basic
    Load Transfer Ratio (LTR) indicators. LTR is defined as the normalized
    difference of normal forces between the left and right wheels of an axle.
    A value near ``±1`` indicates an imminent rollover event.

    The function optionally uses logged vehicle states such as lateral
    acceleration and roll angle if present inside the `SimulationResult`.

    Parameters
    ----------
    result : SimulationResult
        Container returned by a simulation run, expected to include:
        - ``result.tires.Fz``: tire normal loads of shape ``(T, 4)``
          ordered as ``[FL, FR, RL, RR]``.
        - (Optional) ``result.vehicle.ay``: lateral acceleration at CG (m/s²).
        - (Optional) ``result.vehicle.roll``: vehicle roll angle φ (rad).

    Returns
    -------
    Dict[str, float]
        A dictionary containing:
        - ``LTR_front`` : Load Transfer Ratio time series for the front axle.
        - ``LTR_rear``  : Load Transfer Ratio time series for the rear axle.

    Raises
    ------
    ValueError
        If ``result.tires.Fz`` is missing or ``None``.

    Notes
    -----
    The returned LTR values are dimensionless and lie approximately in
    the range ``[-1, 1]``. Positive values indicate a transfer to the left tires,
    and negative values indicate a transfer to the right tires.

    Example
    -------
    >>> metrics = compute_rollover_metrics(res)
    >>> metrics["LTR_front"].max()
    0.73  # Example peak load transfer
    """
    Fz = result.tires.Fz  # (T, 4) or None
    if Fz is None:
        raise ValueError("Rollover metrics require tires.Fz to be logged.")

    # Unpack wheels: [FL, FR, RL, RR]
    Fz_FL = Fz[:, 0]
    Fz_FR = Fz[:, 1]
    Fz_RL = Fz[:, 2]
    Fz_RR = Fz[:, 3]

    eps = 1e-6  # avoid division by zero

    # LTR front & rear time series
    LTR_front = (Fz_FL - Fz_FR) / (Fz_FL + Fz_FR + eps)
    LTR_rear  = (Fz_RL - Fz_RR) / (Fz_RL + Fz_RR + eps)

    return {
        "LTR_front": LTR_front,
        "LTR_rear": LTR_rear,
    }
