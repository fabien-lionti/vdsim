# simulation/integrators/step.py

import numpy as np

def integrate_one_step(model, x, u, dt, method="euler"):
    """
    Single-step propagation with Euler or RK4.

    Returns:
        x_next (np.ndarray)
        outputs (dict) from model.get_dx__dt()
    """
    method = method.lower()

    if method == "euler":
        dx, outputs = model.get_dx__dt(x, u)
        return x + dt * np.asarray(dx), outputs

    elif method == "rk4":
        k1, _ = model.get_dx__dt(x, u)
        k2, _ = model.get_dx__dt(x + 0.5 * dt * k1, u)
        k3, _ = model.get_dx__dt(x + 0.5 * dt * k2, u)
        k4, outputs = model.get_dx__dt(x + dt * k3, u)

        dx = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        return x + dt * dx, outputs

    else:
        raise ValueError(f"Unknown method '{method}'")
