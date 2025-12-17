import numpy as np
from collections import defaultdict
from typing import Dict
from simulation.results import SimulationResult, VehicleLog, TireLog


def euler_integration(
    model,
    x0: np.ndarray,
    u: np.ndarray,
    time_array: np.ndarray,
) -> SimulationResult:
    """
    Perform Euler integration with time-varying control inputs and log
    state trajectory plus model outputs at each time step.

    Args:
        model: Vehicle model instance exposing
            dx, outputs = model.get_dx__dt(x, u_k)
        x0 (np.ndarray): Initial state vector, shape (n_states,).
        u (np.ndarray): Control input array, shape (T, n_inputs).
        time_array (np.ndarray): Time vector, shape (T,).

    Returns:
        SimulationResult: Container with vehicle-level and tire-level logs.
    """
    time_array = np.asarray(time_array)
    u = np.asarray(u)
    x = np.array(x0, dtype=float)

    T = time_array.shape[0]
    n_states = x.shape[0]
    n_inputs = u.shape[1]

    dt = time_array[1] - time_array[0]  # assume constant time step

    # Logs
    x_traj = [x.copy()]
    dx_list = []
    u_list = []

    # key -> list of values over time (T-1)
    outputs_buffers = defaultdict(list)

    for k in range(1, T):
        u_k = u[k - 1]
        u_list.append(u_k.copy())

        dx, outputs = model.get_dx__dt(x, u_k)
        dx = np.asarray(dx, dtype=float)
        dx_list.append(dx.copy())

        # log all outputs (forces, slips, accelerations, aero, etc.)
        for key, value in outputs.items():
            outputs_buffers[key].append(np.asarray(value))

        # Euler step
        x += dt * dx
        x_traj.append(x.copy())

    # Assemble trajectories
    x_traj = np.stack(x_traj)  # (T, n_states)

    # Pad dx and u with NaNs at t=0 to align with time/x
    dx_arr = np.vstack(
        [np.full((1, n_states), np.nan), np.stack(dx_list)]
    )  # (T, n_states)

    u_arr = np.vstack(
        [np.full((1, n_inputs), np.nan), np.stack(u_list)]
    )  # (T, n_inputs)

    # Convert outputs buffers -> arrays (padded at t=0 with NaNs)
    outputs_arrays: Dict[str, np.ndarray] = {}
    for key, values in outputs_buffers.items():
        arr = np.stack(values)  # (T-1, ...)
        pad_shape = (1,) + arr.shape[1:]
        pad = np.full(pad_shape, np.nan)
        arr = np.concatenate([pad, arr], axis=0)  # (T, ...)
        outputs_arrays[key] = arr

    # --- Build TireLog (seulement ce que le modèle fournit) ---
    tires = TireLog(
        Fx=outputs_arrays.get("Fx"),
        Fy=outputs_arrays.get("Fy"),
        Fz=outputs_arrays.get("Fz"),
        kappa=outputs_arrays.get("kappa"),
        alpha=outputs_arrays.get("alpha"),
    )

    # --- Build VehicleLog ---
    vehicle = VehicleLog(
        time=time_array,
        x=x_traj,
        dx=dx_arr,
        u=u_arr,
        ax=outputs_arrays.get("ax"),
        ay=outputs_arrays.get("ay"),
        az=outputs_arrays.get("az"),
        Fx_aero=outputs_arrays.get("Fx_aero"),
        Fy_aero=outputs_arrays.get("Fy_aero"),
        Fz_aero=outputs_arrays.get("Fz_aero"),
        beta=outputs_arrays.get("beta"),
    )

    return SimulationResult(
        vehicle=vehicle,
        tires=tires,
        # raw_debug=outputs_arrays,  # tous les outputs bruts si tu veux y accéder
    )

# def euler_integration(
#     model,
#     x0: np.ndarray,
#     u: np.ndarray,
#     time_array: np.ndarray,
# ) -> SimulationResult:
#     """
#     Perform Euler integration with time-varying control inputs and log
#     all model outputs at each time step.

#     This function assumes that `self` is a vehicle model implementing:
#         dx, outputs = self.get_dx__dt(x, u_k)

#     where `outputs` is a dict mapping string keys (e.g. "Fx", "ax", "yaw_rate")
#     to numpy arrays or scalars for that time step.

#     Args:
#         x0 (np.ndarray): Initial state vector, shape (n_states,).
#         u (np.ndarray): Control input array, shape (T, n_inputs).
#         time_array (np.ndarray): Time vector, shape (T,).

#     Returns:
#         SimulationResult: Container with vehicle-level and tire-level logs.
#     """
#     time_array = np.asarray(time_array)
#     u = np.asarray(u)
#     x = np.array(x0, dtype=float)

#     T = time_array.shape[0]
#     n_states = x.shape[0]
#     n_inputs = u.shape[1]

#     dt = time_array[1] - time_array[0]  # assume constant time step

#     # Logs
#     x_traj = [x.copy()]
#     dx_list = []
#     u_list = []

#     # key -> list of values over time (T-1)
#     outputs_buffers = defaultdict(list)

#     for k in range(1, T):
#         u_k = u[k - 1]
#         u_list.append(u_k.copy())

#         dx, outputs = model.get_dx__dt(x, u_k)
#         dx = np.asarray(dx, dtype=float)
#         dx_list.append(dx.copy())

#         # log all outputs
#         for key, value in outputs.items():
#             outputs_buffers[key].append(np.asarray(value))

#         # Euler step
#         x += dt * dx
#         x_traj.append(x.copy())

#     # Assemble trajectories
#     x_traj = np.stack(x_traj)  # (T, n_states)

#     # Pad dx and u with NaNs at t=0 to align with time/x
#     dx_arr = np.vstack(
#         [np.full((1, n_states), np.nan), np.stack(dx_list)]
#     )  # (T, n_states)

#     u_arr = np.vstack(
#         [np.full((1, n_inputs), np.nan), np.stack(u_list)]
#     )  # (T, n_inputs)

#     # Convert outputs buffers -> arrays (padded at t=0 with NaNs)
#     outputs_arrays: Dict[str, np.ndarray] = {}
#     for key, values in outputs_buffers.items():
#         arr = np.stack(values)  # (T-1, ...)
#         pad_shape = (1,) + arr.shape[1:]
#         pad = np.full(pad_shape, np.nan)
#         arr = np.concatenate([pad, arr], axis=0)  # (T, ...)
#         outputs_arrays[key] = arr