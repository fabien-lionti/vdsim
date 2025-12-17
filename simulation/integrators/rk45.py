import numpy as np
from collections import defaultdict
from typing import Dict
from simulation.results import SimulationResult, VehicleLog, TireLog

def rk4_integration(
    model,
    x0: np.ndarray,
    u: np.ndarray,
    time_array: np.ndarray,
) -> SimulationResult:
    """
    Perform Runge–Kutta 4th order (RK4) integration with time-varying
    control inputs and log state trajectory plus model outputs.

    Assumes that `model` implements:
        dx, outputs = model.get_dx__dt(x, u_k)

    where:
        - dx: np.ndarray, shape (n_states,)
        - outputs: dict[str, np.ndarray] with physical quantities to log
          (e.g. "Fx", "Fy", "Fz", "kappa", "alpha", "ax", "ay", "Fx_aero", ...).

    Args:
        model: Vehicle model instance.
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

        # --- RK4 stages ---
        # k1: on logge les outputs à ce point
        k1, outputs = model.get_dx__dt(x, u_k)
        k1 = np.asarray(k1, dtype=float)

        for key, value in outputs.items():
            outputs_buffers[key].append(np.asarray(value))

        # k2, k3, k4: on n'utilise que dx (pas besoin de re-logger)
        k2, _ = model.get_dx__dt(x + 0.5 * dt * k1, u_k)
        k2 = np.asarray(k2, dtype=float)

        k3, _ = model.get_dx__dt(x + 0.5 * dt * k2, u_k)
        k3 = np.asarray(k3, dtype=float)

        k4, _ = model.get_dx__dt(x + dt * k3, u_k)
        k4 = np.asarray(k4, dtype=float)

        dx_equiv = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        dx_list.append(dx_equiv.copy())

        # RK4 state update
        x += dt * dx_equiv
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

    # --- Tire-level logs (ce que le modèle fournit) ---
    tires = TireLog(
        Fx=outputs_arrays.get("Fx"),
        Fy=outputs_arrays.get("Fy"),
        Fz=outputs_arrays.get("Fz"),
        kappa=outputs_arrays.get("kappa"),
        alpha=outputs_arrays.get("alpha"),
        omega=None,  # omega est dans l'état, donc on ne le re-log pas ici
    )

    # --- Vehicle-level logs ---
    vehicle = VehicleLog(
        time=time_array,
        x=x_traj,
        dx=dx_arr,
        u=u_arr,
        ax=outputs_arrays.get("ax"),
        ay=outputs_arrays.get("ay"),
        az=outputs_arrays.get("az"),
        beta=outputs_arrays.get("beta"),
        Fx_aero=outputs_arrays.get("Fx_aero"),
        Fy_aero=outputs_arrays.get("Fy_aero"),
        Fz_aero=outputs_arrays.get("Fz_aero"),
        # les autres champs (yaw, roll, etc.) resteront à None
        # et peuvent être dérivés de x/dx si tu en as besoin plus tard
    )

    return SimulationResult(
        vehicle=vehicle,
        tires=tires,
    )

# def rk4_integration(
#     self,
#     x0: np.ndarray,
#     u: np.ndarray,
#     time_array: np.ndarray,
# ) -> SimulationResult:
#     """
#     Perform Runge–Kutta 4th order (RK4) integration with time-varying
#     control inputs and log all model outputs at each time step.

#     Assumes `self` implements:
#         dx, outputs = self.get_dx__dt(x, u_k)

#     where `outputs` is a dict mapping string keys (e.g. "Fx", "ax",
#     "yaw_rate") to numpy arrays or scalars for that time step.

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

#         # --- RK4 stages ---
#         # k1: on logge les outputs de ce point (début d'intervalle)
#         k1, outputs = self.get_dx__dt(x, u_k)
#         k1 = np.asarray(k1, dtype=float)

#         # log outputs from k1
#         for key, value in outputs.items():
#             outputs_buffers[key].append(np.asarray(value))

#         # k2, k3, k4: on ignore les outputs (ou on pourrait les utiliser ailleurs)
#         k2, _ = self.get_dx__dt(x + 0.5 * dt * k1, u_k)
#         k2 = np.asarray(k2, dtype=float)

#         k3, _ = self.get_dx__dt(x + 0.5 * dt * k2, u_k)
#         k3 = np.asarray(k3, dtype=float)

#         k4, _ = self.get_dx__dt(x + dt * k3, u_k)
#         k4 = np.asarray(k4, dtype=float)

#         dx_equiv = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
#         dx_list.append(dx_equiv.copy())

#         # RK4 state update
#         x += dt * dx_equiv
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

#     # Build TireLog (keys optional depending on the model)
#     tires = TireLog(
#         Fx=outputs_arrays.get("Fx"),
#         Fy=outputs_arrays.get("Fy"),
#         Fz=outputs_arrays.get("Fz"),
#         kappa=outputs_arrays.get("kappa"),
#         alpha=outputs_arrays.get("alpha"),
#         omega=outputs_arrays.get("omega"),
#     )

#     # Build VehicleLog
#     vehicle = VehicleLog(
#         time=time_array,
#         x=x_traj,
#         dx=dx_arr,
#         u=u_arr,
#         ax=outputs_arrays.get("ax"),
#         ay=outputs_arrays.get("ay"),
#         az=outputs_arrays.get("az"),
#         yaw=outputs_arrays.get("yaw"),
#         yaw_rate=outputs_arrays.get("yaw_rate"),
#         yaw_acc=outputs_arrays.get("yaw_acc"),
#         roll=outputs_arrays.get("roll"),
#         roll_rate=outputs_arrays.get("roll_rate"),
#         roll_acc=outputs_arrays.get("roll_acc"),
#         pitch=outputs_arrays.get("pitch"),
#         pitch_rate=outputs_arrays.get("pitch_rate"),
#         pitch_acc=outputs_arrays.get("pitch_acc"),
#         beta=outputs_arrays.get("beta"),
#         Fx_aero=outputs_arrays.get("Fx_aero"),
#         Fy_aero=outputs_arrays.get("Fy_aero"),
#         Fz_aero=outputs_arrays.get("Fz_aero"),
#     )

#     return SimulationResult(
#         vehicle=vehicle,
#         tires=tires,
#         raw_debug=outputs_arrays,  # tu peux renommer ce champ si tu veux
#     )
