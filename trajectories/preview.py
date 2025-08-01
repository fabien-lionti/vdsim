import matplotlib.pyplot as plt

def plot_trajectory(traj, label="Trajectory"):
    x, y, psi = traj[:, 0], traj[:, 1], traj[:, 2]
    plt.plot(x, y, label=label)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    plt.legend()