import numpy as np

def straight_line(length=100, dt=0.1, vx=10.0):
    T = int(length / (vx * dt))
    x = np.linspace(0, length, T)
    y = np.zeros_like(x)
    psi = np.zeros_like(x)
    return np.vstack([x, y, psi]).T

def circular(radius=20, v=10.0, dt=0.1):
    omega = v / radius
    T = int((2 * np.pi * radius) / (v * dt))
    t = np.arange(T) * dt
    x = radius * np.cos(omega * t)
    y = radius * np.sin(omega * t)
    psi = omega * t
    return np.vstack([x, y, psi]).T

def sinusoidal(amplitude=5, wavelength=20, vx=10.0, total_time=10.0, dt=0.1):
    t = np.arange(0, total_time, dt)
    x = vx * t
    y = amplitude * np.sin(2 * np.pi * x / wavelength)
    dy_dx = (2 * np.pi * amplitude / wavelength) * np.cos(2 * np.pi * x / wavelength)
    psi = np.arctan(dy_dx)
    return np.vstack([x, y, psi]).T

def lemniscate(a=20, vx=10.0, dt=0.1, total_time=20):
    t = np.arange(0, total_time, dt)
    omega = 2 * np.pi / total_time
    x = a * np.sin(omega * t)
    y = a * np.sin(omega * t) * np.cos(omega * t)
    dx = a * omega * np.cos(omega * t)
    dy = a * omega * (np.cos(2 * omega * t) - np.sin(2 * omega * t)) / 2
    psi = np.arctan2(dy, dx)
    return np.vstack([x, y, psi]).T
