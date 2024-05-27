import numpy as np
from scipy.stats import uniform, truncnorm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from numba import njit

# Simulation parameters
width = 1
height = 1
limits = np.array([width, height])
xlim = (0, width)
ylim = (0, height)
r1 = 0.1
r2 = 0.2
r3 = 0.3
rho1 = 0.001
rho2 = 0.125
rho3 = 0.001
rho4 = 0.001
alpha = 0.99
beta = 0.01
dt = 0.001
c = 5

def main():
    # System
    N = 30
    pos = np.random.rand(N, 2)
    vel = np.random.uniform(-1, 1, size=(N, 2))

    t = 1
    num_steps = 100

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    quiver = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], angles='xy', scale_units='xy', scale=40, width=0.01, headlength=1, headwidth=1)

    def update_plot(frame, t):
        t = update(pos, vel, N, t)
        quiver.set_offsets(pos)
        quiver.set_UVC(vel[:, 0], vel[:, 1])
        title = ax.text(0.5, 1.05, '', ha='center', transform=ax.transAxes)
        return quiver, title

    ani = FuncAnimation(fig, update_plot, fargs=(t,), frames=num_steps, interval=10, blit=True)
    plt.tight_layout()
    plt.show()
    
    N = 30
    pos = np.random.rand(N, 2)
    vel = np.random.uniform(-1, 1, size=(N, 2))

    t = 1
    num_steps = 100

    N_values = np.arange(1, 50, 10)  # Change this to the range of N values you want to use
    mean_speeds = np.zeros_like(N_values, dtype=float)

    for i, N in enumerate(N_values):
        pos = np.random.rand(N, 2)
        vel = np.random.uniform(-1, 1, size=(N, 2))

        va_values = np.zeros(num_steps)  # Array to store va at each time step

        for j in range(num_steps):
            update(pos, vel, N)  
            normalized_velocities = vel / np.linalg.norm(vel, axis=1, keepdims=True)
            va = np.mean(np.abs(normalized_velocities))/N
            va_values[j] = va  # Store va at this time step

        mean_speeds[i] = np.mean(va_values)  # Temporal mean of va

    area = width*height
    # Calculate densities
    density_values = N_values / area

    # Plot mean self-propulsion speed against density
    plt.figure()
    plt.plot(density_values, mean_speeds)
    plt.xlabel('Density')
    plt.ylabel('Temporal mean of mean self-propulsion speed')
    plt.show()
# @njit
def update(pos, vel, N):
    # Zoning
    com1 = np.zeros((N, 2))
    n1 = np.zeros((N,))
    avgvel2 = np.zeros((N, 2))
    n2 = np.zeros((N,))
    com3 = np.zeros((N, 2))
    n3 = np.zeros((N,))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i][0] - pos[j][0]
            dx = dx - width * np.round(dx / width)
            dy = pos[i][1] - pos[j][1]
            dy = dy - height * np.round(dy / height)
            sqdist = dx**2 + dy**2
            if sqdist < r1**2:
                com1[i] += pos[j]
                n1[i] += 1 
                com1[j] += pos[i]
                n1[j] += 1
            elif sqdist < r2**2:
                avgvel2[i] += vel[j]
                n2[i] += 1
                avgvel2[j] += vel[i]
                n2[j] += 1
            elif sqdist < r3**2:
                com3[i] += pos[j]
                n3[i] += 1
                com3[j] += pos[i]
                n3[j] += 1
        
        if n1[i] > 0:
            com1[i] /= n1[i]
        if n2[i] > 0:
            avgvel2[i] /= n2[i]
        if n3[i] > 0:
            com3[i] /= n3[i]

        # Unit vectors
        if np.linalg.norm(pos[i] - com1[i]) > 0:
            v1 = pos[i] - com1[i]
            e1 = v1 / np.linalg.norm(v1)
        else:
            e1 = np.zeros(2)
        
        if np.linalg.norm(avgvel2[i] + vel[i]) > 0:
            v2 = avgvel2[i] + vel[i]
            e2 = v2 / np.linalg.norm(v2)
        else:
            e2 = np.zeros(2)
        
        if np.linalg.norm(com3[i] - pos[i]) > 0:
            v3 = com3[i] - pos[i]
            e3 = v3 / np.linalg.norm(v3)
        else:
            e3 = np.zeros(2)

        # Update
        vel[i] += rho1 * e1 + rho2 * e2 + rho3 * e3 + rho4 * vel[i]
        vel[i] /= np.linalg.norm(vel[i]) # Normalizing velocity
        pos[i] = (pos[i] + alpha * dt * vel[i] + beta * brownian(0.1)) % limits

def brownian(rmax):
    dr = truncnorm.rvs(0, rmax, scale=c**2 * dt)
    theta = uniform.rvs(0, 2*np.pi)
    return dr * np.array([np.cos(theta), np.sin(theta)])

if __name__ == '__main__':
    main()