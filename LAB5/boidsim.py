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
r1 = 0.05
r2 = 0.1
r3 = 0.15
rho1 = 0.2
rho2 = 0.3
rho3 = 0.3
rho4 = 0.2
alpha = 0.7
beta = 0.3
dt = 0.01
c = 1
def main():
    # System
    N = 30
    pos = np.random.rand(N, 2)
    vel = np.random.uniform(-1, 1, size=(N, 2))

    num_steps = 100

    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    quiver = ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], angles='xy', scale_units='xy', scale=40, width=0.01, headlength=1, headwidth=1)

    def update_plot(frame):
        update(pos, vel, N)
        quiver.set_offsets(pos)
        quiver.set_UVC(vel[:, 0], vel[:, 1])
        title = ax.text(0.5, 1.05, '', ha='center', transform=ax.transAxes)
        return quiver, title

    ani = FuncAnimation(fig, update_plot, frames=num_steps, interval=10, blit=True)
    plt.show()
    
    # Set all parameters to small non-zero values, except for beta and rho2
    alpha = 0.01
    rho1 = 0.01
    rho3 = 0.01
    rho4 = 0.01
    beta_values = np.linspace(0.1, 10, 10)  # Range of values for beta
    density_values = np.linspace(0.1, 10, 10)  # Range of values for average density

    # Initialize arrays to store the average velocities
    avg_velocities_beta = np.zeros_like(beta_values)
    avg_velocities_density = np.zeros_like(density_values)

    # For each combination of beta and average density
    for i, beta in enumerate(beta_values):
        for j, density in enumerate(density_values):
           # Adjust the number of birds based on the average density
            N = int(density * width * height)
            if N == 0:
                continue
            pos = np.random.rand(N, 2)
            vel = np.random.uniform(-1, 1, size=(N, 2))

            # Run the simulation for a certain number of steps
            num_steps = 100
            velocities = np.zeros(num_steps)
            for step in range(num_steps):
                update(pos, vel, N)
                # Calculate the average velocity at this step
                velocities[step] = np.linalg.norm(vel.sum(axis=0)) / N

            # Average the velocities over all steps
            avg_velocity = velocities.mean()
            avg_velocities_beta[i] = avg_velocity
            avg_velocities_density[j] = avg_velocity

    # Plot the average velocity vs beta and vs average density
    plt.figure()
    plt.plot(beta_values, avg_velocities_beta)
    plt.xlabel('Beta')
    plt.ylabel('Average Velocity')
    plt.show()

    plt.figure()
    plt.plot(density_values, avg_velocities_density)
    plt.xlabel('Average Density')
    plt.ylabel('Average Velocity')
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
        pos[i] = (pos[i] + dt * vel[i] + brownian(0.05)) % limits

def brownian(rmax):
    dr = truncnorm.rvs(0, rmax, scale=1/(2*np.pi))
    theta = uniform.rvs(scale=2*np.pi)
    return dr * np.array([np.cos(theta), np.sin(theta)])

if __name__ == '__main__':
    main()