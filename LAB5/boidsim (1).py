import numpy as np
from scipy.stats import uniform, rayleigh
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
rho1 = 0.01
rho2 = 0.125
rho3 = 0.01
rho4 = 0.01
c = 0.01
alpha = 0.99
beta = 0.01
dt = 0.01

def main():
    # System
    N = 30
    pos = np.random.rand(N, 2)
    vel = np.random.uniform(-1, 1, size=(N, 2))

    t = 1
    num_steps = 100

    beta_values = np.linspace(0, 5, 10)  # Change this to the range of beta values you want to use
    V_values = []

    for beta in beta_values:
        alpha = 1 - beta
        V_temp = []
        for _ in range(num_steps):
            update(pos, vel, N, t, beta)
            V = compute_V(vel)
            V_temp.append(V)
        V_values.append(np.mean(V_temp))

    plt.plot(beta_values, V_values)
    plt.xlabel('Beta')
    plt.ylabel('V(t)')
    plt.show()
    
#@njit
def compute_V(vel):
    magnitudes = np.sqrt(np.sum(vel**2, axis=1))
    V = np.mean(magnitudes)
    return V

# @njit
def update(pos, vel, N, t,beta):
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
        pos[i] = (alpha * (pos[i] + dt * vel[i]) + beta * brownian(0.1, t)) % limits

    t += dt
    return t

def brownian(rmax, t):
    sigma = c**2 * t
    scale = np.sqrt(sigma / 2)
    u = uniform.rvs(scale=rayleigh.cdf(rmax, scale=scale))
    dr = rayleigh.ppf(u, scale=scale)
    theta = uniform.rvs(scale=2*np.pi)
    return dr * np.array([np.cos(theta), np.sin(theta)])

if __name__ == '__main__':
    main()