import numpy as np
import matplotlib.pyplot as plt
# Parameters
s = 10
r = 28
b = 8/3

def lorenz(x, y, z, s=s, r=r, b=b):
    """
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot  

def rk4_step(f, v, dt):
    """Take one RK4 step. Return updated solution."""
    k1 = dt * np.array(f(*v))
    k2 = dt * np.array(f(*(v + 0.5 * k1)))
    k3 = dt * np.array(f(*(v + 0.5 * k2)))
    k4 = dt * np.array(f(*(v + k3)))

    v = v + (k1 + 2*k2 + 2*k3 + k4) / 6
    return v

# Initial conditions
x0, y0, z0 = 0.03, 0.01, 10

# Time parameters
dt = 0.01  # Time step
T = 100    # Total time
num_steps = int(T / dt)  # Number of time steps

# Arrays to store results
x_values = np.zeros(num_steps)
y_values = np.zeros(num_steps)
z_values = np.zeros(num_steps)

# Initial values
v = np.array([x0, y0, z0])

# Time evolution using RK4
for i in range(num_steps):
    v = rk4_step(lorenz, v, dt)
    x_values[i], y_values[i], z_values[i] = v

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, lw=0.5)
ax.scatter3D(x_values[0], y_values[0], z_values[0],'.',marker='o',color='b') #The starting point
ax.scatter3D(x_values[-1], y_values[-1], z_values[-1],'r.',marker='*') #The final point after num_steps of "running" the model
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()
