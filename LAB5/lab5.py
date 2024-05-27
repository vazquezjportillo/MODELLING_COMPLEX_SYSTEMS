import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_particles = 100
space_size = 100.0
r1 = 5.0
r2 = 15.0
r3 = 30.0
dt = 0.1  # time step
num_steps = 1000

# Initialize positions and velocities
positions = np.random.rand(num_particles, 2) * space_size
velocities = (np.random.rand(num_particles, 2) - 0.5) * 10.0  # random velocities

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def apply_periodic_boundary(positions, space_size):
    return positions % space_size

def compute_center_of_mass(positions):
    return np.mean(positions, axis=0)

def update_positions(positions, velocities, dt, space_size):
    positions += velocities * dt
    return apply_periodic_boundary(positions, space_size)

def update_velocities(positions, velocities, r1, r2, r3):
    new_velocities = np.copy(velocities)
    for i, pos in enumerate(positions):
        # Zone 1: Repulsion
        zone1_particles = [positions[j] for j in range(num_particles) if i != j and distance(pos, positions[j]) < r1]
        if zone1_particles:
            center_of_mass1 = np.mean(zone1_particles, axis=0)
            direction1 = pos - center_of_mass1
            direction1 /= np.linalg.norm(direction1)
            new_velocities[i] += direction1
        
        # Zone 2: Alignment (Vicsek Model)
        zone2_particles = [velocities[j] for j in range(num_particles) if i != j and r1 <= distance(pos, positions[j]) < r2]
        if zone2_particles:
            average_velocity2 = np.mean(zone2_particles, axis=0)
            direction2 = average_velocity2 / np.linalg.norm(average_velocity2)
            new_velocities[i] += direction2
        
        # Zone 3: Attraction
        zone3_particles = [positions[j] for j in range(num_particles) if r2 <= distance(pos, positions[j]) < r3]
        if zone3_particles:
            center_of_mass3 = np.mean(zone3_particles, axis=0)
            direction3 = center_of_mass3 - pos
            direction3 /= np.linalg.norm(direction3)
            new_velocities[i] += direction3

        # Normalize velocity
        new_velocities[i] /= np.linalg.norm(new_velocities[i])

    return new_velocities

# Simulation loop
for step in range(num_steps):
    positions = update_positions(positions, velocities, dt, space_size)
    velocities = update_velocities(positions, velocities, r1, r2, r3)

    # Visualization (every 10 steps)
    if step % 10 == 0:
        plt.clf()
        plt.scatter(positions[:, 0], positions[:, 1], s=10)
        plt.xlim(0, space_size)
        plt.ylim(0, space_size)
        plt.pause(0.01)

plt.show()