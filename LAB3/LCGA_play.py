import numpy as np
import time
import matplotlib.pyplot as plt

# Global lattice Size
N = 100

class Particle:
    """
        Particle class, dont really know if this is necessary but it 
        makes it easy to add properties of the particles
    """
    def __init__(self, velocity, state):
        self.vel = velocity
        self.state = state


class Hex_cell:
    """
        Hexagonal cell identified by pos = (i, j) index. Holds 
        particles in velocity channels or in the bathtub (restchannel). 
    """
    def __init__(self, pos):
        self.pos = pos
        self.particles = np.zeros(7, dtype = Particle) # 1 rest (index 0) , 6 velocity index (1-6)
        self.neighbours = self.allocate_neighbours()
        self.state = 0  # Add this line

    def allocate_neighbours(self):
        """
            Get the neighbours of hex cell at pos using periodic BC. 
        """
        i, j = self.pos[0], self.pos[1]
        top, bottom = (i - 1)%N, (i + 1)%N
        right, left = (j + 1)%N, (j - 1)%N 
        return [self.pos, (i, right), (top, right), (top, j), (top, left), (i, left), (bottom, j)]
    
    def ad_particle(self, particle: Particle):
        """
            Ad particle to cell, inserted at apropriate index in velocity channel
            indexed by the velocity (tuple) of the particle. 
        """

        if self.particles[particle.vel] == 0:
            self.particles[particle.vel] = particle
            return
        

def create_hex_lattice():
    """
        Build the hexagonal lattice/grid. 
    """
    
    lattice = np.ndarray((N,N),dtype=Hex_cell)

    for index, _ in np.ndenumerate(lattice):
        new_cell = Hex_cell(index)
        lattice[index] = new_cell

    return lattice

def allocate_random_particles(lattice):
    """
        Insert particles into the lattice at random. The allowed states 
        are S (Succeptible) or I (Infected). 
    """

    P = 6 # Number of attempts to ad particles

    for index, cell in np.ndenumerate(lattice):
        for _ in range(P):
            r = np.random.random()
            random_vel = np.random.randint(1, 7)

            if r < 0.05:
                random_state = "I"
            else:
                random_state = "S"

            new_partile = Particle(random_vel, random_state)
            cell.ad_particle(new_partile)

    return lattice

def contact_operation(lattice):
    """
        The first of the simulation operations. Go through each particle 
        and update it's state. 
    """
    # Simulation parameters
    a = 0.05
    r = 0.05

    for _, cell in np.ndenumerate(lattice):

        # Count number of SIR in cell
        state_count = {"S": 0, "I": 0, "R": 0}
        for particle in cell.particles:
            if particle != 0:
                state_count[particle.state] += 1

        # Update state of each particle
        for particle in cell.particles:
            if particle != 0:
                rand = np.random.rand()
                if particle.state == "S":
                    prob = 1 - (1 - r)**state_count["I"]
                    if rand < prob:
                        particle.state = "I"
                elif particle.state == "I":
                    if rand < a:
                        particle.state = "R"

def count_states(lattice):
    state_count = {"S": 0, "I": 0, "R": 0}
    
    for _, cell in np.ndenumerate(lattice):
        for particle in cell.particles:
            if particle != 0:
                state_count[particle.state] += 1
        # Update the cell's state based on the states of its particles
        if state_count["I"] > state_count["R"]:
            if state_count["I"] > state_count["S"]:
                cell.state = 1  # I
            else:
                cell.state = 0  # S
        elif state_count["R"] > state_count["S"]:
            cell.state = 2  # R
        else:
            cell.state = 0  # S

    return np.array(list(state_count.values()))

def redistribute_operation(lattice):
    """
        Second step of simulation. Shuffles particles within the cell.
        Preserves particle count. 
    """

    for index, cell in np.ndenumerate(lattice):
        new_distribution = np.array([i for i in range(7)])
        np.random.shuffle(new_distribution)

        new_cell = Hex_cell(index)
        for i, particle in enumerate(cell.particles):
            if particle != 0:
                new_cell.particles[new_distribution[i]] = particle

        lattice[index] = new_cell

def traversal_operation(lattice):
    """
        TODO

        Final step of simulation. Should move particles in velocity channel
        into the neighbouring cells if there is space. 
    """

    # Allocate new lattice
    new_lattice = np.ndarray((N,N),dtype=Hex_cell)
    for cell_ind, cell in np.ndenumerate(new_lattice):
        new_lattice[cell_ind] = Hex_cell(cell_ind)

    # Move all particles
    for cell_ind, cell in np.ndenumerate(lattice):
        for index ,particle in enumerate(cell.particles):
            if particle != 0:
                new_lattice[cell.neighbours[index]].particles[index] = particle


    return new_lattice

def plot_hex_grid(lattice):
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green']  # S, I, R
    for index, cell in np.ndenumerate(lattice):
        x, y, state = index[0], index[1], cell.state
        y += 0.5 * (x % 2)  # shift every second row
        ax.scatter(x, y, color=colors[state], s=20, marker = 'h')  # s parameter controls the size of the markers

    ax.set_aspect('equal')
    plt.show()

def count_states(lattice):
    state_mapping = {"S": 0, "I": 1, "R": 2}
    state_counts = {"S": 0, "I": 0, "R": 0}
    for _, cell in np.ndenumerate(lattice):
        state_sum = 0
        particle_count = 0
        for particle in cell.particles:
            if particle != 0:
                # Map the state to an integer before adding it to state_sum
                state_sum += state_mapping[particle.state]
                particle_count += 1
        if particle_count > 0:
            average_state = state_sum / particle_count
            # Set the cell's state based on the average state of its particles
            if average_state < 0.5:
                cell.state = 0  # S
                state_counts["S"] += 1
            elif average_state < 1.5:
                cell.state = 1  # I
                state_counts["I"] += 1
            else:
                cell.state = 2  # R
                state_counts["R"] += 1
    return state_counts

def main():
    T_max = 100    
    plot_steps = [0, 20, 40, 60, 80, 100]  # Time steps at which to plot the lattice

    start_time = time.time()
    lattice = create_hex_lattice()
    allocate_random_particles(lattice)
    print("Initial states:", count_states(lattice))
    
    # Plot the initial state of the lattice
    if 0 in plot_steps:
        plot_hex_grid(lattice)

    states_S = np.zeros(T_max)
    states_I = np.zeros(T_max)
    states_R = np.zeros(T_max)
    t_range = range(T_max)

    for t in t_range:
        contact_operation(lattice)
        redistribute_operation(lattice)
        lattice = traversal_operation(lattice)
        state_counts = count_states(lattice)
        states_S[t] = state_counts["S"]
        states_I[t] = state_counts["I"]
        states_R[t] = state_counts["R"]

        # Plot the state of the lattice at the specified time steps
        if t in plot_steps:
            plot_hex_grid(lattice)

    print("Final states:", count_states(lattice))
    print(f"Execution time: {time.time() - start_time}")

    plt.figure()
    plt.title("SIR Simulation")
    plt.plot(t_range, states_S, label = "S")
    plt.plot(t_range, states_I, label = "I")
    plt.plot(t_range, states_R, label = "R")

    plt.xlabel(r"Time $(t)$")
    plt.ylabel(r"Population")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    main()