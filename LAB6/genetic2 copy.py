import numpy as np
import painter2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from numba import njit

eps = 1e-10

def competition(generation, scores):
    new_gen = np.zeros(generation.shape)
    cumsum = np.cumsum(scores / (np.sum(scores) + eps))

    for i in range(generation.shape[0]):
        r = np.random.random()
        where = np.where(cumsum > r)[0][0]  # Select first index where cumulative sum exceeds r
        new_gen[i, :] = generation[where, :]

    return new_gen

def crossover(generation, cross_rate):
    n_chroms, chrom_len = generation.shape
    new_gen = np.copy(generation)

    parents = [i for i in range(n_chroms) if np.random.random() < cross_rate]

    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            p1, p2 = parents[i], parents[i + 1]
            cross_point = np.random.randint(1, chrom_len - 1)
            new_gen[p1, cross_point:], new_gen[p2, cross_point:] = generation[p2, cross_point:], generation[p1, cross_point:]

    return new_gen

def mutation(generation, mut_rate):
    n_chroms, chrom_len = generation.shape
    new_gen = np.copy(generation)

    for _ in range(int(n_chroms * chrom_len * mut_rate)):
        chrom_idx = np.random.randint(0, n_chroms)
        gene_idx = np.random.randint(0, chrom_len)
        new_gen[chrom_idx, gene_idx] = np.random.randint(0, 4)

    return new_gen

def plot_chromosomes(generation):
    plt.figure(figsize=(10, 6))
    plt.imshow(generation, cmap='viridis', aspect='auto')
    plt.colorbar(label='Action')
    plt.xlabel('Gene')
    plt.ylabel('Chromosome')
    plt.title('Final Generation of Chromosomes')
    plt.show()


def animate_trajectory(room, xpos, ypos, N):
    cmap = ListedColormap(['white', 'black', 'white'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(room, cmap=cmap, origin='upper')

    line, = ax.plot([], [], 'b.-')
    position_marker, = ax.plot([], [], 'ro')  # Red dot for current position

    plt.gca().set_xticks(np.arange(-.5, room.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-.5, room.shape[0], 1), minor=True)
    plt.gca().grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    plt.gca().xaxis.set_ticks_position('none') 
    plt.gca().yaxis.set_ticks_position('none') 

    def init():
        line.set_data([], [])
        position_marker.set_data([], [])  # Initialize red dot
        return line, position_marker

    def update(frame):
        line.set_data(ypos[:frame], xpos[:frame])
        position_marker.set_data(ypos[frame], xpos[frame])  # Update red dot position
        return line, position_marker

    # Increase speed by decreasing interval value
    interval = 50  # milliseconds
    ani = FuncAnimation(fig, update, frames=min(len(xpos),700), init_func=init, blit=True, repeat=False, interval=interval)
    plt.title('Trajectory of a Successful Chromosome')
    plt.xlabel('Y Position')
    plt.ylabel('X Position')

    # Set filename based on N
    filename = f'trajectory_animation_N_{N}.mp4'
    
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])


def main():
    np.random.seed(42)  # Set seed for reproducibility

    n_chrom = 100
    n_states = 54
    room = np.zeros((30, 60))

    cross_rate = 0.25
    mut_rate = 0.1
    N = 200

    generation = np.random.choice([0, 1, 2, 3], (n_chrom, n_states))

    for i in range(N):
        scores = np.zeros(n_chrom)

        current = np.copy(generation)

        # Get scores from painter
        for j in range(n_chrom):
            for _ in range(10):  # Average score over 10 runs to reduce noise
                scores[j] += painter2.painter_play(generation[j], room)[0]

        scores /= 10
        generation = competition(generation, scores)
        generation = crossover(generation, cross_rate)
        generation = mutation(generation, mut_rate)

        avg_score = np.mean(scores)
        print(f"Avg. score of iteration {i}: \t{avg_score:.4f}")
        print(f'Total change: {np.sum(np.abs(current - generation))}')

    # Plot the final generation of chromosomes
    plot_chromosomes(generation)

    # Select one of the successful chromosomes
    best_idx = np.argmax(scores)
    best_chromosome = generation[best_idx]

    # Simulate the trajectory of the best chromosome
    score, xpos, ypos, env = painter2.painter_play(best_chromosome, room)

    # Animate the trajectory
    animate_trajectory(env, xpos, ypos,N)

if __name__ == "__main__":
    main()
