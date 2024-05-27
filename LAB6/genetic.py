import numpy as np
import painter
import matplotlib.pyplot as plt
from numba import njit

eps = 1e-10

def competition(generation, scores):

    new_gen = np.zeros(generation.shape)
    cumsum = np.cumsum(scores/np.sum(scores))

    for i in range(generation.shape[0]):
        r = np.random.random()
        where = np.where(cumsum <= r)[0]

        # To handle the case when where = 0, then it should be index 0
        if len(where) > 0:
            new_gen[i, :] = generation[where.max() + 1, :]
        else: 
            new_gen[i, :] = generation[0, :]
        
    return new_gen

def crossover(generation, cross_rate):
    parents = []
    n_chroms, chrom_len = generation.shape

    new_gen = np.copy(generation)

    for i in range(n_chroms):
        if np.random.random() < cross_rate:
            parents.append(i)

    for i in range(len(parents)):
        c = np.random.randint(0, chrom_len - 1)
        new_gen[parents[i], c:] = generation[parents[(i + 1)%len(parents)], c:]

    return new_gen

def mutation(generation, mut_rate):
    n_chroms, chrom_len = generation.shape
    genes = n_chroms*chrom_len
    mutated_genes = int(genes*mut_rate*mut_rate)

    new_gen = np.copy(generation)

    for i in range(mutated_genes):
        r = np.random.randint(0, mutated_genes - 1)
        mutated_index = divmod(r, chrom_len)

        new_gen[mutated_index[0], mutated_index[1]] = np.random.randint(0, 4)

    return new_gen

def main():
    n_chrom = 100
    n_states = 54
    room = np.zeros((30, 60))

    cross_rate = 0.25
    mut_rate = 0.1
    N = 10
    
    generation = np.zeros((n_chrom, n_states))
    for i in range(n_chrom):
            generation[i, :] = np.random.choice([0, 1, 2, 3], 54)

    for i in range(N):
        scores = np.zeros(n_chrom)

        current = np.copy(generation)

        # Get scores from painter
        for j in range(n_chrom):
            for _ in range(10):
                scores[j] += painter.painter_play(generation[j], room)[0]

        scores /= 10

        generation = competition(generation, scores)
        generation = crossover(generation, cross_rate)
        generation = mutation(generation, mut_rate)

        avg_score = np.mean(scores)*100
        print(f"Avg. score of iteration {i}: \t{avg_score}")
        print(f'Total change: {abs(np.sum(current - generation))}')
    
    # Plot the final set of chromosomes
    plt.figure(figsize=(10, 5))
    plt.imshow(generation, cmap='Set3', vmin=0, vmax=2, aspect='auto')
    plt.colorbar(ticks=[0, 1, 2], label='State')
    plt.xlabel('State')
    plt.ylabel('Chromosome')
    plt.title('Final set of chromosomes')
    plt.show()
    
    best_chromosome_index = np.argmax(scores)
    best_chromosome = generation[best_chromosome_index]

    # Simulate the trajectory of the best chromosome
    room_states = painter.painter_play(best_chromosome, room, record_trajectory=True)

    # Plot the trajectory
    for i, room_state in enumerate(room_states):
        plt.figure()
        plt.imshow(room_state, cmap='gray')
        plt.title(f'Step {i}')
        plt.show()


if __name__ == "__main__":
    main()