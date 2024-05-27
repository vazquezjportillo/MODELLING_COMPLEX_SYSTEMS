import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def is_connected(graph):
    adj_matrix = nx.adjacency_matrix(graph).todense()
    num_edges = graph.number_of_edges()
    
    for i in range(adj_matrix.shape[0]):
        reachable = np.copy(adj_matrix[i])
        for _ in range(2, num_edges + 1):
            reachable = np.maximum(reachable, np.dot(reachable, adj_matrix))
        
        if np.any(reachable == 0):
            return False
    return True
        

n = 50
eps = 0.1
p = (1-eps)*np.log(n)/n
p = p - 0.1*p
p2 = (1+eps)*np.log(n)/n
p2 = p2 + 0.3*p2
num_trials = 100

disconnected = 0
connected = 0
# Initialize lists to store results
connected_ratios = []
p_values = np.linspace(0, 1, 20)  # Adjust as needed

p_all_disconnected = None
p_all_connected = None

for p in p_values:
    disconnected = 0
    connected = 0
    for _ in range(num_trials):
        graph = nx.erdos_renyi_graph(n, p)
        if not is_connected(graph):
            disconnected += 1
        else:
            connected += 1
    ratio = connected / disconnected if disconnected != 0 else connected
    connected_ratios.append(ratio)

    # Update the threshold values
    if disconnected == num_trials and p_all_disconnected is None:
        p_all_disconnected = p
    if connected == num_trials:
        p_all_connected = p

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(p_values, connected_ratios, label='Connected/Disconnected Ratio')
if p_all_disconnected is not None:
    plt.axvline(x=p_all_disconnected, color='r', linestyle='--', label='All Disconnected')
if p_all_connected is not None:
    plt.axvline(x=p_all_connected, color='g', linestyle='--', label='All Connected')
plt.xlabel('p')
plt.ylabel('Connected/Disconnected Ratio')
plt.legend()
plt.show()