import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

n = 50
eps = 0.1
num_trials = 100

# Initialize lists to store results
connected_ratios = []
p_values = np.linspace(0, 1, 20)  # Adjust as needed

# Initialize variables to store the threshold values
p_all_disconnected = None
p_all_connected = None

for p in p_values:
    disconnected = 0
    connected = 0
    for _ in range(num_trials):
        graph = nx.erdos_renyi_graph(n, p)
        if not nx.is_connected(graph):
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

# Continue from your existing code

if p_all_disconnected is not None:
    plt.axvline(x=p_all_disconnected, color='r', linestyle='--', label='All Disconnected')

plt.legend()
plt.show()

p = (1-eps)*np.log(n)/n
p2 = (1+eps)*np.log(n)/n

# Draw the graph for p
# Draw the graph for p
graph_p = nx.erdos_renyi_graph(n, p)
plt.figure(figsize=(10, 5))
pos_p = nx.spring_layout(graph_p)  # Calculate the positions only once
connected_nodes_p = [node for node in graph_p.nodes if graph_p.degree(node) > 0]
disconnected_nodes_p = [node for node in graph_p.nodes if graph_p.degree(node) == 0]
nx.draw_networkx_nodes(graph_p, pos=pos_p, nodelist=connected_nodes_p, node_color='g', node_size=100, alpha=1.0)
nx.draw_networkx_nodes(graph_p, pos=pos_p, nodelist=disconnected_nodes_p, node_color='r', node_size=50, alpha=0.5)
nx.draw_networkx_edges(graph_p, pos=pos_p, edge_color='b')
nx.draw_networkx_labels(graph_p, pos=pos_p, font_size=8)  # Add labels to the nodes
plt.title('Graph Visualization for p', fontsize=20)
plt.show()

# Draw the graph for p2
graph_p2 = nx.erdos_renyi_graph(n, p2)
plt.figure(figsize=(10, 5))
pos_p2 = nx.spring_layout(graph_p2)  # Calculate the positions only once
connected_nodes_p2 = [node for node in graph_p2.nodes if graph_p2.degree(node) > 0]
disconnected_nodes_p2 = [node for node in graph_p2.nodes if graph_p2.degree(node) == 0]
nx.draw_networkx_nodes(graph_p2, pos=pos_p2, nodelist=connected_nodes_p2, node_color='g', node_size=100, alpha=1.0)
nx.draw_networkx_nodes(graph_p2, pos=pos_p2, nodelist=disconnected_nodes_p2, node_color='r', node_size=50, alpha=0.5)
nx.draw_networkx_edges(graph_p2, pos=pos_p2, edge_color='b')
nx.draw_networkx_labels(graph_p2, pos=pos_p2, font_size=8)  # Add labels to the nodes
plt.title('Graph Visualization for p2', fontsize=20)
plt.show()