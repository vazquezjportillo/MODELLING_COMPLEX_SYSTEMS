import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

n = 50
eps = 0.1
p = (1-eps)*np.log(n)/n
p2 = (1+eps)*np.log(n)/n
num_trials = 100

disconnected = 0
connected = 0
connected_p = 0  
connected_p2 = 0  

for _ in range(num_trials):
    graph = nx.erdos_renyi_graph(n, p)
    graph2 = nx.erdos_renyi_graph(n, p2)
    if nx.is_connected(graph):
        connected_p += 1
    if not nx.is_connected(graph):
        disconnected += 1
    if nx.is_connected(graph2):
        connected += 1
        connected_p2 += 1

print('Probability 1: ', p)
print('Number of connected graphs for p: ', connected_p)
print('Percentage of connected graphs for p: ', (connected_p/num_trials)*100)
print('Probability 2: ', p2)
print('Number of connected graphs for p2: ', connected_p2)
print('Percentage of connected graphs for p2: ', (connected_p2/num_trials)*100)

p_values = np.linspace(0, 0.3, 100) 
percentages = []  

for p_value in p_values:
    connected = 0
    for _ in range(num_trials):
        graph = nx.erdos_renyi_graph(n, p_value)
        if nx.is_connected(graph):
            connected += 1
    percentage = (connected / num_trials) * 100  
    percentages.append(percentage)

plt.figure(figsize=(10, 5))
plt.plot(p_values, percentages)
plt.axvline(x=p, color='r', linestyle='--', label=r'$p=\frac{(1-\epsilon)*ln(n)}{n}$') 
plt.axvline(x=p2, color='g', linestyle='--', label=r'$p=\frac{(1+\epsilon)*ln(n)}{n}$')  
plt.title(f'n = {n}, epsilon = {eps}', fontsize=20)  
plt.legend(fontsize=20) 
plt.tick_params(axis='both', which='major', labelsize=20)  
plt.show()
