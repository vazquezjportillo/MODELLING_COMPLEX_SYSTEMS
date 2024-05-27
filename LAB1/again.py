import numpy as np
import matplotlib.pyplot as plt

interval = (2.8, 4)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
result = np.zeros(reps)

fig, biax = plt.subplots()
fig.set_size_inches(16, 9)

result[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        result[i + 1] = r * result[i] * (1 - result[i])

    biax.plot([r] * numtoplot, result[reps - numtoplot :], "b.", markersize=0.02)

biax.set(xlabel="r", ylabel="x", title="logistic map")
plt.show()

import numpy as np

# Identify bifurcation points
bifurcation_points = []  # List to store bifurcation points
prev_result = result[0]  # Initialize previous result
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        result[i + 1] = r * result[i] * (1 - result[i])
    if abs(result[-1] - prev_result) > threshold:  # If there's a large jump in x
        bifurcation_points.append(r)  # This is a bifurcation point
    prev_result = result[-1]

# Calculate δ
differences = np.diff(bifurcation_points)  # Differences between bifurcation points
ratios = differences[:-1] / differences[1:]  # Ratios of successive differences
delta = np.mean(ratios)  # Average of ratios

# Calculate α
widths = []  # List to store widths of bifurcations
for r in bifurcation_points:
    # Calculate width of bifurcation as difference between max and min x
    width = max(result[r == r_values]) - min(result[r == r_values])
    widths.append(width)
alpha_ratios = widths[:-1] / widths[1:]  # Ratios of successive widths
alpha = np.mean(alpha_ratios)  # Average of ratios

print("Feigenbaum constants for the new map:")
print("δ =", delta)
print("α =", alpha)