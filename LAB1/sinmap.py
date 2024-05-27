import numpy as np
import matplotlib.pyplot as plt

interval = (0.4, 1)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
lims = np.zeros(reps)
range_r = np.arange(interval[0], interval[1], accuracy)
lamb = np.zeros(len(range_r))

fig, (biax, lambax) = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(16, 9)

lims[0] = np.random.rand()
index = 0
for r in range_r:
    partial_lamb = 0
    for i in range(reps - 1):
        lims[i + 1] = r * np.sin(np.pi * lims[i])
        derivative = r * np.pi * np.cos(np.pi * lims[i])
        partial_lamb += np.log(abs(derivative))
    lamb[index] = partial_lamb / reps
    index += 1
    biax.plot([r] * numtoplot, lims[reps - numtoplot :], "b.", markersize=0.02)

# Print the value of the function for the first positive value of lamb (bigger than eps = 0.01)
eps = 0.01
for i in range(len(lamb)):
    if lamb[i] > eps:
        print("The value of the function for the first positive value of lambda is: ", range_r[i])
        break

# Plotting
biax.set_title("Sin Map", fontsize=20)
biax.set_xlabel("r", fontsize=16)
biax.set_ylabel("x", fontsize=16)
biax.tick_params(axis='both', which='major', labelsize=14)

biax.plot(range_r, 0.5*np.ones(len(range_r)), "g--")

lambax.plot(range_r, lamb, "r-")
lambax.plot(range_r, np.zeros(len(range_r)), "k--")

lambax.set_title("Lambda", fontsize=20)
lambax.set_xlabel("r", fontsize=16)
lambax.set_ylabel("lambda", fontsize=16)
lambax.tick_params(axis='both', which='major', labelsize=14)

lambax.set_ylim(-7, 1)

plt.savefig('plotsin.png', dpi = 150)

plt.show()