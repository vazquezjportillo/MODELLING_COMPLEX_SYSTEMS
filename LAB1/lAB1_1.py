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

biax.set(xlabel="r", ylabel="x", title="Logistic map  f=rx(1-x)")
plt.show()