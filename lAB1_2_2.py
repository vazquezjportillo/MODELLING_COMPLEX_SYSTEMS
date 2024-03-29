import numpy as np

def logistic_map(r, x):
    return r * x * (1 - x)

def perturbed_logistic_map(r, x, a):
    return logistic_map(r, x) + a * x**5

def sine_map(r, x):
    return r * np.sin(np.pi * x)

interval = (2.8, 4)  # start, end
accuracy = 0.0001
reps = 600  # number of repetitions
numtoplot = 200
a = 0.1  # Small constant for perturbed logistic map
logistic = np.zeros(reps)
logistic_perturbed = np.zeros(reps)
sine = np.zeros(reps)


# Compute bifurcation diagram for logistic map
logistic[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        logistic[i + 1] = logistic_map(r, logistic[i])

# Compute bifurcation diagram for perturbed logistic map
logistic_perturbed[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        logistic_perturbed[i + 1] = perturbed_logistic_map(r, logistic_perturbed[i], a)
   
# Compute bifurcation diagram for sine map
sine[0] = np.random.rand()
for r in np.arange(interval[0], interval[1], accuracy):
    for i in range(reps - 1):
        sine[i + 1] = sine_map(r, sine[i])