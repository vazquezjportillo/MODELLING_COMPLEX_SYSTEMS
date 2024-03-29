import numpy as np
import matplotlib.pyplot as plt


def perturbed_logistic_map(r, x, a):
    return r*x*(1-x) + a * x**5

def dperturbed_logistic_map(r, x, a):
    return r*(1-2*x) + 5*a*x**4

def logistic_map(r, x):
    return r * x * (1 - x)

def dlogistic_map(r, x):
    return r * (1 - 2 * x)

r_values = np.linspace(2.4, 4, 1000)
lyapunov_exponents_log = []
lyapunov_exponents_perturlog = []
a = 0.1

for r in r_values:
    x = np.random.random()
    for _ in range(1000):  # Discard the first 1000 iterations
        x = logistic_map(r, x)
    sum = 0
    for _ in range(10000):  # Next 10000 iterations
        sum += np.log(abs(dlogistic_map(r, x)))
        x = logistic_map(r, x)
    lyapunov_exponents_log.append(sum / 10000)
    
for r in r_values:
    x = np.random.random()
    for _ in range(1000):  # Discard the first 1000 iterations
        x = perturbed_logistic_map(r, x,a)
    sum = 0
    for _ in range(10000):  # Next 10000 iterations
        sum += np.log(abs(dperturbed_logistic_map(r, x,a)))
        x = perturbed_logistic_map(r, x,a)
    lyapunov_exponents_perturlog.append(sum / 10000)

plt.plot(r_values, lyapunov_exponents_log)
plt.plot(r_values, lyapunov_exponents_perturlog)
plt.xlabel('r')
plt.ylabel('Lyapunov exponent')
plt.show()

#This code calculates the Lyapunov exponent 
# for the logistic map for r values between
# 2.4 and 4. The Lyapunov exponent is negative 
# when the system is stable (the trajectories
# converge) and positive when the system is chaotic 
# (the trajectories diverge). The periodic windows in the 
# bifurcation diagram correspond to regions where the system
# is stable, which is why the Lyapunov exponent is negative 
# there. The chaotic regions correspond to regions where the 
# system is chaotic, which is why the Lyapunov exponent is 
# positive there.