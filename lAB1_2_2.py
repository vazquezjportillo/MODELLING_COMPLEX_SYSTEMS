import numpy as np
from scipy.optimize import newton
import numpy as np

def logistic_map(mu, x):
    return mu * x * (1 - x)

def dlogistic_map(mu, x):
    return mu * (1 - 2 * x)

def perturbed_logistic_map(r, x):
    return r*np.sin(x*np.pi)

def dperturbed_logistic_map(r, x):
    return r*np.cos(x*np.pi)*np.pi

def f(n, mu):
    logistic_val = np.zeros(2**n)
    logistic_val[0] = logistic_map(mu, 0.5)
    dlogistic_val = 0.25
    for i in range(1, 2**n):
        logistic_val[i] = logistic_map(mu, logistic_val[i-1])
        devpart = dlogistic_map(mu, logistic_val[i-1])
        dlogistic_val *= devpart
    return logistic_val[-1] - 0.5, dlogistic_val

def fperturbed(n, mu):
    perturbed_logistic_val = np.zeros(2**n)
    perturbed_logistic_val[0] = perturbed_logistic_map(mu, 0.5)
    dperturbed_logistic_val = np.sin(0.5*np.pi)
    for i in range(1, 2**n):
        perturbed_logistic_val[i] = perturbed_logistic_map(mu, perturbed_logistic_val[i-1])
        perturbed_devpart = dperturbed_logistic_map(mu, perturbed_logistic_val[i-1])
        dperturbed_logistic_val *= perturbed_devpart
    return perturbed_logistic_val[-1] - 0.5, dperturbed_logistic_val

niter = 7
mu_values = np.zeros(niter)
mu_values[0] = 2
mu_values[1] = 3.23607
mu_values_perturbed = np.zeros(niter)
mu_values_perturbed[0] = 0.5
mu_values_perturbed[1] = 0.7777
deltareal = 4.6692016

for n in range(2, niter):
    x0 = mu_values[n-1] + (mu_values[n-1] - mu_values[n-2]) / deltareal
    mu_values[n] = newton(lambda x: f(n, x)[0], x0, fprime=lambda x: f(n, x)[1])
    
for n in range(2, niter):
    x0 = mu_values_perturbed[n-1] + (mu_values_perturbed[n-1] - mu_values_perturbed[n-2]) / deltareal
    mu_values_perturbed[n] = newton(lambda x: fperturbed(n, x)[0], x0, fprime=lambda x: fperturbed(n, x)[1], maxiter=4000)

delta_values = [(mu_values[i+1] - mu_values[i]) / (mu_values[i] - mu_values[i-1]) for i in range(1, len(mu_values)-1) if mu_values[i] != mu_values[i-1]]
delta = np.mean(delta_values)
delta = 1/delta

delta_values_perturbed = [(mu_values_perturbed[i+1] - mu_values_perturbed[i]) / (mu_values_perturbed[i] - mu_values_perturbed[i-1]) for i in range(1, len(mu_values_perturbed)-1) if mu_values_perturbed[i] != mu_values_perturbed[i-1]]
delta_perturbed = np.mean(delta_values_perturbed)
delta_perturbed = 1/delta_perturbed


print("Feigenbaum delta constant:", delta)
print("Feigenbaum delta constant perturbed:", delta_perturbed)

last_deriv_value = f(niter-1, mu_values[niter-1])[1]
pre_last_deriv_value = f(niter-2, mu_values[niter-2])[1]
alpha = last_deriv_value / pre_last_deriv_value
print("Feigenbaum alpha constant:", alpha)
