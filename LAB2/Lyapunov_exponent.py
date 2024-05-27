# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/jvp/miniconda3/lib/python3.12/site-packages')
import lyapynov

s = 10
r = 28
b = 8/3
x0 = np.array([0.03, 0.01, 10])
t0 = 0
dt = 0.01

def f(x, t, s=s, r=r, b=b):
    v = np.zeros_like(x)
    v[0] = s*(x[1] - x[0])
    v[1] = r*x[0] - x[1] - x[0]*x[2]
    v[2] = x[0]*x[1] - b*x[2]
    return v  

def jac(x, t, s=s, r=r, b=b):
    v = np.zeros((x.shape[0], x.shape[0]))
    v[0,0], v[0,1] = -s, s
    v[1,0], v[1,1], v[1,2] = r - x[2], -1., -x[0]
    v[2,0], v[2,1], v[2,2] = x[1], x[0], -b
    return v

Lorenz_system = lyapynov.ContinuousDS(x0, t0, f, jac, dt)
Lorenz_system.forward(10**6, False)

mLCE, history = lyapynov.mLCE(Lorenz_system, 0, 10**6, True)

print("mLCE: {:.3f}".format(mLCE))

plt.figure(figsize = (10,6))
plt.plot(history[:5000])
plt.xlabel("Number of time steps")
plt.ylabel("Largest Lyapunov exponent")
plt.title("Evolution of the largest Lyapunov exponent for the first 5000 time steps")
plt.show()