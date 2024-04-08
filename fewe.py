import numpy as np
epsilon = 0.6   
sigma = 5.67*10**-8
Q = 340
t_mean = 289.208
alfa0 = 0.3

A = (-epsilon * sigma)/(t_mean**2)
B = -epsilon * sigma
C = (Q/t_mean**2)
D = Q-2*alfa0*Q

print(A)
print(B)
print(C)
print(epsilon*sigma)