#parameters 

beta = 0.5
epsilon = 0.01
ub = -3.0*log(epsilon * beta) / beta
# ub = 1
offset = 0.0
h = 2.0
c = 1.0
z = 0.7
gamma = -1 - offset
dim = 1
drift_b = 2.0
sigma = 1.0

#level = 6
M = [1024^2, 1024, 128, 32, 16, 12, 10]
M1 = 7
M2 = 10

mmt = 0.4
delta = 0.0

rate = 1.0