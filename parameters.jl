#parameters 

beta = 0.2  #discount factor
epsilon = 0.01 #ignore this 
#ub = -log(epsilon * beta) / beta
ub = -log(epsilon) #ignore this
#ub = 3.5
offset = 0.0 #ignore this
h = 2.0 #holding cost
c = 1.0 #pushing cost
z = 0.6 #initial state
gamma = -1 - offset #this is the drift = -1.0
dim = 1 #ignore this
drift_b = 100.0 #this is your κ, upper bound on drift
sigma = 1.0 #standard deviation of brownian motion
myshape = 0.5


#ignore all values below

#level = 6
M = [40000, 1024, 128, 32, 16, 16, 8]
M1 = 7
M2 = 10

mmt = 0.1
penalty1 = 0.1
penalty2 = 0.2

rate = 0.5