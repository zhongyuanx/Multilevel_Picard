struct PSS_Ex{T1 <: AbstractFloat, T2 <: Int, T3 <: Function, T4 <: Function}
    z0::Array{T1} #initial state
    dim::T2 #dimension of problem
    beta::T1 #discount rate
    drift::Array{T1} #drifts
    mu::Array{T1} #service rates
    Sigma::Array{T1} #covariance matrix
    cost::T3 #holding cost function
    ctrl::T4 #nonlinear control function
end

beta = 0.1  
epsilon = 0.01 
ub = 0.5
h = [1.0, 2.0]
z = [1.0, 1.0]
gamma = [-1.0, -1.0]
sigma = [1.0, 1.0]

mu = [1.0, 1.0, 1.0]
#mu1, mu2, mu3 = 1.0, 1.0, 1.0
G = [1.0, -mu[3]/mu[2]]
drift_b = 5.0

myshape = 0.5

mmt = 0.2

M = [40000, 1024, 128, 32, 16, 8, 8]