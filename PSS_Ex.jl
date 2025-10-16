#=
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
=#

dim = 2
beta = 0.0
#epsilon = 0.01 
ub = 0.2
#h = [1.0, 1.5, 2.0]
#h1 = 1.0
#h = [h1+0.2*(i-1) for i in 1:dim]
h = [1.0, 2.0]
#h = repeat([1.0*i for i in 1:5], outer=4)
#z = [1.01, 1.0]
z = 1.0*ones(dim)
#gamma = [-1.0, -1.0, -1.0]
gamma = -ones(dim)
sigma = 1.0*ones(dim)
#sigma = [1.0, 1.0, 1.0]

#mu = [1.0, 1.0, 1.0, 1.0, 1.0]
mu = ones(2*dim-1)
#mu1, mu2, mu3 = 1.0, 1.0, 1.0
#G = [1.0 0; -mu[4]/mu[2] 1.0; 0 -mu[5]/mu[3]]
G = zeros(dim, dim-1)
for i in 1:dim-1
    G[i,i] = 1.0
    G[i+1,i] = -mu[i+dim]/mu[i+1]
end
drift_b = 5.0

alpha = 1.0

#myshape = 0.5

#mmt = 0.2

#M = [786432, 6144, 384, 240, 72, 12, 12, 8, 4, 4]
M = [196608, 768, 192, 96, 60, 24, 12, 8, 4, 4]