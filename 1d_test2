using Random, Distributions, Statistics
include("rbm.jl")

Random.seed!(1234)

offset = 0
h = 2
beta = 1
z = 1
gamma = -1 - offset
epsilon = 0.01
ub = -log(epsilon * beta) / beta
# ub = 1
dim = 1
push_cost = 1
drift_b = 2
sigma = 1

function cost(state)
    return h * state
end

function my_C(beta, lb, ub)
    return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
end

function my_g(v, drift_bd, offset)
    #return offset * v + drift_b * min(push_cost - v, 0) # if v > push_cost else 0
    return offset * v + drift_bd * min(push_cost - v, 0)
end

function picard_iter(v1, v2, drift_bd1, drift_bd2, offset=0)
    return my_g(v1, drift_bd1, offset) - my_g(v2, drift_bd2, offset)
end

function mlp_v(t, T, z, gamma, sigma, level, M)
    output = zeros(dim + 1)
    # println(t, x, level)
    if level == 0
        return output
    end

    # number of simulated instances
    ns = M ^ level

    for i in 1:ns
        my_Srand = rand()
        my_taurand = rand()
        my_taunorm = rand(Normal())
        my_rbmrand = rand()
        my_rbmnorm = rand(Normal())
        my_S1 = my_rtime(beta, my_Srand, t, T)
        my_S2 = my_rtime(beta, 1-my_Srand, t, T)
        my_tau1 = bm_hitting(z, gamma, my_taurand, my_taunorm, sigma, t)
        my_tau2 = bm_hitting(z, gamma, 1-my_taurand, -my_taunorm, sigma, t)
        my_B1, my_R1 = stopped_rbm(z, gamma, my_tau1, my_S1, my_rbmrand, my_rbmnorm, sigma, t)
        my_B2, my_R2 = stopped_rbm(z, gamma, my_tau2, my_S2, 1-my_rbmrand, -my_rbmnorm, sigma, t)
        output += cost(my_R1) * [sqrt(my_S1 - t), my_B1 / sigma] + cost(my_R2) * [sqrt(my_S2 - t), my_B2 / sigma]
    end
    output = output / (2*ns) * my_C(beta, t, T)

    for l in 1:(level - 1)
        # number of simulated instances
        ns = M ^ (level - l)
        #offset = (l-1)/2

        temp = zeros(dim + 1)
        for i in 1:ns
            my_Srand = rand()
            my_taurand = rand()
            my_taunorm = rand(Normal())
            my_rbmrand = rand()
            my_rbmnorm = rand(Normal())
            my_S1 = my_rtime(beta, my_Srand, t, T)
            my_S2 = my_rtime(beta, 1-my_Srand, t, T)
            my_tau1 = bm_hitting(z, gamma, my_taurand, my_taunorm, sigma, t)
            my_tau2 = bm_hitting(z, gamma, 1-my_taurand, -my_taunorm, sigma, t)
            my_B1, my_R1 = stopped_rbm(z, gamma, my_tau1, my_S1, my_rbmrand, my_rbmnorm, sigma, t)
            my_B2, my_R2 = stopped_rbm(z, gamma, my_tau2, my_S2, 1-my_rbmrand, -my_rbmnorm, sigma, t)
            v11 = mlp_v(my_S1, T, my_R1, gamma, sigma, l, M)
            v21 = mlp_v(my_S1, T, my_R1, gamma, sigma, l - 1, M)
            v12 = mlp_v(my_S2, T, my_R2, gamma, sigma, l, M)
            v22 = mlp_v(my_S2, T, my_R2, gamma, sigma, l - 1, M)
            temp += picard_iter(v11[2], v21[2], drift_b, drift_b) * [sqrt(my_S1 - t), my_B1 / sigma] + picard_iter(v12[2], v22[2], drift_b, drift_b) * [sqrt(my_S2 - t), my_B2 / sigma]
        end
        temp = temp / (2*ns) * my_C(beta, t, T)
        output += temp
    end

    return output
end

alpha = (sqrt(gamma^2 + 2 * beta * (sigma^2)) + gamma) / (sigma^2)

function DV1(z)
    return h / beta * (1 - exp(-alpha * z))
end

V = h * z / beta + h * gamma / beta^2 + h * exp(-alpha * z) / (beta * alpha)

println("T =", ub, "; sigma =", sigma)
println(V, ", ", DV1(z))

test_array1 = []
test_array2 = []
for i in 1:10
    temp = mlp_v(0, ub, z, gamma, sigma, 7, 6)
    println(temp)
    push!(test_array1, temp[1])
    push!(test_array2, temp[2])
end

println(std(test_array1))
println(std(test_array2))