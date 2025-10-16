using Random, Distributions, Statistics, Distributed

include("rbm2.jl")
include("parameters.jl")

NUM_SIM = 1000000

function my_C(beta, lb, ub)
    return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
end

function rtime_sim(t, T, beta, n)
    my_S = zeros(n+1)
    my_S[1]=t
    for i in 2:(n+1)
        my_S[i] = my_rtime(beta, my_S[i-1], T)
    end
    return my_S
end

function nconst_sim(t, T, beta, n)
    my_S = rtime_sim(t, T, beta, n)
    prod = ones(n+1)
    for i in 1:n
        prod[i+1]=(my_C(beta, my_S[i], T))*min(1, 1/sqrt(my_S[i+1]))*prod[i]
    end
    return prod
end

function nconst_est(t, T, beta, n, NUM_SIM)
    a = zeros(n+1)
    for i in 1:NUM_SIM
        a += nconst_sim(t, T, beta, n)/NUM_SIM
    end
    #println(a)
    return a[2:(n+1)]
end

function binomial_sum(t, T, mmt, beta, n, NUM_SIM)
    a = nconst_est(t, T, beta, n, NUM_SIM)
    my_sum = 0.0
    for i in 1:n
        my_sum += binomial(n, i)*mmt^i*(1-mmt)^(n-i)*a[i]
    end
    return my_sum
end

#println(nconst_est(0, ub, beta, 20, NUM_SIM))

for i in 20:40
    println(binomial_sum(0, ub, mmt, beta, i, NUM_SIM))
end
