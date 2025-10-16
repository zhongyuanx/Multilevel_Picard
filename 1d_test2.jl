@time begin

using Random, Distributions, Statistics, Distributed
include("rbm2.jl")
include("parameters.jl")

Random.seed!(1234)

# beta = 1.0
# epsilon = 0.01
# ub = -log(epsilon * beta) / beta
# # ub = 1
# offset = 0
# h = 2.0
# push_cost = 1.0
# z = 2.0
# gamma = -1 - offset
# dim = 1
# drift_b = 2.0
# sigma = 1.0

# level = 6
# M = 6

# mmt = 0.6
# delta = 0.0

function cost(state)
    return h * state
end

function my_C(beta, lb, ub)
    return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
end

function my_g(v, drift_bd, offset)
    #return offset * v + drift_b * min(push_cost - v, 0) # if v > push_cost else 0
    return offset * v + drift_bd * min(c - v, 0)
end

function picard_iter(v1, v2, drift_bd1, drift_bd2, offset=0)
    return my_g(v1, drift_bd1, offset) - my_g(v2, drift_bd2, offset)
end

function mmtm(l, initial, delta)
    # lower momentum at larger levels. initial = initial momentum; terminal = terminal momentum
    return initial+delta*l
end

function mlp_v(t, T, z, gamma, sigma, level, M)
    output = zeros(dim + 1)
    # println(t, x, level)
    if level == 0
        return output
    end

    # number of simulated instances
    ns = M ^ level
    #mmt = mmtm(0, init_mmt, delta)

    for i in 1:ns
        my_S = my_rtime(beta, t, T)
        my_tau = bm_hitting(z, gamma, sigma, t)
        my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
        output += (1-mmt)*(cost(my_R))*[sqrt(my_S - t), my_B / sigma]
        #output += cost(my_R) * [sqrt(my_S - t), my_B / sigma]
    end
    output = output / (ns) * my_C(beta, t, T) #+ mmt*[z, 1]

    for l in 1:(level - 1)
        # number of simulated instances
        ns = 2 * M ^ (level - l)

        temp = zeros(dim + 1)
        #alpha = my_C(beta, t, T)
        alpha = 2
        threshold = floor(Int, (alpha*(1-mmt))^2*ns/((alpha*(1-mmt))^2 + mmt^2))
        for _ in 1:threshold
            my_S = my_rtime(beta, t, T)
            my_tau = bm_hitting(z, gamma, sigma, t)
            my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
            v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
            v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
            temp += (1-mmt)/threshold*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b), offset) * [sqrt(my_S - t), my_B / sigma]
        end
        
        for _ in 1:(ns-threshold)
            test1 = mlp_v(t, T, z, gamma, sigma, l, M)
            test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
            temp += mmt/(ns-threshold)*(test1-test2)
        end

        # for i in 1:ns
        #     my_S = my_rtime(beta, t, T)
        #     my_tau = bm_hitting(z, gamma, sigma, t)
        #     my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
        #     v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
        #     v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
        #     #try an Anderson scheme
        #     test1 = mlp_v(t, T, z, gamma, sigma, l, M)
        #     test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
        #     #temp += (1-m1)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] + m1 * (test1-test2) + (m2-m1)*(my_C(beta, t, T)*(cost(my_R)+my_g(v2[2], l-1, offset))*[sqrt(my_S - t), my_B / sigma] - test2)
        #     temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b), offset) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
        #     #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] #+ mmt * (test1-test2)
        #     # if l==1
        #     #     temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma]
        #     # else
        #     #     test1 = mlp_v(t, T, z, gamma, sigma, l, M)
        #     #     test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
        #     #     temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
        #     # end
        # end
        # temp = temp / (ns) #* my_C(beta, t, T)
        output += temp
    end

    return output
end


function mlp_dist(t, T, z, gamma, sigma, level, M)
    temp = mlp_v(t, T, z, gamma, sigma, level-1, M)/M
    my_S = my_rtime(beta, t, T)
    my_tau = bm_hitting(z, gamma, sigma, t)
    my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
    v1 = mlp_v(my_S, T, my_R, gamma, sigma, level-1, M)
    v2 = mlp_v(my_S, T, my_R, gamma, sigma, level-2, M)
    test1 = mlp_v(t, T, z, gamma, sigma, level-1, M)
    test2 = mlp_v(t, T, z, gamma, sigma, level-2, M)
    temp += (1-mmt)/M*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((level-1)^(0.4), drift_b), min((level-2)^(0.4), drift_b), offset) * [sqrt(my_S - t), my_B / sigma] + mmt/M * (test1-test2)
    return temp
end


#for j in 1:10
    #result = zeros(dim+1)
    #println(mlp_v(0, ub, z, gamma, sigma, level, M))
    #for i in 1:M
        #result += mlp_dist(0, ub, z, gamma, sigma, level, M)
    #end
    #println(result)
#end



# for i in 1:1
#     result = @distributed (+) for j=1:M
#         mlp_dist(0, ub, z, gamma, sigma, level, M)
#     end
#     println(result, "\n")
# end


# function my_mlp_v(t, T, z, gamma, sigma, level, M1, M2)
#     ns = M2
#     temp = zeros(dim+1)
#     for i in 1:ns
#         my_S = my_rtime(beta, t, T)
#         my_tau = bm_hitting(z, gamma, sigma, t)
#         my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
#         v1 = mlp_v(my_S, T, my_R, gamma, sigma, level-1, M2)
#         v2 = mlp_v(my_S, T, my_R, gamma, sigma, level-2, M2)
#         test1 = mlp_v(t, T, z, gamma, sigma, level-1, M2)
#         test2 = mlp_v(t, T, z, gamma, sigma, level-2, M2)
#         temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((level-1)^(0.5), drift_b), min((level-2)^(0.5), drift_b), offset) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
#     end
#     temp = temp/ns

#     temp2 = zeros(dim+1)
#     for i in 1:M1
#         temp2+=mlp_v(t, T, z, gamma, sigma, level-1, M1)/M1
#     end
#     print(temp2, ", ", temp)

#     return temp+temp2
# end



# alpha = (sqrt(gamma^2 + 2 * beta * (sigma^2)) + gamma) / (sigma^2)

# function DV1(z)
#     return h / beta * (1 - exp(-alpha * z))
# end

# V = h * z / beta + h * gamma / beta^2 + h * exp(-alpha * z) / (beta * alpha)

# #print(gamma)

println("T = ", ub, "; drift upper bound = ", drift_b, "; momentum = ", mmt, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma, "; level = ", level, "; M = ", M) #, "; M2 = ", M2)
# # #println(V, ", ", DV1(z))

test_array1 = []
test_array2 = []
for i in 1:20
    temp = mlp_v(0, ub, z, gamma, sigma, level, M)
    println(temp)
    push!(test_array1, temp[1])
    push!(test_array2, temp[2])
end

println("average V = ", mean(test_array1), "; std = ", std(test_array1))
println("average DV = ", mean(test_array2), "; std = ", std(test_array2))

end