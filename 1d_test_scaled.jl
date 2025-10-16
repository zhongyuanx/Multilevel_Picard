@time begin

    using Random, Distributions, Statistics, Distributed
    include("rbm2.jl")
    include("parameters.jl")

    scaled_beta = beta*ub
    
    Random.seed!(1234)

    function cost(state)
        return h * state
    end
    
    function my_C(beta, lb, ub)
        return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
    end
    
    function my_g(v, drift_bd, cost)
        return drift_bd * min(cost - v, 0)
    end
    
    function picard_iter(v1, v2, drift_bd1, drift_bd2, cost)
        return my_g(v1, drift_bd1, cost) - my_g(v2, drift_bd2, cost)
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
            my_S = my_rtime(scaled_beta, t, T)
            my_tau = bm_hitting(z, ub*gamma, sqrt(ub)*sigma, t)
            my_B, my_R = stopped_rbm(z, ub*gamma, my_tau, my_S, sqrt(ub)*sigma, t)
            output += (1-mmt)*(cost(my_R))*[sqrt(my_S - t), my_B /(sqrt(ub)*sigma)]
            #output += cost(my_R) * [sqrt(my_S - t), my_B / sigma]
        end
        output = output / (ns) * my_C(scaled_beta, t, T) #+ mmt*[z, 1]
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
            #m1 = mmtm(l, init_mmt, delta) #momentum for level l
            #m2 = mmtm(l-1, init_mmt, delta) #momentum for level l-1
    
            temp = zeros(dim + 1)
            for i in 1:ns
                my_S = my_rtime(scaled_beta, t, T)
                my_tau = bm_hitting(z, ub*gamma, sqrt(ub)*sigma, t)
                my_B, my_R = stopped_rbm(z, ub*gamma, my_tau, my_S, sqrt(ub)*sigma, t)
                v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
                v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
                #println("level = ", l, "; S = ", my_S, "; v1 = ", v1[2], "; v2 = ", v2[2], "; g(v1)-g(v2) = ", picard_iter(v1[2], v2[2], drift_b, drift_b, c), "\n")
                #try an Anderson scheme
                #test1 = mlp_v(t, T, z, ub*gamma, sqrt(ub)*sigma, l, M)
                #test2 = mlp_v(t, T, z, ub*gamma, sqrt(ub)*sigma, l-1, M)
                #temp += (1-m1)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] + m1 * (test1-test2) + (m2-m1)*(my_C(beta, t, T)*(cost(my_R)+my_g(v2[2], l-1, offset))*[sqrt(my_S - t), my_B / sigma] - test2)
                temp += (1-mmt)*my_C(scaled_beta, t, T)*picard_iter(v1[2], v2[2], ub*drift_b, ub*drift_b, c/ub) * [sqrt(my_S - t), my_B / (sqrt(ub)*sigma)] #+ mmt * (test1-test2)
            end
            temp = temp / (ns) #* my_C(beta, t, T)
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
    
    println("T = ", ub, "; drift upper bound = ", drift_b, "; momentum = ", mmt, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma, "; level = ", level, "; M = ", M) #, "; M2 = ", M2)
    # # #println(V, ", ", DV1(z))
    
    test_array1 = []
    test_array2 = []
    for i in 1:100
        temp = mlp_v(0, 1, z, gamma, sigma, level, M)
        println(temp)
        push!(test_array1, temp[1])
        push!(test_array2, temp[2])
    end
    
    println("average V = ", mean(test_array1)/(1-mmt), "; std = ", std(test_array1)/(1-mmt))
    println("average DV = ", mean(test_array2)/(1-mmt), "; std = ", std(test_array2)/(1-mmt))
    
    end