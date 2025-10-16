@time begin

    using Random, Distributions, Statistics, Distributed, LinearAlgebra, FastGaussQuadrature, QuadGK, SpecialFunctions
    include("rbm2.jl")
    include("parameters.jl")
    
    Random.seed!(1234)
    
    function cost(state)
        return h * state
    end
    
    function my_C(beta, lb, ub)
        return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
    end

    function my_C2(beta, lb, ub)
        return 1/beta*(1-exp(-beta*(ub-lb)))
    end

    function my_C3(lb, ub, myshape)
        return SpecialFunctions.gamma(myshape) - SpecialFunctions.gamma(myshape, ub-lb)
    end

    function my_g(v, drift_bd)
        return -drift_bd * (max(v - c, 0))
    end
    
    function picard_iter(v1, v2, drift_bd1, drift_bd2)
        return my_g(v1, drift_bd1) - my_g(v2, drift_bd2)
    end

    # N_nodes = 4

    # S, w = gauss(x -> exp(-x)/x^(0.17), N_nodes, 0, ub)

    # N_samples = 10000
    
    # darray = []
    # for _ in 1:50
    #     output = 0.0
    #     for i in 1:N_nodes
    #         for _ in 1:N_samples
    #             my_tau = bm_hitting(z, gamma, sigma, 0.0)
    #             my_B, my_R = stopped_rbm(z, gamma, my_tau, S[i]/beta, sigma, 0.0)
    #             output += cost(my_R)*my_B/S[i]^(0.83)*w[i]/N_samples
    #         end
    #     end
    #     push!(darray, output)
    #     #println(output)
    # end
    # println("Number of nodes = ", N_nodes, "; average = ", mean(darray), "; std = ", std(darray))




        
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
            #my_S = my_rtime2(1.0, 0.0, T-t)
            #my_S = my_rtime(1.0, 0.0, T-t)
            my_S = my_rtime3(0.0, T-t, myshape)
            my_tau = bm_hitting(z, gamma, sigma, 0.0)
            my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S/beta, sigma, 0.0)
            #output += (1-mmt)*(cost(my_R))*[sqrt(my_S - t), my_B / sigma]
            output += cost(my_R) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^myshape]
        end
        #output = output / (ns) * my_C2(1.0, 0, T-t) #+ mmt*[z, 1]
        #output = output / (ns) * my_C(1.0, 0, T-t)
        output = output / (ns) * my_C3(0, T-t, myshape)
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
    
            temp = zeros(dim + 1)
            for i in 1:ns
                #my_S = my_rtime2(1.0, 0.0, T-t)
                #my_S = my_rtime(1.0, 0.0, T-t)
                my_S = my_rtime3(0.0, T-t, myshape)
                my_tau = bm_hitting(z, gamma, sigma, 0.0)
                my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S/beta, sigma, 0.0)
                v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
                v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma]
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [sqrt(my_S - t), my_B / sigma]
                #try an Anderson scheme
                test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                if l==1
                    #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma]
                    #temp += my_C2(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0, mmt*my_B / sigma / my_S] + [0, (1.0-mmt)*test1[2]]
                    #temp += my_C(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*sqrt(my_S), mmt*my_B / sigma / sqrt(my_S)] + [0, (1.0-mmt)*test1[2]]
                    temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*test1[2]]
                    #temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], drift_b, drift_b) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*test1[2]]
                else
                    #test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                    test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
                    #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
                    #temp += my_C2(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0, mmt*my_B / sigma / my_S] + [0, (1.0-mmt)*(test1[2]-test2[2])]
                    #temp += my_C(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*sqrt(my_S), mmt*my_B / sigma / sqrt(my_S)] + [0, (1.0-mmt)*(test1[2]-test2[2])]
                    temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*(test1[2]-test2[2])]
                    #temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], drift_b, drift_b) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*(test1[2]-test2[2])]
                end
            end
            temp = temp / (ns) #* my_C(beta, t, T)
            output += temp
        end
    
        return output
    end

    function mlp_mlt_call(t, T, z, gamma, sigma, level, M, thread_id, NUM_THREADS)
        output2 = zeros(dim+1)
        # number of simulated instances
        ns = M ^ level
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
    
        for _ in 1:loop_num
            #my_S = my_rtime2(1.0, 0.0, T-t)
            #my_S = my_rtime(1.0, 0.0, T-t)
            my_S = my_rtime3(0.0, T-t, myshape)
            my_tau = bm_hitting(z, gamma, sigma, 0.0)
            my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S/beta, sigma, 0.0)
            output2 += (cost(my_R))*[1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)]
            #output += cost(my_R) * [sqrt(my_S - t), my_B / sigma]
        end
        #output2 = output2 / (ns) * my_C2(1.0, 0.0, T-t) #+ mmt*[z, 1]
        #output2 = output2 / (ns) * my_C(1.0, 0.0, T-t)
        output2 = output2 / (ns) * my_C3(0.0, T-t, myshape)

        for l in 1:(level - 1)
            # number of simulated instances
            ns2 = M ^ (level - l)
            loop_num = _get_loop_num(ns2, thread_id, NUM_THREADS)
            temp = zeros(dim + 1)
            for _ in 1:loop_num
                #my_S = my_rtime2(1.0, 0.0, T-t)
                #my_S = my_rtime(1.0, 0.0, T-t)
                my_S = my_rtime3(0.0, T-t, myshape)
                my_tau = bm_hitting(z, gamma, sigma, 0.0)
                my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S/beta, sigma, 0.0)
                v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
                v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma]
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1,drift_b)) * [sqrt(my_S - t), my_B / sigma]
                #try an Anderson scheme
                test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                if l==1
                    #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma]
                    #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l*(0.5), drift_b), min((l-1)*(0.5), drift_b)) * [sqrt(my_S - t), mmt * my_B / sigma] + [0, (1.0-mmt)*test1[2]]
                    #temp += my_C2(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0, mmt * my_B / sigma / my_S] + [0, (1.0-mmt)*test1[2]]
                    #temp += my_C(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*sqrt(my_S), mmt * my_B / sigma / sqrt(my_S)] + [0, (1.0-mmt)*test1[2]]
                    temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*(my_S)^(1.0 - myshape)/beta, mmt * my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*test1[2]]
                    #temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], drift_b, drift_b) * [1.0*(my_S)^(1.0 - myshape)/beta, mmt * my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt)*test1[2]]
                else
                    #test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                    test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
                    #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.4), drift_b), min((l-1)^(0.4), drift_b)) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
                    #temp += my_C2(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0, mmt*my_B / sigma / my_S] + [0, (1.0-mmt) * (test1[2]-test2[2])]
                    #temp += my_C(1.0, 0.0, T-t)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*sqrt(my_S), mmt*my_B / sigma / sqrt(my_S)] + [0, (1.0-mmt) * (test1[2]-test2[2])]
                    temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], min(l, drift_b), min(l-1, drift_b)) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt) * (test1[2]-test2[2])]
                    #temp += my_C3(0.0, T-t, myshape)*picard_iter(v1[2], v2[2], drift_b, drift_b) * [1.0*(my_S)^(1.0-myshape)/beta, mmt*my_B / sigma / (my_S)^(myshape)] + [0, (1.0-mmt) * (test1[2]-test2[2])]
                end
            end
            temp = temp / (ns2) #* my_C(beta, t, T)
            output2 += temp
        end

        return output2
    end

    #decides how many iteration given thread id and num
    function _get_loop_num(num, thread_id, NUM_THREADS) 
        if num < NUM_THREADS
            # each thread only goes once through the loop
            loop_num = thread_id > num ? 0 : 1
        else
            remainder =  num % NUM_THREADS
            if (remainder > 0) && (thread_id <= remainder) 
                # each thread goes num / NUM_THREADS + the remainder 
                loop_num = num / NUM_THREADS + 1
            else
                loop_num = num / NUM_THREADS
            end
        end
        return loop_num
    end

    function mlp_mlt_test(t, T, z, gamma, sigma, level, M)
        output = zeros(dim+1)
        NUM_THREADS = 5
        for thread_id in 1:NUM_THREADS
            output+=mlp_mlt_call(t, T, z, gamma, sigma, level, M, thread_id, NUM_THREADS)
        end
        return output
    end



    function mlp_mlt(t, T, z, gamma, sigma, level, M)
        NUM_THREADS = Threads.nthreads()
        tasks = [Threads.@spawn(mlp_mlt_call(t, T, z, gamma, sigma, level, M, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
        a = sum([fetch(task) for task in tasks])
        return a
    end
    
    
    println("T = ", ub, "; drift upper bound = ", drift_b, "; momentum = ", mmt, "; discount factor = ", beta, "; drift = ", gamma, "; sigma = ", sigma) #"; level = ", level, "; M = ", M) #, "; M2 = ", M2)
    # # #println(V, ", ", DV1(z))
    
    for x in 0.6
        for level in 1:4
            test_array1 = []
            test_array2 = []
            println("initial state = ", x, "; level = ", level, "; M = ", M[level])
            for i in 1:25
                #temp = mlp_mlt_test(0, ub, z, gamma, sigma, level, M)
                temp = mlp_mlt(0, ub, x, gamma, sigma, level, M[level])
                #temp = mlp_v(0, ub, z, gamma, sigma, level, M)
                #println(temp)
                push!(test_array1, temp[1])
                push!(test_array2, temp[2])
                if i%5==0
                    println("average V = ", mean(test_array1), "; std = ", std(test_array1))
                    println("average DV = ", mean(test_array2), "; std = ", std(test_array2))
                end         
            end
        end
    end
end