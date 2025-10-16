@time begin

    using Random, Distributions, Statistics, Distributed
    include("rbm2.jl")
    include("parameters6.jl")
    
    Random.seed!(1234)

    println("Threads = ", Threads.nthreads())
    
    function cost(state)
        return h * state
    end
    
    function my_C(beta, lb, ub)
        return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
    end
    
    function my_g(v, drift_bd)
        # if v<=c
        #     return 0
        # elseif v<=c+1/drift_bd
        #     return -(v-c)^2*0.5*drift_bd^2
        # else
        #     return -(v-c)*drift_bd+0.5
        # end
        return offset * v + drift_bd * min(c - v, 0)
    end
    
    function picard_iter(v1, v2, drift_bd1, drift_bd2)
        return my_g(v1, drift_bd1) - my_g(v2, drift_bd2)
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
            output += (cost(my_R))*[sqrt(my_S - t), mmt*my_B / sigma]
            #output += cost(my_R) * [sqrt(my_S - t), my_B / sigma]
        end
        output = output / (ns) * my_C(beta, t, T) #+ mmt*[z, 1]
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
    
            temp = zeros(dim + 1)
            # if l<2
            #     my_mmt = mmt
            #     rate = 1.0
            # elseif l==2
            #     my_mmt = mmt-0.1
            #     rate = my_mmt/mmt
            # else
            #     my_mmt = mmt-0.1
            #     rate = 1.0
            # end
            for i in 1:ns
                my_S = my_rtime(beta, t, T)
                my_tau = bm_hitting(z, gamma, sigma, t)
                my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
                v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
                v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
                #try an Anderson scheme
                test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
                #temp += (1-m1)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] + m1 * (test1-test2) + (m2-m1)*(my_C(beta, t, T)*(cost(my_R)+my_g(v2[2], l-1, offset))*[sqrt(my_S - t), my_B / sigma] - test2)
                #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.5), drift_b), min((l-1)^(0.5), drift_b)) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
                temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((l)^(0.5), drift_b), min((l-1)^(0.5), drift_b)) * [sqrt(my_S - t), mmt*my_B / sigma] + [0, (rate-mmt) * (test1[2]-test2[2])]
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((l)^(0.7), drift_b), min((l-1)^(0.7), drift_b)) * [sqrt(my_S - t), 1/(l+1)*my_B / sigma] + [0, (l-1)/(l+1)* (test1[2]-test2[2])]
            end
            temp = temp / (ns) #* my_C(beta, t, T)
            output += temp
            #output = temp
        end
    
        return output
    end

    function mlp_mlt_incr_call(t, T, z, gamma, sigma, level, M, thread_id, NUM_THREADS)
        output3 = zeros(dim+1)
        loop_num = _get_loop_num(M, thread_id, NUM_THREADS)
        for _ in 1:loop_num
            my_S = my_rtime(beta, t, T)
            my_tau = bm_hitting(z, gamma, sigma, t)
            my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
            v1 = mlp_v(my_S, T, my_R, gamma, sigma, level-1, M)
            v2 = mlp_v(my_S, T, my_R, gamma, sigma, level-2, M)
            #test1 = mlp_v(t, T, z, gamma, sigma, level-1, M)
            #test2 = mlp_v(t, T, z, gamma, sigma, level-2, M)
            output3 += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(level-1, drift_b), min(level-2, drift_b)) * [sqrt(my_S - t), (1-mmt)*my_B / sigma] #+ mmt * (test1-test2)
        end
        
        output3 = output3 / M 
    end            

    function mlp_mlt_call(t, T, z, gamma, sigma, level, M, thread_id, NUM_THREADS)
        output2 = zeros(dim+1)
        # number of simulated instances
        ns = M ^ level
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
    
        for _ in 1:loop_num
            my_S = my_rtime(beta, t, T)
            my_tau = bm_hitting(z, gamma, sigma, t)
            my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
            output2 += (cost(my_R))*[sqrt(my_S - t), mmt*my_B / sigma]
            #output2 += cost(my_R) * [sqrt(my_S - t), my_B / sigma]
        end
        output2 = output2 / (ns) * my_C(beta, t, T) #+ mmt*[z, 1]

        for l in 1:(level - 1)
            # number of simulated instances
            ns2 = M ^ (level - l)
            loop_num = _get_loop_num(ns2, thread_id, NUM_THREADS)
            temp = zeros(dim + 1)
            # if l<2
            #     my_mmt = mmt
            #     rate = 1.0
            # elseif l==2
            #     my_mmt = mmt/2.0
            #     rate = 0.5
            # else
            #     my_mmt = mmt/2.0
            #     rate = 1.0
            # end
            for _ in 1:loop_num
                my_S = my_rtime(beta, t, T)
                my_tau = bm_hitting(z, gamma, sigma, t)
                my_B, my_R = stopped_rbm(z, gamma, my_tau, my_S, sigma, t)
                v1 = mlp_v(my_S, T, my_R, gamma, sigma, l, M)
                v2 = mlp_v(my_S, T, my_R, gamma, sigma, l - 1, M)
                #try an Anderson scheme
                test1 = mlp_v(t, T, z, gamma, sigma, l, M)
                test2 = mlp_v(t, T, z, gamma, sigma, l-1, M)
                #temp += (1-m1)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], drift_b, drift_b, offset) * [sqrt(my_S - t), my_B / sigma] + m1 * (test1-test2) + (m2-m1)*(my_C(beta, t, T)*(cost(my_R)+my_g(v2[2], l-1, offset))*[sqrt(my_S - t), my_B / sigma] - test2)
                #temp += (1-mmt)*my_C(beta, t, T)*picard_iter(v1[2], v2[2], min(l^(0.5), drift_b), min((l-1)^(0.5), drift_b)) * [sqrt(my_S - t), my_B / sigma] + mmt * (test1-test2)
                temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((l)^(0.5), drift_b), min((l-1)^(0.5), drift_b)) * [sqrt(my_S - t), mmt*my_B / sigma] + [0, (rate-mmt) * (test1[2]-test2[2])]
                #temp += my_C(beta, t, T)*picard_iter(v1[2], v2[2], min((l)^(0.7), drift_b), min((l-1)^(0.7), drift_b)) * [sqrt(my_S - t), 1/(l+1)*my_B / sigma] + [0, (l-1)/(l+1)* (test1[2]-test2[2])]
            end
            temp = temp / (ns2) #* my_C(beta, t, T)
            output2 += temp
            #output2 = temp
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
    
    
    println("T = ", ub, "; drift upper bound = ", drift_b, "; momentum = ", mmt, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma, "; v1")  #, "; M2 = ", M2)
    # # #println(V, ", ", DV1(z))

    for level in 7:7
        test_array1 = []
        test_array2 = []
        println("level = ", level, "; M = ", M[level])
        for i in 1:25
            #temp = mlp_mlt_test(0, ub, z, gamma, sigma, level, M)
            temp = mlp_mlt(0, ub, z, gamma, sigma, level, M[level])
            #temp = mlp_v(0, ub, z, gamma, sigma, level, M)
            println(temp)
            push!(test_array1, temp[1])
            push!(test_array2, temp[2])
            # if i%10==0
            #     println("average V = ", mean(test_array1), "; std = ", std(test_array1))
            #     println("average DV = ", mean(test_array2), "; std = ", std(test_array2))
            # end
        end
        println("average V = ", mean(test_array1), "; std = ", std(test_array1))
        println("average DV = ", mean(test_array2), "; std = ", std(test_array2))         
    end
end