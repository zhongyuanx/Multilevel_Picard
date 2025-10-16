@time begin

    using Random, Distributions, Statistics, Distributed, LinearAlgebra
    include("rbm2.jl")
    include("PSS_Ex.jl")
    #include("parameters.jl")
    
    Random.seed!(1234)

    #specify holding cost rate h
    #specify control matrix G
    
    function cost(state::Vector{Float64})
        #holding cost
        return dot(h, state)
    end
    
    function my_C(beta, lb, ub)
        #normalizing constant
        return sqrt(pi / beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * beta * (ub - lb))))
    end
    
    function my_ctrl(v::Vector{Float64}, rG::Array{Float64}, drift_bd::Float64)
        #This is the hamiltonian
        Gtv = transpose(rG) * v[2:end]
        u_dim = length(Gtv)
        return drift_bd * sum(min(Gtv[i], 0.0) for i in 1:u_dim)
    end
    
    function picard_iter(v1, v2, rG, drift_bd1, drift_bd2)
        return my_ctrl(v1, rG, drift_bd1) - my_ctrl(v2, rG, drift_bd2)
    end
        
    function pss_mlp(t, T, z, gamma, sigma, G, beta, level, M)
        dim = length(z)
        output = zeros(dim + 1)
        # println(t, x, level)
        if level <= 0
            return output
        end
    
        # number of simulated instances
        ns = M ^ level
        #mmt = mmtm(0, init_mmt, delta)
    
        for _ in 1:ns
            my_tau = zeros(dim)
            my_B = zeros(dim)
            my_R = zeros(dim)
            my_S = my_rtime(beta, t, T)
            for i in 1:dim
                my_tau[i] = bm_hitting(z[i], gamma[i], sigma[i], t)
                my_B[i], my_R[i] = stopped_rbm(z[i], gamma[i], my_tau[i], my_S, sigma[i], t)
            end
            output += cost(my_R) * vcat([sqrt(my_S - t)], my_B ./ sigma)
        end
        output = output / (ns) * my_C(beta, t, T) #+ mmt*[z, 1]
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
    
            temp = zeros(dim + 1)
            for i in 1:ns
                my_S = my_rtime(beta, t, T)
                my_tau = zeros(dim)
                my_B = zeros(dim)
                my_R = zeros(dim)
                for i in 1:dim
                    my_tau[i] = bm_hitting(z[i], gamma[i], sigma[i], t)
                    my_B[i], my_R[i] = stopped_rbm(z[i], gamma[i], my_tau[i], my_S, sigma[i], t)
                end
                v1 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, l, M)
                v2 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, l - 1, M)
                temp += my_C(beta, t, T)*picard_iter(v1, v2, G, min(drift_b*l/4,drift_b), min(drift_b*(l-1)/4,drift_b)) * vcat([sqrt(my_S - t)], my_B ./ sigma)
                #temp += my_C(beta, t, T)*picard_iter(v1, v2, G, drift_b, drift_b) * vcat([sqrt(my_S - t)], my_B ./ sigma)
            end
            temp = temp / (ns) #* my_C(beta, t, T)
            output += temp
        end
    
        return output
    end

    function pss_mlp_test(t, T, z, gamma, sigma, G, beta, level, M, b)
        # This is an alternative implementation of pss_mlp if M = b.
        # b is number of simulated copies allocated to the function. 
        # Basically, this uses the idea that v(n) = F(v(n-1)) = v(n-1) + F(v(n-1))-F(v(n-2))
        # Use average of M samples of to estimate v(n-1, M). Use average of M samples to estimate F(v(n-1,M))-F(v(n-2,M)).
        # Corner case is when level==1, in which case we just return the value of pss_mlp.
        dim = length(z)
        output_test = zeros(dim+1)
        if level == 1
            return b/M*pss_mlp(t, T, z, gamma, sigma, G, beta, level, M)
        end

        for i in 1:b
            v0 = pss_mlp(t, T, z, gamma, sigma, G, beta, level-1, M)
            output_test+= v0/M

            my_S = my_rtime(beta, t, T)
            my_tau = zeros(dim)
            my_B = zeros(dim)
            my_R = zeros(dim)
            for j in 1:dim
                my_tau[j] = bm_hitting(z[j], gamma[j], sigma[j], t)
                my_B[j], my_R[j] = stopped_rbm(z[j], gamma[j], my_tau[j], my_S, sigma[j], t)
            end
            v1 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, level-1, M)
            v2 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, level-2, M)
            output_test += my_C(beta, t, T)/M*picard_iter(v1, v2, G, min(drift_b*(level-1)/4,drift_b), min(drift_b*(level-2)/4,drift_b)) * vcat([sqrt(my_S - t)], my_B ./ sigma)
        end

        return output_test
    end


    function pss_mlp_mlt_call1(t, T, z, gamma, sigma, G, beta, level, M, thread_id, NUM_THREADS)
        dim = length(z)
        output2 = zeros(dim+1)
        # number of simulated instances
        ns = M ^ level
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
    
        for _ in 1:loop_num
            my_S = my_rtime(beta, t, T)
            my_tau = zeros(dim)
            my_B = zeros(dim)
            my_R = zeros(dim)
            for i in 1:dim
                my_tau[i] = bm_hitting(z[i], gamma[i], sigma[i], t)
                my_B[i], my_R[i] = stopped_rbm(z[i], gamma[i], my_tau[i], my_S, sigma[i], t)
            end
            output2 += (cost(my_R))*vcat([sqrt(my_S - t)], my_B ./ sigma)
        end
        output2 = output2 / (ns) * my_C(beta, t, T)

        for l in 1:(level - 2)
            # number of simulated instances
            ns2 = M^(level-l)
            loop_num = _get_loop_num(ns2, thread_id, NUM_THREADS)
            temp = zeros(dim + 1)
            for _ in 1:loop_num
                my_S = my_rtime(beta, t, T)
                my_tau = zeros(dim)
                my_B = zeros(dim)
                my_R = zeros(dim)
                for i in 1:dim
                    my_tau[i] = bm_hitting(z[i], gamma[i], sigma[i], t)
                    my_B[i], my_R[i] = stopped_rbm(z[i], gamma[i], my_tau[i], my_S, sigma[i], t)
                end
                v1 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, l, M)
                v2 = pss_mlp(my_S, T, my_R, gamma, sigma, G, beta, l - 1, M)
                temp += my_C(beta, t, T)*picard_iter(v1, v2, G, min(drift_b*l/4, drift_b), min(drift_b*(l-1)/4, drift_b)) * vcat([sqrt(my_S - t)], my_B ./ sigma)
            end
            temp = temp / (ns2) #* my_C(beta, t, T)
            output2 += temp
        end

        #now for l=level-1. here we want to break down the iterations. 
        #PLACEHOLDER FOR THINGS TO FILL IN FOR l=level-1

        return output2
    end

    function pss_mlp_mlt_call2(SM, T, zM, gamma, sigma, G, beta, level, M, thread_id, NUM_THREADS)
        # This function takes care of the parallel computation at level l=level-1
        # This only works if M^2 is divisible by NUM_THREADS and NUM_THREADS is divisible by M
        # Say k1 = NUM_THREADS/M. The first k1 threads will compute the first iteration, the next k1 threads the next iteration, and so on.
        # Say k2 = M^2/NUM_THREADS. Each thread will compute k2 iterations.
        
        # We make use of pss_mlp_test for this multithreaded approach. 
        # We need to compute v(level-1, M), v(level-2, M), and then get F(v(level-1))-F(v(level-2)) later in the main function pss_mlp_mlt.
        # Important observation: computing v(l,M) is simply averaging over M samples of v(l-1,M) and F(v(l-1, M))-F(v(l-2, M)), for l = level-1, level-2.
        # This follows from pss_mlp_test.
        # We use NUM_THREADS/M threads to compute v(level-1,M) and v(level-2,M). 
        # As a result, each thread computes M/(NUM_THREADS/M) = M^2/NUM_THREADS samples for v(level-1, M) and same for v(level-2, M).
        result = zeros(dim+1, 2)
        k = div(NUM_THREADS, M) # number of threads per iteration
        ind = div(thread_id-1, k) + 1 # which iteration to compute: sM[ind], zM[:, ind]
        b = div(M^2, NUM_THREADS) # how many simulated copies to compute per thread
        result[:, 1] = pss_mlp_test(SM[ind], T, zM[:, ind], gamma, sigma, G, beta, level-1, M, b)
        result[:, 2] = pss_mlp_test(SM[ind], T, zM[:, ind], gamma, sigma, G, beta, level-2, M, b)
        return result
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


    function pss_mlp_mlt(t, T, z, gamma, sigma, G, beta, level, M)
        NUM_THREADS = Threads.nthreads()
        tasks = [Threads.@spawn(pss_mlp_mlt_call1(t, T, z, gamma, sigma, G, beta, level, M, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
        a = sum([fetch(task) for task in tasks])
        # so far, this takes care of levels 1 to level-2.
        # Now we need to take care of level-1, which is a bit more complicated.

        #first generate the random times sM, the initial states zM, stopping times tauM, and the stopped Brownian motion BM
        dim = length(z)
        SM = [my_rtime(beta, t, T) for _ in 1:M] # random times
        tauM = zeros(dim, M) # hitting times
        for i in 1:dim
            tauM[i,:] = [bm_hitting(z[i], gamma[i], sigma[i], t) for _ in 1:M] # hitting times
        end

        zM = zeros(dim, M)
        BM = zeros(dim, M)
        for i in 1:dim
            for j in 1:M
                BM[i,j], zM[i,j] = stopped_rbm(z[i], gamma[i], tauM[i,j], SM[j], sigma[i], t)
            end
        end

        tasks2 = Vector{Task}(undef, NUM_THREADS)
        results2 = fetch.([Threads.@spawn(pss_mlp_mlt_call2(SM, T, zM, gamma, sigma, G, beta, level, M, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]) # to store results for v(level-1, M) and v(level-2, M)


        #for thread_id in 1:NUM_THREADS
            # each thread computes M/(NUM_THREADS/M) = M^2/NUM_THREADS samples for v(level-1, M) and same for v(level-2, M).
        #    results2[:,(2*thread_id-1):(2*thread_id)] = fetch(Threads.@spawn(pss_mlp_mlt_call2(SM, T, zM, gamma, sigma, G, beta, level, M, thread_id, NUM_THREADS)))
            #temp = fetch(tasks2[thread_id])
            #ind = div(thread_id-1, k) + 1 # which sample the thread computes
            #results2[:,(2*ind-1):(2*ind)] += temp # store results for v(level-1, M) and v(level-2, M)
        #end
        # Now we have the results for v(level-1, M) and v(level-2, M) in results2.

        # Now we need to compute F(v(level-1, M))-F(v(level-2, M)) and add it to a.
        results = zeros(dim+1,2*M)
        k = div(NUM_THREADS, M) # number of threads per iteration
        for i in 1:NUM_THREADS
            ind = div(i-1, k) + 1
            results[:, (2*ind-1)] += results2[i][:,1] # v(level-1, M)
            results[:, (2*ind)] += results2[i][:,2]   # v(level-2, M)
        end

        for i in 1:M
            v1 = results[:, 2*i-1]
            v2 = results[:, 2*i]
            a += my_C(beta, t, T)/M * picard_iter(v1, v2, G, min(drift_b*(level-1)/4, drift_b), min(drift_b*(level-2)/4, drift_b)) * vcat([sqrt(SM[i] - t)], BM[:, i] ./ sigma)
        end
        return a
    end
    
    
    println("T = ", ub, "; drift upper bound = ", drift_b, "; holding cost = ", h, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma) #"; level = ", level, "; M = ", M) #, "; M2 = ", M2)
    # # #println(V, ", ", DV1(z))

    for j in 1:10
        println(pss_mlp_mlt(0, ub, z, gamma, sigma, G, beta, 6, 4))
        #println(pss_mlp_test(0, ub, z, gamma, sigma, G, beta, 1, 40000))
    end

    #=
    for level in 3:3
        local test_array1 = []
        local test_array2 = []
        local test_array3 = []
        #test_array4 = []
        println("level = ", level, "; M = ", M[level])
        for i in 1:25
            temp = pss_mlp_mlt(0, ub, z, gamma, sigma, G, beta, level, M[level])
            #println(temp)
            push!(test_array1, temp[1])
            push!(test_array2, temp[2])
            push!(test_array3, temp[3])
            #push!(test_array4, temp[4])
            if i%5==0
                println("average V = ", mean(test_array1), "; std = ", std(test_array1))
                println("average DV1 = ", mean(test_array2), "; std = ", std(test_array2))
                println("average DV2 = ", mean(test_array3), "; std = ", std(test_array3))
                #println("average DV3 = ", mean(test_array4), "; std = ", std(test_array4))
            end         
        end
    end
    =#

end