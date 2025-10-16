@time begin

    using Random, Distributions, Statistics, Distributed, LinearAlgebra
    #include("rbm2.jl")
    include("PSS_Ex.jl")
    include("euler_sample.jl")
    #include("parameters.jl")
    
    Random.seed!(1234)

    #specify holding cost rate h
    #specify control matrix G

    sigma = Matrix(1.0I, dim, dim)
    inv_sigma = inv(sigma)
    R = Matrix(1.0I, dim, dim)
    
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
    
    function pss_mlp(t, T, z, gamma, sigma, inv_sigma, G, beta, level, M, dsteps = 50)
        dim = length(z)
        output = zeros(dim + 1)
        # println(t, x, level)
        if level == 0
            return output
        end
    
        # number of simulated instances
        ns = M ^ level
        #mmt = mmtm(0, init_mmt, delta)
    
        for _ in 1:ns
            my_S = rtime(beta, t, T)
            my_R, my_B = RF_Output(z, gamma, sigma, R, Drift, t, T, my_S, dsteps)
            #output += cost(my_R) * vcat([sqrt(my_S - t)], mmt*my_B ./ sigma)
            output += cost(my_R) * vcat([sqrt(my_S - t)], inv_sigma*my_B)
        end
        output = output / (ns) * my_C(beta, t, T) #+ mmt*[z, 1]
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
    
            temp = zeros(dim + 1)
            for i in 1:ns
                my_S = rtime(beta, t, T)
                my_R, my_B = RF_Output(z, gamma, sigma, R, Drift, t, T, my_S, dsteps)                
                v1 = pss_mlp(my_S, T, my_R, gamma, sigma, inv_sigma, G, beta, l, M, dsteps)
                v2 = pss_mlp(my_S, T, my_R, gamma, sigma, inv_sigma, G, beta, l - 1, M, dsteps)
                temp += my_C(beta, t, T)*picard_iter(v1, v2, G, min(drift_b*l/4,drift_b), min(drift_b*(l-1)/4,drift_b)) * vcat([sqrt(my_S - t)], inv_sigma*my_B)
            end
            temp = temp / (ns) #* my_C(beta, t, T)
            output += temp
        end
    
        return output
    end

    function pss_mlp_mlt_call(t, T, z, gamma, sigma, inv_sigma, G, beta, level, M, dsteps, thread_id, NUM_THREADS)
        dim = length(z)
        output2 = zeros(dim+1)
        # number of simulated instances
        ns = M ^ level
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
    
        for _ in 1:loop_num
            my_S = rtime(beta, t, T)
            my_R, my_B = RF_Output(z, gamma, sigma, R, Drift, t, T, my_S, dsteps)
            output2 += (cost(my_R))*vcat([sqrt(my_S - t)], inv_sigma*my_B)
        end
        output2 = output2 / (ns) * my_C(beta, t, T)

        for l in 1:(level - 1)
            # number of simulated instances
            ns2 = M^(level-l)
            #if l < level-1
            #    ns2 = M ^ (level - l)
            #else
            #    ns2 = ceil(Int, M/2)
            #end
            loop_num = _get_loop_num(ns2, thread_id, NUM_THREADS)
            temp = zeros(dim + 1)
            for _ in 1:loop_num
                my_S = rtime(beta, t, T)
                my_R, my_B = RF_Output(z, gamma, sigma, R, Drift, t, T, my_S, dsteps)
                v1 = pss_mlp(my_S, T, my_R, gamma, sigma, inv_sigma, G, beta, l, M, dsteps)
                v2 = pss_mlp(my_S, T, my_R, gamma, sigma, inv_sigma, G, beta, l - 1, M, dsteps)
                temp += my_C(beta, t, T)*picard_iter(v1, v2, G, min(drift_b*l/4,drift_b), min(drift_b*(l-1)/4,drift_b)) * vcat([sqrt(my_S - t)], inv_sigma*my_B)
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

    function pss_mlp_mlt(t, T, z, gamma, sigma, inv_sigma, G, beta, level, M, dsteps)
        NUM_THREADS = Threads.nthreads()
        tasks = [Threads.@spawn(pss_mlp_mlt_call(t, T, z, gamma, sigma, inv_sigma, G, beta, level, M, dsteps, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
        a = sum([fetch(task) for task in tasks])
        return a
    end
    

    println("T = ", ub, "; drift upper bound = ", drift_b, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma) #"; level = ", level, "; M = ", M) #, "; M2 = ", M2)
    # # #println(V, ", ", DV1(z))
    
    dsteps = 50
    for level in 1:2
        test_array1 = []
        test_array2 = []
        test_array3 = []
        println("level = ", level, "; M = ", M[level])
        for i in 1:20
            temp = pss_mlp_mlt(0, ub, z, gamma, sigma, inv_sigma, G, beta, level, M[level], dsteps)
            #println(temp)
            push!(test_array1, temp[1])
            push!(test_array2, temp[2])
            push!(test_array3, temp[3])
            if i%10==0
                println("average V = ", mean(test_array1), "; std = ", std(test_array1))
                println("average DV1 = ", mean(test_array2), "; std = ", std(test_array2))
                println("average DV2 = ", mean(test_array3), "; std = ", std(test_array3))
            end         
        end
    end
    
end