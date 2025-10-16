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
        if level == 0
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
            #output += cost(my_R) * vcat([sqrt(my_S - t)], mmt*my_B ./ sigma)
            output += vcat(cost(my_R)*sqrt(my_S - t), h.*my_R.*my_B./sigma)
        end
        output = output / ns * my_C(beta, t, T)
    
        return output
    end
    
    println("T = ", ub, "; drift upper bound = ", drift_b, "; holding cost = ", h, "; discount factor = ", beta, "; initial state = ", z, "; drift = ", gamma, "; sigma = ", sigma) #"; level = ", level, "; M = ", M) #, "; M2 = ", M2)
    
    for level in 1:1
        local test_array1 = []
        local test_array2 = []
        local test_array3 = []
        #test_array4 = []
        println("level = ", level, "; M = ", M[level])
        for i in 1:10
            temp = pss_mlp(0, ub, z, gamma, sigma, G, beta, level, M[level])
            #println(temp)
            push!(test_array1, temp[1])
            push!(test_array2, temp[2])
            push!(test_array3, temp[3])
            #push!(test_array4, temp[4])
            if i%10==0
                println("average V = ", mean(test_array1), "; std = ", std(test_array1))
                println("average DV1 = ", mean(test_array2), "; std = ", std(test_array2))
                println("average DV2 = ", mean(test_array3), "; std = ", std(test_array3))
                #println("average DV3 = ", mean(test_array4), "; std = ", std(test_array4))
            end         
        end
    end
    

end