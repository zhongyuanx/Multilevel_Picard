@time begin

    using Random, Distributions, Statistics, Distributed, LinearAlgebra, FastGaussQuadrature, QuadGK, SpecialFunctions
    #include("rbm2.jl")
    #include("parameters.jl")
    
    Random.seed!(1234)

    # mu = [1.0, 1.0]
    # z = [0.5, 0.5]
    # lambda = z.*mu
    # theta = [0.5,0.5]
    # h = [1.0, 1.5]
    # p = [1.0, 1.5]
    # c = h + theta.*p
    # zeta = -mu
    # sigma = sqrt.(2.0*mu.*z)
    # my_beta = 0.1
    # x0 = [0.0, 1.0]
    # dim = length(x0)
    
    # mu = [1.0]
    # z = [1.0]
    # lambda = z.*mu
    # theta = [0.5]
    # h = [1.0]
    # p = [1.0]
    # c = h + theta.*p
    # zeta = -mu
    # sigma = sqrt.(2.0*mu.*z)
    # my_beta = 0.1
    # x0 = [1.0]
    # dim = length(x0)

    # mu = [1.0, 1.5]
    # z = [0.6, 0.4]
    # lambda = z.*mu
    # theta = [0.5, 1.0]
    # h = [1.0, 1.0]
    # p = [2.0, 1.0]
    # c = h + theta.*p
    # zeta = -mu
    # #z = (lambda./mu)/sum(lambda./mu)
    # sigma = sqrt.(2.0*mu.*z)
    # my_beta = 0.1
    # x0 = [1.0, 1.0]
    # dim = length(x0)

    dim = 2
    mu = ones(dim)
    z = ones(dim)/dim
    lambda = z.*mu
    theta = 0.5*ones(dim)
    h = ones(dim)
    p = 1.5*ones(dim)
    c = h + theta.*p
    zeta = -mu
    sigma = sqrt.(2.0*mu.*z)
    my_beta = 0.1
    x0 = ones(dim)/dim

    M = [5000, 1024, 128, 32, 16, 16, 8] #M=4321 test case
    
    epsilon = 0.01
    ub = -log(epsilon)/my_beta
    #ub = 1.0

    mmt = 1.0

    function my_rtime(my_beta, lb=0, ub=Inf)
        # simulate random sampling times
        return lb+(rand(Truncated(Normal(), -sqrt(2*my_beta*(ub-lb)), sqrt(2*my_beta*(ub-lb)))))^2/(2*my_beta)
    end

    function my_C(my_beta, lb, ub)
        return sqrt(pi / my_beta) * (1 - 2 * cdf(Normal(0, 1), -sqrt(2 * my_beta * (ub - lb))))
    end
    
    function my_g(state, v)
        return max(sum(state),0)*minimum(c.+(mu-theta).*v[2:end])
    end
    
    function picard_iter(state, v1, v2)
        return my_g(state, v1) - my_g(state, v2)
    end

    function cov_matrix(sigma, mu, t)
        #v22 = t>1e-7 ? (1-exp(-2.0*mu*t))/(2.0*mu*t)/sigma^2 : sigma^(-2.0)
        v22 = (1-exp(-2.0*mu*t))/(2.0*mu*t)/sigma^2
        return [sigma^2/(2.0*mu)*(1-exp(-2.0*mu*t)) sqrt(t)*exp(-mu*t); sqrt(t)*exp(-mu*t) v22]
    end
        
    function mlp_v(t, T, x, level, M)
        output = zeros(dim + 1)
        # println(t, x, level)
        if level == 0
            return output
        end
    
        # number of simulated instances
        ns = M ^ level
        #mmt = mmtm(0, init_mmt, delta)
    
        for i in 1:ns
            my_S = my_rtime(my_beta, t, T)
            my_randn = randn(2, dim)
            my_W = sigma./sqrt.(2.0*mu) .*sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randn[1,:]
            my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
            a = exp.(-(my_S-t)*mu)./sigma.* (my_S-t<1e-10 ? ones(dim) : sqrt.(2.0*(my_S-t)*mu./(1.0.-exp.(-2.0*(my_S-t)*mu))))
            #b = sqrt.((my_S-t<1e-4 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))./sigma./sigma-a.*a)
            my_bel = a.*my_randn[1,:] #+ b.*my_randn[2,:]

            # my_W = zeros(dim)
            # my_bel = zeros(dim)
            # for j in 1:dim
            #     my_W[j], my_bel[j] = rand(MvNormal(zeros(2), cov_matrix(sigma[j], mu[j], my_S-t)))
            # end
            
            # my_randomness = randn(dim)
            # my_W = sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randomness
            # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W.*sigma./sqrt.(2.0*mu)
            # if my_S-t<1e-7
            #     my_bel = my_randomness./sigma
            # else
            #     my_bel = my_W./sqrt.(2.0*mu)./sigma./sqrt(my_S - t)
            # end
            output += mmt*my_g(my_state, zeros(dim+1)) * vcat([sqrt(my_S - t)], my_bel)
        end
        output = output * my_C(my_beta, t, T) / ns #+ mmt*[z, 1]
    
        for l in 1:(level - 1)
            # number of simulated instances
            ns = M ^ (level - l)
    
            temp = zeros(dim + 1)
            for i in 1:ns
                my_S = my_rtime(my_beta, t, T)
                my_randn = randn(2, dim)
                my_W = sigma./sqrt.(2.0*mu) .*sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randn[1,:]
                my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
                a = exp.(-(my_S-t)*mu)./sigma.* (my_S-t<1e-10 ? ones(dim) : sqrt.(2.0*(my_S-t)*mu./(1.0.-exp.(-2.0*(my_S-t)*mu))))
                #b = sqrt.((my_S-t<1e-10 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))./sigma./sigma-a.*a)
                my_bel = a.*my_randn[1,:] #+ b.*my_randn[2,:]

                # my_W = zeros(dim)
                # my_bel = zeros(dim)
                # for j in 1:dim
                #     my_W[j], my_bel[j] = rand(MvNormal(zeros(2), cov_matrix(sigma[j], mu[j], my_S-t)))
                # end
                # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
                # my_randomness = randn(dim)
                # my_W = sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randomness
                # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W.*sigma./sqrt.(2.0*mu)
                # if my_S-t<1e-7
                #     my_bel = my_randomness./sigma
                # else
                #     my_bel = my_W./sqrt.(2.0*mu)./sigma./sqrt(my_S - t)
                # end                
                v1 = mlp_v(my_S, T, my_state, l, M)
                v2 = mlp_v(my_S, T, my_state, l - 1, M)
                test1 = mlp_v(t, T, x, l, M)
                test2 = mlp_v(t, T, x, l-1, M)
                temp += mmt*my_C(my_beta, t, T)*picard_iter(my_state, v1, v2) * vcat([sqrt(my_S - t)], my_bel) + (1-mmt)*(test1-test2)
            end
            temp = temp / (ns) #* my_C(my_beta, t, T)
            output += temp
        end
    
        return output
    end

    function mlp_mlt_call(t, T, x, level, M, thread_id, NUM_THREADS)
        output2 = zeros(dim+1)
        # number of simulated instances
        ns = M ^ level
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
        #println(loop_num)
    
        for i in 1:loop_num
            my_S = my_rtime(my_beta, t, T)
            my_randn = randn(2, dim)
            my_W = sigma./sqrt.(2.0*mu) .*sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randn[1,:]
            my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
            a = exp.(-(my_S-t)*mu)./sigma.* (my_S-t<1e-10 ? ones(dim) : sqrt.(2.0*(my_S-t)*mu./(1.0.-exp.(-2.0*(my_S-t)*mu))))
            # if i==1081
            #     println(a)
            #     println(my_S-t, ", ", my_S-t<1e-10 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))
            #     println((my_S-t<1e-8 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))./sigma./sigma-a.*a)
            # end
            #b = sqrt.((my_S-t<1e-8 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))./sigma./sigma-a.*a)
            my_bel = a.*my_randn[1,:] #+ b.*my_randn[2,:]

            # my_W = zeros(dim)
            # my_bel = zeros(dim)
            # for j in 1:dim
            #     my_W[j], my_bel[j] = rand(MvNormal(zeros(2), cov_matrix(sigma[j], mu[j], my_S-t)))
            # end
            # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
            # my_randomness = randn(dim)
            # my_W = sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randomness
            # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W.*sigma./sqrt.(2.0*mu)
            # if my_S-t<1e-7
            #     my_bel = my_randomness./sigma
            # else
            #     my_bel = my_W./sqrt.(2.0*mu)./sigma./sqrt(my_S - t)
            # end
            output2 += mmt*my_g(my_state, zeros(dim+1)) * vcat([sqrt(my_S - t)], my_bel)
        end
        output2 = output2 * my_C(my_beta, t, T)/ns #+ mmt*[z, 1]

        for l in 1:(level - 1)
            # number of simulated instances
            ns2 = M^(level-l)
            loop_num = _get_loop_num(ns2, thread_id, NUM_THREADS)
            temp = zeros(dim + 1)
            for _ in 1:loop_num
                my_S = my_rtime(my_beta, t, T)
                my_randn = randn(2, dim)
                my_W = sigma./sqrt.(2.0*mu) .*sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randn[1,:]
                my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
                a = exp.(-(my_S-t)*mu)./sigma.* (my_S-t<1e-10 ? ones(dim) : sqrt.(2.0*(my_S-t)*mu./(1.0.-exp.(-2.0*(my_S-t)*mu))))
                #b = sqrt.((my_S-t<1e-10 ? ones(dim) : (1.0.-exp.(-2.0*(my_S-t)*mu))./(2.0*(my_S-t)*mu))./sigma./sigma-a.*a)
                my_bel = a.*my_randn[1,:] #+ b.*my_randn[2,:]

                # my_W = zeros(dim)
                # my_bel = zeros(dim)
                # for j in 1:dim
                #     my_W[j], my_bel[j] = rand(MvNormal(zeros(2), cov_matrix(sigma[j], mu[j], my_S-t)))
                # end
                # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W
                # my_randomness = randn(dim)
                # my_W = sqrt.(1.0.-exp.(-2.0*(my_S-t)*mu)).*my_randomness
                # my_state = x.*exp.(-(my_S-t)*mu)+zeta./mu.*(1.0.-exp.(-(my_S-t)*mu)) + my_W.*sigma./sqrt.(2.0*mu)
                # if my_S-t<1e-7
                #     my_bel = my_randomness./sigma
                # else
                #     my_bel = my_W./sqrt.(2.0*mu)./sigma./sqrt(my_S - t)
                # end
                v1 = mlp_v(my_S, T, my_state, l, M)
                v2 = mlp_v(my_S, T, my_state, l - 1, M)
                test1 = mlp_v(t, T, x, l, M)
                test2 = mlp_v(t, T, x, l - 1, M)
                temp += mmt*my_C(my_beta, t, T)*picard_iter(my_state, v1, v2) * vcat([sqrt(my_S - t)], my_bel) + (1-mmt)*(test1-test2)
            end
            temp = temp / (ns2) #* my_C(my_beta, t, T)
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

    function mlp_mlt(t, T, x, level, M)
        NUM_THREADS = Threads.nthreads()
        tasks = [Threads.@spawn(mlp_mlt_call(t, T, x, level, M, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
        a = sum([fetch(task) for task in tasks])
        return a
    end
    
    
    println("T = ", ub, "; discount factor = ", my_beta, "; initial state = ", x0)
    # # #println(V, ", ", DV1(z))
    
    for level in 1:3
        test_array1 = []
        test_array2 = []
        test_array3 = []
        println("level = ", level, "; M = ", M[level])
        for i in 1:10
            #temp = mlp_mlt_test(0, ub, z, gamma, sigma, level, M)
            temp = mlp_mlt(0, ub, x0, level, M[level])
            #temp = mlp_v(0, ub, z, gamma, sigma, level, M)
            #println(temp)
            push!(test_array1, temp[1])
            push!(test_array2, temp[2])
            push!(test_array3, temp[3])
            if i%5==0
                println("average V = ", mean(test_array1), "; std = ", std(test_array1))
                println("average DV1 = ", mean(test_array2), "; std = ", std(test_array2))
                println("average DV2 = ", mean(test_array3), "; std = ", std(test_array3))
            end         
        end
        #println(c.+(mu-theta).*[mean(test_array2), mean(test_array3)])
    end
    
end