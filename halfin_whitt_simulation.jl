@time begin
    #Simulating a 1D OU process
    
    using Random, Distributions, Statistics, Plots, LinearAlgebra
    
    Random.seed!(1234)
    
    mu = [1.0, 1.0]
    z = [0.5, 0.5]
    lambda = z.*mu
    theta = [0.5,0.5]
    h = [1.0, 1.5]
    p = [1.0, 1.5]
    c = h + theta.*p
    zeta = -mu
    sigma = sqrt.(2.0*mu.*z)
    beta = 0.1
    x0 = [0.0, 1.0]
    dim = length(x0)

    # mu = [1.0, 1.5]
    # z = [0.6, 0.4]
    # lambda = z.*mu
    # theta = [0.5, 1.0]
    # h = [1.0, 1.5]
    # p = [1.0, 1.5]
    # c = h + theta.*p
    # zeta = -mu
    # #z = (lambda./mu)/sum(lambda./mu)
    # sigma = sqrt.(2.0*mu.*z)
    # beta = 0.1
    # x0 = [0.0, 1.0]
    # dim = length(x0)

    epsilon = 0.01
    ub = -log(epsilon)/beta

    function drift(x)
        result = zeta - mu.*x + [max(sum(x),0)*(mu[1]-theta[1]), 0.0]
        return result
    end

    function ou_path(x0, sigma, T, step=1e-3)
        nsteps = ceil(Int, T/step)
        ou = zeros(dim, nsteps+1)
        ou[:,1] = x0
        
        dist = Normal()
        incr = sigma[1] * sqrt(step) .* rand(dist, (dim, nsteps))
        
        cost = 0.0
        for i in 1:(nsteps)
            ou[:,i+1] = ou[:,i]+drift(ou[:,i])*step+incr[:,i]
            cost+=exp(-beta*i*step)*max(sum(ou[:,i+1]), 0)*c[1]*step
        end

        return cost
    end

    function ou_est(x0, sigma, T, ns, step=1e-3)
        result = 0.0
        for _ in 1:ns
            result+=ou_path(x0, sigma, T, step)/ns
        end
        return result
    end

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

    function ou_est_call(x0, sigma, T, ns, step, thread_id, NUM_THREADS)
        loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)

        return ou_est(x0, sigma, T, loop_num, step)*loop_num/ns
    end

    function ou_est_mlt(x0, sigma, T, ns, step)
        NUM_THREADS = Threads.nthreads()
        tasks = [Threads.@spawn(ou_est_call(x0, sigma, T, ns, step, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
        a = sum([fetch(task) for task in tasks])
        return a
    end


    ns = 1000
    my_array = []
    for _ in 1:10
        temp = ou_est_mlt(x0, sigma, ub, ns, 1e-4)
        println(temp)
        push!(my_array, temp)
    end
    println(mean(my_array), std(my_array))



end