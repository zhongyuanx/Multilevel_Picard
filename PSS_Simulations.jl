@time begin
#Simulating heavy-traffic parallel server systems with least control

#This only works for an open chain

using Random, Distributions, Statistics, Plots, LinearAlgebra

include("PSS_Ex.jl")

function skorohod(arr::Vector{Float64})
    #function computes the skorokhod regulator map (the reflection part)
    return accumulate(max, -arr, init=0.0)
end

function least_ctrl(arr::Array{Float64}) #least control paths for generalized d-dimensional N system (open chain)
    d = size(arr, 1)
    for i in 2:d
        skhd = skorohod(arr[i-1,:])
        arr[i,:] = arr[i,:].+skhd
    end
end

function BEL_rbm(arr::Array{Float64})
    n = length(arr)
    tau = findfirst(x -> x<=0.0, arr)
    tau = isnothing(tau) ? n+1 : tau
    arr[tau:end] = arr[tau]*ones(n-tau+1)
    arr = [arr[i]/i for i in 1:n]
    return arr
end

function bm_path(z0, drift, sigma, T, step=1e-3)
    nsteps = ceil(Int, T/step)
    bm = zeros(nsteps+1)
    bm[1] = z0
    
    dist = Normal()
    incr = drift*step .+ sigma * sqrt(step) .* rand(dist, nsteps) #randn(nsteps)

    bm[2:end] = z0 .+ cumsum(incr)
    return bm
end

function rbm_path(z0, drift, sigma, T, step=1e-3)
    nsteps = ceil(Int, T/step)
    bm = zeros(nsteps+1)
    rbm = zeros(nsteps+1)
    bm[1] = z0
    rbm[1] = z0

    dist = Normal()
    incr = drift*step .+ sigma * sqrt(step) .* rand(dist, nsteps)

    bm[2:end] = z0 .+ cumsum(incr)
    tau = findfirst(x -> x<=0.0, bm)
    tau = isnothing(tau) ? nsteps+1 : tau
    rbm = bm .+ skorohod(bm)
    return rbm
end

function cost(h, beta, arr::Vector{Float64}, step=1e-3)
    discount = [exp(-beta*step*(i-1))*step for i in 1:length(arr)]
    return h*dot(arr, discount)
end

function value_func(z0, drift, sigma, beta, h, T, Ns, step=1e-3)
    nsteps = ceil(Int, T/step)
    myoutput = 0.0
    for _ in 1:Ns
        my_rbm = rbm_path(z0, drift, sigma, T, step)
        myoutput+=cost(h, beta, my_rbm, step)/Ns
    end
    return myoutput
end

#println(value_func(0.5, -1.0, 1.0, 0.2, 2.0, -log(0.01)/0.2, 10000, 1e-3))


#the function below is a test. missing the service rate information (mu).

function VandDV(free::Matrix{Float64}, path::Matrix{Float64})
    dim, n = size(free) 
    reflection = zeros(dim, n)
    path = zeros(dim, n)

    reflection[1,:] = skorohod(free[1,:])
    for i in 2:dim
        path[i,:] = free[i,:] - reflection[i-1,:]
        reflection[i,:] = skorohod(path[i,:])
    end

    lc_path = path .+ reflection
    return lc_path
end


function least_ctrl_path(z::Vector{Float64}, driftv::Vector{Float64}, sigmam::Matrix{Float64}, rG::Array{Float64}, T::Float64, step=1e-3)
    nsteps = ceil(Int, T/step)
    dim = length(z)
    bm = zeros(dim, nsteps+1)
    reflection = zeros(dim, nsteps+1) 
    bm[:, 1] = z
    
    dist = MvNormal(step*driftv, step*(sigmam'*sigmam))
    incr = rand(dist, nsteps)
    bm[:, 2:end] = z .+ cumsum(incr, dims=2)

    reflection[1,:] = skorohod(bm[1,:])
    for i in 2:dim
        bm[i,:] = bm[i,:] + rG[i,i-1]*reflection[i-1,:]
        reflection[i,:] = skorohod(bm[i,:])
    end

    lc_path = bm .+ reflection
    return lc_path
end

#z = [1.0, 1.0]
#drift = [-1.0, -1.0]
#sigma = [1.0 0 0; 0 1.0 0; 0 0 1.0]
sigma = Matrix(1.0I, dim, dim)
#h = [1.5, 1.0]
#beta = 0.1
#T = 1.0

function Vfunc(z::Vector{Float64}, driftv::Vector{Float64}, sigmam::Matrix{Float64}, rG::Array{Float64}, T::Float64, h::Vector{Float64}, beta::Float64, Ns::Int, step=1e-3)
    nsteps = ceil(Int, T/step)
    discount = [exp(-beta*step*(i-1))*step for i in 1:(nsteps+1)]
    mycost = []
    for _ in 1:Ns
        mypath = least_ctrl_path(z, driftv, sigmam, rG, T, step)
        push!(mycost, dot(h, mypath*discount))
    end
    return mycost
end

function Vfunc_mlt_call(z::Vector{Float64}, driftv::Vector{Float64}, sigmam::Matrix{Float64}, rG::Array{Float64}, T::Float64, h::Vector{Float64}, beta::Float64, Ns::Int, step, thread_id, NUM_THREADS)
    loop_num = _get_loop_num(ns, thread_id, NUM_THREADS)
    #We are returning the vector of costs for each thread
    #println(loop_num)
    return Vfunc(z, driftv, sigmam, rG, T, h, beta, loop_num, step)
end


function _get_loop_num(num, thread_id, NUM_THREADS)
    if num < NUM_THREADS
        # each thread only goes once through the loop
        loop_num = thread_id > num ? 0 : 1
    else
        remainder =  num % NUM_THREADS
        if (remainder > 0) && (thread_id <= remainder)
            # each thread goes num / NUM_THREADS + the remainder
            loop_num = div(num, NUM_THREADS) + 1
        else
            loop_num = div(num, NUM_THREADS)
        end
    end
    return loop_num
end

function Vfunc_mlt(z::Vector{Float64}, driftv::Vector{Float64}, sigmam::Matrix{Float64}, rG::Array{Float64}, T::Float64, h::Vector{Float64}, beta::Float64, Ns::Int, step=1e-3)
    NUM_THREADS = Threads.nthreads()
    tasks = [Threads.@spawn(Vfunc_mlt_call(z, driftv, sigmam, rG, T, h, beta, Ns, step, thread_id, NUM_THREADS)) for thread_id in 1:NUM_THREADS]
    mycost = [fetch(t) for t in tasks]
    return reduce(vcat, mycost)
end

#println(z, gamma, sigma, G, ub, h, beta)
println("Base drift = ", gamma, "; sigma = ", sigma, "; control matrix = ", G, "; horizon = ", ub, "; holding cost = ", h, "; discount rate = ", beta)
ns = 10000

mycost=Vfunc_mlt(z, gamma, sigma, G, ub, h, beta, ns, 1e-6)
println("initial state = ", z, "mean = ", mean(mycost), "; standard error = ", std(mycost)/sqrt(ns))



# for i in 0:10
#     local z = [0.1*i, 0.1*i]
#     local mycost=Vfunc(z, gamma, sigma, G, ub, h, beta, ns, 1e-5)
#     println("initial state = ", z, "mean = ", mean(mycost), "; standard error = ", std(mycost)/sqrt(ns))
# end

#time = 0:0.001:4
#plot(time, my_rbm)

end