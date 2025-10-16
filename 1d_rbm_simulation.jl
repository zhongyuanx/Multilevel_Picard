#This code simulates reflecting Brownian motion. 

using Distributions, Plots

function skorohod(arr::Vector{Float64})
    #function computes the skorokhod regulator map (the reflection part)
    return accumulate(max, -arr, init=0.0)
end

function bm_path(z0, delta, drift, sigma, T, step=1e-3)
    nsteps = ceil(Int, T/step)
    bm1 = zeros(nsteps+1)
    bm2 = zeros(nsteps+1)
    bm1[1] = z0
    
    dist = Normal()
    incr = drift*step .+ sigma * sqrt(step) .* rand(dist, nsteps) #randn(nsteps)

    bm1[2:end] = z0 .+ cumsum(incr)
    bm2 = bm1 + delta*ones(nsteps+1)
    return nsteps, bm1, bm2
end

function rbm_path(z0, delta, drift, sigma, T, step=1e-3)
    nsteps, bm1, bm2 = bm_path(z0, delta, drift, sigma, T, step)
    rbm1 = bm1.+skorohod(bm1)
    rbm2 = bm2.+skorohod(bm2)
    return nsteps, rbm1, rbm2
end


z0, delta, drift, sigma, T, step = 1.0, 0.2, -1.0, 1.0, 1.0, 1e-4

nsteps, my_bm1, my_bm2 = rbm_path(z0, delta, drift, sigma, T, step)

time = range(0.0, T, nsteps+1)
#p1=plot(time, my_bm1, yticks = (0:0.5:1.5, ["0", "0.5", "1.0", "1.5"]), legend=false)
plot(time, [my_bm1, my_bm2], yticks = (0:0.5:1.5, ["0", "0.5", "1.0", "1.5"]), legend=false)
xlabel!("Time")
