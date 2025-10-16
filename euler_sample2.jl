@time begin

using Random, Distributions, Statistics, Plots, LinearAlgebra

struct RF_Input{T1 <: AbstractFloat, T2 <: Int, T3 <: Function}
    x0::Array{T1} #initial state
    dim::T2  #problem dimension
    Tub::T1 #upper bound of time interval
    Tlb::T1 #lower bound of time interval
    Drift::T3 #drift rate function
    sigma::Matrix{T1} #diffusion coefficients
    R::Matrix{T1} #reflection matrix
end

function rtime(beta, Tlb, Tub)
    # simulate random sampling times
    if beta > 0
        return Tlb+(rand(Truncated(Normal(), -sqrt(2*beta*(Tub-Tlb)), sqrt(2*beta*(Tub-Tlb)))))^2/(2*beta)
    elseif beta == 0
        return Tlb + rand()^2 * (Tub - Tlb)  # random time sampling if beta is zero
    else
        error("Beta must be non-negative.")
    end
end

function Drift(x, gamma)
    # This function computes the drift at state x
    # inputs: x (state), Drift (drift rate function)
    return gamma
end

function RF_Output(x0, gamma, sigma, R, Drift, Tlb, Tub, S, dsteps::Int = 25)
    # This function computes the outputs of the reflected diffusion process
    # inputs: x0 (initial state), Tlb (lower bound of time interval),
    #         Tub (upper bound of time interval), S (sampling time),
    #         Drift (drift rate function), sigma (diffusion coefficients),
    #         R (reflection matrix), dsteps (number of steps for discretization)

    #outputs: S (random time), tau (hitting time to zero tau), Z(S) (reflected diffusion path at time S), 
    #         BM(S\wedge tau) (Brownian motion at min of S and tau)

    dim = length(x0)
    if S-Tlb < 1e-8
        # if S is very close to Tlb, return initial state and zero Brownian motion
        return x0, zeros(dim)
    end

    dh = (S - Tlb) / dsteps
    path = zeros(dim,dsteps+1)
    bm = zeros(dim,dsteps+1)
    #dS = ceil(Int, (S-Tlb)/dh)  # number of steps to reach S
    tau = ones(Int, dim)*(dsteps+1) # initialize tau to upper bound

    path[:, 1] = x0  # initial state at time Tlb

    for i in 2:dsteps+1
        incr = rand(MvNormal(zeros(dim), dh*(sigma'*sigma)))
        bm[:, i] = bm[:, i-1] + incr
        path[:, i] = path[:, i-1] + dh*Drift(path[:, i-1], gamma) + incr
        for j in 1:dim
            if path[j, i] < 0.0
                # reflection at zero
                if tau[j] == dsteps+1  # if tau is still at upper bound
                    tau[j] = i  # simulate hitting time
                end
                path[j, i] = max(path[j, i],0.0)
            end
        end
    end

    stopped_bm = [bm[j, tau[j]] for j in 1:dim]
    #stopped_bm = zeros(dim)
    #tau = Int.(tau)  # convert tau to Int for indexing
    #for j in 1:dim
        #if tau[j]>dsteps+1 || dS > dsteps+1
            #println("j = ", j, ", tau = ", tau[j], ", Tlb = ", Tlb, ", Tub = ", Tub, ", dh = ", dh, ", S = ", S, ", dS = ", dS)
        #end
        #stopped_bm[j] = bm[j, tau[j]]
        # if tau[j] < dsteps+1, then we take the value at tau[j]. 
        # else, tau[j] must be dsteps+1, in which case tau[j] is simply the discretized S.
    #end
    
    return path[:, dsteps+1], stopped_bm/sqrt(S - Tlb)
end

end