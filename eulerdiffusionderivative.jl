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

function Drift(x, gamma, drift_bd, alpha)
    # This function computes the drift at state x
    # inputs: x (state), Drift (drift rate function)
    return gamma + 0.5*drift_bd*(tanh(alpha*(2.0*x[2]-x[1]))+1.0)*[1.0, -1.0]
    # If x2 >> x1, server 1 should help, so set drift to upper bound. If x2 << x1, no help so set drift to zero. 
end

function Grad_Drift(x, gamma, drift_bd, alpha)
    return 0.5*drift_bd*(sech(alpha*(x[2]-x[1])))^2*alpha*[-1.0 2.0; 2.0 -1.0]  # gradient of drift function
    # two-dimensional special case. 
end

function RF_Output(x0, gamma, sigma, inv_sigma, R, drift_bd, alpha, Tlb, Tub, S, dsteps::Int = 25)
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
        return x0, inv_sigma * rand(MvNormal(zeros(dim), sigma'*sigma))
    end

    dh = (S - Tlb) / dsteps
    path = zeros(dim,dsteps+1)
    path[:, 1] = x0  # initial state at time Tlb

    deriv = zeros(dim,dim, dsteps+1)
    deriv[:, :, 1] = Matrix(1.0I, dim, dim)  # initial derivative is identity matrix
    
    bel = zeros(dim)  # bismut-elworthy-li term

    for i in 2:dsteps+1
        incr = rand(MvNormal(zeros(dim), dh*(sigma'*sigma)))
        bel += deriv[:, :, i-1] * incr  # accumulate bismut-elworthy-li term
        path[:, i] = path[:, i-1] + dh*Drift(path[:, i-1], gamma, drift_bd, alpha) + incr
        deriv[:, :, i] = deriv[:, :, i-1] + dh*Grad_Drift(path[:, i-1], gamma, drift_bd, alpha)*deriv[:, :, i-1]
        for j in 1:dim
            if path[j, i] < 0.0 # if the jth component of the path is negative, reflect it
                path[j, i] = 0.0
                deriv[j, :, i] = transpose(zeros(dim))  # jth row of deriv at time i is zero row vector; hence transpose. 
            end
        end
    end
    
    return path[:, dsteps+1], inv_sigma*bel/sqrt(S - Tlb)
end