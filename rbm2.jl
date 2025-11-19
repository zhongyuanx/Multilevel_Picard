#This program implements the O(1) simulation algorithm for the triple (88) in Appendix D of the paper.

using Distributions

function my_rtime(beta, lb=0, ub=Inf)
    # simulate random sampling times
    if beta>0
        return lb+(rand(Truncated(Normal(), -sqrt(2*beta*(ub-lb)), sqrt(2*beta*(ub-lb)))))^2/(2*beta)
    elseif beta==0
        return lb+(rand())^2*(ub-lb)
    else
        error("Beta must be non-negative.")
    end
end

function bm_hitting(z, gamma, sigma=1, elapsed=0)
    # simulate tau = hitting time of brownian motion with initial state z, drift gamma and variance sigma^2 to zero starting at elapsed
    # note that the hitting time density is an inverse Gaussian with parameters -z/gamma and z^2/sigma^2
    if z==0
        return 0
    end
    
    if gamma < 0
        return elapsed + rand(InverseGaussian(-z/gamma, z^2 / sigma^2))
    else 
        std_Gauss = rand(Normal())
        return z^2/sigma^2/std_Gauss^2
    end
end 



function stopped_rbm(z, gamma, tau, time, sigma=1, elapsed=0)
    # tau can only be hitting time to zero here; needs to be very specific.
    rs_time = time - elapsed
    rs_tau = tau - elapsed
    std_Exp = rand(Exponential(1.0))
    std_Gauss = rand(Normal())
    
    if rs_time >= rs_tau
        my_bm_scaled = -(z + gamma * rs_tau) / sigma / sqrt(rs_time)
        my_bm = my_bm_scaled * sqrt(rs_time)
        #my_bm_rs_time = rand(Normal(-gamma * (rs_time - rs_tau), sqrt(rs_time - rs_tau) * sigma))
        my_bm_rs_time = -gamma * (rs_time - rs_tau) + sqrt(rs_time - rs_tau) * sigma * std_Gauss
        my_rbm = sqrt(my_bm_rs_time^2 + 2 * (sigma^2) * (rs_time - rs_tau) * std_Exp) / 2 - my_bm_rs_time / 2
    else
        #std_Gauss = rand(Normal())
        #std_Exp = rand(Exponential(1.0))
        my_rbm = sqrt(2 * (sigma^2) * rs_time * (rs_tau - rs_time) / rs_tau * std_Exp + ((rs_tau - rs_time) * z / rs_tau + sigma * sqrt(rs_time * (rs_tau - rs_time) / rs_tau) * std_Gauss)^2)
        my_bm_scaled = ((z^2) * ((rs_time^1.5) / (rs_tau^2) - 2 * sqrt(rs_time) / rs_tau) + 2 * z * sigma * (((rs_tau - rs_time) / rs_tau)^1.5) * std_Gauss + sigma^2 * (rs_tau - rs_time) / rs_tau * sqrt(rs_time) * (std_Gauss^2 + 2 * std_Exp)) / (my_rbm + z) / sigma - gamma * sqrt(rs_time) / sigma
        # the complicated expressions for my_rbm and my_bm_scaled are due to numerical instabilities associated with their simulations.
        # this happens when the random time rs_time is very close to zero. 
        my_bm = my_bm_scaled * sqrt(rs_time)
    end
    
    #return my_bm, my_rbm
    return my_bm_scaled, my_rbm
end
