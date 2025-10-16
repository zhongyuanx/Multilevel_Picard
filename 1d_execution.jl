@time begin

include("parameters.jl")
#include("1d_test2.jl")
include("1d_test_mlt2.jl")
include("1d_explicit.jl")

println("Drift control: drift upper bound = ", drift_b, "; discount factor = ", beta, "; drift = ", gamma, "; sigma = ", sigma, "; holding cost = ", h, "; push cost = ", c, "\n")

for z in 0.9:0.01:1.0
    println("Initial state z = ", z, "; b = ", drift_sol.u, ", V(z) = ", VF(z, beta, gamma, h, c, drift_sol.u, drift_b, C2, C3), ", V'(z) = ", DVF(z, beta, gamma, h, c, drift_sol.u, drift_b, C2, C3), "\n")
    my_array1 = []
    my_array2 = []
    for i in 1:10
        temp = mlp_mlt(0, ub, z, gamma, sigma, level, M)
        push!(my_array1, temp[1])
        push!(my_array2, temp[2])
    end
    println("T = ", ub, "; momentum = ", mmt,  "; level = ", level, "; M = ", M)
    println("average V = ", mean(my_array1), "; std = ", std(my_array1))
    println("average DV = ", mean(my_array2), "; std = ", std(my_array2))
    println("\n")
end

end