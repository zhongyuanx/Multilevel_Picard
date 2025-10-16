@time begin

using CSV, DataFrames

include("PSS_mlp.jl")
include("PSS_Simulations.jl")

println("Base drift = ", gamma, "; sigma = ", sigma, "; control matrix = ", G, "; horizon = ", ub, "; holding cost = ", h, "; discount rate = ", beta)


end_n = 10
#my_output = zeros((end_n+1)^2,8) # output for mlp
my_output = zeros((end_n+1)^2,4) # output for least control

#Code for least control
for a in 0:end_n
    for b in 0:end_n
        x = 0.1*[a, b]
        my_output[(end_n+1)*a+b+1,1:2] = x
        local my_cost=Vfunc_mlt(x, gamma, sigma, G, ub, h, beta, ns, 1e-6)
        my_output[(end_n+1)*a+b+1,3:4] = [mean(my_cost) std(my_cost)]
    end
end

#Code for mlp
# level = 4
# for a in 0:end_n
#     for b in 0:end_n
#         x = 0.1*[a b]
#         my_output[(end_n+1)*a+b+1,1:2] = x
#         local test_array0 = []
#         local test_array1 = []
#         local test_array2 = []
#         for i in 1:9
#             temp = pss_mlp_mlt(0, ub, x, gamma, sigma, drift_b, G, beta, level, M[level])
#             push!(test_array0, temp[1])
#             push!(test_array1, temp[2])
#             push!(test_array2, temp[3])
#         end
#         my_output[(end_n+1)*a+b+1,3:8] = [mean(test_array0) mean(test_array1) mean(test_array2) std(test_array0) std(test_array1) std(test_array2)]
#     end
# end


CSV.write("output_least.csv", DataFrame(my_output, :auto), header=false)

end