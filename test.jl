@time begin

    using Random, Distributions, Statistics, Distributed, LinearAlgebra
    include("rbm2.jl")
    include("PSS_Ex.jl")
    #include("parameters.jl")
    
    #Random.seed!(1234)

    dim = 10
    ns = 10^4
    function mytest(dim, ns)
        output = zeros(dim)
        for _ in 1:ns
            myrand = zeros(dim)
            for i in 1:dim
                myrand[i] = rand(Normal())
            end
            output += myrand
        end
        return output / ns
    end
    
    local test_array1 = []
    local test_array2 = []

    for j in 1:10
        temp = mytest(dim, ns)
        push!(test_array1, temp[1])
    end

    println("average = ", mean(test_array1), "; std = ", std(test_array1))

end