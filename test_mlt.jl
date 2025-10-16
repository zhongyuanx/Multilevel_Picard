NUMTHREADS = Threads.nthreads()

function test_mlt(thread_id, NUMTHREADS)
    return [zeros(4) ones(4)]
end

function test_mlt2(thread_id, NUMTHREADS)
    return thread_id^3
end

task = Vector{Task}(undef, NUMTHREADS)
task2 = Vector{Task}(undef, NUMTHREADS)
result = fetch.([Threads.@spawn(test_mlt(thread_id, NUMTHREADS)) for thread_id in 1:NUMTHREADS])
#println(result)

#=
for thread_id in 1:NUMTHREADS
    task[thread_id] = Threads.@spawn(test_mlt(thread_id, NUMTHREADS))
    result[thread_id] = fetch(task[thread_id])
    task2[thread_id] = Threads.@spawn(test_mlt2(thread_id, NUMTHREADS))
    result[thread_id] += fetch(task2[thread_id])
    println("Thread ID: $thread_id, Result: $result")
end
=#

function square(x)
    return [x^2 1]
end

test = [square(x) for x in 1:3]
println(test[:,1], test[2])