using Combinatorics

function degeneracy(cluster)
    idx = Dict()
    multiple = Dict()
    for (i, a) in enumerate(cluster)
        if get(multiple, a, nothing) == nothing
            multiple[a] = 0
            idx[a] = []
        end
        multiple[a] += 1
        push!(idx[a], i)
    end

    total_mul = 1
    for (k, v) in pairs(multiple)
        total_mul *= factorial(v)
    end

    permutedidx = Dict()
    for s in keys(idx)
        permutedidx[s] = permutations(idx[s])
    end

    allpermutations = []
    orgidx = 1:length(cluster) |> collect
    for combinations in Iterators.product(values(permutedidx)...)
        copied = copy(orgidx)
        for (s, combo) in zip(keys(permutedidx), combinations)
            copied[idx[s]] = combo
        end

        push!(allpermutations, copied)
    end

    # return multiple, total_mul, allpermutations
    return allpermutations
end
