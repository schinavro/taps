module Means

export Zero, Average, NNP

mutable struct Zero
    hyperparameters
    data
    idx::Int64
end

function (mean::Zero)(X)
    return 0.
end

mutable struct Average
    hyperparameters
    data
    idx::Int64
    function Average()
        new(nothing, zeros(1, 1), 1)
    end
end

function (mean::Average)(X)
    potentials = mean.data[mean.idx, :]
    return sum(potentials) / length(potentials)
end


struct NNP
    hyperparameters
    data
    idx::Int64
end

"""
Neural Network Potential
"""
function (nnp::NNP)(X, θ)
    return W*θ .+ b
end


end
