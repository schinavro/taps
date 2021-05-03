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
    hyperparameters::Any
    data::Any
    function Average(;hyperparameters=nothing, data=nothing)
        new(hyperparameters, data)
    end
end

function (mean::Average)(X::Array; gradients=false, hessian=false)
    if gradients || hessian
        return 0.
    end
    return Real(sum(mean.data)) / size(mean.data)[end]
end

mutable struct Average2
    hyperparameters::Any
    data::Any
    D::Int
    A::Int
    function Average2(;hyperparameters=nothing, data=nothing, D=nothing, A=nothing)
        new(hyperparameters, data, D, A)
    end
end

function (mean::Average2)(X::Array; potential=false, gradients=false, hessian=false)
    if gradients || hessian
        return 0.
    end
    Vave = Real(sum(mean.data)) / size(mean.data)[end]
    if potential
        return Vave
    end
    mean.D
    dVave = 0.
end


mutable struct AtomicAverage
    hyperparameters
    data
    idx::Int64
    function AtomicAverage()
        new(nothing, zeros(1, 1), 1)
    end
end

function (mean::AtomicAverage)(X)
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
