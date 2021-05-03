

using Base
using LinearAlgebra: diag

using Database, Kernels, Means, Descriptors

Kernels = include("./kernels.jl")
Means = include("./means.jl")

"""
Gaussian Process
"""
mutable struct Gaussian <: Model
    kernel::Any
    kernel_hyperparameters::Any
    kernel_bounds::Any
    kernel_kwargs::Any
    mean::Any
    mean_hyperparameters::Any
    mean_bounds::Any
    mean_kwargs::Any
    descriptor::Any
    descriptor_kwargs::Any
    numbers::Any
    imgdata_filename::String
    data_ids::Any
    optimized::Bool
    _cache::Dict


    function Gaussian(;
        kernel="Standard", kernel_hyperparameters=nothing,
        kernel_bounds=nothing, kernel_kwargs=Dict(),
        mean="Zero", mean_hyperparameters=nothing,
        mean_bounds=nothing, mean_kwargs=Dict(),
        descriptor="Descriptor", descriptor_kwargs=Dict(),
        numbers=nothing, imgdata_filename="imgdata.db",
        data_ids=Dict("image"=>[1, 2]),
        optimized=false,
        kwargs...)

        kernel = getfield(Kernels, Symbol(kernel))(;kernel_kwargs...)
        mean = getfield(Means, Symbol(mean))(;mean_kwargs...)
        descriptor = getfield(Descriptors, Symbol(descriptor))(;descriptor_kwargs...)
        new(kernel, kernel_hyperparameters, kernel_bounds, kernel_kwargs,
            mean, mean_hyperparameters, mean_bounds, mean_kwargs,
            descriptor, descriptor_kwargs,
            numbers, imgdata_filename, data_ids, optimized, Dict())
    end
end

function (model::Gaussian)(coords::Array{Float64, 3},
                               properties::Array{String, 1})
    # data::Dict{String, Any}, "coords", "potentials"...
    data = Database.read_data(model.imgdata_filename, model.data_ids)
    𝐤, 𝐦, 𝐆 = model.kernel, model.mean, model.descriptor
    # 𝐗 ::Array{Float64, 3};   3xAxM -> 3A x M
    # 𝐗'::Array{Float64, 3};   3xAxM -> 3A x M
    # 𝐘 ::Array{Float64, 2};   M
    𝐗, 𝐘 = 𝐆.data2𝐗𝐘(data)
    𝐗′ = 𝐆(coords)
    # 𝐗 = 𝐆(𝐗)
    # 𝐗′::Array{Float64, 3};  3xAxN -> 3A x N
    # N = size(coords)[end]
    # D = 3 * A
    # D = 𝐆.D
    # D, N = size(coords)
    results = Dict()

    # 𝐤.hyperparameters = _kernel_hyperparameters(model.kernel_hyperparameters)
    𝐤.hyperparameters = 𝐆.model2ker_hyper(model.kernel_hyperparameters)
    # 𝐦.hyperparameters = _mean_hyperparameters(model.mean_hyperparameters)
    𝐦.hyperparameters = 𝐆.model2mean_hyper(model.mean_hyperparameters)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    if !model.optimized || model._cache == Dict() ||
         get(model._cache, "K_y_inv", nothing) == nothing
        𝐊 = 𝐤(𝐗, 𝐗, true)
        𝐊⁻¹ = inv(𝐊)                            # M x M
        model._cache["K"] = 𝐊                   # M x M
        model._cache["K_y_inv"] = 𝐊⁻¹           # M x M
        model._cache["Λ"] = 𝐊⁻¹ * (𝐘 .- 𝐦(𝐗))  # M
        model.optimized = true
    end
    𝐊, 𝐊⁻¹, 𝚲 = model._cache["K"], model._cache["K_y_inv"], model._cache["Λ"]

    if "potential" in properties
        𝐊₊ = 𝐤(𝐗, 𝐗′)
        𝛍 = 𝐦(𝐗′) .+ 𝐊₊' * 𝚲
        results[:potential] = 𝛍
    end

    if "gradients" in properties
        ∂𝐊₊ = 𝐤(𝐗, 𝐗′; gradients=true)
        ∂𝛍 = 𝐦(𝐗′; gradients=true) .+ ∂𝐊₊' * 𝚲
        results[:gradients] = 𝐆.reshape_gradients(∂𝛍)
        # permutedims(reshape(∂𝛍, N, D), [2, 1])
    end

    if "hessian" in properties
        ∂²𝐊₊ = 𝐤(𝐗, 𝐗′; hessian=true)
        ∂²𝛍 = 𝐦(𝐗′; hessian=true) .+ ∂²𝐊₊' * 𝚲
        results[:hessian] = 𝐆.reshape_hessian(∂²𝛍)
        # permutedims(reshape(∂²𝛍, N, D, 3), [3, 2, 1])
    end

    if "covariance" in properties
        𝐊₊ = 𝐤(𝐗, 𝐗′)
        𝐊₊₊ = 𝐤(𝐗′, 𝐗′)
        _cov = diag(𝐊₊₊) .- diag((𝐊₊' * 𝐊⁻¹ * 𝐊₊))
        cov = copy(_cov)
        flags = _cov .< 0
        cov[flags] .= 0.
        results[:covariance] =  1.96 .* sqrt.(cov) / 2
    end
    return results
end


kvpair = Dict(:kernel => :𝒌, :kernel_hyperparameters => :𝛉ₖ,
              :mean => :𝒎, :mean_hyperparameters => :𝛉ₘ,
              :projector => :𝐏)
mutable struct GraphGaussian <: AtomicModel
    𝒌::Kernel
    𝛉ₖ::Hyperparameter
    𝒎::Mean
    𝛉ₘ::Hyperparameter
    𝐏::Projector
    function GraphGaussian(;kwargs...)
        new_kwargs =
        for (key, value) in pairs(kwargs)

        end
        new()
    end
end

function setproperty!(model::GraphGaussian, name::Symbol, val)
    name = kvpair[name]
    Base.setproperty!(model, name, val)
end
