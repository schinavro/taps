module Models

export MullerBrown, Gaussian, Model, AtomicGaussian

using LinearAlgebra: diag

abstract type Model end

struct MullerBrown <: Model
    model_label::String
    model_potential_unit::String
    data_ids::Dict{String, Array{Int64, 1}}
    A::Array{Float64, 1}
    a::Array{Float64, 1}
    b::Array{Float64, 1}
    c::Array{Float64, 1}
    x0::Array{Float64, 1}
    y0::Array{Float64, 1}
    function MullerBrown(model_label::String, model_potential_unit::String,
                         data_ids::Dict{String, Array{Int, 1}};
                         A::Array{Float64, 1}=[-200., -100., -170, 15],
                         a::Array{Float64, 1}=[-1, -1, -6.5, 0.7],
                         b::Array{Float64, 1}=[0., 0., 11., 0.6],
                         c::Array{Float64, 1}=[-10, -10, -6.5, 0.7],
                         x0::Array{Float64, 1}=[1, 0, -0.5, -1],
                         y0::Array{Float64, 1}=[0, 0.5, 1.5, 1]
                         )
        new(model_label, model_potential_unit, data_ids,
            A, a, b, c, x0, y0)
    end
    function MullerBrown(input_dict::Dict)

        properties = Array{String, 1}(indict["properties"])
        model_model = String(indict["model_model"])
        model_label = String(indict["model_label"])
        model_potential_unit  = String(indict["model_potential_unit"])
        data_ids = Dict{String, Array{Int, 1}}(indict["data_ids"])
        _kwargs = Dict{Any, Any}(indict["model_kwargs"])
        model_kwargs = Dict{Symbol, Any}()
        for (key, value) = pairs(_kwargs); model_kwargs[Symbol(key)] = value; end
        coords_epoch = indict["coords_epoch"]
        coords_unit = indict["coords_unit"]
        finder_finder = indict["finder_finder"]
        finder_prj = indict["finder_prj"]
        finder_label = indict["finder_label"]
        new(model_label, model_potential_unit, data_ids,
            A, a, b, c, x0, y0)
    end
end

function (model::MullerBrown)(coords::Array{Float64, 2},
                              properties::Array{String, 1})
    D, N = size(coords)
    Vk = zeros(N, 4)
    results = Dict()

    A, a, b, c, x0, y0 = model.A, model.a, model.b, model.c, model.x0, model.y0

    for i=1:N
        𝑥, 𝑦 = coords[:, i]
        Vk[i, :] = @. A * exp(a*(𝑥 - x0)^2 + b*(𝑥-x0)*(𝑦-y0) + c * (𝑦-y0)^2)
    end

    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    if "potential" in properties
        results[:potential] = sum(Vk, dims=2) / 100
    end

    if "gradients" in properties
        dV = zeros(Float64, 4)
        for i=1:N
            Fx = sum(@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0))
            Fy = sum(@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0))
            dV[i] = [Fx, Fy] / 100
        end
        results[:gradients] = dV
    end

    if "hessian" in properties
        nothing
    end

    return results
end


Database = include("./database.jl")
# abstract type Kernel end
# abstract type Mean end
Kernels = include("./kernels.jl")
Means = include("./means.jl")

# import Base: convert, promote_rule
# convert(::Tuple{Float64, Float64}, x::Array{Any, 1}) = (x[1], x[2])
# promote_rule(::Type{Array{Any, 1}}, ::Type{Tuple{Float64, Float64}}) = Tuple{Float64, Float64}

"""
Gaussian Process
"""
mutable struct Gaussian <: Model
    model_label::String
    model_potential_unit::String
    data_ids::Dict{String, Array{Int64, 1}}
    imgdata_filename::String
    kernel::Any
    mean::Any
    mean_type::String
    optimized::Bool
    hyperparameters::Dict{String, Float64}
    # hyperparameters_bounds::Dict{String, Tuple{Float64, Float64}}
    hyperparameters_bounds::Dict{String, Any}
    regression_method::String
    _cache::Dict

    function Gaussian(model_label::String, model_potential_unit::String,
                      data_ids::Dict{String, Array{Int64, 1}};
                      imgdata_filename="imagedata.db",
                      kernel="Standard",
                      mean="Mean",
                      mean_type="average",
                      optimized=false,
                      hyperparameters=nothing,
                      hyperparameters_bounds=nothing,
                      regression_method=nothing)
        kernel = getfield(Kernels, Symbol(kernel))(hyperparameters)
        mean = getfield(Means, Symbol(mean))(data_ids, mean_type)
        new(model_label, model_potential_unit, data_ids,
            imgdata_filename, kernel, mean, mean_type, optimized,
            hyperparameters, hyperparameters_bounds, regression_method, Dict())
    end
    function Gaussian(indict::Dict)
        model_label = indict["model_label"]
        model_potential_unit = indict["model_potential_unit"]
        data_ids = indict["data_ids"]
        imgdata_filename = get(indict, "imgdata_filename", "imagedata.db")
        kernel = get(indict, "kernel", "Standard")
        mean = get(indict, "mean", "Mean")
        mean_type = get(indict, "mean_type", "average")
        optimized = get(indict, "optimized", false)
        hyperparameters = get(indict, "hyperparameters", nothing)
        hyperparameters_bounds = get(indict, "hyperparameters_bounds", nothing)
        regression_method = get(indict, "regression_method", nothing)
        kernel = getfield(Kernels, Symbol(kernel))(hyperparameters)
        mean = getfield(Means, Symbol(mean))(data_ids, mean_type)
        new(model_label, model_potential_unit, data_ids,
            imgdata_filename, kernel, mean, mean_type, optimized,
            hyperparameters, hyperparameters_bounds, regression_method, Dict())
    end
end

function (model::Gaussian)(coords::Array{Float64, 2},
                           properties::Array{String, 1})
    D, N = size(coords)
    results = Dict()

    data = Database.read_data(model.imgdata_filename, model.data_ids)
    # coords : D x N, data['coords']: D x M
    Xn = shape_x(coords)
    Xm = shape_x(data["coords"])
    Y = shape_y(data)
    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    k = model.kernel
    m = model.mean
    # k, m = model.kernel, model.mean
    if !model.optimized || model._cache == Dict() ||
         get(model._cache, "K_y_inv", nothing) == nothing
            # model.hyperparameters = self.regression(paths)
            # k.hyperparameters.update(self.hyperparameters)
            # k.hyperparameters.update(self.regression(data))
            model._cache["K_y_inv"] = inv(k(Xm, Xm; noise=true))    # (D+1)Mx(D+1)M
            model.optimized = true
    end
    K_y_inv = model._cache["K_y_inv"]

    if "potential" in properties
        K_s = k(Xm, Xn; potential_only=true)
        potential = m(Xn;data=data, hess=false) .+ K_s' * K_y_inv * (Y .- m(Xm; data=data))
        results[:potential] = potential
    end

    if "potentials" in properties
        potentials = zeros(Float64, A)
        for i=1:A
            K_s = k(Xm, Xn; potential_only=true)
            potentials[i] = m(Xn, hess=false) + K_s' * K_y_inv * (Y .- m(Xm; data=data))
        end

        results[:potentials] = potentials
    end

    if "gradients" in properties
        dK_s = k(Xm, Xn; gradient_only=true)
        mu_f = dK_s' * K_y_inv * (Y .- m(Xm; data=data))
        # results[:gradients] = reshape(mu_f, D, N)
        results[:gradients] = permutedims(reshape(mu_f, N, D), [2, 1])
    end

    if "hessian" in properties
        ddK_s = k(Xm, Xn; hessian_only=true)
        H = ddK_s' * K_y_inv * (Y .- m(Xm; data=data))
        # results[:hessian] = reshape(H, D, D, N)
        results[:hessian] = permutedims(reshape(H, N, D, D), [3, 2, 1])
    end

    if "covariance" in properties
        K = k(Xn, Xn; orig=true)
        K_s = k(Xm, Xn)
        K_s_T = k(Xn, Xm)
        _cov = diag(K) .- diag((K_s_T * K_y_inv * K_s)[1:N, 1:N])
        cov = copy(_cov)
        flags = _cov .< 0
        cov[flags] .= 0.
        results[:covariance] =  1.96 .* sqrt.(cov) / 2
    end
    return results
end

function shape_x(coords)
    shape = size(coords) # D x M
    rank = ndims(coords)
    M = shape[end]
    D = rank == 2 ? shape[1] : shape[1] * shape[2]
    return reshape(coords, D, M)
end

function shape_y(data)
    V = data["potential"]   # M
    dV = data["gradients"]' # D x M -> M x D
    vec(hcat(V, dV))      # M + MxD -> M(1+D)
end

mutable struct AtomicGaussian <: Model
    kernel::Any
    kernel_hyperparameters::Any
    mean::Any
    mean_hyperparameters::Any
    imgdata_filename::String
    data_ids::Any
    optimized::Bool
    _cache::Dict

    function AtomicGaussian(;
        kernel="Standard", kernel_hyperparameters=nothing,
        mean="Zero", mean_hyperparameters=nothing,
        imgdata_filename="imgdata.db",
        data_ids=Dict("image"=>[1, 2]),
        optimized=false,
        kwargs...)

        kernel = getfield(Kernels, Symbol(kernel))()
        mean = getfield(Means, Symbol(mean))()
        new(kernel, kernel_hyperparameters, mean, mean_hyperparameters,
            imgdata_filename, data_ids, optimized, Dict())
    end
end


function (model::AtomicGaussian)(coords::Array{Float64, 3}, properties::Array{String, 1})
    # data::Dict{String, Any}, "coords", "potentials"...
    data = Database.read_data(model.imgdata_filename, model.data_ids)
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    𝐗, 𝐘 = data2𝐗𝐘(data)
    # 𝐗′::Array{Float64, 3};  3xAxN -> A x 3(A-1) x N

    A, M = size(𝐘)
    N = size(coords)[end]
    D = 3 * A

    𝐤, 𝐦 = model.kernel, model.mean

    results = Dict()

    𝐤.hyperparameters = _kernel_hyperparameters(model.kernel_hyperparameters)
    𝐦.hyperparameters = _mean_hyperparameters(model.mean_hyperparameters)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    if !model.optimized || model._cache == Dict() ||
         get(model._cache, "K_y_inv", nothing) == nothing
        model._cache["K_y_inv"] = zeros(Float64, A, M, M)
        model._cache["Λ"] = zeros(Float64, A, M)

        for a = 1:A
            Xm, Y = 𝐗[a, :, :], 𝐘[a, :]
            𝐤.idx, 𝐦.idx = a, a
            𝐊⁻¹ = inv(𝐤(Xm, Xm, true))              # M x M
            model._cache["K_y_inv"][a, :, :] = 𝐊⁻¹        # A x MxM
            model._cache["Λ"][a, :] = 𝐊⁻¹ * (Y .- 𝐦(Xm))  # A x M
        end
        model.optimized = true
    end
    𝚲 = model._cache["Λ"]

    """
    coords : 3 x A x N
    return : N; Potential or AxN if potentials are true
    """
    function 𝑓(coords; potentials=false)
        𝐗′ = coords2𝐗′(coords) # 3 x A x N -> A x 3(A-1) x N
        μ = zeros(Float64, A, N)
        for a = 1:A
            Xm, Xn, Y, Λ = 𝐗[a, :, :], 𝐗′[a, :, :], 𝐘[a, :], 𝚲[a, :, :]
            𝐤.idx, 𝐦.idx = a, a
            μ[a, :] = 𝐦(Xn) .+ 𝐤(Xm, Xn)' * Λ
        end
        if potentials
            return μ
        end
        return dropdims(sum(μ, dims=1), dims=1)
    end

    if "potential" in properties || "potentials" in properties
        potentials = zeros(Float64, A, N)
        μ = 𝑓(coords, potentials=true)
        results[:potentials] = μ
        results[:potential] = dropdims(sum(μ, dims=1), dims=1)
    end

    if "gradients" in properties || "hessian" in properties
        gradients = zeros(3, A, N)
        "hessian" in properties ? hessian = zeros(3, A, 3, A, N) : nothing
        X = reshape(coords, D, N) # 3 x A x N -> D x N
        h = 1e-4
        h² = h^2
        he(i) = begin
            temp = zeros(Float64, D, N)
            temp[i, :] .+= h
            return temp
        end
        f☾X☽ = get(results, :potential, 𝑓(X))
        f☾X⁺h☽ = zeros(Float64, D, N)
        for i = 1:D
            ai = ((i - 1) ÷ 3) + 1
            di = ((i - 1) % 3) + 1
            hei = he(i)
            f☾X⁺h☽[i, :] = 𝑓(X .+ hei)
            f☾X⁺hei☽ = f☾X⁺h☽[i, :]
            if "hessian" in properties for j = 1:i
                aj = ((j - 1) ÷ 3) + 1
                dj = ((j - 1) % 3) + 1
                hej = he(j)
                f☾X⁺hej☽ = f☾X⁺h☽[j, :]
                f☾X⁺hei⁺hej☽ = 𝑓(X .+ hei .+ hej)
                hessian[dj, aj, di, ai, :] = @. (f☾X⁺hei⁺hej☽-f☾X⁺hei☽-f☾X⁺hej☽+f☾X☽)/h²
                ####### I am that lazy
                if i != j
                    hessian[di, ai, dj, aj, :] = hessian[dj, aj, di, ai, :]
                end
            end end
            gradients[di, ai, :] = @. (f☾X⁺hei☽ - f☾X☽) / h
        end
        results[:gradients] = gradients
        "hessian" in properties ? results[:hessian] = hessian : nothing
    end

    if "covariance" in properties || "covariacnes" in properties
        covariances = zeros(Float64, A, N)
        K_y_inv = model._cache["K_y_inv"]
        𝐗′ = coords2𝐗′(coords) # 3 x A x N -> A x 3(A-1) x N
        for a = 1:A
            Xm, Xn = 𝐗[a, :, :], 𝐗′[a, :, :], 𝐘[a, :]
            𝐤.idx = a
            K = 𝐤(Xn, Xn)
            K_s = 𝐤(Xm, Xn)
            K_s_T = 𝐤(Xn, Xm)
            cov = diag(K) .- diag((K_s_T * K_y_inv[a, :, :] * K_s)[1:N, 1:N])
            covariances[a, :] = map(x -> max(x, 0), cov)
        end

        covariances = 1.96 .* sqrt.(covariances) / 2
        results[:covariances] = covariances
        # results[:covariance] = dropdims(sum(covariances, dims=1), dims=1)
        results[:covariance] = dropdims(maximum(covariances, dims=1), dims=1)
    end

    return results
end

"""
if A < 30
    3 x A x N -> A x 3(A-1) x N
"""
function coords2𝐗′(coords)
    if ndims(coords) == 2
        coords = reshape(coords, 3, size(coords)[1] ÷ 3, size(coords)[end])
    end
    # 3 x A x N
    d, A, N = size(coords)
    # A x 3 x (A-1) x N
    disp = ones(Float64, A, 3, (A-1), N)
    for i = 1:A
        for j = 1:A
            if i == j
                continue
            elseif i > j
                _j = j
            elseif i < j
                _j = j - 1
            end
            disp[i, :, _j, :] = coords[:, j, :] - coords[:, i, :]
        end
    end
    return reshape(disp, A, 3(A-1), N)
end

function data2𝐗𝐘(data)
    coords = data["coords"]
    if ndims(coords) == 2
        coords = reshape(coords, size(coords)[1] ÷ 3, 3, size(coords)[end])
        coords = permutedims(coords, [2, 1, 3])
    end
    # 3 x A x N
    d, A, N = size(coords)
    # A x 3 x (A-1) x N
    disp = ones(Float64, A, 3, (A-1), N)
    for i = 1:A
        for j = 1:A
            if i == j
                continue
            elseif i > j
                _j = j
            elseif i < j
                _j = j - 1
            end
            disp[i, :, _j, :] = coords[:, j, :] - coords[:, i, :]
        end
    end
    # (A x 3(A-1) x M, A x M)
    return (reshape(disp, A, 3(A-1), N), data["potentials"])
end


"""
hyperparameters = {'numbers': [28, 79, 78],
                       sig_28: 1, sig_79: 1, sig_78: 1,
                       noise_28: 1e-4, noise_79: 1e-4, noise_78: 1e-4,
                       (28, 28): l1, (28, 79): l2, (28, 78): l3,
                       (79, 79): l4, (79, 78): l5, (78, 78): l6}
"""
function _kernel_hyperparameters(hyperparameters::Dict)
    numbers = hyperparameters["numbers"]
    A = size(numbers)[1]
    hyper_kernel = zeros(Float64, A, 3(A-1) + 2)
    for (i, ni) in enumerate(numbers)
        hyper_kernel[i, end-1] = hyperparameters["sig_$ni"][i]
        hyper_kernel[i, end] = hyperparameters["noise_$ni"][i]
        masked = numbers[[i!=j for j in 1:A]]
        for (j, nj) in enumerate(masked)
            ll = hyperparameters["ll_$nj"][i]
            hyper_kernel[i, 3(j-1)+1:3(j)] .= ll
        end
    end
    # A x A(A-1) + 2
    return hyper_kernel
end

"""
hyperparameters : A x 3(A-1) + 2
numbers : A
return : Dict
"""
function _kernel_hyperparameters(hyperparameters::Array, numbers::Array)
    newparameters = Dict()
    newparameters["numbers"] = numbers
    A = length(numbers)
    species = Set(numbers)

    for ni in species
        newparameters["sig_$ni"] = zeros(Float64, A)
        newparameters["noise_$ni"] = zeros(Float64, A)
        newparameters["ll_$ni"] = zeros(Float64, A)
    end

    for (i, ni) in enumerate(numbers)
        newparameters["sig_$ni"][i] = hyperparameters[i, end-1]
        newparameters["noise_$ni"][i] = hyperparameters[i, end]
        masked = numbers[[i!=j for j in 1:A]]
        for (j, nj) in enumerate(masked)
            newparameters["ll_$nj"][i] = hyperparameters[i, 3*j]
        end
    end
    return newparameters
end


function _kernel_param_dict2list(hyperparameters)
    numbers = hyperparameters["numbers"]
end


"""
data :  A x M potentials array
return : A    average potential array
"""
function _mean_hyperparameters(hyperparameters)
    return hyperparameters
end

end
