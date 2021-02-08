module Models

export MullerBrown, Gaussian

using LinearAlgebra: diag

struct MullerBrown
    model_label::String
    model_potential_unit::String
    model_data_ids::Dict{String, Array{Int64, 1}}
    A::Array{Float64, 1}
    a::Array{Float64, 1}
    b::Array{Float64, 1}
    c::Array{Float64, 1}
    x0::Array{Float64, 1}
    y0::Array{Float64, 1}
    function MullerBrown(model_label::String, model_potential_unit::String,
                         model_data_ids::Dict{String, Array{Int, 1}};
                         A::Array{Float64, 1}=[-200., -100., -170, 15],
                         a::Array{Float64, 1}=[-1, -1, -6.5, 0.7],
                         b::Array{Float64, 1}=[0., 0., 11., 0.6],
                         c::Array{Float64, 1}=[-10, -10, -6.5, 0.7],
                         x0::Array{Float64, 1}=[1, 0, -0.5, -1],
                         y0::Array{Float64, 1}=[0, 0.5, 1.5, 1]
                         )
        new(model_label, model_potential_unit, model_data_ids,
            A, a, b, c, x0, y0)
    end
end

function (model::MullerBrown)(coords::Array{Float64, 2},
                              properties::Array{String, 1})
    N, D = size(coords)
    Vk = zeros(N, 4)
    results = Dict()

    A, a, b, c, x0, y0 = model.A, model.a, model.b, model.c, model.x0, model.y0

    for i=1:N
        𝑥, 𝑦 = coords[i, :]
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

"""
Gaussian Process
"""
mutable struct Gaussian
    model_label::String
    model_potential_unit::String
    model_data_ids::Dict{String, Array{Int64, 1}}
    imgdata_filename::String
    kernel::Any
    mean::Any
    mean_type::String
    optimized::Bool
    hyperparameters::Dict{String, Float64}
    hyperparameters_bounds::Dict{String, Tuple{Float64, Float64}}
    regression_method::String
    _cache::Dict

    function Gaussian(model_label::String, model_potential_unit::String,
                      model_data_ids::Dict{String, Array{Int64, 1}};
                      imgdata_filename="imagedata.db",
                      kernel="Standard",
                      mean="Mean",
                      mean_type="average",
                      optimized=false,
                      hyperparameters=nothing,
                      hyperparameters_bounds=nothing,
                      regression_method=nothing)
        kernel = getfield(Kernels, Symbol(kernel))(hyperparameters)
        mean = getfield(Means, Symbol(mean))(model_data_ids, mean_type)
        new(model_label, model_potential_unit, model_data_ids,
            imgdata_filename, kernel, mean, mean_type, optimized,
            hyperparameters, hyperparameters_bounds, regression_method, Dict())
    end
end

function (model::Gaussian)(coords::Array{Float64, 2},
                           properties::Array{String, 1})
    N, D = size(coords)
    results = Dict()

    print("Data read from", model.imgdata_filename)
    println("recieved: ", model.model_data_ids["image"])
    data = Database.read_data(model.imgdata_filename, model.model_data_ids)
    # coords : N x D, data['coords']: D x M
    println("Size of coords, data['coords']: ", size(coords), size(data["coords"]))
    Xn = shape_x(coords)
    Xm = shape_x(data["coords"])
    Y = shape_y(data)
    println("Size of Xm, Xn, Y: ", size(Xm), size(Xn), size(Y))
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
        println(typeof(Xm), typeof(Xn))
        K_s = k(Xm, Xn; potential_only=true)
        println(size(K_s), size(K_y_inv), size(Y .- m(Xm; data=data)),
                size(m(Xn;data=data, hess=false)))
        potential = m(Xn;data=data, hess=false) .+ K_s' * K_y_inv * (Y .- m(Xm; data=data))
        results[:potential] = potential
    end

    if "potentials" in properties
        println("Calculating potentials")
        potentials = zeros(Float64, A)
        for i=1:A
            K_s = k(Xm, Xn; potential_only=true)
            potentials[i] = m(Xn, hess=false) + K_s' * K_y_inv * (Y .- m(Xm; data=data))
        end

        results[:potentials] = potentials
    end

    if "gradients" in properties
        println("Calculating gradient")
        dK_s = k(Xm, Xn; gradient_only=true)
        mu_f = dK_s' * K_y_inv * (Y .- m(Xm; data=data))
        results[:gradients] = permutedims(reshape(mu_f, D, N), [2, 1])
    end

    if "hessian" in properties
        println("Calculating hessian")
        K_s = k(Xm, Xn; hessian_only=true)
        H = K_s' * K_y_inv * (Y .- m(Xm; data=data))
        results[:hessian] = permutedims(reshape(H, D, D, N), [3, 2, 1])
    end

    if "covariance" in properties
        println("Calculating covariance")
        K = k(Xn, Xn; orig=true)
        K_s = k(Xm, Xn)
        K_s_T = k(Xn, Xm)
        cov = diag(K) .- diag((K_s_T * K_y_inv * K_s)[1:N, 1:N])
        cov[cov .< 0] .= 0.
        results[:covariance] =  1.96 .* sqrt.(cov) / 2
    end
    return results
end

function regression(kernel, mean, data)
    k, m = kernel, mean
    X = data["X"]
    hyperparam =
    K = k(X, X, noise=true, hyperparameters=hyperparam)
    detK = diagonal(cholesky(K))
    dataK = linalg.det(K)
end

struct likelihood
    k
    X
    Y_m
    function likelihood(k, m, X, Y)
        Y_m = Y - m(X)
        new(k, X, Y_m)
    end
end

function (obj::likelihood)log_likelihood(hyperparameters)
    k, X, Y_m = obj.k, obj.X, obj.Y_m
    K = k(X, X, orig=true, hyperparameters=hyperparameters)
    detK = diagonal(cholesky(K))
    return sum(log(detK)) + 0.5 * (Y_m' * (inv(K) * Y_m))
end

function (obj::likelihood)gradient_likelihood(hyperparameters)
    k, X, Y_m = obj.k, obj.X, obj.Y_m
    K = k(X, X, hyperparameters)
    return log_detK + 0.5 * (Y_m' * (inv(K) * Y_m))
end

function (obj::likelihood)pseudo_gradient_likelihood(hyperparameters)
    k, X, Y_m = obj.k, obj.X, obj.Y_m
    K = k(X, X, orig=true, hyperparameters=hyperparameters)
    detK = diagonal(cholesky(K))
    return sum(log(detK)) + 0.5 * (Y_m' * (inv(K) * Y_m))
end

function shape_x(coords)
    shape = size(coords) # M x D
    rank = length(shape)
    M = shape[1]
    D = rank == 2 ? shape[2] : shape[2] * shape[3]
    return reshape(coords, M, D)
end

function shape_y(data)
    V = data["potential"] # M
    dV = data["gradients"] # M x D
    vec(hcat(V, dV))
end


end
