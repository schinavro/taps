module Models

export MullerBrown, Gaussian

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
                         model_data_ids::Dict{String, Array{Int64, 1}};
                         A::Array{Float64, 1}=[-200., -100., -170, 15],
                         a::Array{Float64, 1}=[-1, -1, -6.5, 0.7],
                         b::Array{Float64, 1}=[0., 0., 11., 0.6],
                         c::Array{Float64, 1}=[-10, -10, -6.5, 0.7],
                         x0::Array{Float64, 1}=[1, 0, -0.5, -1],
                         y0::Array{Float64, 1}=[0, 0.5, 1.5, 1]
                         )
        new(model_label, model_potential_unit, model_data_ids,
            A=A, a=a, b=b, c=c, x0=x0, y0=y0)
    end
end

function (model::MullerBrown)(coords::Array{Float64, 2},
                              properties::Array{String, 1})
    N = size(coords)
    Vk = zeros(N, 4)
    results = Dict()

    comm, rank, nprc = model.comm, model.rank, model.nprc
    A, a, b, c, x0, y0 = model.A, model.a, model.b, model.c, model.x0, model.y0

    Vk = zeros(Float64, 4)
    for i=1:N
        𝑥, 𝑦 = coords[i]
        Vk[i] = @. A * exp(a*(𝑥 - x0)^2 + b*(𝑥-x0)*(𝑦-y0) + c * (𝑦-y0)^2)
    end

    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    if "potential" in properties
        results["potentials"] = sum(Vk, axis=2) / 100
    end

    if "gradients" in properties
        dV = zeros(Float64, 4)
        for i=1:N
            Fx = sum(@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0))
            Fy = sum(@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0))
            dV[i] = [Fx, Fy] / 100
        end
        results["gradients"] = dV
    end

    if "hessian" in get_properties
    end

    return results
end


struct Gaussian
    model_label::String
    model_potential_unit::String
    model_data_ids::Dict{String, Array{Int64, 1}}
    kernel::String,
    mean::String,
    mean_type::String,
    optimized::Bool,
    hyperparameters::Dict{String, Array{Float64, 1}},
    hyperparameters_bounds::Dict{String, Tuple{Float64}},
    regression_method::String,
    likelihood_type::String

    k, m = model.kernel, model.mean

    function Gaussian(model_label::String, model_potential_unit::String,
                      model_data_ids::Dict{String, Array{Int64, 1}};
                      kernel::String,
                      mean::String,
                      mean_type::String,
                      optimized::Bool,
                      hyperparameters::Dict{String, Array{Float64, 1}},
                      hyperparameters_bounds::Dict{String, Tuple{Float64}},
                      regression_method::String,
                      likelihood_type::String)
        new(model_label, model_potential_unit, model_data_ids)
    end
end

function (model::Gaussian)(coords::Array{Float64, 2},
                           properties::Array{String, 1})

    N = size(coords)
    comm, rank, nprc = model.comm, model.rank, model.nprc
    data_ids = model.model_data_ids

    results = Dict()

    data = get_data(data_ids)


    if !optimized
        hyperparameters = regression(data, model.hyperparameters,
                                     model.likelihood_type,
                                     model.hyperparameters_bounds,
                                     model.regression_method)
    end


    Vk = zeros(Float64, 4)
    for i=1:N
        𝑥, 𝑦 = coords[i]
        k[i] = @. kernel
    end

    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    if "potential" in properties
        results["potentials"] = sum(Vk, axis=2) / 100
    end

    if "gradients" in properties
        dV = zeros(Float64, 4)
        for i=1:N
            Fx = sum(@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0))
            Fy = sum(@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0))
            dV[i] = [Fx, Fy] / 100
        end
        results["gradients"] = dV
    end

    if "hessian" in get_properties
    end

    return results
end

function (model::Gaussian)(coords::Array{Float64, 3},
                           properties::Array{String, 1})

    N = size(coords)
    comm, rank, nprc = model.comm, model.rank, model.nprc
    data_ids = model.model_data_ids

    results = Dict()

    data = get_data(data_ids)

    if !optimized
        hyperparameters = regression(data, model.hyperparameters,
                                     model.likelihood_type,
                                     model.hyperparameters_bounds,
                                     model.regression_method)
    end


    Vk = zeros(Float64, 4)
    for i=1:N
        𝑥, 𝑦 = coords[i]
        k[i] = @. kernel
    end

    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    if "potential" in properties
        results["potentials"] = sum(Vk, axis=2) / 100
    end

    if "gradients" in properties
        dV = zeros(Float64, 4)
        for i=1:N
            Fx = sum(@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0))
            Fy = sum(@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0))
            dV[i] = [Fx, Fy] / 100
        end
        results["gradients"] = dV
    end

    if "hessian" in get_properties
    end

    return results
end

end
