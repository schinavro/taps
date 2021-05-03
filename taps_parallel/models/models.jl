
export MullerBrown

@inline get_properties(model::Model, paths::Paths, properties::String, args...; kwargs...) = model(paths, paths.coords, [properties], args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::Array{String}, args...; kwargs...) = model(paths, paths.coords, properties, args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::String, coords::Coords, args...; kwargs...) = model(paths, coords, [properties], args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::Array{String}, coords::Coords, args...; kwargs...) = model(paths, coords, properties, args...; kwargs...)

@inline get_displacements(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "displacements", args...; kwargs...)
@inline get_momentum(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "momoentum", args...; kwargs...)
@inline get_kinetic_energy(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "kinetic_energy", args...; kwargs...)
@inline get_kinetic_energy_gradient(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "kinetic_energy_gradient", args...; kwargs...)
@inline get_velocity(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "velocity", args...; kwargs...)
@inline get_acceleration(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "acceleration", args...; kwargs...)

function get_mass(model::Model, paths::Paths, args...; kwargs...)
    get_properties(model, paths, "mass", args...; kwargs...)
end
function get_effective_mass(paths::Paths, args...; kwargs...)
    get_effective_mass(paths.model, args...; kwargs...)
end

""" Calculate potential( energy) """
@inline get_potential(model::Model, paths::Paths, args...; kwargs...) = get_properties(model, paths, "potential", args...; kwargs...)
@inline get_potentials(model::Model, paths::Paths, args...; kwargs...) = get_properties(model, paths, "potentials", args...; kwargs...)
@inline get_gradients(model::Model, paths::Paths, args...; kwargs...) = get_properties(model, paths, "gradients", args...; kwargs...)
@inline get_hessian(model::Model, paths::Paths, args...; kwargs...) = get_properties(model, paths, "hessain", args...; kwargs...)
""" Calculate covariance. It only applies when potential is guassian"""
@inline get_covariance(model::Model, paths::Paths, args...; kwargs...) = get_properties(model, paths, "covariance", args...; kwargs...)

struct MullerBrown <: Model
    A::Array{Float64, 1}
    a::Array{Float64, 1}
    b::Array{Float64, 1}
    c::Array{Float64, 1}
    x0::Array{Float64, 1}
    y0::Array{Float64, 1}
end

MullerBrown(;
   A=[-200., -100., -170, 15], a=[-1, -1, -6.5, 0.7], b=[0., 0., 11., 0.6],
   c=[-10, -10, -6.5, 0.7], x0=[1, 0, -0.5, -1], y0=[0, 0.5, 1.5, 1]) =
                                                 MullerBrown(A, a, b, c, x0, y0)

@inline (model::MullerBrown)(paths::Paths, coords::Coords, properties::Array{String, 1}) = (model::MullerBrown)(coords::Coords, properties::Array{String, 1})
@inline (model::MullerBrown)(coords::Coords, properties::Array{String, 1}) = (model::MullerBrown)(convert(Cartesian{eltype(coords), 2}, coords), properties::Array{String, 1})
function (model::MullerBrown)(coords::Cartesian{T, 2}, properties::Array{String, 1}) where {T<:Number}
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
