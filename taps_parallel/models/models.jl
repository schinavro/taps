
export MullerBrown

function Base.getproperty(model::Model, name::Symbol, x)
    if length(String(name)) > 7 && String(name)[end-6:end] == "_kwargs"
        name = Symbol(String(name)[1:end-7])
        field = getfield(model, name)
        x = x == nothing ? Dict([(f, nothing) for f in fieldnames(field)]) : x
        ans = Dict()
        for (key, value) in pairs(x)
            ans[key] = getproperty(field, key, value)
        end
        return ans
    elseif parentmodule(fieldtype(typeof(model), name)) ∉ [Base, Core]
        typename = typeof(getfield(model, name))
        return "$typename"
    else
        return getfield(model, name)
    end
end

@inline get_properties(model::Model, paths::Paths, properties::String, args...; kwargs...) = model(paths, paths.coords, [properties], args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::Array{Any, 1}, args...; kwargs...) = model(paths, paths.coords, properties, args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::String, coords::Coords, args...; kwargs...) = model(paths, coords, [properties], args...; kwargs...)
@inline get_properties(model::Model, paths::Paths, properties::Array{Any, 1}, coords::Coords, args...; kwargs...) = model(paths, coords, properties, args...; kwargs...)

@inline get_kinetics(model::Model, paths::Paths, properties::String, args...; kwargs...) = paths.coords(paths, paths.coords, [properties], args...; kwargs...)
@inline get_kinetics(model::Model, paths::Paths, properties::Array{Any, 1}, args...; kwargs...) = paths.coords(paths, paths.coords, properties, args...; kwargs...)
@inline get_kinetics(model::Model, paths::Paths, properties::String, coords::Coords, args...; kwargs...) = paths.coords(paths, coords, [properties], args...; kwargs...)
@inline get_kinetics(model::Model, paths::Paths, properties::Array{Any, 1}, coords::Coords, args...; kwargs...) = paths.coords(paths, coords, properties, args...; kwargs...)

@inline get_distances(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "displacements", args...; kwargs...)
@inline get_momentums(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "momoentum", args...; kwargs...)
@inline get_kinetic_energies(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "kinetic_energy", args...; kwargs...)
@inline get_kinetic_energy_gradients(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "kinetic_energy_gradient", args...; kwargs...)
@inline get_velocities(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "velocity", args...; kwargs...)
@inline get_accelerations(model::Model, paths::Paths, args...; kwargs...) = get_kinetics(model, paths, "acceleration", args...; kwargs...)

function get_masses(model::Model, paths::Paths, args...; kwargs...)
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
    results::Dict
end

MullerBrown(;
   A=[-2., -1., -1.7, 0.15], a=[-1, -1, -6.5, 0.7], b=[0., 0., 11., 0.6],
   c=[-10, -10, -6.5, 0.7], x0=[1, 0, -0.5, -1], y0=[0, 0.5, 1.5, 1],
   results=Dict()) = MullerBrown(A, a, b, c, x0, y0, results)

@inline (model::MullerBrown)(paths::Paths, coords::Coords, properties::Array{Any, 1}) = (model::MullerBrown)(coords::Coords, properties::Array{Any, 1})
@inline (model::MullerBrown)(coords::Coords, properties::Array{Any, 1}) = (model::MullerBrown)(convert(Cartesian{eltype(coords), 2}, coords), properties::Array{Any, 1})

"""
coords : Cartesian; N x 2
"""
function (model::MullerBrown)(coords::Cartesian{T, 2}, properties::Array{Any, 1}) where {T<:Number}
    N, D = size(coords)
    results = Dict()

    # 4 -> 1x4
    A, a, b, c = model.A[nax, :], model.a[nax, :], model.b[nax, :], model.c[nax, :]
    x0, y0 = model.x0[nax, :], model.y0[nax, :]
    # for i=1:N
    𝑥 = coords[:, 1]
    𝑦 = coords[:, 2]
    # Vk = zeros(N, 4)

    𝑥_x0, 𝑦_y0 = 𝑥 .- x0, 𝑦 .- y0
    Vk = @. A * exp(a*(𝑥 - x0)^2 + b*(𝑥-x0)*(𝑦-y0) + c * (𝑦-y0)^2)
    # end

    # "potential" in properties ? results["potential"] = sum(Vk) / 100 : nothing
    if "potential" in properties
        results[:potential] = sum(Vk, dims=2)
    end

    if "gradients" in properties
        dV = zeros(Float64, 4)

        Fx = dropdims(sum((@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0)), dims=2), dims=2)
        Fy = dropdims(sum((@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0)), dims=2), dims=2)
        dV = [Fx, Fy]
        results[:gradients] = dV
    end

    if "hessian" in properties
        nothing
    end

    if "mass" in properties
        results[:mass] = ones(N)
    end

    merge!(model.results, results)

    return results
end

include("./ase.jl")
