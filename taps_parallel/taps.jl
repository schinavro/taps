module Taps
#import Base.getproperty, Base.setproperty!
abstract type Coords{T, N} <: AbstractArray{T, N} end
abstract type Model end
abstract type Database end
# abstract type Finder end

export Paths, getproperty, setproperty!, get_displacements, get_momentum, get_kinetics
export Coords, Cartesian, Atomic, Fouriered
export Model, Gaussian
export ImageData

mutable struct Paths
    coords::Coords
    model::Model
    prefix::String
    directory::String
    imgdata::Database
    tag::Dict
    cache::Dict
end

function Paths(;coords="Cartesian", coords_kwargs=Dict(), model="MullerBrown", model_kwargs=Dict(),
       prefix="paths", directory=".", imgdata="ImgData", imgdata_kwargs=Dict(),
       tag=Dict(), cache=Dict())
    coords = getfield(Main, Symbol(coords))(;coords_kwargs...)
    model = getfield(Main, Symbol(model))(;model_kwargs...)
    imgdata = getfield(Main, Symbol(imgdata))(;imgdata_kwargs...)
    Paths(coords, model, prefix, directory, imgdata, tag, cache)
end

function Base.getproperty(paths::Paths, name::Symbol, x)
    if length(String(name)) > 7 && String(name)[end-6:end] == "_kwargs"
        name = Symbol(String(name)[1:end-7])
        field = getfield(paths, name)
        x = x == nothing ? Dict([(f, nothing) for f in fieldnames(field)]) : x
        ans = Dict()
        for (key, value) in pairs(x)
            ans[key] = getproperty(field, key, value)
        end
        return ans
    elseif parentmodule(fieldtype(typeof(paths), name)) ∉ [Base, Core]
        typename = typeof(getfield(paths, name))
        return "$typename"
    else
        return getfield(paths, name)
    end
end

function Base.setproperty!(obj::T, name::Symbol, x) where {T<:Union{Paths, Coords, Database, Model}}
    if length(String(name)) > 7 && String(name)[end-6:end] == "_kwargs"
        name = Symbol(String(name)[1:end-7])
        field = getfield(obj, name)
        for (key, value) in pairs(x)
            setproperty!(field, key, value)
        end
    elseif parentmodule(fieldtype(typeof(obj), name)) ∉ [Base, Core]
        # Change the class
        if typeof(x) == String
            newfield = getfield(Main, Symbol(x))()
            setfield!(obj, name, newfield)
        # Complex update
        elseif typeof(x) <: Dict
            # :coords => {kwargs...} case. Update kwargs.
            if get(x, :name, nothing) == nothing
                field = getfield(obj, name)
                for (key, value) in pairs(x)
                    setproperty!(field, name, x)
                end
            # :coords => {:name=> "Sine", :kwargs => {...}} case. Reconstruct coords
            else
                newname = x[:name]
                new_kwargs = x[:kwargs]
                newfield = getfield(Main, Symbol(newname))(;new_kwargs...)
                setfield!(obj, name, newfield)
            end
        end
    else
        setfield!(obj, name, x)
    end
end

@inline get_displacements(paths::Paths, args...; kwargs...) = get_displacements(paths.model, paths, args...; kwargs...)
@inline get_momentum(paths::Paths, args...; kwargs...) = get_momentum(paths.model, paths, args...; kwargs...)
@inline get_kinetic_energy(paths::Paths, args...; kwargs...) = get_kinetic_energy(paths.model, paths, args...; kwargs...)
@inline get_kenetic_energy_graident(paths::Paths, args...; kwargs...) = get_kenetic_energy_graident(paths.model, paths, args...; kwargs...)
@inline get_velocity(paths::Paths, args...; kwargs...) = get_velocity(paths.model, paths, args...; kwargs...)
@inline get_acceleration(paths::Paths, args...; kwargs...) = get_acceleration(paths.model, paths, args...; kwargs...)
const get_accelerations = get_acceleration
@inline get_mass(paths::Paths, args...; kwargs...) = get_mass(paths.model, paths, args...; kwargs...)
@inline get_effective_mass(paths::Paths, args...; kwargs...) = get_effective_mass(paths.model, paths, args...; kwargs...)
""" Directly calls the :meth:`get_properties` in ``paths.model``"""
@inline get_properties(paths::Paths, args...; kwargs...) = get_properties(paths.model, paths, args...; kwargs...)
""" Calculate potential """
@inline get_potential(paths::Paths, args...; kwargs...) = get_potential(paths.model, paths, args...; kwargs...)
""" Calculate potential( energy) """
const get_potential_energy = get_potential
""" Calculate potentials, individual energy of each atoms"""
@inline get_potentials(paths::Paths, args...; kwargs...) = get_potentials(paths.model, paths, args...; kwargs...)
""" Equivalanet to Calculate potentials"""
const get_potential_energies = get_potentials
""" Calculate potential gradient(s)"""
@inline get_gradients(paths::Paths, args...; kwargs...) = get_gradients(paths.model, paths, args...; kwargs...)
""" Calculate - potential gradient"""
const get_forces = get_gradients
""" Calculate potential gradient"""
const get_gradient = get_gradients

""" Calculate Hessian of a potential"""
@inline get_hessian(paths::Paths, args...; kwargs...) = get_hessian(paths.model, paths, args...; kwargs...);
""" Calculate kinetic + potential energy"""
function get_total_energy(paths::Paths, args...; kwargs...)
    V = get_potential_energy(paths.model, paths, args...; kwargs...)
    T = get_kinetic_energy(paths.model, paths, args...; kwargs...)
    return V + T
end

""" Calculate covariance. It only applies when potential is guassian"""
@inline get_covariance(paths::Paths, args...; kwargs...) = get_covariance(paths.model, paths, args...; kwargs...)

""" Get index of highest potential energy simplified"""
function get_higest_energy_idx(paths::Paths)
    E = get_potential_energy(path)
    return argmax(E)
end

""" Get index of lowest of covariance simplified"""
function get_lowest_confident_idx(paths::Paths)
    cov = get_covariance(path)
    return argmax(diag(cov))
end

"""" list of int; Get rowid of data"""
@inline get_data(paths::Paths, args...; kwargs...) = get_data(paths.model::Model, paths, args...; kwargs...)

include("./utils/utils.jl")
include("./coordinates/transformations.jl")
include("./coordinates/coordinates.jl")
include("./database.jl")
include("./models/models.jl")

end
