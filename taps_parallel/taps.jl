module Taps

abstract type Coords{T, N} <: AbstractArray{T, N} end
abstract type Model end
abstract type Finder end

export Paths, get_displacements, get_momentum, get_kinetics
export Coords, Cartesian, Atomic, Fouriered
export Model, Gaussian

mutable struct Paths
    coord::Coords
    prefix::String; directory;
    model::Model
    finder; imgdata; tag
    cache
end

function get_displacements(paths::Paths, args...; kwargs...)
    get_displacements(paths.model, path, args...; kwargs...)
end

function get_momentum(paths::Paths, args...; kwargs...)
    get_momentum(paths.model, args...; kwargs...)
end

function get_kinetic_energy(paths::Paths, args...; kwargs...)
    get_kinetic_energy(paths.model, args...; kwargs...)
end

function get_kenetic_energy_graident(paths::Paths, args...; kwargs...)
    get_kenetic_energy_graident(paths.model, args...; kwargs...)
end

function get_velocity(paths::Paths, args...; kwargs...)
    get_velocity(paths.model, args...; kwargs...)
end

function get_acceleration(paths::Paths, args...; kwargs...)
    get_acceleration(paths.model, args...; kwargs...)

end
get_accelerations = get_acceleration

function get_mass(paths::Paths, args...; kwargs...)
    get_mass(paths.model, args...; kwargs...)

end

function get_effective_mass(paths::Paths, args...; kwargs...)
    get_effective_mass(paths.model, args...; kwargs...)
end

""" Directly calls the :meth:`get_properties` in ``paths.model``"""
function get_properties(paths::Paths, args...; kwargs...)
    get_properties(paths.model, args...; kwargs...)
end

""" Calculate potential( energy) """
function get_potential_energy(paths::Paths, args...; kwargs...)
    get_potential_energy(paths.model, args...; kwargs...)
end

""" Calculate potential """
function get_potential(paths::Paths, args...; kwargs...)
    get_potential(paths.model, args...; kwargs...)
end

""" Equivalanet to Calculate potentials"""
function get_potential_energies(paths::Paths, args...; kwargs...)
    get_potential_energies(paths.model, args...; kwargs...)
end

""" Calculate potentials, individual energy of each atoms"""
function get_potentials(paths::Paths, args...; kwargs...)
    get_potentials(paths.model, args...; kwargs...)
end

""" Calculate - potential gradient"""
function get_forces(paths::Paths, args...; kwargs...)
    get_forces(paths.model, args...; kwargs...)
end

""" Calculate potential gradient"""
function get_gradient(paths::Paths, args...; kwargs...)
    get_gradient(paths.model, args...; kwargs...)
end

""" Calculate potential gradient(s)"""
function get_gradients(paths::Paths, args...; kwargs...)
    get_gradients(paths.model, args...; kwargs...)
end

""" Calculate Hessian of a potential"""
function get_hessian(paths::Paths, args...; kwargs...)
    return get_hessian(paths.model, args...; kwargs...);
end

""" Calculate kinetic + potential energy"""
function get_total_energy(paths::Paths, args...; kwargs...)
    V = get_potential_energy(paths.model, args...; kwargs...)
    T = get_kinetic_energy(paths.model, args...; kwargs...)
    return V + T
end

""" Calculate covariance. It only applies when potential is guassian"""
function get_covariance(paths::Paths, args...; kwargs...)
    get_covariance(paths.model, args...; kwargs...)
end

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
function get_data(paths::Paths, args...; kwargs...)
    get_data(paths.model, args...; kwargs...)
end

include("./utils.jl")
include("./coordinates/trasnformations.jl")
include("./coordinates/coordinate.jl")

include("./database.jl")
include("./models/models.jl")

end
