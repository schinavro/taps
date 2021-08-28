module Projectors
using ..Coords: AbstractCoords

abstract type AbstractProjector end

"""
Coordinate transformation module
"""
struct Projector{Domain, Codomain} where {Domain, Codomain <: AbstractCoords}
    pipeline::Union{Projector, Nothing}
    domain::Type{Domain}
    codomain::Type{Codomain}
end

prj(x) = 2x
inv(x) = x/2

struct Mask <: AbstractProjector

end

end
