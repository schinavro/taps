using LinearAlgebra
using NearestNeighbors

include("./descriptors.jl")

Base.size(A::Coords) = size(A.coords)
Base.getindex(A::Coords, I::Vararg{Int, N}) where {N} = getindex(A.coords, I...)
Base.setindex!(A::Coords, v, I::Vararg{Int, N}) where {N} = setindex!(A.coords, v, I...)
Base.isapprox(c1::T, c2::T; kwargs...) where {T<:Coords} = isapprox(c1.coords, c2.coords; kwargs...) && isapprox(c1.epoch, c2.epoch; kwargs...) && c1.unit == c2.unit

function Base.getproperty(coords::Coords, name::Symbol, x)
    if length(String(name)) > 7 && String(name)[end-6:end] == "_kwargs"
        name = Symbol(String(name)[1:end-7])
        field = getfield(coords, name)
        x = x == nothing ? Dict([(f, nothing) for f in fieldnames(field)]) : x
        ans = Dict()
        for (key, value) in pairs(x)
            ans[key] = getproperty(field, key, value)
        end
        return ans
    elseif parentmodule(fieldtype(typeof(coords), name)) ∉ [Base, Core]
        typename = typeof(getfield(coords, name))
        return "$typename"
    else
        return getfield(coords, name)
    end
end

function Base.getproperty(obj::T, sym::Symbol) where {T <: Coords}
    if sym == :D
        return prod(size(obj.coords)[2:end])
    elseif sym == :N
        return size(obj.coords)[1]
    elseif sym == :A
        D = prod(size(obj.coords)[2:end])
        if D % 3 == 0
            return max(D ÷ 3, 1)
        end
        return 1
    else
        return getfield(obj, sym)
    end
end

struct Cartesian{T, N} <: Coords{T, N}
    coords::Array{T, N}
    epoch::Real
    unit::String
    cache::Dict
end

Base.similar(A::Cartesian, ::Type{T}, dims::Dims) where {T} = Cartesian{T, length(dims)}(similar(A.coords, T, dims), A.epoch, A.unit, A.cache)
Base.convert(::Type{Cartesian{T, 2}}, coords::Cartesian{T, N}) where {T, N} = Cartesian{T, 2}(reshape(coords.coords, coords.N, prod(size(coords)[2:end])), coords.epoch, coords.unit, coords.cache)

@inline Cartesian(;coords=zeros(2, 1), epoch=1., unit="Å", cache=Dict()) = Cartesian(coords, epoch, unit, cache)
@inline Cartesian(A::Array{T, N}, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Cartesian{T, N}(A, epoch, unit, cache)
@inline Cartesian{T}(A::Array{T, N}, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Cartesian{T, N}(A, epoch, unit, cache)
@inline Cartesian{T, N}(A::Array{T, N}) where {T<:Number, N} = Cartesian{T, N}(A, 1., "Å", Dict())

# @inline Cartesian{T, 3}(A::Cartesian{T, 2}) where {T<:Number, N} = Cartesian{T, 2}(reshape(coords.coords, coords.N, prod(size(coords)[2:end])), A.epoch, A.unit)
# @inline Cartesian{T, 2}(A::Cartesian{T, 3}) where {T<:Number, N} = Cartesian{T, 2}(reshape(coords.coords, coords.N, prod(size(coords)[2:end])), A.epoch, A.unit)
"""Return displacements of each steps.
Get coords(array) and return length N array. Useful for plotting E/dist

Parameters
----------
coords : array
    size of NxD or NxAx3
epoch : float
    total time spend during transition
index : slice obj; Default `np.s_[:]`
    Choose the steps want it to be returned. Default is all steps.
"""
function get_displacements(coords::Cartesian; epoch=nothing, index=1⁝0)
    axis = 1
    p = copy(coords.coords)
    d = map(norm, eachslice(diff(p, dims=axis), dims=axis))
    d = cat(0., d, dims=axis)
    return accumulate(sum, d, dims=1)[index]
end

""" Return velocity at each step
Get coords and return DxN or 3xAxN array, two point moving average.

:math:`v[i] = (x[i+1] - x[i]) / dt`

Parameters
----------
coords : array
    size of DxN or 3xAxN
epoch : float
    total time step.
index : slice obj; Default `np.s_[:]`
    Choose the steps want it to be returned. Default is all steps.
"""
function get_velocity(coords::Cartesian; epoch=nothing, index=1⁝0)
    axis = 1
    N = size(coords)[axis]
    epoch = epoch == nothing ? coords.epoch : epoch
    dt = epoch / N
    p = copy(coords.coords)
    return if index == 1⁝0
        p = cat(p, p[nax, end, :, :], dims=axis)
        @. (p[2:end, :, :] - p[1:end-1, :, :]) / dt
    elseif index == 2:-1
        @. (p[3:end, :, :] - p[2:end-1, :, :]) / dt
    else
        i = collect(1:N)[index]
        i[end] == N ? p = cat(p, p[nax, end, :], dims=axis) : nothing
        @. (p[i, :, :] - p[i - 1, :, :]) / dt
    end
end

""" Return acceleration at each step
Get Dx N ndarray, Returns 3xNxP - 1 array, use three point to get
acceleration

:math:`a[i] = (2x[i] - x[i+1] - x[i-1]) / dtdt`

Parameters
----------
coords : array
    size of DxN or 3xAxN
epoch : float
    total time step.
index : slice obj; Default `np.s_[:]`
    Choose the steps want it to be returned. Default is all steps.
"""
function get_acceleration(coords::Cartesian; epoch=nothing, index=1⁝0)
    axis = 1
    N = size(coords)[axis]
    epoch = epoch == nothing ? coords.epoch : epoch
    dt = epoch / N
    ddt = dt * dt
    p = copy(coords.coords)
    return if index == 1⁝0
        p = cat(p[nax, 1, :], p, p[nax, end, :], dims=axis)
        (2 * p[2:end-1, :, :] - p[1:N-2, :, :] - p[2:end, :, :]) / ddt
    elseif index == 2⁝-1
        (2 * p[1:-1, :, :] - p[1:-2, :, :] - p[2:N-2, :, :]) / ddt
    else
        i = collect(1:N)[index]
        i[1]==1 ? begin p= cat(p[nax, 1, :], p, axis=axis); i .+= 1 end : nothing
        i[end]==N ? p = cat(p, p[nax, end, :], axis=axis) : nothing
        @. (2 * p[i, :, :] - p[i - 1, :, :] - p[i + 1, :, :]) / ddt
    end
end

# struct Cartesian2D{T, 2} <: Coords{T, 2}
#     coords::Array{T, 2}
#     epoch::Real
#     unit::String
# end

struct Fouriered{T, N} <: Coords{T, N}
    coords::Array{T, N}
    init::Array
    fin::Array
    epoch::Real
    unit::String
    cache::Dict
end

Base.similar(A::Fouriered, ::Type{T}, dims::Dims) where {T} = Fouriered{T, length(dims)}(similar(A.coords, T, dims), A.init, A.fin, A.epoch, A.unit, A.cache)

Fouriered(A::Array{T, N}, init::Array=nothing, fin::Array=nothing, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Fouriered{T, N}(A, init, fin, epoch, unit, cache)
Fouriered{T}(A::Array{T, N}, init::Array=nothing, fin::Array=nothing, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Fouriered{T, N}(A, init, fin, epoch, unit, cache)
Fouriered{T, N}(A::Array{T, N}) where {T<:Number, N} = Fouriered{T, N}(A, nothing, nothing, 1., "Å", Dict())

struct Atomic{T, N} <: Coords{T, N}
    coords::Array{T, N}
    center::Union{Array, Real}
    epoch::Real
    unit::String
    cache::Dict
end

Base.similar(A::Atomic, ::Type{T}, dims::Dims) where {T} = Atomic{T, length(dims)}(similar(A.coords, T, dims), A.center, A.epoch, A.unit, A.cache)
#Base.show(io::IO, coords::Atomic) = print(io, "Fouriered(r=$(x.r), θ=$(x.θ) rad, ϕ=$(x.ϕ) rad)")

Atomic(A::Array{T, N}, center::Union{Array, Real}=0., epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Atomic{T, N}(A, center, epoch, unit, cache)
Atomic{T}(A::Array{T, N}, center::Union{Array, Real}=0., epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T<:Number, N} = Atomic{T, N}(A, center, epoch, unit, cache)
Atomic{T, N}(A::Array{T, N}) where {T<:Number, N} = Atomic{T, N}(A, 0., 1., "Å", Dict())

struct SparseAtomic{T, N} <: Coords{T, N}
    coords::Array{T, N}
    epoch::Real
    unit::String
    cache::Dict
end

Base.similar(A::SparseAtomic, ::Type{T}, dims::Dims) where {T} = SparseAtomic{T, length(dims)}(similar(A.coords, T, dims), A.epoch, A.unit, A.cache)
Base.show(io::IO, coords::SparseAtomic{T, N}) where {T<:Descriptor, N} = print(io, "SparseAtomic", T, N)
Base.show(io::IO, coords::Array{T, N}) where {T<:Descriptor, N} = print(io, "SparseAtomic", T, N)

SparseAtomic(A::Array{T, N}, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T, N} = SparseAtomic{T, N}(A, epoch, unit, cache)
SparseAtomic{T}(A::Array{T, N}, epoch::Real=1, unit::String="Å", cache::Dict=Dict()) where {T, N} = SparseAtomic{T, N}(A, epoch, unit, cache)
SparseAtomic{T, N}(A::Array{T, N}) where {T, N} = SparseAtomic{T, N}(A, 1., "Å", Dict())

"`SparseAtomicFromCartesian()` - transformation from 3D point to `SparseAtomic` type"
struct SparseAtomicFromCartesian <: Transformation; end
"`CartesianFromSparseAtomic()` - transformation from `SparseAtomic` type to `Cartesian` type"
struct CartesianFromSparseAtomic <: Transformation; end

function (::SparseAtomicFromCartesian)(coords::Cartesian)
    N, A, _ = size(coords.coords)
    dtype = eltype(coords.coords)
    numbers = get(coords.cache, "numbers", [1 for i=1:A])
    cutoff = get(coords.cache, "cutoff", 7)
    periodic = get(coords.cache, "periodic", [true, true, true])
    cell = get(coords.cache, "cell", I(3))

    # 3 x 28(A) x N
    miccell = getmiccell(coords.coords, cell, periodic)

    sparseatomic = SparseAtomic(Array{AtomicDescriptor{dtype, 1}, 2}(nothing, N, A),
                                coords.epoch, coords.unit, coords.cache)
    sparseatomic.cache["miccell"] = miccell
    sparseatomic.cache["balltree"] = []
    sparseatomic.cache["neighbors"] = []

    for i=1:N
        positions = coords.coords[i, :, :]
        balltree = BallTree(miccell[:, :, i])
        neighbors = inrange(balltree, positions', cutoff, true)
        envnumberss = [numbers[(neighbor .% A) .+ 1] for neighbor in neighbors]

        push!(sparseatomic.cache["balltree"], balltree)
        push!(sparseatomic.cache["neighbors"], neighbors)

        for a=1:A
            number = numbers[a]
            position = positions[a, :]

            selfidx = searchsortedfirst(neighbors[a], a)
            envidcs = deleteat!(neighbors[a], selfidx)
            envnumbers = deleteat!(envnumberss[a], selfidx)
            # micpos = Array(miccell[:, :, i]')
            envpositions = Array(miccell[:, envidcs, i]')

            sparseatomic.coords[i, a] = AtomicDescriptor{dtype, 1}(
                a, number, position, envidcs, envnumbers, envpositions, cutoff,
                Dict("numbers"=>numbers, "balltree"=>balltree, "cell"=>cell,
                     "periodic"=>periodic, "miccell"=>view(miccell, :, :, i)))
        end
    end
    sparseatomic
end

function transform_deriv(::SparseAtomicFromCartesian, x::Cartesian)
    # M1 = transform_deriv(FourieredFromCartesian(), CartesianFromAtomic()(x))
    # M2 = transform_deriv(CartesianFromAtomic(), x)
    # return M1*M2
end
transform_deriv_params(::SparseAtomicFromCartesian, x::Cartesian) = error("FourieredFromAtomic has no parameters")

function (::CartesianFromSparseAtomic)(coords::SparseAtomic)
    N, A = size(coords.coords)
    dtype = eltype(coords.coords[1, 1].position)
    cart = Cartesian(Array{dtype, 3}(undef, N, A, 3))
    for i=1:N for a=1:A
        cart[i, a, :] = coords.coords[i, a].position
    end end
    cart
end

Base.convert(::Type{SparseAtomic}, coords::Cartesian) = SparseAtomicFromCartesian()(coords)
Base.convert(::Type{Cartesian}, coords::SparseAtomic) = CartesianFromSparseAtomic()(coords)

function transform_deriv(::SparseAtomicFromCartesian, x::Cartesian)
    # M1 = transform_deriv(FourieredFromCartesian(), CartesianFromAtomic()(x))
    # M2 = transform_deriv(CartesianFromAtomic(), x)
    # return M1*M2
end
transform_deriv_params(::SparseAtomicFromCartesian, x::Cartesian) = error("FourieredFromAtomic has no parameters")

################## Image ##################
"""
 Images coordinates;


"""
struct Images{T, N} <: Coords{T, N}
    coords::Array{T, N}
    epoch::Real
    unit::String
    images::Array{D where {D<:Descriptor}, 1}
    numbers::Vector{Number}
    cell::Union{Nothing, Array}
    periodic::Tuple{Bool, Bool, Bool}
    cache::Dict
end

function Images(;coords=nothing)

end



"`FourieredFromCartesian()` - transformation from 3D point to `Fouriered` type"
struct FourieredFromCartesian <: Transformation; end
"`CartesianFromFouriered()` - transformation from `Fouriered` type to `SVector{3}` type"
struct CartesianFromFouriered <: Transformation; end
"`AtomicFromCartesian()` - transformation from 3D point to `Atomic` type"
struct AtomicFromCartesian <: Transformation; end
"`CartesianFromAtomic()` - transformation from `Atomic` type to `SVector{3}` type"
struct CartesianFromAtomic <: Transformation; end
"`AtomicFromFouriered()` - transformation from `Fouriered` type to `Atomic` type"
struct AtomicFromFouriered <: Transformation; end
"`FourieredFromAtomic()` - transformation from `Atomic` type to `Fouriered` type"
struct FourieredFromAtomic <: Transformation; end

Base.show(io::IO, trans::FourieredFromCartesian) = print(io, "FourieredFromCartesian()")
Base.show(io::IO, trans::CartesianFromFouriered) = print(io, "CartesianFromFouriered()")
Base.show(io::IO, trans::AtomicFromCartesian) = print(io, "AtomicFromCartesian()")
Base.show(io::IO, trans::CartesianFromAtomic) = print(io, "CartesianFromAtomic()")
Base.show(io::IO, trans::AtomicFromFouriered) = print(io, "AtomicFromFouriered()")
Base.show(io::IO, trans::FourieredFromAtomic) = print(io, "FourieredFromAtomic()")


# Cartesian <-> Fouriered
function (::FourieredFromCartesian)(coords::Cartesian)
    N = coords.N
    Nk = N - 2
    init = coords.coords[1⁝1]
    fin = coords.coords[0⁝0]
    dist = fin - init
    line = dist .* collect(1:N-2) ./ (N-1)
    fcoords = coords.coords[2⁝-1] .- line
    fcoords = FFTW.r2r(fcoords, FFTW.RODFT00, 1) ./ sqrt(2*(size(fcoords)[1]+1))
    Fouriered(fcoords, init, fin, coords.epoch, coords.unit, coords.cache)
end

function transform_deriv(::FourieredFromCartesian, coords::Cartesian)
    N = coords.N
    Nk = N - 2
    init = coords.coords[1⁝1]
    fin = coords.coords[0⁝0]
    dist = fin - init
    line = dist .* collect(1:N-2) ./ (N-1)
    fcoords = coords.coords[2⁝-1] .- line
    fcoords = FFTW.r2r(fcoords, FFTW.REDFT00, 1) ./ sqrt(2*(size(fcoords)[1]+1))
    Fouriered(fcoords, init, fin, coords.epoch, coords.unit, coords.cache)
end

transform_deriv_params(::FourieredFromCartesian, x::AbstractVector) = error("FourieredFromCartesian has no parameters")

function (::CartesianFromFouriered)(fcoords::Fouriered)
    Nk = fcoords.Nk
    N = N + 2
    coords = zeros(eltype(fcoords), size(fcoords))
    coords[1⁝1] = fcoords.init
    coords[0⁝0] = fcoords.fin
    dist = fin - init
    line = dist .* collect(1:N-2) ./ (N-1)
    coords[1⁝0] = line + FFTW.r2r(fcoords.coords, FFTW.RODFT00, 1) ./ sqrt(2*(size(fcoords)[1]+1))

    Cartesian(coords, fcoords.epoch, fcoords.unit, fcoords.cache)
end

function transform_deriv(::CartesianFromFouriered, fcoords::Fouriered{T}) where T
    Nk = fcoords.Nk
    N = N + 2
    coords = zeros(eltype(fcoords), size(fcoords))
    coords[1⁝1] = fcoords.init
    coords[0⁝0] = fcoords.fin
    dist = fin - init
    line = dist .* collect(1:N-2) ./ (N-1)
    coords[1⁝0] = line + FFTW.r2r(fcoords.coords, FFTW.REDFT00, 1) ./ sqrt(2*(size(fcoords)[1]+1))

    Cartesian(coords, fcoords.epoch, fcoords.unit, fcoords.cache)
end

transform_deriv_params(::CartesianFromFouriered, x::Fouriered) = error("CartesianFromFouriered has no parameters")

# Cartesian <-> Atomic
function (::AtomicFromCartesian)(coords::Cartesian)
    ndims(coords.coords) == 3 || error("Cartesian Dimension should be NxAx3 ")
    center = coords.coords[:, 1, :]
    acoords = similar(coords.coords, eltype(coords.coords), N, 3(A-1), A)
    A, NE, neighbors = graph.A, graph.NE, graph.graph
    new_coords = zeros(T, A, 3, NE, N)
    for i=1:A for j=1:NE
        nidx = neighbors[i, j]
        new_coords[i, :, j, :] = coords[:, nidx, :] - coords[:, i, :]
    end end

    for i=1:A
        mask[i] = false
        disp[i, :, :, :] = wrap_coords[:, mask, :]
        mask[i] = true
        acoords[:, mask, i] = coords[:, mask, i] .- mask
    end
    Atomic(acoords, center, coords.epoch, coords.unit, coords.cache)
end

function transform_deriv(::AtomicFromCartesian, x::AbstractVector)
    x
end
transform_deriv_params(::AtomicFromCartesian, x::AbstractVector) = error("AtomicFromCartesian has no parameters")
#####
function (::CartesianFromAtomic)(acoords::Atomic)

end
#####
function transform_deriv(::CartesianFromAtomic, x::Atomic{T}) where {T}

end
transform_deriv_params(::CartesianFromAtomic, x::Atomic) = error("CartesianFromAtomic has no parameters")

# Fouriered <-> Atomic (TODO direct would be faster)
function (::AtomicFromFouriered)(x::Fouriered)
    AtomicFromCartesian()(CartesianFromFouriered()(x))
end
function transform_deriv(::AtomicFromFouriered, x::Fouriered)
    M1 = transform_deriv(AtomicFromCartesian(), CartesianFromFouriered()(x))
    M2 = transform_deriv(CartesianFromFouriered(), x)
    return M1*M2
end
transform_deriv_params(::AtomicFromFouriered, x::Fouriered) = error("AtomicFromFouriered has no parameters")

function (::FourieredFromAtomic)(x::Atomic)
    FourieredFromCartesian()(CartesianFromAtomic()(x))
end
function transform_deriv(::FourieredFromAtomic, x::Atomic)
    M1 = transform_deriv(FourieredFromCartesian(), CartesianFromAtomic()(x))
    M2 = transform_deriv(CartesianFromAtomic(), x)
    return M1*M2
end
transform_deriv_params(::FourieredFromAtomic, x::Atomic) = error("FourieredFromAtomic has no parameters")


Base.inv(::FourieredFromCartesian)   = CartesianFromFouriered()
Base.inv(::CartesianFromFouriered)   = FourieredFromCartesian()
Base.inv(::AtomicFromCartesian) = CartesianFromAtomic()
Base.inv(::CartesianFromAtomic) = AtomicFromCartesian()
Base.inv(::AtomicFromFouriered) = FourieredFromAtomic()
Base.inv(::FourieredFromAtomic) = AtomicFromFouriered()

# Inverse composition
compose(::FourieredFromCartesian,   ::CartesianFromFouriered)   = IdentityTransformation()
compose(::CartesianFromFouriered,   ::FourieredFromCartesian)   = IdentityTransformation()
compose(::AtomicFromCartesian, ::CartesianFromAtomic) = IdentityTransformation()
compose(::CartesianFromAtomic, ::AtomicFromCartesian) = IdentityTransformation()
compose(::AtomicFromFouriered, ::FourieredFromAtomic) = IdentityTransformation()
compose(::FourieredFromAtomic, ::AtomicFromFouriered) = IdentityTransformation()

# Cyclic compositions
compose(::FourieredFromCartesian,   ::CartesianFromAtomic) = FourieredFromAtomic()
compose(::CartesianFromFouriered,   ::FourieredFromAtomic) = CartesianFromAtomic()
compose(::AtomicFromCartesian, ::CartesianFromFouriered)   = AtomicFromFouriered()
compose(::CartesianFromAtomic, ::AtomicFromFouriered) = CartesianFromFouriered()
compose(::AtomicFromFouriered, ::FourieredFromCartesian)   = AtomicFromCartesian()
compose(::FourieredFromAtomic, ::AtomicFromCartesian) = FourieredFromCartesian()

# For convenience
Base.convert(::Type{Fouriered}, v::AbstractArray) = FourieredFromCartesian()(v)
Base.convert(::Type{Atomic}, v::AbstractArray) = AtomicFromCartesian()(v)

Base.convert(::Type{V}, s::Fouriered) where {V <: AbstractArray} = convert(V, CartesianFromFouriered()(s))
Base.convert(::Type{V}, c::Atomic) where {V <: AbstractArray} = convert(V, CartesianFromAtomic()(c))
# Base.convert(::Type{V}, s::Fouriered) where {V <: StaticArray} = convert(V, CartesianFromFouriered()(s))
# Base.convert(::Type{V}, c::Atomic) where {V <: StaticArray} = convert(V, CartesianFromAtomic()(c))

Base.convert(::Type{Fouriered}, c::Atomic) = FourieredFromAtomic()(c)
Base.convert(::Type{Atomic}, s::Fouriered) = AtomicFromFouriered()(s)



using Combinatorics

struct TwoBodyFromSparseAtomicDescriptor <: Transformation; twobodycutoff::Union{Nothing, Number, Array}; end
struct ThreeBodyFromSparseAtomicDescriptor <: Transformation; threebodycutoff::Union{Nothing, Number, Array}; end
struct TwoThreeBodyFromSparseAtomicDescriptor <: Transformation; twobodycutoff::Union{Nothing, Number, Array}; threebodycutoff::Union{Nothing, Number, Array}; end

Base.convert(::Type{TwoBody}, coords::SparseAtomic; twobodycutoff=nothing) = TwoBodyFromSparseAtomicDescriptor(twobodycutoff)(coords)
Base.convert(::Type{ThreeBody}, coords::SparseAtomic; threebodycutoff=nothing) = ThreeBodyFromSparseAtomicDescriptor(threebodycutoff)(coords)
Base.convert(::Type{TwoThreeBody}, coords::SparseAtomic; twobodycutoff=nothing, threebodycutoff=nothing) = TwoThreeBodyFromSparseAtomicDescriptor(twobodycutoff, threebodycutoff)(coords)

macro ini(chart)
    return :(Dict([(key, Vector{Number}([])) for key in keys($chart)]))
end

function (twobody::TwoBodyFromSparseAtomicDescriptor)(sparseatomic::SparseAtomic{T, NN}) where {T<:AtomicDescriptor, NN}
    coords = sparseatomic.coords
    N, A = size(coords)
    dtype = eltype(coords[1, 1])
    rk = length(size(sparseatomic))
    twobodyarr = Array{TwoBody{dtype, 1}, rk}(nothing, N, A)

    # <!--- Cut off radious swap
    numbers = sparseatomic.cache["numbers"]
    miccell = sparseatomic.cache["miccell"]

    balltrees = sparseatomic.cache["balltree"]
    neighborss = sparseatomic.cache["neighbors"]

    twobodycutoff = twobody.twobodycutoff
    if typeof(twobodycutoff)<:Union{Number, Nothing}
        twobodycutoffs = Array{typeof(twobodycutoff), 1}(undef, A)
        fill!(twobodycutoffs, twobodycutoff)
    end
    # Cutoff radius swap initialize-->

    # <!--- Initialize descriptor dictionary
    numbers = sparseatomic.cache["numbers"]
    species = Set(numbers) |> collect
    chart = Dict()
    multiplicity = Dict()
    permutations = Dict()
    counter = 1
    for cluster in with_replacement_combinations(species, 3)
        chart[Tuple(sort(cluster))] = counter
        multiplicity[Tuple(sort(cluster))] =
        permutations[Tuple(sort(cluster))] = []
        counter += 1
    end
    # Initialize descriptor dictionary -->

    for i=1:N
        balltree = balltrees[i]
        neighbors = neighborss[i]
        for a=1:A
            atom = coords[i, a]
            idx = atom.idx
            number = numbers[a]
            position = atom.position

            if twobodycutoffs[a] != nothing && twobodycutoff != atom.cutoff
                cutoff = twobodycutoffs[a]
                envidcs_plusself = inrange(balltree, position, cutoff, true)

                selfidx = searchsortedfirst(envidcs_plusself, a)
                envidcs = deleteat!(envidcs_plusself, selfidx)
                envnumbers = numbers[(envidcs .% A) .+ 1]
                envpositions = Array(miccell[:, envidcs, i]')
            else
                cutoff = atom.cutoff
                envidcs = desc.envidcs
                envnumbers = desc.envnumbers
                envpositions = desc.envpositions
            end

            M = length(envnumbers)

            s1 = number
            # Create a dictionary maps sends species to distances
            descriptor = Dict("twobody"=>[], "twobodychart"=>Set())
            for j=1:M
                s2 = envnumbers[j]

                if s1 <= s2
                    cluster = (s1, s2)
                    p1, p2 = position, envpositions[j, :]
                else
                    cluster = (s2, s1)
                    p1, p2 = envpositions[j, :], position
                end

                xi, yi, zi = p1
                xi1, yi1, zi1 = p2
                xi_i1, yi_i1, zi_i1 = xi-xi1, yi-yi1, zi-zi1
                rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)

                # xi = position .- envpositions[j, :]
                datum = Dict("cluster"=>cluster, "xi"=>xi, "yi"=>yi, "zi"=>zi,
                             "xi1"=>xi1, "yi1"=>yi1, "zi1"=>zi1,
                             "xi_i1"=>zi_i1, "yi_i1"=>yi_i1, "zi_i1"=>zi_i1,
                             "rii1"=>rii1)
                push!(descriptor["twobody"], datum)
                union!(descriptor["twobodychart"], [cluster])
            end
            twobodyarr[i, a] = TwoBody{dtype, 1}(idx, atom.number, atom.position,
                                                 envidcs, envnumbers, envpositions, cutoff, descriptor)
        end
    end
    SparseAtomic{TwoBody{dtype, 1}, NN}(twobodyarr, sparseatomic.epoch, sparseatomic.unit, sparseatomic.cache)
end

function (threebody::ThreeBodyFromSparseAtomicDescriptor)(sparseatomic::SparseAtomic{T, NN}) where {T<:AtomicDescriptor, NN}
    coords = sparseatomic.coords
    N, A = size(coords)
    dtype = eltype(coords[1, 1])
    rk = length(size(sparseatomic))
    threebodyarr = Array{ThreeBody{dtype, 1}, rk}(nothing, N, A)

    # <!--- Cut off radious swap
    numbers = sparseatomic.cache["numbers"]
    miccell = sparseatomic.cache["miccell"]
    balltrees = sparseatomic.cache["balltree"]
    neighborss = sparseatomic.cache["neighbors"]

    threebodycutoff = threebody.threebodycutoff
    if typeof(threebodycutoff)<:Union{Number, Nothing}
        threebodycutoffs = Array{typeof(threebodycutoff), 1}(undef, A)
        fill!(threebodycutoffs, threebodycutoff)
    end
    # Cutoff radius swap initialize-->

    # <!---Initialize permutations
    numbers = sparseatomic.cache["numbers"]
    species = Set(numbers) |> collect
    chart = Dict()
    multiplicity = Dict()
    permutations = Dict()
    counter = 1
    for cluster in with_replacement_combinations(species, 3)
        chart[Tuple(sort(cluster))] = counter
        multiplicity[Tuple(sort(cluster))] =
        permutations[Tuple(sort(cluster))] = []
        counter += 1
    end
        # Initialize -->

    for i=1:N
        balltree = balltrees[i]
        neighbors = neighborss[i]
        for a=1:A
            desc = coords[i, a]
            idx = coords[i, a].idx
            number = coords[i, a].number
            position = coords[i, a].position

            if threebodycutoffs[a] != nothing && threebodycutoff != atom.cutoff
                cutoff = threebodycutoffs[a]
                envidcs_plusself = inrange(balltree, position, cutoff, true)

                selfidx = searchsortedfirst(envidcs_plusself, a)
                envidcs = deleteat!(envidcs_plusself, selfidx)
                envnumbers = numbers[(envidcs .% A) .+ 1]
                envpositions = Array(miccell[:, envidcs, i]')
            else
                cutoff = atom.cutoff
                envidcs = desc.envidcs
                envnumbers = desc.envnumbers
                envpositions = desc.envpositions
            end
            envidcs = coords[i, a].envidcs
            envnumbers = coords[i, a].envnumbers
            envpositions = coords[i, a].envpositions

            M = length(envnumbers)

            s1 = number

            # Create a dictionary maps sends species to distances
            descriptor = Dict("threebody"=>[], "threebodychart"=>Set())
            for j=1:M
                s2 = envnumbers[j]
                if s1 <= s2
                    p1, p2 = position, envposition[j, :]
                else
                    s2, s1 = s1, s2
                    p1, p2 = envpositions[j, :], position
                end

                xi, yi, zi = p1
                xi1, yi1, zi1 = p2
                xi_i1, yi_i1, zi_i1 = xi-xi1, yi-yi1, zi-zi1

                _rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)

                for k=j+1:M
                    s3 = envnumbers[k]
                    p3 = envposition[k, :]
                    if s2 <= s3
                        cluster = (s1, s2, s3)
                        xi2, yi2, zi2 = p3

                        xi_i2, yi_i2, zi_i2 = xi-xi2, yi-yi2, zi-zi2
                        xi1_i2, yi1_i2, zi1_i2 = xi1-xi2, yi1-yi2, zi1-zi2
                        # rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)
                        rii1 = _rii1
                        rii2 = sqrt(xi_i2*xi_i2 + yi_i2*yi_i2 + zi_i2*zi_i2)
                        ri1i2 = sqrt(xi1_i2*xi1_i2 + yi1_i2*yi_i2 + zi1_i2*zi_i2)
                    elseif s3 < s1
                        cluster = (s3, s1, s2)
                        xi, yi, zi = p3
                        xi1, yi1, zi1 = p1
                        xi2, yi2, zi2 = p2

                        xi_i2, yi_i2, zi_i2 = xi-xi2, yi-yi2, zi-zi2
                        xi1_i2, yi1_i2, zi1_i2 = xi_i1, yi_i1, zi_i1
                        xi_i1, yi_i1, zi_i1 = xi-xi1, yi-yi1, zi-zi1

                        rii1 = sqrt(xi_i1*xi_i1 + yi_i1*y_i1 + zi_i1*z_i1)
                        rii2 = sqrt(xi_i2*xi_i2 + yi_i2*yi_i2 + zi_i2*zi_i2)
                        ri1i2 = _rii1

                        xi_i2, yi_i2, zi_i2 = xi-xi2, yi-yi2, zi-zi2
                    else
                        cluster = (s1, s3, s2)
                        xi1, yi1, zi1 = p3
                        xi2, yi2, zi2 = p2

                        xi_i2, yi_i2, zi_i2 = xi_i1, yi_i1, zi_i1
                        xi_i1, yi_i1, zi_i1 = xi-xi2, yi-yi2, zi-zi2
                        xi1_i2, yi1_i2, zi1_i2 = xi1-xi2, yi1-yi2, zi1-zi2

                        rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)
                        rii2 = _rii1
                        ri1i2 = sqrt(xi1_i2*xi_i2 + yi1_i2*yi1_i2 + zi1_i2*zi1_i2)
                    end

                    push!(descriptor2["xi"][cluster], xi)
                    push!(descriptor2["yi"][cluster], yi)
                    push!(descriptor2["zi"][cluster], zi)

                    push!(descriptor2["xi1"][cluster], xi1)
                    push!(descriptor2["yi1"][cluster], yi1)
                    push!(descriptor2["zi1"][cluster], zi1)

                    push!(descriptor2["xi2"][cluster], xi2)
                    push!(descriptor2["yi2"][cluster], yi2)
                    push!(descriptor2["zi2"][cluster], zi2)

                    push!(descriptor2["xi_i1"][cluster], xi_i1)
                    push!(descriptor2["yi_i1"][cluster], yi_i1)
                    push!(descriptor2["zi_i1"][cluster], zi_i1)

                    push!(descriptor2["xi_i2"][cluster], xi_i2)
                    push!(descriptor2["yi_i2"][cluster], yi_i2)
                    push!(descriptor2["zi_i2"][cluster], zi_i2)

                    push!(descriptor2["xi1_i2"][cluster], xi1_i2)
                    push!(descriptor2["yi1_i2"][cluster], yi1_i2)
                    push!(descriptor2["zi1_i2"][cluster], zi1_i2)

                    push!(descriptor2["rii1"][cluster], rii1)
                    push!(descriptor2["rii2"][cluster], rii2)
                    push!(descriptor2["ri1i2"][cluster], ri1i2)

                    datum = Dict("cluster"=>cluster, "xi"=>xi, "yi"=>yi, "zi"=>zi,
                                 "xi1"=>xi1, "yi1"=>yi1, "zi1"=>zi1,
                                 "xi2"=>xi2, "yi2"=>yi2, "zi2"=>zi2,
                                 "xi_i1"=>zi_i1, "yi_i1"=>yi_i1, "zi_i1"=>zi_i1,
                                 "xi_i2"=>zi_i2, "yi_i2"=>yi_i2, "zi_i2"=>zi_i2,
                                 "xi1_i2"=>zi1_i2, "yi1_i2"=>yi1_i2, "zi1_i2"=>zi1_i2,
                                 "rii1"=>rii1, "rii2"=>rii2, "ri1i2"=>ri1i2)
                    push!(descriptor["threebody"], datum)
                    union!(descriptor["threebodychart"], [cluster])
                end
            end
            threebodyarr[i, a] = ThreeBody{dtype, 1}(idx,
                desc.number, desc.position, desc.envidcs, desc.envnumbers,
                desc.envpositions, desc.cutoff, descriptor2)
        end
    end
    SparseAtomic{ThreeBody{dtype, 1}, NN}(threebody, sparseatomic.epoch,
                                  sparseatomic.unit, sparseatomic.cache)
end


Base.convert(::Type{SparseAtomic{TD, N}}, coords::SparseAtomic{TC, N};
             twobodycutoff=7, threebodycutoff=3.5) #=
=# where {TD<:TwoThreeBody, TC<:AtomicDescriptor, N} = #=
=# TwoThreeBodyFromSparseAtomicDescriptor(twobodycutoff, threebodycutoff)(coords)

function (twothreebody::TwoThreeBodyFromSparseAtomicDescriptor)(
    sparseatomic::SparseAtomic{T, NN}) where {T<:AtomicDescriptor, NN}

    twothreebody = Array{TwoThreeBody, 2}(sparseatomic.coords)
    dtype = eltype(sparseatomic.coords[1, 1])
    SparseAtomic{TwoThreeBody{dtype, 1}, NN}(twothreebody, sparseatomic.epoch,
                             sparseatomic.unit, sparseatomic.cache)
end





#sparsetwobody = convert(TwoBody, sparseatomic, twobodycutoff=3)
