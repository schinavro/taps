using LinearAlgebra: norm


Base.size(A::Coords) = size(A.coords)
Base.getindex(A::Coords, I::Vararg{Int, N}) where {N} = getindex(A.coords, I...)
Base.setindex!(A::Coords, v, I::Vararg{Int, N}) where {N} = setindex!(A.coords, v, I...)
Base.isapprox(c1::T, c2::T; kwargs...) where {T<:Coords} = isapprox(c1.coords, c2.coords; kwargs...) && isapprox(c1.epoch, c2.epoch; kwargs...) && c1.unit == c2.unit



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
end

Base.similar(A::Cartesian, ::Type{T}, dims::Dims) where {T} = Cartesian{T, length(dims)}(similar(A.coords, T, dims), A.epoch, A.unit)
Base.convert(::Type{Cartesian{T, 2}}, coords::Cartesian{T, N}) where {T, N} = Cartesian{T, 2}(reshape(coords.coords, coords.N, prod(size(coords)[2:end])), coords.epoch, coords.unit)

Cartesian(A::Array{T, N}, epoch::Real=1, unit::String="Å") where {T<:Number, N} = Cartesian{T, N}(A, epoch, unit)
Cartesian{T}(A::Array{T, N}, epoch::Real=1, unit::String="Å") where {T<:Number, N} = Cartesian{T, N}(A, epoch, unit)
Cartesian{T, N}(A::Array{T, N}) where {T<:Number, N} = Cartesian{T, N}(A, 1., "Å")

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
function get_displacements(coords::Coords; epoch=nothing, index=1⁝0)
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
function get_velocity(coords::Coords; epoch=nothing, index=1⁝0)
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
function get_acceleration(coords::Coords; epoch=nothing, index=1⁝0)
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

struct Fouriered{T, N} <: Coords{T, N}
    coords::Array{T, N}
    init::Array
    fin::Array
    epoch::Real
    unit::String
end

Base.similar(A::Fouriered, ::Type{T}, dims::Dims) where {T} = Fouriered{T, length(dims)}(similar(A.coords, T, dims), A.init, A.fin, A.epoch, A.unit)

Fouriered(A::Array{T, N}, init::Array=nothing, fin::Array=nothing, epoch::Real=1, unit::String="Å") where {T<:Number, N} = Fouriered{T, N}(A, init, fin, epoch, unit)
Fouriered{T}(A::Array{T, N}, init::Array=nothing, fin::Array=nothing, epoch::Real=1, unit::String="Å") where {T<:Number, N} = Fouriered{T, N}(A, init, fin, epoch, unit)
Fouriered{T, N}(A::Array{T, N}) where {T<:Number, N} = Fouriered{T, N}(A, nothing, nothing, 1., "Å")

struct Atomic{T, N} <: Coords{T, N}
    coords::Array{T, N}
    center::Union{Array, Real}
    epoch::Real
    unit::String
end

Base.similar(A::Atomic, ::Type{T}, dims::Dims) where {T} = Atomic{T, length(dims)}(similar(A.coords, T, dims), A.center, A.epoch, A.unit)
#Base.show(io::IO, coords::Atomic) = print(io, "Fouriered(r=$(x.r), θ=$(x.θ) rad, ϕ=$(x.ϕ) rad)")

Atomic(A::Array{T, N}, center::Union{Array, Real}=0., epoch::Real=1, unit::String="Å") where {T<:Number, N} = Atomic{T, N}(A, center, epoch, unit)
Atomic{T}(A::Array{T, N}, center::Union{Array, Real}=0., epoch::Real=1, unit::String="Å") where {T<:Number, N} = Atomic{T, N}(A, center, epoch, unit)
Atomic{T, N}(A::Array{T, N}) where {T<:Number, N} = Atomic{T, N}(A, 0., 1., "Å")


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
    Fouriered(fcoords, init, fin, coords.epoch, coords.unit)
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
    Fouriered(fcoords, init, fin, coords.epoch, coords.unit)
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

    Cartesian(coords, fcoords.epoch, fcoords.unit)
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

    Cartesian(coords, fcoords.epoch, fcoords.unit)
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
    Atomic(acoords, center, coords.epoch, coords.unit)
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
