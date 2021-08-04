
nax = [CartesianIndex()]
# include("./antenna.jl")
# include("./parser.jl")

"""
For Relative indexing like arr[1:-1]
"""
struct Slice
    start::Int
#    step::Int
    stop::Int
end

# const sslice = ⁝
⁝(start::Int=1, stop::Int=0) = Slice(start, stop)
#⁝(start::Int=1, step::Int=0, stop::Int=0) = Slice(start, 1, stop)

function Base.getindex(A::T, s::Slice) where {T<:AbstractArray}
    N = size(A)[1]
    a = s.start < 1 ? N + s.start : s.start
    b = s.stop < 1 ? N + s.stop : s.stop

    R = ndims(A)
    return if R == 3
        getindex(A, a:b, :, :)
    elseif R == 2
        getindex(A, a:b, :)
    elseif R == 1
        getindex(A, a:b)
    end
end

"""
coord : 3xA array
cell : 3x3  ase-like array
"""
function cart2scaled(coord::Array, cell::Array)
    return cell' \ coord
end

"""
coords : 3xAxN array
cell : 3x3  ase-like array
return : 3xAxN array
"""
function cart2scaled(coords::Array{Float64, 3}, cell::Array{Float64, 2})
    scaleds = zeros(Float64, size(coords)...)
    # cell_t = cell'
    cell⁻¹ = inv(cell')
    for i = 1:size(coords)[end]
        scaleds[:, :, i] =  cell⁻¹ * coords[:, :, i]
    end
    return scaleds
end

"""
scaled : 3xA array
cell : 3x3  ase-like array
"""
function scaled2cart(scaled::Array{Float64, 2}, cell::Array{Float64, 2})
    return cell' * scaled
end

"""
scaled : 3xAxN array
cell : 3x3  ase-like array
return : 3xAxN array
"""
function scaled2cart(scaleds::Array{Float64, 3}, cell::Array{Float64, 2})
    coords = zeros(Float64, size(scaleds)...)
    for i = 1:size(scaleds)[end]
        coords[:, :, i] = cell' * scaleds[:, :, i]
    end
    return coords
end

"""
coord : 3xA array
cell : 3x3  ase-like array
n : Int index of atom to be centered
return : 3xA array
"""
function center_wrap(coord::Array{Float64, 2}, cell::Array{Float64, 2}, n::Int)
    scaled = cart2scaled(coord, cell)
    scaled .-= ((scaled[:, n] .+ 0.5) .% 1.) .- 0.5
    return scaled2cart(scaled, cell)
end

"""
coords : 3xAxN array
cell : 3x3  ase-like array
n : Int; index of atom to be centered
return : 3xAxN array
"""
function center_wrap(coords::Array{Float64, 3}, cell::Array{Float64, 2}, n::Int)
    scaleds = cart2scaled(coords, cell)
    scaleds .-= ((scaleds[:, n, nax, :] .+ 0.5) .% 1.) .- 0.5
    return scaled2cart(scaleds, cell)
end

"""
period = [false, false, false]
periodic = supercell_periodic_help(period)
"""
function supercell_periodic_help(period)
    per = Set([])
    for x in [-1, 0, 1] for y in [-2, 0, 2] for z in [3, 0, -3]
        dir = [x, y, z][period]
        dir == [0, 0, 0][period] ? continue : nothing
        union!(per, Set([dir]))
    end end end
    hcat([p for p in per]...)
end



"""
Minimum image convention cell padding
"""
function getmiccell(coords, cell, period)
    N, A, _ = size(coords)
    periodic = supercell_periodic_help(period)

    pD, pN = periodic == [] ? (0, 0) : size(periodic)
    supercell = similar(coords, 3, A * (1+pN), N)
    positionss = permutedims(coords, (3, 2, 1))
    supercell[:, 1:A, :] = positionss
    for i=1:pN
        translate = zeros(3)
        for j=1:pD
            p = periodic[j, i]
            p == 0 ? continue : nothing
            s = sign(p)
            idx = abs(p)
            translate += s .* cell[idx, :]
        end
        scell = positionss .+ translate
        supercell[:, A*i+1:A*(i+1), :] = scell
    end
    return supercell
end
