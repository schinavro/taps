
mutable struct TwoBodyDB <: Database
    m::Int
    n::Int

    rawDB
    chart::Vector{Tuple}
    degeneracy::Dict

    fx::Vector{Number}
    fy::Vector{Number}
    fz::Vector{Number}

    idx::Dict
    idxset::Dict

    xi::Dict
    yi::Dict
    zi::Dict

    ri::Dict

    xi1::Dict
    yi1::Dict
    zi1::Dict

    ri1::Dict

    xi_i1::Dict
    yi_i1::Dict
    zi_i1::Dict

    rii1::Dict
end

function TwoBodyDB(numbers::Array{Int, 1})
    chart, degeneracy = getchart(Tuple(numbers), 2)
    m = length(chart)
    n = 0

    fx = Vector{Number}([])
    fy = Vector{Number}([])
    fz = Vector{Number}([])

    # Indexing to force
    idx = Dict{Tuple{Int, Int}, Vector{Int}}([(key, Vector{Int}([])) for key in chart])
    idxset = Dict{Tuple{Int, Int}, Set}([(key, Set()) for key in chart])

    xi = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    ri = Dict{Tuple{Int, Int}, Array{Array{Float64, 1}, 1}}([(key, Array{Array{Float64, 1}, 1}([])) for key in chart])

    xi1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    # ri1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    ri1 = Dict{Tuple{Int, Int}, Array{Array{Float64, 1}, 1}}([(key, Array{Array{Float64, 1}, 1}([])) for key in chart])

    xi_i1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi_i1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi_i1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    rii1 = Dict{Tuple{Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    TwoBodyDB(m, n, [], chart, degeneracy, fx, fy, fz, idx, idxset,
              xi, yi, zi, ri, xi1, yi1, zi1, ri1, xi_i1, yi_i1, zi_i1, rii1)
end

update!(db::TwoBodyDB,
        coords::Cartesian{T1, 2},
        forces::Array{T2, 2}; kwargs...) where {T1<:Number, T2<:Number} = #=
        =# update!(db, coords[nax, :, :], forces[nax, :, :]; kwargs...)
update!(db::TwoBodyDB, coords::Cartesian{T1, 3},
        forces::Array{T2, 3}; kwargs...) where {T1<:Number, T2<:Number}#=
    =# = update!(db, convert(TwoBody, coords; kwargs...), forces)

function update!(db::TwoBodyDB, coords::SparseAtomic, forces::Array{T, 3}) where {T<:Number}
    M, A, _ = size(forces)
    # size between coords and forces is inconsistent
    @assert (M, A) == Tuple(size(coords)[1:2])
    # "No data found"
    @assert M > 0

    chart = keys(db.idx)
    numbers = coords.cache["numbers"]

    count = db.n + 1

    for m=1:M
        for a=1:A
            push!(db.rawDB, coords[m, a])
            push!(db.fx, forces[m, a, 1]);
            push!(db.fy, forces[m, a, 2]);
            push!(db.fz, forces[m, a, 3])
            datum = coords[m, a].descriptor
            for cluster in chart
                append!(db.xi[cluster],    datum[cluster]["xi"])
                append!(db.yi[cluster],    datum[cluster]["yi"])
                append!(db.zi[cluster],    datum[cluster]["zi"])
                append!(db.ri[cluster],    datum[cluster]["ri"])
                append!(db.xi1[cluster],   datum[cluster]["xi1"])
                append!(db.yi1[cluster],   datum[cluster]["yi1"])
                append!(db.zi1[cluster],   datum[cluster]["zi1"])
                append!(db.ri1[cluster],   datum[cluster]["ri1"])
                append!(db.xi_i1[cluster], datum[cluster]["xi_i1"])
                append!(db.yi_i1[cluster], datum[cluster]["yi_i1"])
                append!(db.zi_i1[cluster], datum[cluster]["zi_i1"])
                append!(db.rii1[cluster],  datum[cluster]["rii1"])

                idx = similar(Vector{Int}, length(datum[cluster]["xi"]))
                fill!(idx, count)
                append!(db.idx[cluster], idx)
                union!(db.idxset[cluster], count)
            end

            count += 1
            db.n += 1
        end
    end
end

"""
 idx refers to the number where its enviroment located in the array.
"""
mutable struct ThreeBodyDB <: Database
    m::Int                  # Number of rows
    n::Int                  # Number of columns

    rawDB
    chart::Vector{Tuple}
    degeneracy::Dict

    fx::Vector{Number}
    fy::Vector{Number}
    fz::Vector{Number}

    idx::Dict
    idxset::Dict

    xi::Dict
    yi::Dict
    zi::Dict
    ri::Dict

    xi1::Dict
    yi1::Dict
    zi1::Dict
    ri1::Dict

    xi2::Dict
    yi2::Dict
    zi2::Dict
    ri2::Dict

    xi_i1::Dict
    yi_i1::Dict
    zi_i1::Dict

    xi_i2::Dict
    yi_i2::Dict
    zi_i2::Dict

    xi1_i2::Dict
    yi1_i2::Dict
    zi1_i2::Dict

    rii1::Dict
    rii2::Dict
    ri1i2::Dict

end


function ThreeBodyDB(numbers::Array{Int, 1})
    chart, degeneracy = getchart(Tuple(numbers), 3)
    m = length(chart)
    n = 0

    fx = Vector{Number}([])
    fy = Vector{Number}([])
    fz = Vector{Number}([])

    # Indexing to force
    idx = Dict{Tuple{Int, Int, Int}, Vector{Int}}([(key, Vector{Int}([])) for key in chart])
    idxset = Dict{Tuple{Int, Int, Int}, Set}([(key, Set()) for key in chart])

    xi = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    ri = Dict{Tuple{Int, Int, Int}, Array{Array{Float64, 1}, 1}}([(key, Array{Array{Float64, 1}, 1}([])) for key in chart])

    xi1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    ri1 = Dict{Tuple{Int, Int, Int}, Array{Array{Float64, 1}, 1}}([(key, Array{Array{Float64, 1}, 1}([])) for key in chart])

    xi2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    ri2 = Dict{Tuple{Int, Int, Int}, Array{Array{Float64, 1}, 1}}([(key, Array{Array{Float64, 1}, 1}([])) for key in chart])

    xi_i1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi_i1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi_i1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    xi_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    xi1_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    yi1_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    zi1_i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    rii1 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    rii2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])
    ri1i2 = Dict{Tuple{Int, Int, Int}, Vector{Number}}([(key, Vector{Number}([])) for key in chart])

    ThreeBodyDB(m, n, [], chart, degeneracy, fx, fy, fz, idx, idxset, xi, yi, zi, ri, xi1, yi1, zi1, ri1, xi2, yi2, zi2, ri2,
                xi_i1, yi_i1, zi_i1, xi_i2, yi_i2, zi_i2, xi1_i2, yi1_i2, zi1_i2,
                rii1, rii2, ri1i2)
end

update!(db::ThreeBodyDB, coords::Cartesian{T, 2}, forces::Array{Number, 2}; kwargs...) where {T<:Number} = update!(db, coords[nax, :, :], forces[nax, :, :]; kwargs...)
update!(db::ThreeBodyDB, coords::Cartesian{T, 3}, forces::Array{Number, 3}; kwargs...) where {T<:Number} = update!(db, convert(TwoBody, coords; kwargs...), forces)

function update!(db::ThreeBodyDB, coords::SparseAtomic, forces::Array{T, 3}) where {T<:Number}
    M, A, _ = size(forces)
    # "size between coords and forces is inconsistent"
    @assert (M, A) == Tuple(size(coords)[1:2])
    # No data found"
    @assert M > 0

    chart = keys(db.idx)
    numbers = coords.cache["numbers"]

    count = db.n + 1
    for m=1:M
        for a=1:A
            push!(db.rawDB, coords[m, a])
            push!(db.fx, forces[m, a, 1]); push!(db.fy, forces[m, a, 2]);
            push!(db.fz, forces[m, a, 3])

            datum = coords[m, a].descriptor
            for cluster in chart
                append!(db.xi[cluster],    datum[cluster]["xi"])
                append!(db.yi[cluster],    datum[cluster]["yi"])
                append!(db.zi[cluster],    datum[cluster]["zi"])
                append!(db.ri[cluster],    datum[cluster]["ri"])
                append!(db.xi1[cluster],   datum[cluster]["xi1"])
                append!(db.yi1[cluster],   datum[cluster]["yi1"])
                append!(db.zi1[cluster],   datum[cluster]["zi1"])
                append!(db.ri1[cluster],   datum[cluster]["ri1"])
                append!(db.xi2[cluster],   datum[cluster]["xi2"])
                append!(db.yi2[cluster],   datum[cluster]["yi2"])
                append!(db.zi2[cluster],   datum[cluster]["zi2"])
                append!(db.ri2[cluster],   datum[cluster]["ri2"])
                append!(db.xi_i1[cluster], datum[cluster]["xi_i1"])
                append!(db.yi_i1[cluster], datum[cluster]["yi_i1"])
                append!(db.zi_i1[cluster], datum[cluster]["zi_i1"])
                append!(db.xi_i2[cluster], datum[cluster]["xi_i2"])
                append!(db.yi_i2[cluster], datum[cluster]["yi_i2"])
                append!(db.zi_i2[cluster], datum[cluster]["zi_i2"])
                append!(db.xi1_i2[cluster],datum[cluster]["xi1_i2"])
                append!(db.yi1_i2[cluster],datum[cluster]["yi1_i2"])
                append!(db.zi1_i2[cluster],datum[cluster]["zi1_i2"])
                append!(db.rii1[cluster],  datum[cluster]["rii1"])
                append!(db.rii2[cluster],  datum[cluster]["rii2"])
                append!(db.ri1i2[cluster], datum[cluster]["ri1i2"])


                idx = similar(Vector{Int}, length(datum[cluster]["xi"]))
                fill!(idx, count)
                append!(db.idx[cluster], idx)
                union!(db.idxset[cluster], count)
            end

            count += 1
            db.n += 1
        end
    end
end

mutable struct TwoThreeBodyDB <: Database
    twobody::TwoBodyDB
    threebody::ThreeBodyDB
end

function TwoThreeBodyDB(numbers)
    twobody = TwoBodyDB(numbers)
    threebody = ThreeBodyDB(numbers)
    TwoThreeBodyDB(twobody, threebody)
end

update!(db::TwoThreeBodyDB, coords::Cartesian{T, 2},
        forces::Array{Number, 2}; kwargs...) where {T<:Number}#=
        =# = update!(db, coords[nax, :, :], forces[nax, :, :]; kwargs...)
update!(db::TwoThreeBodyDB, coords::Cartesian{T, 3},
        forces::Array{Number, 3}; kwargs...) where {T<:Number}#=
        =# = update!(db, convert(TwoBody, coords; kwargs...), forces)

function update!(db::TwoThreeBodyDB, coords, forces)
    update!(db.twobody, coords, forces)
    update!(db.threebody, coords, forces)
end
