
using Memoize

abstract type Descriptor{T<:Number, DN} end

Base.eltype(::Descriptor{T, DN}) where {T, DN} = T

"""
idx : Index of self
specie : Periodic number of self
position : Cartesian position of self
envidcs : Index of atoms within BallTree
envnumbers : Cartesian coordinates of atoms within BallTree
envpositions : Periodic number of each atoms
cutoff
descriptor : cache
"""
struct AtomicDescriptor{T, DN} <: Descriptor{T, DN}
    idx::Int
    number::Int
    position::Array{T, 1}
    ##
    # numbers
    # cell
    # pbc
    # balltree
    #
    ## below I want to delte it later
    envidcs::Array{Int, 1}
    envnumbers::Array{Int, 1}
    envpositions::Array{T, 2}
    cutoff::Number
    descriptor::Dict
end

@inline AtomicDescriptor() = AtomicDescriptor{Float64, 1}(1, 1, [0., 0., 0.], [1], [1], zeros(1, 3), 7, Dict())

Base.convert(::Type{AtomicDescriptor{T, DN}}, not::Nothing) where {T<:Number, DN} = AtomicDescriptor()
Base.convert(::Type{AtomicDescriptor{T}}, not::Nothing) where {T<:Number} = AtomicDescriptor()
Base.convert(::Type{AtomicDescriptor}, not::Nothing) = AtomicDescriptor()

#### <!--- Utility function neeeded for manybody descriptors
function degeneracy(cluster)
    idx = Dict()
    multiple = Dict()
    for (i, a) in enumerate(cluster)
        if get(multiple, a, nothing) == nothing
            multiple[a] = 0
            idx[a] = []
        end
        multiple[a] += 1
        push!(idx[a], i)
    end

    total_mul = 1
    for (k, v) in pairs(multiple)
        total_mul *= factorial(v)
    end

    permutedidx = Dict()
    for s in keys(idx)
        permutedidx[s] = permutations(idx[s])
    end

    allpermutations = []
    orgidx = 1:length(cluster) |> collect
    for combinations in Iterators.product(values(permutedidx)...)
        copied = copy(orgidx)
        for (s, combo) in zip(keys(permutedidx), combinations)
            copied[idx[s]] = combo
        end

        push!(allpermutations, copied)
    end

    # return multiple, total_mul, allpermutations
    return allpermutations
end

@memoize function getchart(numbers::Tuple, n::Int)
    species = Set(numbers) |> collect
    chart = Vector{Tuple}([])
    # multiplicity = Dict()
    permutations = Dict()
    for cluster in with_replacement_combinations(species, n)
        srtdcluster = Tuple(sort(cluster))
        push!(chart, srtdcluster)
        permutations[srtdcluster] = degeneracy(srtdcluster)
    end
    return (chart, permutations)
end

macro ini(chart)
    return :(Dict([(key, Vector{Number}([])) for key in $chart]))
end
#### -->


"""
idx : Index of self
specie : Periodic number of self
position : Cartesian position of self
envidcs : Index of atoms within BallTree
envnumbers : Cartesian coordinates of atoms within BallTree except self
envpositions : Periodic number of each atoms except self
cutoff
descriptor : cache
"""
struct TwoBody{T, DN} <: Descriptor{T, DN}
    idx::Int
    number::Int
    position::Array{T, 1}
    envidcs::Array{Int, 1}
    envnumbers::Array{Int, 1}
    envpositions::Array{T, 2}
    cutoff::Number
    descriptor::Dict
end

@inline TwoBody() = TwoBody{Float64, 1}(1, 1, [0., 0., 0.], [1], [1], zeros(1, 3), 7, Dict())

Base.convert(::Type{TwoBody{T, DN}}, not::Nothing) where {T<:Number, DN} = TwoBody()
Base.convert(::Type{TwoBody{T}}, not::Nothing) where {T<:Number} = TwoBody()
Base.convert(::Type{TwoBody}, not::Nothing) = TwoBody()

Base.copy(d::TwoBody{T, DN}) where {T<:Number, DN} = #=
    =# TwoBody{T, DN}(d.idx, d.number, copy(d.position), copy(d.envidcs),
                      copy(d.envnumbers), copy(d.envpositions), d.cutoff,
                      copy(d.descriptor))

struct TwoBodyFromAtomicDescriptor <: Transformation; twobodycutoff; end
Base.convert(::Type{TwoBody}, atomic::AtomicDescriptor) = TwoBodyFromAtomicDescriptor(twobodycutoff)(atomic)
function (twobody::TwoBodyFromAtomicDescriptor)(atomic)
    atom = atomic
    number = atom.number
    numbers = atom.descriptor["numbers"]
    # chart = atom.descriptor["chart"]
    A = length(atom.descriptor["numbers"])
    idx = atom.idx
    position = atom.position

    balltree = atom.descriptor["balltree"]
    miccell = atom.descriptor["miccell"]

    envidcs_plusself = inrange(balltree, position, twobody.twobodycutoff, true)

    selfidx = searchsortedfirst(envidcs_plusself, idx)
    envidcs = deleteat!(envidcs_plusself, selfidx)
    envnumbers = numbers[(envidcs .% A) .+ 1]
    envpositions = view(miccell, :, envidcs)'

    M = length(envnumbers)

    # Create a dictionary maps sends species to distances
    # print(getchart(Tuple(numbers), 2))
    chart, degeneracy = getchart(Tuple(numbers), 2)
    descriptor = Dict([(
        key, Dict("idx"   =>Vector{Number}([]),
                  "xi"    =>Vector{Number}([]),
                  "yi"    =>Vector{Number}([]),
                  "zi"    =>Vector{Number}([]),
                  "ri"    =>Array{Array{Float64, 1}}([]),
                  "xi1"   =>Vector{Number}([]),
                  "yi1"   =>Vector{Number}([]),
                  "zi1"   =>Vector{Number}([]),
                  "ri1"   =>Array{Array{Float64, 1}}([]),
                  "xi_i1" =>Vector{Number}([]),
                  "yi_i1" =>Vector{Number}([]),
                  "zi_i1" =>Vector{Number}([]),
                  "rii1"  =>Vector{Number}([]),
                  "permutations" => degeneracy[key])) for key in chart])

    s1 = number
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

        push!(descriptor[cluster]["xi"], xi)
        push!(descriptor[cluster]["yi"], yi)
        push!(descriptor[cluster]["zi"], zi)

        push!(descriptor[cluster]["ri"], p1)

        push!(descriptor[cluster]["xi1"], xi1)
        push!(descriptor[cluster]["yi1"], yi1)
        push!(descriptor[cluster]["zi1"], zi1)

        push!(descriptor[cluster]["ri1"], p2)

        push!(descriptor[cluster]["xi_i1"], xi_i1)
        push!(descriptor[cluster]["yi_i1"], yi_i1)
        push!(descriptor[cluster]["zi_i1"], zi_i1)

        push!(descriptor[cluster]["rii1"], rii1)
    end
    TwoBody{eltype(atomic), 1}(idx, number,position, envidcs,
                    envnumbers, envpositions, twobody.twobodycutoff, descriptor)
end



"""
idx : Index of self
specie : Periodic number of self
position : Cartesian position of self
envidcs : Index of atoms within BallTree
envnumbers : Cartesian coordinates of atoms within BallTree exept self
envpositions : Periodic number of each atoms exept self
cutoff
descriptor : cache
"""
struct ThreeBody{T, DN} <: Descriptor{T, DN}
    idx::Int
    number::Int
    position::Array{T, 1}
    envidcs::Array{Int, 1}
    envnumbers::Array{Int, 1}
    envpositions::Array{T, 2}
    cutoff::Number
    descriptor::Dict
end

@inline ThreeBody() = ThreeBody{Float64, 1}(1, 1, [0., 0., 0.], [1], [1], zeros(1, 3), 7, Dict())

Base.convert(::Type{ThreeBody{T, DN}}, not::Nothing) where {T<:Number, DN} = ThreeBody()
Base.convert(::Type{ThreeBody{T}}, not::Nothing) where {T<:Number} = ThreeBody()
Base.convert(::Type{ThreeBody}, not::Nothing) = ThreeBody()

Base.copy(d::ThreeBody{T, DN}) where {T<:Number, DN} = #=
=# ThreeBody{T, DN}(d.idx, d.number, copy(d.position), copy(d.envidcs),
      copy(d.envnumbers), copy(d.envpositions), d.cutoff, copy(d.descriptor))

struct ThreeBodyFromAtomicDescriptor; threebodycutoff; end
Base.convert(::Type{ThreeBody}, atomic::AtomicDescriptor) = ThreeBodyFromAtomicDescriptor(threebodycutoff)(atomic)

function (threebody::ThreeBodyFromAtomicDescriptor)(atomic)
    atom = atomic
    number = atom.number
    numbers = atom.descriptor["numbers"]
    A = length(atom.descriptor["numbers"])
    idx = atom.idx
    position = atom.position

    balltree = atom.descriptor["balltree"]
    miccell = atom.descriptor["miccell"]

    envidcs_plusself = inrange(balltree, position, threebody.threebodycutoff, true)

    selfidx = searchsortedfirst(envidcs_plusself, idx)
    envidcs = deleteat!(envidcs_plusself, selfidx)
    envnumbers = numbers[(envidcs .% A) .+ 1]
    envpositions = view(miccell, :, envidcs)'

    M = length(envnumbers)

    # Create a dictionary maps sends species to distances
    chart, degeneracy = getchart(Tuple(numbers), 3)
    descriptor = Dict([(
        key, Dict("idx"   =>Vector{Number}([]),
                  "xi"    =>Vector{Number}([]),
                  "yi"    =>Vector{Number}([]),
                  "zi"    =>Vector{Number}([]),
                  "ri"    =>Array{Array{Float64, 1}, 1}([]),
                  "xi1"   =>Vector{Number}([]),
                  "yi1"   =>Vector{Number}([]),
                  "zi1"   =>Vector{Number}([]),
                  "ri1"    =>Array{Array{Float64, 1}, 1}([]),
                  "xi2"   =>Vector{Number}([]),
                  "yi2"   =>Vector{Number}([]),
                  "zi2"   =>Vector{Number}([]),
                  "ri2"    =>Array{Array{Float64, 1}, 1}([]),
                  "xi_i1" =>Vector{Number}([]),
                  "yi_i1" =>Vector{Number}([]),
                  "zi_i1" =>Vector{Number}([]),
                  "xi_i2" =>Vector{Number}([]),
                  "yi_i2" =>Vector{Number}([]),
                  "zi_i2" =>Vector{Number}([]),
                  "xi1_i2"=>Vector{Number}([]),
                  "yi1_i2"=>Vector{Number}([]),
                  "zi1_i2"=>Vector{Number}([]),
                  "rii1"  =>Vector{Number}([]),
                  "rii2"  =>Vector{Number}([]),
                  "ri1i2" =>Vector{Number}([]),
                  "permutations" => degeneracy[key])) for key in chart])
    # descriptor["chart"] = chart
    # descriptor["degeneracy"] = degeneracy

    s1 = number
    for j=1:M
        s2 = envnumbers[j]

        if s1 <= s2
            cluster = (s1, s2)
            p1, p2 = position, envpositions[j, :]
        else
            cluster = (s2, s1)
            s2, s1 = s1, s2
            p1, p2 = envpositions[j, :], position
        end

        xi, yi, zi = p1
        xi1, yi1, zi1 = p2
        xi_i1, yi_i1, zi_i1 = xi-xi1, yi-yi1, zi-zi1
        _rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)

        for k=j+1:M
            s3 = envnumbers[k]
            p3 = envpositions[k, :]
            if s2 <= s3
                cluster = (s1, s2, s3)
                xi2, yi2, zi2 = p3

                xi_i2, yi_i2, zi_i2 = xi-xi2, yi-yi2, zi-zi2
                xi1_i2, yi1_i2, zi1_i2 = xi1-xi2, yi1-yi2, zi1-zi2
                # rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)
                rii1 = _rii1
                rii2 = sqrt(xi_i2*xi_i2 + yi_i2*yi_i2 + zi_i2*zi_i2)
                ri1i2 = sqrt(xi1_i2*xi1_i2 + yi1_i2*yi_i2 + zi1_i2*zi_i2)
            elseif s3 <= s1
                cluster = (s3, s1, s2)
                xi, yi, zi = p3
                xi1, yi1, zi1 = p1
                xi2, yi2, zi2 = p2

                xi_i2, yi_i2, zi_i2 = xi-xi2, yi-yi2, zi-zi2
                xi1_i2, yi1_i2, zi1_i2 = xi_i1, yi_i1, zi_i1
                xi_i1, yi_i1, zi_i1 = xi-xi1, yi-yi1, zi-zi1

                rii1 = sqrt(xi_i1*xi_i1 + yi_i1*yi_i1 + zi_i1*zi_i1)
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
            push!(descriptor[cluster]["xi"], xi)
            push!(descriptor[cluster]["yi"], yi)
            push!(descriptor[cluster]["zi"], zi)
            push!(descriptor[cluster]["ri"], p1)

            push!(descriptor[cluster]["xi1"], xi1)
            push!(descriptor[cluster]["yi1"], yi1)
            push!(descriptor[cluster]["zi1"], zi1)
            push!(descriptor[cluster]["ri1"], p2)

            push!(descriptor[cluster]["xi2"], xi2)
            push!(descriptor[cluster]["yi2"], yi2)
            push!(descriptor[cluster]["zi2"], zi2)
            push!(descriptor[cluster]["ri2"], p3)

            push!(descriptor[cluster]["xi_i1"], xi_i1)
            push!(descriptor[cluster]["yi_i1"], yi_i1)
            push!(descriptor[cluster]["zi_i1"], zi_i1)

            push!(descriptor[cluster]["xi_i2"], xi_i2)
            push!(descriptor[cluster]["yi_i2"], yi_i2)
            push!(descriptor[cluster]["zi_i2"], zi_i2)

            push!(descriptor[cluster]["xi1_i2"], xi1_i2)
            push!(descriptor[cluster]["yi1_i2"], yi1_i2)
            push!(descriptor[cluster]["zi1_i2"], zi1_i2)

            push!(descriptor[cluster]["rii1"], rii1)
            push!(descriptor[cluster]["rii2"], rii2)
            push!(descriptor[cluster]["ri1i2"], ri1i2)


        end
    end
    ThreeBody{eltype(atomic), 1}(idx, number, position, envidcs, envnumbers,
          envpositions, threebody.threebodycutoff, descriptor)
end



"""
idx : Index of self
specie : Periodic number of self
position : Cartesian position of self
envidcs : Index of atoms within BallTree
envnumbers : Cartesian coordinates of atoms within BallTree
envpositions : Periodic number of each atoms
cutoff
descriptor : cache
"""
struct TwoThreeBody{T, DN} <: Descriptor{T, DN}
    twobody::TwoBody
    threebody::ThreeBody
    descriptor::Dict
    # idx::Int
    # number::Int
    # position::Array{T, 1}
    # envidcs::Array{Int, 1}
    # envnumbers::Array{Int, 1}
    # envpositions::Array{T, 2}
    # cutoff::Number
    # descriptor::Dict
end

@inline TwoThreeBody() = TwoThreeBody{Float64, 1}(TwoBody(), ThreeBody(), Dict())

Base.convert(::Type{TwoThreeBody{T, DN}}, not::Nothing) where {T<:Number, DN} = TwoThreeBody()
Base.convert(::Type{TwoThreeBody{T}}, not::Nothing) where {T<:Number} = TwoThreeBody()
Base.convert(::Type{TwoThreeBody}, not::Nothing) = TwoThreeBody()

Base.copy(d::TwoThreeBody{T, DN}) where {T<:Number, DN} = #=
=# TwoThreeBody{T, DN}(copy(d.twobody), copy(d.threebody), copy(d.descriptor))

struct TwoThreeBodyFromAtomicDescriptor; twobodycutoff; threebodycutoff; end
Base.convert(::Type{TwoThreeBody}, atomic::AtomicDescriptor;
             twobodycutoff=7, threebodycutoff=3.5) =  #=
    =#TwoThreeBodyFromAtomicDescriptor(twobodycutoff, threebodycutoff)(atomic)

function (twothreebody::TwoThreeBodyFromAtomicDescriptor)(atomic)

    twobody = TwoBodyFromAtomicDescriptor(twothreebody.twobodycutoff)(atomic)
    threebody = ThreeBodyFromAtomicDescriptor(twothreebody.threebodycutoff)(atomic)
    dtype = eltype(atomic)
    descriptor = Dict()
    for (k, v) in pairs(twobody.descriptor)
        descriptor[k] = v
    end
    for (k, v) in pairs(threebody.descriptor)
        descriptor[k] = v
    end
    TwoThreeBody{dtype, 1}(twobody, threebody, descriptor)
end
