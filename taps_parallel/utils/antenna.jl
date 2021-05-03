
@inline compactify(array::Array{T}) where {T <: Number} = array
@inline compactify(array::T) where {T <: Union{Number, String, Bool, Nothing, Tuple}} = array

function compactify(array::Array{Any, 1})
    # Check all the type are same. If not, return as it is
    checker = typeof(array[1])
    for arr in array
        checker == typeof(arr) ? continue : nothing
        return array
        break
    end
    # Make new array with new element type array
    return compactify(Array{checker, 1}(array))
end

function compactify(array::Array{T, 1}) where {T <: Array}
    # Check if it can be compact
    len = length(array[1])
    compactable = true
    for arr in array
        len == length(arr) ? continue : compactable = false
        break
    end
    if compactable
        compat = array |> compactify
        return cat(compat..., dims=ndims(compat[1])+1)
    else
        return map(compactify, array)
    end
end

function compactify(array::T) where {T <: Dict}
    new_dict = Dict()
    for (key, value) in pairs(array)
        value = compactify(value)
        if typeof(value) <: Array{TT, 1} where {TT <: Number}
            nothing
        elseif typeof(value) <: Array{TT} where {TT <: Number}
            value = permutedims(value, ndims(value):-1:1 |> collect)
        end
        new_dict[Symbol(key)] = value
    end
    new_dict
end

"""
Helper function for instruction
"""
struct Instruction instruction::String end
Instruction(instruction::Array{UInt8}) = Instruction(String(instruction))

function Base.eval(io::SocketIO, instruction::Instruction)
    Mod = parentmodule(typeof(io))
    type = MPI.bcast(instruction.instruction[1], io.root, io.comm)
    if type in [b"*", b"1", b"2", b"3"]
        instruction = MPI.bcast(instruction.instruction, io.root, io.comm)
        inst = instruction
    end
    return if type == b"*" # WildCard Instruction
        eval(Metal.eval(String(inst)))
    elseif type == b"0"    # Zero argument
        getfield(Mod, Symbol(inst[2:end]))(io)
    elseif type == b"1"    # One arguments ex, b"1write,energy"
        instruct, arg = split(inst[2:end], ",")
        getfield(Mod, Symbol(instruct))(io, args)
    elseif type == b"2"    # More than One arguments ex, b"1write,energy,forces"
        instruct, args = split(inst[2:end], ",", limit=2)
        getfield(Mod, Symbol(instruct))(io, args...)
    elseif type == b"3"    # Arguments built with a JSON
        instruct, args = split(inst[2:end], ",", limit=2)
        kwargs = Utils.compactify(JSON.parse(String(args)))
        getfield(Mod, Symbol(instruct))(io; kwargs...)
    elseif type == b"4"    # Arguments built with a Binary Array
        ########SCATTER
        instruct, args = split(inst[2:end], ",", limit=2)
        array = Utils.read_array(args)
        getfield(Mod, Symbol(instruct))(io, args)
    elseif type == b"5"    # Arguments built with a numpy arr and JSON arr
        instruct, args = split(inst[2:end], ",", limit=2)
        narr, njson = reinterpret(Int64, read(io.tcp, 16))
        arrbytes = inst[16+2:njson]
        jsonbytes = inst[16+2:njson]
        kwargs = Utils.compactify(JSON.parse(String(jsonbytes)))
        args = Utils.read_array(arrbytes)
        getfield(Mod, Symbol(instruct))(io, args; kwargs...)
    elseif type == b"6"    # N args with a JSON kwargs
        instruct, args = split(inst[2:end], ",", limit=2)
        narr, njson = reinterpret(Int64, read(io.tcp, 16))
        arrbytes = inst[16+2:njson]
        jsonbytes = inst[16+2:njson]
        kwargs = Utils.compactify(JSON.parse(String(jsonbytes)))
        args = Utils.read_array(arrbytes)
        getfield(Mod, Symbol(instruct))(io, args; kwargs...)
    end
end

"""
Helper function for sending data
"""
function mpi_parser(arr, N, nprc)
    # Efficient Coords MPI mpi initialize
    remainder = N % nprc
    quant = zeros(Int64, nprc)
    partn = zeros(Int64, nprc)
    temp = 0
    for i=1:nprc
        quant[i] = N ÷ nprc
        if remainder > 0
            quant[i] += 1
            remainder -= 1 # scope of variable
        end
        partn[i] = temp
        temp += quant[i]
    end
    return N
end

function gather(io::SocketIO, property)
    sendarr = io.cache["results"][property]
    _shape = size(sendarr)
    _N = _shape[1]
    N = MPI.reduce(_N, sum, io.root, io.comm)
    shape = (N, _shape[2:end]...)
    N, D, quant, partn = Utils.mpi_parser_param(_results[property])
end

function scatter(io::SocketIO, array)
    crd_byts = rank == root ? read(tcp, D * N * 8) : nothing
    # If N is too small, just calculate on every node.
    if N < nprc
        _N = N
        _shape = A == 1 ? (D, _N) : (3, A, _N)
        coords = zeros(Float64, D, N)
        crd_byts = MPI.bcast(crd_byts, root, comm)
        coords = reshape(reinterpret(Float64, crd_byts), _shape...)
        coords = Array{Float64}(coords)
    else
        _N = quant[rank + 1]
        _shape = A == 1 ? (D, _N) : (3, A, _N)
        coords = zeros(Float64, quant[rank + 1] * D)
        if rank == root
            crd_byts = MPI.VBuffer(crd_byts, quant .* 8D, partn .* 8D, mUInt8)
        end
        MPI.Scatterv!(crd_byts, coords, root, comm)
        coords = reshape(reinterpret(Float64, coords), _shape...)
        coords = Array{Float64}(coords)
    end
    model_name = input_dict[:model]
    model_kwargs = input_dict[:model_kwargs]
    model = getfield(Models, model_name)(;model_kwargs...)
end
