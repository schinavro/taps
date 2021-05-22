
# module SocketIOs
# export SocketIO, run_taps

import JSON
import Base.read, Base.write, Base.sleep

using Sockets

include("../utils/antenna.jl")
Taps = include("../taps.jl")

using .Taps


mutable struct SocketIO
            host; port; clienthost; clientport; tcp; tcp_server; keep_open; is_tcp_opened; MPI; comm; root; nprc; rank; mFloat64; mUInt8; cache::Dict;
end

function Base.setproperty!(value::SocketIO, name::Symbol, x)
    if name == :host
        x = Sockets.IPv4(host)
    end
    setfield!(value, name, x)
end


SocketIO(;host="127.0.0.1", port=6543, clienthost="127.0.0.1", clientport=6544,
          keep_open=true, is_tcp_opened=false, MPI=nothing, comm=nothing,
          root=0, nprc=nothing, rank=nothing,
          mFloat64=nothing, mUInt8=nothing, cache=Dict(), kwargs...) = begin
    host = Sockets.IPv4(host)
    clienthost = Sockets.IPv4(clienthost)
    tcp_server = rank != root ? nothing : is_tcp_opened ? nothing : listen(host, port)
    is_tcp_opened = true
    # It should be nothing at the beginning
    tcp = nothing

    SocketIO(host, port, clienthost, clientport, tcp, tcp_server, keep_open, is_tcp_opened, MPI, comm, root, nprc, rank, mFloat64, mUInt8, cache)
end

function run_taps(io::SocketIO)
    # Pre define static variables, remove `io.`
    host = io.host; port = io.port; tcp_server = io.tcp_server;
    MPI = io.MPI; comm = io.comm; root = io.root; nprc = io.nprc; rank = io.rank;
    mFloat64 = io.mFloat64; mUInt8 = io.mUInt8
    # MPI set menu
    mpi_args = [MPI, comm, root, nprc, rank]
    mpi_type_args = [mFloat64, mUInt8]
    mpi_full_args = [mpi_args..., mpi_type_args...]
    # socket set menu
    socket_args = [host, port, tcp_server]

    # Start
    while io.keep_open
        io.tcp = rank == root ? accept(tcp_server) : nothing
        len = rank == root ? reinterpret(Int64, read(io.tcp, 8))[1] : nothing
        instruction = rank == root ? read(io.tcp, len) : nothing
        rank == root ? poke(mpi_args...) : sleep(mpi_args...)
        rank == root ? echo(io, Array{UInt8, 1}(reinterpret(UInt8, [len]))) : nothing
        operate(io, instruction)
    end
end

function operate(io::SocketIO, instruction::T) where {T<:Union{Array{UInt8, 1}, Nothing}}
    instruction = instruction == nothing ? b"1" : instruction
    instructiontype = [io.MPI.bcast(instruction[1], io.root, io.comm)]
    if instructiontype == b"*" # WildCard Instruction
        instruction = io.MPI.bcast(instruction, io.root, io.comm)
        eval(Meta.parse(String(instruction[2:end])))
    elseif instructiontype in [b"0", b"1", b"2"] # Semi wild card
        instruction = io.MPI.bcast(instruction, io.root, io.comm)
        howlongisurname = reinterpret(Int64, instruction[2:9])[1]
        method = String(instruction[10:9+howlongisurname])
        argsbytes = instruction[10+howlongisurname:end]
        args, kwargs = unpacking(Array{UInt8, 1}(argsbytes[9:end]))
        if instructiontype == b"0"
            eval(Meta.parse(method))(args...; kwargs...)
        elseif instructiontype == b"1"
            eval(Meta.parse(method))(io, args...; kwargs...)
        elseif instructiontype == b"2"
            eval(Meta.parse(method))(io.cache["paths"], args...; kwargs...)
        end
    end
end

function poke(MPI, comm, root, nprc, rank)
    send_temp = Array{Int64}(undef, 1)
    fill!(send_temp, Int64(0))
    for dst=1:nprc-1 MPI.Isend(send_temp, dst, root+32, comm) end
end

function sleep(MPI, comm, root, nprc, rank)
    while !(MPI.Iprobe(root, root+32, comm)[1]) sleep(0.001) end
    recv_temp = Array{Int64}(undef, 1)
    req = MPI.Irecv!(recv_temp, root, root+32, comm)
end

echo(io::SocketIO, instruction::Array{UInt8, 1}) = write(io.tcp, instruction)
ping(io::SocketIO) = io.rank == io.root ? write(io.tcp, reinterpret(UInt8, [time()])) : nothing
shutdown(io::SocketIO) = (io.keep_open = false)

function set_client(io::SocketIO; clienthost="127.0.0.1", clientport=6544)
    io.clienthost = Sockets.IPv4(clienthost)
    io.clientport = clientport
end

"""
Dict(
"coords"=> class_of_coords
"coords_kwargs" => Dict("coords"=>Array, ...)
"model"=> class_of_model
"model_kwargs"=> Dict("model")

"""
function read(io::SocketIO, args...; kwargs...)
    io.cache["paths"] = Paths(args...; kwargs...)
end

function read_parallel(io::SocketIO, args...; kwargs...)
    host, port = io.clienthost, io.clientport
    args = [args...]
    kwargs = Dict{Symbol, Any}([kwargs...])
    ptcp = connect(host, port)
    write(ptcp, reinterpret(UInt8, [io.rank]))
    len = reinterpret(Int64, read(ptcp, 8))[1]
    bytesarr = read(ptcp, len)
    pargs, pkwargs = unpacking(bytesarr)
    # Merge two lists that element is nothing.
    for i=1:length(args)
        args[i] = args[i] == nothing ? pargs[i] : args[i]
    end
    merge!(kwargs, pkwargs)
    read(io, args...; kwargs...)
end

function update(io::SocketIO, args...; kwargs...)
    for (key, value) in pairs(kwargs)
        setproperty!(io.cache["paths"], key, value)
    end
end

function update_parallel(io::SocketIO, args...; kwargs...)
    host, port = io.clienthost, io.clientport
    args = [args...]
    kwargs = Dict{Symbol, Any}([kwargs...])

    ptcp = connect(host, port)
    write(ptcp, reinterpret(UInt8, [io.rank]))
    len = reinterpret(Int64, read(ptcp, 8))[1]
    bytesarr = read(ptcp, len)
    pargs, pkwargs = unpacking(bytesarr)
    # Merge two lists that element is nothing.
    for i=1:length(args)
        args[i] = args[i] == nothing ? pargs[i] : args[i]
    end
    merge!(kwargs, pkwargs)
    update(io, args...; kwargs...)
end

function write(io::SocketIO, args...; sender=0, kwargs...)
    host, port = io.clienthost, io.clientport
    if io.rank == sender
        paths = io.cache["paths"]
        pargs = Array{Any, 1}([])
        pkwargs = Dict()
        for key in args
            push!(pargs, getproperty(paths, key))
        end
        for (key, value) in pairs(kwargs)
            pkwargs[key] = getproperty(paths, key, value)
        end
        argsbytes = packing(pargs...; pkwargs...)

        tcp = connect(host, port)
        write(tcp, argsbytes)
    end
end

function write_parallel(io::SocketIO, args...; senders="all", kwargs...)
    host, port = io.clienthost, io.clientport
    args = [args...]
    kwargs = Dict{Symbol, Any}([kwargs...])

#    sender = senders == "all" ? io.rank : io.rank in senders ? io.rank : nothing

#    if sender == io.rank
        ptcp = connect(host, port)
        write(ptcp, reinterpret(UInt8, [io.rank]))
        len = reinterpret(Int64, read(ptcp, 8))[1]
        bytesarr = read(ptcp, len)
        pargs, pkwargs = unpacking(bytesarr)

        args = cat(args, pargs, dims=1)
        merge!(kwargs, pkwargs)

        paths = io.cache["paths"]
        pathargs = Array{Any, 1}([])
        pathkwargs = Dict()
        for key in args
            push!(pathargs, getproperty(paths, key))
        end
        for (key, value) in pairs(kwargs)
            pathkwargs[key] = getproperty(paths, key, value)
        end
        argsbytes = packing(pathargs...; pathkwargs...)
        write(ptcp, argsbytes)
end
#
# function get_properties(io::SocketIO, properties, args...; kwargs...)
#     paths = io.cache["paths"]
#     get_properties(paths, properties, args...; kwargs...)
# end

# end
