"""
using Pkg; Pkg.add("LBFGSB"); Pkg.add("FFTW"); Pkg.add("ASE")
import Conda
run(`conda create -n conda_jl python conda`)
ENV["CONDA_JL_HOME"] = "/home/schinavro/anaconda3/envs/conda_jl"

ENV["PYTHON"] = "/home/schinavro/anaconda3/bin/python"
Pkg.build("PyCall")
"""
module SocketIOs
using Sockets

import Base
import Base.println, Base.print

import JSON
Taps = include("../taps.jl")
using Taps

mutable struct SocketIO
    host; port; tcp; tcp_server; keep_open; is_tcp_opened; MPI; comm; root; nprc; rank; mFloat64; mUInt8; cache::Dict;
end
SocketIO(;host="127.0.0.1", port=6543, keep_open=true, is_tcp_opened=false,
          MPI=nothing, comm=nothing, root=0, nprc=nothing rank=nothing,
          mFloat64=nothing, mUInt8=nothing, cache=Dict(), kwargs...) = begin
    host = Sockets.IPv4(host)
    tcp_server = rank == root && is_tcp_opened ? listen(host, port) : nothing
    is_tcp_opened = true
    # It should be nothing at the beginning
    tcp = nothing
    SocketIO(host, port, tcp; tcp_server, keep_open, is_tcp_opened, MPI, comm, root, nprc, rank, mFloat64, mUInt8, cache)
end

function Base.setproperty!(value::SocketIO, name::Symbol, x)
    if name == :host
        x = Sockets.IPv4(host)
    end
    setfield!(value, name, x)
end

function run_taps(io::SocketIO)
    # Pre define static variables, remove `io.`
    host = io.host; port = io.port; tcp_server = io.tcp_server;
    MPI = io.MPI; comm = io.comm; root = io.root; nprc = io.nprc; rank = io.rank;
    mFloat64 = io.Float64; mUInt8 = io.mUInt8
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
        instruction = rank == root ? Instruction(read(io.tcp, len)) : nothing
        rank == root ? poke(mpi_args...) : sleep(mpi_args...)
        rank == root ? echo(io, instruction) : nothing
        eval(io, instruction)
        #getfield(typeof(io), Symbol(instruction))(io)
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

echo(io::SocketIO, instruction) = write(io.tcp, instruction)
shutdown(io::SocketIO) = (io.keep_open = false)

function read(io::SocketIO; kwargs...)
    c = io.cache
    get(c, "kwargs", nothing) == nothing ? c = Dict() : merge!(c["kwargs"], kwargs)
end

function read(arrbytes)
    rank = reinterpret(Int64, arrbytes[1:8])[1]
    shape = reinterpret(Int64, arrbytes[9:9 + 8rank])[1]
    symbol = reinterpret(String, arrbytes[9 + 8rank:n])
    array = reinterpret(Float64, arrbytes[n:end])
    coords = getfield(Coords, Symbol(symbol))(reshape(array, shape))
end

function read(io::SocketIO, coords::Array; kwargs...)
    c = io.cache
    c["coords"] = coords
    read(io; kwargs)
end

function write(io::SocketIO; kwargs...)
    write_input(io; kwargs...)
    write_results(io; kwargs...)
end

function write(io::SocketIO, coupang::Dict)
    package = Array{UInt8, 1}(JSON.json(coupang))
    weight = Int64(length(package))
    weight = reinterpret(UInt8, [weight])
    write(io.tcp, [weight; package])
end

function write(io::SocketIO, symbol::String, array::Array)
    rank = reinterpret(UInt8, [ndims(value)])
    shape = reinterpret(UInt8, [size(value)...])
    meta = [rank..., shape..., Array{UInt8, 1}(String(symbol))...]
    write(io.tcp, [reinterpret(UInt8, [length(meta)])..., meta...])
    write(io.tcp, value)
end

function write_input(io::SocketIO; orders=nothing, kwargs...)
    if rank == root
        paths = io.cache["paths"]
        coupang = Dict()
        for order in orders; coupang[order] = getproperty(paths, Symbol(order)); end
        write(io, coupang)
    end
    MPI.Barrier(comm)
end

function write_results(io::SocketIO; properties=[])
    results = Dict()
    for property in properties
        result = gather(io, property)
        results[property] = result
        rank == root ? write(io, property, result) : nothing
    end
end

function update(io::SocketIO; kwargs...)
    merge!(io.cache["input"], kwargs)
    for (key, value) in pairs(kwargs)
        setproperty!(io.cache["paths"], key, value)
    end
end

function update(io::SocketIO, coords::Array; kwargs...)
    paths = io.cache["paths"]
    paths.coords = coords
    update(io; kwargs...)
end

function update_model(io::SocketIO, kwargs...)
    merge!(io.cache["input"]["model_kwargs"], kwargs)
    model = io.cache["paths"].model
    for (key, value) in pairs(kwargs)
        setproperty!(model, key, value)
    end
end

function construct(io::SocketIO)
    io.cache["paths"] = Paths(io.cache["coords"]; io.cache["kwargs"]...)
end


# import MPI
# using Sockets: UDPSocket, listen, accept #isopen, async

# udp = rank == root ? bind(UDPSocket(), host, port + 1) : nothing
# function println(s::UDPSocket, line::Any)
#     _ = send(s, host, port + 2, [line..., b"\n"...])
# end
# function print(s::UDPSocket, line::Any)
#     _ = send(s, host, port + 2, line)
# end
