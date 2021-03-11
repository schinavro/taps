"""
using Pkg; Pkg.add("LBFGSB"); Pkg.add("FFTW"); Pkg.add("ASE")
import Conda
run(`conda create -n conda_jl python conda`)
ENV["CONDA_JL_HOME"] = "/home/schinavro/anaconda3/envs/conda_jl"

ENV["PYTHON"] = "/home/schinavro/anaconda3/bin/python"
Pkg.build("PyCall")
"""

import NPZ
import Pickle
import JSON


import MPI
# using Sockets: UDPSocket, listen, accept #isopen, async
using Sockets
import Base.println, Base.print

Models = include("./models.jl")
Regression = include("./regression.jl")

MPI.Init()

comm = MPI.COMM_WORLD
nprc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
root = 0

mFloat64 = MPI.Datatype(Float64)
mUInt8 = MPI.Datatype(UInt8)

host = Sockets.IPv4(ARGS[1])
port = parse(Int64, ARGS[2])
tcp_server = rank == root ? listen(host, port) : nothing
udp = rank == root ? bind(UDPSocket(), host, port + 1) : nothing

function println(s::UDPSocket, line::Any)
    _ = send(s, host, port + 2, [line..., b"\n"...])
end
function print(s::UDPSocket, line::Any)
    _ = send(s, host, port + 2, line)
end

function _gather_results(_results, properties)
    results = Dict()
    if N < nprc
        for property in properties
            results[Symbol(property)] = _results[Symbol(property)]
        end
        return results
    end

    if "potential" in properties
        potential = zeros(Float64, N)
        sendbuf = _results[:potential]
        recvbuf = MPI.VBuffer(potential, quant, partn, mFloat64)
        MPI.Gatherv!(sendbuf, recvbuf, root, comm)
        results[:potential] = potential
    end

    if "potentials" in properties
        potentials = zeros(Float64, N*A)
        sendbuf = _results[:potentials]
        recvbuf = MPI.VBuffer(potentials, quant .* A, partn .* A, mFloat64)
        MPI.Gatherv!(sendbuf, recvbuf, root, comm)
        results[:potentials] = reshape(potentials, A, N)
    end

    if "gradients" in properties
        gradients = zeros(Float64, N*D)
        sendbuf = _results[:gradients]
        recvbuf = MPI.VBuffer(gradients, quant .* D, partn .* D, mFloat64)
        MPI.Gatherv!(sendbuf, recvbuf, root, comm)
        if A == 1
            results[:gradients] = reshape(gradients, D, N)
        elseif A > 1
            results[:gradients] = reshape(gradients, 3, A, N)
        end
    end

    if "hessian" in properties
        sendbuf = _results[:hessian]
        recvbuf = nothing
        hessian = nothing
        D² = D * D
        if A == 1
            if rank == root
                hessian = zeros(Float64, N*D²)
                recvbuf = MPI.VBuffer(hessian, quant .* D², partn .* D²,
                                      mFloat64)
            end
            MPI.Gatherv!(sendbuf, recvbuf, root, comm)
            results[:hessian] = rank == root ? reshape(hessian, D, D, N) : nothing
        elseif A > 1
            if rank == root
                hessian = zeros(Float64, N*D²)
                recvbuf = MPI.VBuffer(hessian, quant .* D², partn .* D²,
                      mFloat64)
            end
            MPI.Gatherv!(sendbuf, recvbuf, root, comm)
            results[:hessian] = rank == root ? reshape(hessian, 3, A, 3, A, N) : nothing
        end
    end

    if "covariance" in properties
        sendbuf = _results[:covariance]
        recvbuf = nothing
        covariance = nothing
        if rank == root
            covariance = zeros(Float64, N)
            recvbuf = MPI.VBuffer(covariance, quant, partn, mFloat64)
        end
        MPI.Gatherv!(sendbuf, recvbuf, root, comm)
        results[:covariance] = rank == root ? reshape(covariance, N) : nothing
    end

    return results
end
#######
while true
    global D, A, N, _N, coords, quant, partn, tcp, input_dict, _shape, model_kwargs
    global model, remainder, temp
    local DAN_bytes, crd_byts, coords_buf
    local input_bytes, instruction
    rank == root ? tcp = accept(tcp_server) : tcp = nothing
    instruction = rank == root ? readavailable(tcp) : nothing
    # MPI.Barrier(comm)
    instruction = MPI.bcast(instruction, root, comm)
    # rank == root ? println(udp, instruction) : nothing


    if instruction == b""
        nothing
    elseif instruction == b"shutdown\n"
        _ = rank == root ? write(tcp, b"Shutting down julia\n") : nothing
        break
    elseif instruction == b"standby\n"
        _ = rank == root ? write(tcp, b"Roger standby\n") : nothing
    elseif instruction == b"construct input\n"
        _ = rank == root ? write(tcp, b"Roger standby\n") : nothing
        input_bytes = nothing
        if rank == root
            input_size = reinterpret(Int64, read(tcp, 8))[1]
            input_bytes = read(tcp, input_size)
        end
        # MPI.Barrier(comm)
        input_bytes = MPI.bcast(input_bytes, root, comm)
        input_dict = JSON.parse(String(input_bytes))

    elseif instruction == b"update model_kwargs\n"
        _ = rank == root ? write(tcp, b"Roger standby\n") : nothing
        kwargs_bytes = nothing
        if rank == root
            kwargs_size = reinterpret(Int64, read(tcp, 8))[1]
            kwargs_bytes = read(tcp, kwargs_size)
        end
        # MPI.Barrier(comm)
        kwargs_bytes = MPI.bcast(kwargs_bytes, root, comm)
        new_model_kwargs = Dict{Symbol, Any}()
        _k = Dict{Any, Any}(JSON.parse(String(kwargs_bytes)))
        for (key, value) = pairs(_k); new_model_kwargs[Symbol(key)] = value; end
        merge!(model_kwargs, new_model_kwargs)
        for (key, value) in pairs(new_model_kwargs)
            if key == :label
                value = String(value)
            elseif key == :data_ids
                value = Dict{String, Array{Int, 1}}(value)
            end
            setproperty!(model, key, value)
        end

    elseif instruction == b"construct coords\n"
        _ = rank == root ? write(tcp, b"Prepare for coords\n") : nothing
        DAN_bytes = rank == root ? read(tcp, 24) : nothing
        DAN_bytes = MPI.bcast(DAN_bytes, root, comm)
        D, A, N = reinterpret(Int64, DAN_bytes)

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
        # MPI param calc fin

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
    elseif instruction == b"construct model\n"
        model_name = String(input_dict["model"])

        model_kwargs = Dict{Symbol, Any}()
        _kw = Dict{Any, Any}(input_dict["model_kwargs"])
        for (key, value) = pairs(_kw); model_kwargs[Symbol(key)] = value; end
        model = getfield(Models, Symbol(model_name))(;model_kwargs...)

    elseif instruction == b"construct_pathfinder\n"
        nothing

    elseif instruction == b"model regression\n"
        rank == root ? write(tcp, b"Prepare for regression\n") : nothing
        Regression.regression!(model, MPI, comm)
        rank == root ? write(tcp, b"Finished regression\n") : nothing


    elseif instruction[1] == b"0"[1] # Calculate + properties + return"
        local properties = Array{String, 1}(JSON.parse(String(instruction[2:end])))
        # calculate
        _results = model(coords, properties)
        # Gather to root
        results = _gather_results(_results, properties)
        # Retrive results
        if rank == root
            for (key, value) in results
                local rank, shape, meta, header
                rank = reinterpret(UInt8, [ndims(value)])
                shape = reinterpret(UInt8, [size(value)...])
                meta = [rank..., shape..., Array{UInt8, 1}(String(key))...]
                write(tcp, [reinterpret(UInt8, [size(meta)...])..., meta...])
                write(tcp, value)
            end
        end
        MPI.Barrier(comm)

    elseif instruction == b"optimize\n"
        nothing

    elseif instruction == b"retrive_results\n"
        if rank == root
            for (key, value) in results
                local rank, shape, header
                rank = reinterpret(UInt8, [ndims(value)])
                shape = reinterpret(UInt8, [size(value)])
                header = [rank..., shape..., Array{UInt8, 1}(String(key))]
                write(tcp, header)
                write(tcp, value)
            end
        end
        MPI.Barrier(comm)

    elseif instruction[2] == UInt8(1) # calculate + properties + coords + return
        nothing

    else
        reply = "Please choose between `calculate`, `optimize`, `shutdown`, and `standby` "
        rank == root ? println(reply) : nothing
    end
end
rank == root ? close(tcp_server) : nothing
MPI.Barrier(comm)
MPI.Finalize()
######
