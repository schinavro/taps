
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


MPI.Init()

comm = MPI.COMM_WORLD
nprc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
root = 0

mpi_float64 = MPI.Datatype(Float64)

port = parse(Int64, ARGS[1])
tcp_server = rank == root ? listen(port) : nothing
udp = rank == root ? bind(UDPSocket(), ip"127.0.0.1", port + 1) : nothing

function println(s::UDPSocket, line::Any)
    send(s, ip"127.0.0.1", port + 2, [line..., b"\n"...])
end
function print(s::UDPSocket, line::Any)
    send(s, ip"127.0.0.1", port + 2, line)
end

#######
shutdown = false
while !shutdown
    tcp = rank == root ? accept(tcp_server) : nothing
    instruction = rank == root ? readavailable(tcp) : nothing
    MPI.Barrier(comm)
    instruction = MPI.bcast(instruction, root, comm)
    rank == root ? println(udp, instruction) : nothing

    if instruction == b"shutdown\n"
        rank == root ? write(tcp, b"Shutting down julia\n") : nothing
        global shutdown = true
        break
    elseif instruction == b"input_dict\n"
        global input_dict
        local input_bytes
        rank == root ? write(tcp, b"Roger standby\n") : nothing
        input_bytes = nothing
        if rank == root
            input_size = reinterpret(Int64, read(tcp, 8))[1]
            println(udp, "Received $input_size")
            input_bytes = read(tcp, input_size)
        end
        MPI.Barrier(comm)
        input_bytes = MPI.bcast(input_bytes, root, comm)
        input_dict = JSON.parse(String(input_bytes))
        continue
    elseif instruction == b"coords\n"
        global D, A, N, _N, coords
        local DAN_bytes, coords_bytes
        rank == root ? write(tcp, b"Recieving coords\n") : nothing
        coords_bytes = nothing

        DAN_bytes = rank == root ? read(tcp, 24) : nothing
        MPI.Barrier(comm)
        DAN_bytes = MPI.bcast(DAN_bytes, root, comm)
        D, A, N = reinterpret(Int64, DAN_bytes)

        coords_bytes = read(tcp, D * A * N)

        remainder = N % nprc
        local_counts = zeros(Int64, nprc)
        offsets = zeros(Int64, nprc)
        temp = 0
        for i=1:nprc
            local_counts[i] = N ÷ nprc
            if remainder > 0
                local_counts[i] += 1
                global remainder -= 1 # scope of variable
            end
            offsets[i] = temp
            global temp += local_counts[i]
        end

        _N = local_counts[rank + 1]
        _shape = (D, _N)

        sendbuf = nothing
        if rank == root
            # NxD or NxAx3
          coords_total = NPZ.npzread(ARGS[1] * ".npz", ["coords"])["coords"]
          println("Coords size", size(coords_total))
          # DN
          data = Array(vec(coords_total'))
          sendbuf = MPI.VBuffer(data, local_counts .* D, offsets .* D, mpi_float64)
        end

        recbuf = zeros(Float64, local_counts[rank + 1] * D)
        rank == root ? println("Scattering Data") : nothing
        MPI.Scatterv!(sendbuf, recbuf, root, comm)

        coords = Array(reshape(recbuf, _shape...)')
        continue

    elseif instruction == b"calculate\n"
        rank == root ? write(tcp, b"Recieving input data") : nothing
        results = calculate(indict, coords)
        rank == root ? write(tcp, results) : nothing
    elseif instruction == b"results\n"
        
    else
        reply = b"Please choose between `calculate`,
                 `optimize`, `shutdown`, and `standby` "
        rank == root ? write(tcp, reply) : nothing
        continue
    end
end
rank == root ? close(tcp_server) : nothing
MPI.Barrier(comm)
MPI.Finalize()
######

function calculate(indict)

D = Int64(indict["D"])
A = Int64(indict["A"])
N = Int64(indict["N"])
properties = Array{String, 1}(indict["properties"])
model_model = String(indict["model_model"])
model_label = String(indict["model_label"])
model_potential_unit  = String(indict["model_potential_unit"])
model_data_ids = Dict{String, Array{Int, 1}}(indict["model_data_ids"])
_kwargs = Dict{Any, Any}(indict["model_kwargs"])
model_kwargs = Dict{Symbol, Any}()
for (key, value) = pairs(_kwargs); model_kwargs[Symbol(key)] = value; end
coords_epoch = indict["coords_epoch"]
coords_unit = indict["coords_unit"]
finder_finder = indict["finder_finder"]
finder_prj = indict["finder_prj"]
finder_label = indict["finder_label"]



Models = include("./models.jl")

model = getfield(Models, Symbol(model_model))(model_label, model_potential_unit,
                                              model_data_ids;
                                              model_kwargs...)

println("Constrcted Model")


_results = model(coords, properties)

results = Dict()

if "potential" in properties
    potential = zeros(Float64, N)
    sendbuf = _results[:potential]
    println("sendbuf", size(sendbuf))
    recvbuf = MPI.VBuffer(potential, local_counts, offsets, mpi_float64)
    MPI.Gatherv!(sendbuf, recvbuf, root, comm)
    results[:potential] = potential
end

if "potentials" in properties
    potentials = zeros(Float64, N*A)
    sendbuf = _results[:potentials]
    recvbuf = MPI.VBuffer(potentials, local_counts .* A,
                          offsets .* A, mpi_float64)
    MPI.Gatherv!(sendbuf, recvbuf, root, comm)
    results[:potentials] = potentials
end

if "gradients" in properties
    gradients = zeros(Float64, N*D)
    sendbuf = _results[:gradients]
    recvbuf = MPI.VBuffer(gradients, local_counts .* D,
                          offsets .* D, mpi_float64)
    MPI.Gatherv!(sendbuf, recvbuf, root, comm)
    results[:gradients] = reshape(gradients, _shape...)
end

if "hessian" in properties
    sendbuf = _results[:hessian]
    recvbuf = nothing
    hessian = nothing
    DD = D * D
    if rank == root
        hessian = zeros(Float64, N*DD)
        recvbuf = MPI.VBuffer(hessian, local_counts .* DD, offsets .* DD,
                              mpi_float64)
    end
    MPI.Gatherv!(sendbuf, recvbuf, root, comm)
    results[:hessian] = rank == root ? reshape(hessian, N, D, D) : nothing
end

if "covariance" in properties
    sendbuf = _results[:covariance]
    recvbuf = nothing
    covariance = nothing
    NN = N
    if rank == root
        covariance = zeros(Float64, NN)
        recvbuf = MPI.VBuffer(covariance, local_counts .* NN, offsets .* NN,
                              mpi_float64)
    end
    MPI.Gatherv!(sendbuf, recvbuf, root, comm)
    results[:covariance] = rank == root ? reshape(covariance, N) : nothing
end

    return results
end