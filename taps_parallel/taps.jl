
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


import MPI

MPI.Init()

comm = MPI.COMM_WORLD
nprc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
root = 0

mpi_float64 = MPI.Datatype(Float64)

indict = Pickle.load(open(ARGS[1] * ".pkl"))

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
println("I am rank " * string(rank) * " of nprc " * string(nprc))
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

if rank == root
    NPZ.npzwrite("result.npz"; results...)
end

MPI.Barrier(comm)
MPI.Finalize()
