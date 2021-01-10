
"""
using Pkg; Pkg.add("LBFGSB"); Pkg.add("FFTW"); Pkg.add("ASE")
import Conda
run(`conda create -n conda_jl python conda`)
ENV["CONDA_JL_HOME"] = "/home/schinavro/anaconda3/envs/conda_jl"

ENV["PYTHON"] = "/home/schinavro/anaconda3/bin/python"
Pkg.build("PyCall")
"""


using NPZ
import MPI
MPI.Init()

comm = MPI.COMM_WORLD
nprc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
root = 0
"""
int remainder = N % size;
int local_counts[size], offsets[size];
int sum = 0;
for (int i = 0; i < size; i++) {
 local_counts[i] = N / size;
 if (remainder > 0) {
  local_counts[i] += 1;
  remainder--;
 }
 offsets[i] = sum;
 sum += local_counts[i];
}
int localArray[local_counts[rank]];
if (rank == ROOT) {
 for (int i = 0; i < N; i++) {
  A[i] = rand() % 10;
 }
}
MPI_Scatterv(A, local_counts, offsets, MPI_INT, localArray, local_counts[rank], MPI_INT, ROOT, MPI_COMM_WORLD);
//---------------SORT THE localArray-------------------
MPI_Gatherv(localArray, local_counts[rank], MPI_INT, A, local_counts, offsets, MPI_INT, ROOT, MPI_COMM_WORLD);
MPI_Finalize();
return 0;
}
"""

println("I am rank " * string(rank) * " of nprc " * string(nprc))
indict = npzread(ARGS[1])
if rank == root
    println(size(indict["coords"]))
end
# indict = MPI.bcast(indict, root, comm)

coords = indict["coords"]
properties = indict["properties"]
model_model = indict["model_model"]
model_label = indict["model_label"]
model_potential_unit  = indict["model_potential_unit"]
model_data_ids = indict["model_data_ids"]
model_prj = indict["model_prj"]
model_kwargs = indict["model_kwargs"]
# coords_epoch = indict["coords_epoch"]
# coords_unit = indict["coords_unit"]
# finder_finder = indict["finder_finder"]
# finder_prj = indict["finder_prj"]
# finder_label = indict["finder_label"]

Models = include("./models.jl")

model = getfiled(Models, Symbol(model_model))(model_label,
                                              model_potential_unit,
                                              model_data_ids,
                                              model_kwargs...)


results = model(coords, properties)

# Gain!

dir = (fin - init) / t0
ν = π/t0

x(θ, t::Real) = init + dir .* t + 2 * sum((θ .* sin.(ν*t * [1:1:Nk;])), dims=1) / (2*(Nk + 1))   # (1 x D)
ẋ(θ, t::Real) = dir .+ 2 * ν * sum((θ .*[1:1:Nk;].* cos.(ν * t * [1:1:Nk;])), dims=1) / (2 * (Nk + 1))
ẍ(θ, t::Real) = -2 * ν * ν * sum((θ .* ([1:1:Nk;].^2) .* sin.(ν * t * [1:1:Nk;])), dims=1) / (2 * (Nk + 1))
#ak = (rand(Nk, D) .- 0.5) .* 0.5 .* LinRange(1, 0, Nk) .* LinRange(1, 0, Nk)

A = [-200., -100., -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0., 0., 11., 0.6]
c = [-10, -10, -6.5, 0.7]
x0 = [1, 0, -0.5, -1]
y0 = [0, 0.5, 1.5, 1]              # (4, )

"""
Kinetic
"""
function T(θ, t)
    𝐯 = ẋ(θ, t)
    return 0.5 * 1 * sum(𝐯 .* 𝐯)
end
"""
Muller Brown potential
"""
function V(θ, t)
    # 1 x D
    𝑥, 𝑦 = x(θ, t)
    # (4, )
    Vk = A .* exp.(a.*(𝑥 .- x0).^2 + b.*(𝑥.-x0).*(𝑦.-y0) + c .* (𝑦.-y0).^2)
    return sum(Vk) / 100
end
"""
Muller Brown gradient
"""
function ∇V(θ, t)
    # 1 x D
    𝑥, 𝑦 = x(θ, t)
    # (4, )
    𝑥_x0 = 𝑥 .- x0
    𝑦_y0 = 𝑦 .- y0
    # (4, )
    Vk = @. A * exp(a * (𝑥_x0)^2 + b*(𝑥_x0)*(𝑦_y0) + c * (𝑦_y0)^2)
    # (1)
    Fx = sum(@. Vk * (2 * a * 𝑥_x0 + b * 𝑦_y0))
    Fy = sum(@. Vk * (b * 𝑥_x0 + 2 * c * 𝑦_y0))
    # (2, )
    return [Fx, Fy] / 100
end
"""
Muller Brown hessian
"""
function ∇²V(θ, t)
    𝑥, 𝑦 = x(θ, t)            # 1 x D
    # (4, )
    𝑥_x0 = 𝑥 .- x0
    𝑦_y0 = 𝑦 .- y0
    # (4, )
    Vk = @. A * exp(a * (𝑥_x0)^2 + b*(𝑥_x0)*(𝑦_y0) + c * (𝑦_y0)^2)
    dx = @. (2 * a * 𝑥_x0 + b * 𝑦_y0)
    dy = @. (b * 𝑥_x0 + 2 * c * 𝑦_y0)
    # (4, ) -> (1)
    Hxx = sum(@. Vk * (2 * a + dx * dx))
    Hxy = sum(@. Vk * (b + dx * dy))
    Hyy = sum(@. Vk * (2 * c + dy * dy))
    return [Hxx Hxy; Hxy Hyy] / 100
end
"""
Lagrangian
"""
function ADMD(θ, t)
    _T = T(θ, t)
    _V = V(θ, t)
    return _T - _V + muE * (_T+_V - Et) * (_T+_V - Et)
end

"""
Feynman-Kac
"""
muE = 200
Et = -0.47
β = 3
γ = 8 * 3
𝒟 = 1 / (β*γ)

function OnsagerMachlup(θ, t)
    _T = T(θ, t)
    _V = V(θ, t)

    # F = ∇V(θ, t)
    # ℍ = ∇²V(θ, t)
    # 𝐯 = vec(ẋ(θ, t))
    # ℍ𝐯 = ℍ * 𝐯
    # return β^2 * 𝒟 / 4 * (F'F) - 𝒟 * β / 2 * (ℍ𝐯'ℍ𝐯) + muE * (_T+_V - Et) * (_T+_V - Et)
    F = ∇V(θ, t)
    𝐯 = vec(ẋ(θ, t))
    Fv = F * 𝐯
    return β^2 * 𝒟 / 4 * (F'F) - 𝒟 * β / 2 * sqrt(Fv'Fv) + muE * (_T+_V - Et) * (_T+_V - Et)
end

function OnsagerMachlup1(θ, t)
    dV = ∇V(θ, t)
    𝐚 = vec(ẍ(θ, t))
    # ∇V𝐚 = dV .* 𝐚
    #- 2 * (∇V𝐚'∇V𝐚)
    ℍ = ∇²V(θ, t)
    𝐯 = vec(ẋ(θ, t))
    ℍ𝐯 = ℍ * 𝐯
    # 2 * (ℍ𝐯'ℍ𝐯)
    return γ*(𝐚'𝐚) + (dV'dV)/2γ - (ℍ𝐯'𝐚)
end

function OnsagerMachlup4(θ, t)
    dV = ∇V(θ, t)
    𝐚 = vec(ẍ(θ, t))
    # ∇V𝐚 = dV .* 𝐚
    #- 2 * (∇V𝐚'∇V𝐚)
    ℍ = ∇²V(θ, t)
    𝐯 = vec(ẋ(θ, t))
    ℍ𝐯 = ℍ * 𝐯
    # 2 * (ℍ𝐯'ℍ𝐯)
    return (dV'dV) - (ℍ𝐯'ℍ𝐯)
end

function OnsagerMachlup2(θ, t)
    dV = ∇V(θ, t)
    # 𝐯 = vec(ẋ(θ, t))
    𝐚 = vec(ẍ(θ, t))
    ∇V𝐚 = dV .* 𝐚
    # ℍ = ∇²V(θ, t)
    # ℍ𝐯 = ℍ * 𝐚
    # 2 * (ℍ𝐯'ℍ𝐯)
    return (𝐚'𝐚) + (dV'dV) - 2 * (∇V𝐚'∇V𝐚)
end

function OnsagerMachlup2(θ, t)
    dV = ∇V(θ, t)
    dV_ = ∇V(θ, t + 0.001)
    # 𝐯 = vec(ẋ(θ, t))
    𝐚 = vec(ẍ(θ, t))
    ∇V𝐚 = dV .* 𝐚
    # ℍ = ∇²V(θ, t)
    # ℍ𝐯 = ℍ * 𝐚
    # 2 * (ℍ𝐯'ℍ𝐯)
    return (𝐚'𝐚) + (dV'dV) - 2 * (∇V𝐚'∇V𝐚)
end


using QuadGK
"""
Action
"""
function S(θ)
    dt0 = t0 / nprc
    a = rank * dt0
    b = (rank + 1) * dt0
    _S = quadgk(t -> L(θ, t), a, b, rtol=1e-8)[1]
    _S = MPI.Reduce(_S, (S1, S2) -> S1 + S2, root, comm)
    return _S
end
"""
Onsager Machlup action
"""
function S2(θ)
    dt0 = t0 / nprc
    a = rank * dt0
    b = (rank + 1) * dt0
    _S = quadgk(t -> L(θ, t), a, b, rtol=1e-8)[1]
    _S = MPI.Reduce(_S, (S1, S2) -> S1 + S2, root, comm)
    return _S
end


using ForwardDiff

function dS2(g, θ)
    g = vec(ForwardDiff.gradient(S, θ))
    println("dS: ", max(abs.(g)...))
end

function dS(g, θ)
    Nk, D = size(θ)
    n = Nk * D
    # a = div(Nk * D, nprc) * rank + 1
    # b = div(Nk * D, nprc) * (rank + 1)
    # obj.g = MPI.Gather(dSi, root)
    function 𝒮(θ)
        return quadgk(t -> L(θ, t), 0, t0, rtol=1e-8)[1]
    end
    g[1:n] = vec(ForwardDiff.gradient(𝒮, θ))
    println("dS: ", max(abs.(g)...))
end

if indict["L"] == 1
    L = ADMD
elseif indict["L"] == 2
    L = OnsagerMachlup
else
    L = OnsagerMachlup
end

# using Optim
module_mpi = include("./lbfgsb_mpi.jl")
optimizer = module_mpi.L_BFGS_B(Nk * D, 40)
result, resk = optimizer(S, dS, ak, comm, m=20)


# result = optimize(S, ak, BFGS(),
#                       Optim.Options(g_tol = 1e-16, iterations=1000),
#                       autodiff=:forward)
# resk = Optim.minimizer(result)

if rank == root
    # println(map(t -> x(resk, t), LinRange(0, t0, Nk +2)))
    npzwrite("result.npz", Dict("resk" => resk))
end

MPI.Barrier(comm)
MPI.Finalize()
