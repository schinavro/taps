using NPZ
import MPI
MPI.Init()

comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
root = 0

println("I am rank " * string(rank) * " of size " * string(size))
indict = npzread(ARGS[1])
if rank == root
    print(indict)
end
# indict = MPI.bcast(indict, root, comm)

t0 = indict["epoch"]
Nk = indict["Nk"]
D = indict["D"]
init = indict["init"]
fin = indict["fin"]

ak = indict["ak"]

dir = (fin - init) / t0
ν = π/t0

x(θ, t::Real) = init + dir .* t + 2 * sum((θ .* sin.(ν*t * [1:1:Nk;])), dims=1) / (2*(Nk + 1))   # (1 x D)
ẋ(θ, t::Real) = dir .+ 2 * sum((θ *ν.*[1:1:Nk;].* cos.(ν * t * [1:1:Nk;])), dims=1) / (2 * (Nk + 1))
#ak = (rand(Nk, D) .- 0.5) .* 0.5 .* LinRange(1, 0, Nk) .* LinRange(1, 0, Nk)

A = [-200., -100., -170, 15]
a = [-1, -1, -6.5, 0.7]
b = [0., 0., 11., 0.6]
c = [-10, -10, -6.5, 0.7]
x0 = [1, 0, -0.5, -1]
y0 = [0, 0.5, 1.5, 1]              # 1 x 4

"""
Muller Brown potential
"""
function V(θ, t)
    𝑥, 𝑦 = x(θ, t)            # 1 x D
    Vk = A .* exp.(a.*(𝑥 .- x0).^2 + b.*(𝑥.-x0).*(𝑦.-y0) + c .* (𝑦.-y0).^2) / 100
    return sum(Vk)
end
"""
Kinetic
"""
function T(θ, t)
    𝐯 = ẋ(θ, t)
    return 0.5 * 1 * sum(𝐯 .* 𝐯)
end
muE = 10
Et = -1.47

"""
Lagrangian
"""
function L(θ, t)
    _T = T(θ, t)
    _V = V(θ, t)
    return _T - _V + muE * (_T+_V - Et) * (_T+_V - Et)
end

using QuadGK
"""
Action
"""
function S(θ, comm=comm)
    dt0 = t0 / size
    a = rank * dt0
    b = (rank + 1) * dt0
    return quadgk(t -> L(θ, t), a, b, rtol=1e-8)[1]
end

function dS(g, θ, comm=comm)
    Nk, D = size(θ)
    a = div(Nk * D, size) * rank + 1
    b = div(Nk * D, size) * (rank + 1)
    function 𝒮(θi)
        _θ1D = vec(θ)
        _θ1D[a:b] = θi
        return quadgk(t -> L(reshape(_θ1D, Nk, D), t), 0, t0, rtol=1e-8)[1]
    end
    θ1D = vec(θ)
    return ForwardDiff.gradient(𝒮, θ1D[a:b])
end


# using Optim
module_mpi = include("./lbfgsb_mpi.jl")
optimizer = module_mpi.L_BFGS_B(Nk * D, 40)
result, x = optimizer(S, dS, ak, comm)


# result = optimize(S, ak, BFGS(),
#                       Optim.Options(g_tol = 1e-16, iterations=1000),
#                       autodiff=:forward)
# resk = Optim.minimizer(result)


if rank == root
    println(map(t -> x(resk, t), LinRange(0, t0, Nk +2)))
    npzwrite("result.npz", Dict("resk" => resk))
end

MPI.Barrier(comm)
MPI.Finalize()
