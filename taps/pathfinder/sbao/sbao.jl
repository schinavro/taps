using NPZ

indict = npzread(ARGS[1])

println(indict)
println(size(indict["ak"]))

t0 = indict["epoch"]
Nk = indict["Nk"]
D = indict["D"]
init = indict["init"]
fin = indict["fin"]

ak = indict["ak"]

dir = (fin - init) / t0
ν = π/t0

x(θ, t::Real) = init + dir .* t + sum((θ/100 .* sin.(ν*t * [1:1:Nk;])), dims=1)     # (1 x D)
ẋ(θ, t::Real) = dir .+ sum((θ/100. *ν.*[1:1:Nk;].* cos.(ν * t * [1:1:Nk;])), dims=1)
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
function S(θ)
    return quadgk(t -> L(θ, t), 0., t0, rtol=1e-8)[1]
end


using Optim

result = optimize(S, ak, BFGS(), Optim.Options(g_tol = 1e-16, iterations=1000),
                  autodiff=:forward)

resk = Optim.minimizer(result)

npzwrite("result.npz", Dict("resk" => resk))
