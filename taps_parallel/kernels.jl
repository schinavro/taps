module Kernels

export Standard, SE, Perioidc

using LinearAlgebra

abstract type Kernel end
struct Standard <: Kernel
    hyperparameters::Dict{String, Float64}
    data
    function Standard(hyperparameters)
        new(hyperparameters, nothing)
    end
end
"""
Xm : DxM
Xn : DxN
"""
function (ker::Standard)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2};
                            orig=false,
                            noise=false,
                            hyperparameters=nothing,
                            gradient_only=false,
                            hessian_only=false,
                            potential_only=false)

    hyperparameters = hyperparameters == nothing ? ker.hyperparameters :
                                                    hyperparameters

    println("Size of Xm, Xn: ", size(Xm), size(Xn))
    nax = [CartesianIndex()]
    ll = get(hyperparameters, "ll", 1.)
    σ_f = get(hyperparameters, "sigma_f", 1.)
    # Xn = Xn == nothing ? copy(Xm) : Xn
    D, M = size(Xm) # Xm : DxM
    N = size(Xn)[2] # Xn : DxN
    println(size(Xm), size(Xn))
    Xmn = Xm[:, :, nax] .- Xn[:, nax, :]               # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-dists / 2ll)      # M x N
    println(size(K))
    # for m=1:M
    #     for n=1:N
    #         ll = l[m] * l[n]
    #         K[m, n] = sigma_f * exp(-(Xm[m] - Xn[n])^2 / 2ll)
    #     end
    # end
    if orig
        noise_f = get(hyperparameters, "noise_f", 0)
        return @. K + noise_f * I(M)
    end

    # Derivative coefficient D x M x N
    dc_gd = -Xmn / ll
    # DxMxN x 1xMxN -> DxMxN -> MDxN
    Kgd = dc_gd .* K[nax, :, :]
    Kgd = vcat([Kgd[i, :, :] for i in 1:size(Kgd, 1)]...)
    println(size(Kgd))
    if potential_only
        return vcat(K, Kgd)                  # M(1+D) x N
    end
    # DxMxN * 1xMxN -> MxND
    Kdg = -dc_gd .* K[nax, :, :]
    Kdg = hcat([Kdg[i, :, :] for i in 1:size(Kdg, 1)]...)
    # DxMxN -> MxDxN
    Xnm = permutedims(Xmn, [2, 1, 3])
    # DxMx1xN  * 1xMxDxN  -> D x M x D x N
    dc_dd_glob = -Xmn[:, :, nax, :] .* Xnm[nax, :, :, :] / ll^2
    # dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
    # ∂_mn exp(Xn - Xm)^2
    dc_dd_diag = I(D)[:, nax, :, nax] / ll
    # DxMxDxN - DxMxDxN
    Kdd = @. (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
    # DM x DN
    Kdd = reshape(Kdd, D * M, D * N)
    if gradient_only
        # (D+1)M x DN
        return vcat(Kdg, Kdd)
    end
    if hessian_only
        # Delta _ dd
        dnm = [1:D;]
        # MxDxN
        dc_dg = -Xnm / ll
        # DxMxN * MxDxN -> MxDxDxN
        dc_hg_glob = -Xmn[:, :, nax, :] .* Xnm[nax, :, :, :] / ll^2
        # dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
        # DxNxM -> NxDxM -> NxDDxM
        dc_hg_diag = zeros(M, D, D, N)
        # ∂_mm K(Xm,Xn) MxDxDxN
        # dc_hg_diag[:, dnm, dnm, :] = -1 / ll
        for i in 1:D
            dc_hg_diag[:, i, i, :] .= -1 / ll
        end
        # MxDxDxN + MxDxDxN -> MxDxDxN
        Khg = (dc_hg_glob + dc_hg_diag)
        # Bacground term: ∂_mmn K(Xm,Xn) DxMx1x1xN * 1xMxDxDxN -> DxMxDxDxN
        dc_hd_back = dc_gd[:, :, nax, nax, :] * Khg[nax, :, :, :, :]
        # Diagonal term : DxMxDxDxN * 1xMxDxDxN -> DxMxDxDxN
        dc_hd_diag = zeros(D, M, D, D, N)
        # Global term :
        dc_hd_glob = zeros(D, M, D, D, N)
        # dc_hd_glob[dnm, :, dnm, :, :] += Xmn[nax, :, :, :] / ll / ll
        # dc_hd_glob[dnm, :, :, dnm, :] += Xmn[nax, :, :, :] / ll / ll
        for i in 1:D
            dc_hd_glob[i, :, i, :, :] += Xnm ./ ll / ll
            dc_hd_glob[i, :, :, i, :] += Xnm ./ ll / ll
        end

        # print(dc_hd_glob[0, 0, 0, :, 0])
        # print(dc_hd_glob.reshape(2, N, 2))
        Khd = (dc_hd_glob + dc_hd_diag) + dc_hd_back
        # MxDxDxN x Mx1x1xN -> MxDxDxN
        Khg *= K[:, nax, nax, :]
        # DxMxDxDxN * 1xMx1x1xN
        Khd *= K[nax, :, nax, nax, :]
        # MxDDxN -> M x DDN
        Khg = reshape(Khg, M, D * D * N)
        # DxMxDDxN -> DM x DDN
        Khd = reshape(Khd, D * M, D * D * N)
        # print(Khd.shape)
        return vcat(Khg, Khd)  # (D+1)M x DDN
    end
    println("Sizes K, Kgd, Kdg, Kdd", size(K), size(Kgd), size(Kdg), size(Kdd))
    Kext = [K Kdg;
            Kgd Kdd]  # (D+1)M x (D+1)N
    if noise
        noise_f = get(hyperparameters, "sigma_n^e", 0.)
        noise_df = get(hyperparameters, "sigma_n^f", 0.)
        noise = [noise_f .* ones(M)..., noise_df .* ones(D * M)...]
        return Kext .+ noise .* I((D + 1) * M)
    end
    return Kext
end

struct SE
    hyperparameters::Dict{String, Float64}
    function SE(hyperparameters)
        new(hyperparameters)
    end
end
"""
Xm : DxM
Xn : DxN
ll : D
"""
function (ker::SE)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2};
                            orig=false,
                            noise=false,
                            hyperparameters=nothing,
                            gradient_only=false,
                            hessian_only=false,
                            potential_only=false)

    hyperparameters = hyperparameters == nothing ? ker.hyperparameters :
                                                    hyperparameters

    nax = [CartesianIndex()]
    # Xn = Xn == nothing ? copy(Xm) : Xn
    D, M = size(Xm) # Xm : DxM
    N = size(Xn)[2] # Xn : DxN
    ll = [get(hyperparameters, "l$i", 1) for i in 1:D]
    σ_f = get(hyperparameters, "sigma_f", 1.)
    Xmn = (Xm[:, :, nax] .- Xn[:, nax, :]) ./ ll        # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-0.5 * dists)      # M x N
    # for m=1:M
    #     for n=1:N
    #         ll = l[m] * l[n]
    #         K[m, n] = sigma_f * exp(-(Xm[m] - Xn[n])^2 / 2ll)
    #     end
    # end
    if orig
        noise_f = get(hyperparameters, "noise_f", 0)
        return K + (noise_f * I(M))
    end

    # Derivative coefficient D x M x N
    dc_gd = -Xmn
    # DxMxN x 1xMxN -> DxMxN -> DMxN
    Kgd = dc_gd .* K[nax, :, :]
    Kgd = vcat([Kgd[i, :, :] for i in 1:size(Kgd, 1)]...)
    if potential_only
        return vcat(K, Kgd)                  # (D+1)M x N
    end
    # DxMxN * 1xMxN -> MxDN
    Kdg = -dc_gd .* K[nax, :, :]
    Kdg = hcat([Kdg[i, :, :] for i in 1:size(Kdg, 1)]...)
    # DxMxN -> MxDxN
    Xnm = permutedims(Xmn, [2, 1, 3])
    # DxMx1xN  * 1xMxDxN  -> D x M x D x N
    dc_dd_glob = -Xmn[:, :, nax, :] .* Xnm[nax, :, :, :]
    # dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
    # ∂_mn exp(Xn - Xm)^2
    dc_dd_diag = I(D)[:, nax, :, nax] ./ ll
    # DxMxDxN - DxMxDxN
    Kdd = @. (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
    # DM x DN
    Kdd = reshape(permutedims(Kdd, [2, 1, 4, 3]), D * M, D * N)
    if gradient_only
        # (D+1)M x DN
        return vcat(Kdg, Kdd)
    end
    if hessian_only
        # Delta _ dd
        dnm = [1:D;]
        # MxDxN
        dc_dg = -Xnm
        # MxDx1xN * Mx1xDxN -> MxDxDxN
        dc_hg_glob = -Xnm[:, :, nax, :] .* Xnm[:, nax, :, :]
        # dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
        # DxMxN -> MxDxN -> MxDDxN
        dc_hg_diag = zeros(M, D, D, N)
        # ∂_mm K(Xm,Xn) MxDxDxN
        # dc_hg_diag[:, dnm, dnm, :] = -1 ./ ll
        for i in 1:D
            dc_hg_diag[:, i, i, :] .= -1 / ll[i]
        end
        # MxDxDxN + MxDxDxN -> MxDxDxN
        Khg = dc_hg_glob + dc_hg_diag
        # Bacground term: ∂_mmn K(Xm,Xn) DxMx1x1xN * 1xMxDxDxN -> DxMxDxDxN
        dc_hd_back = dc_gd[:, :, nax, nax, :] .* Khg[nax, :, :, :, :]

        # Diagonal term : DxMxDxDxN * 1xMxDxDxN -> DxMxDxDxN
        dc_hd_diag = zeros(D, M, D, D, N)
        # Global term :
        dc_hd_glob = zeros(D, M, D, D, N)

        # dc_hd_glob[dnm, :, dnm, :, :] += Xmn[nax, :, :, :] ./ ll
        # dc_hd_glob[dnm, :, :, dnm, :] += Xmn[nax, :, :, :] ./ ll
        for i in 1:D
            dc_hd_glob[i, :, i, :, :] += Xnm ./ ll[i]
            dc_hd_glob[i, :, :, i, :] += Xnm ./ ll[i]
        end
        # print(dc_hd_glob[0, 0, 0, :, 0])
        # print(dc_hd_glob.reshape(2, N, 2))
        Khd = (dc_hd_glob + dc_hd_diag) + dc_hd_back
        # MxDxDxN x Mx1x1xN -> MxDxDxN
        Khg .*= K[:, nax, nax, :]
        # DxMxDxDxN * 1xMx1x1xN
        Khd .*= K[nax, :, nax, nax, :]
        # MxDDxN -> MxNxDD -> M x NDD
        Khg = reshape(permutedims(Khg, [1, 4, 3, 2]), M, D * D * N)
        # DxMxDDxN -> MxDxNxDD -> MD x NDD
        Khd = reshape(permutedims(Khd, [2, 1, 5, 4, 3]), D * M, D * D * N)
        # print(Khd.shape)
        return vcat(Khg, Khd)  # (D+1)M x DDN
    end
    Kext = [K Kdg;
            Kgd Kdd]  # (D+1)M x (D+1)N
    if noise
        noise_f = get(hyperparameters, "sigma_n^e", 0.)
        noise_df = get(hyperparameters, "sigma_n^f", 0.)
        noise = [noise_f .* ones(M)..., noise_df .* ones(D * M)...]
        return Kext .+ noise .* I((D + 1) * M)
    end
    return Kext
end

mutable struct Atomic <: Kernel
    hyperparameters::Array{Float64, 2}
    data
    idx::Int64
    function Atomic()
        new([1 1 1; 1  1  1], 1)
    end
    function Atomic(hyperparameters::Array{Float64, 2})
        new(hyperparameters, nothing, 1)
    end
end
"""
Xm : DxM
Xn : DxN
hyperparameters : A x A(A-1) + 2; [a, ll1, ll2, ..., σ, noise_f]
"""
function (ker::Atomic)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2})

    nax = [CartesianIndex()]
    D, M = size(Xm) # Xm : DxM
    N = size(Xn)[2] # Xn : DxN

    hyperparameters = ker.hyperparameters[ker.idx, :]
    ll = hyperparameters[1:end-2]
    σ_f = hyperparameters[end-1]

    Xmn = (Xm[:, :, nax] .- Xn[:, nax, :]) ./ ll        # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-0.5 * dists)                      # M x N

    return K
end

function (ker::Atomic)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, noise::Bool)
    @assert noise
    D, M = size(Xm)
    noise_f =  ker.hyperparameters[ker.idx, end]
    K = ker(Xm, Xn)
    return K + (noise_f * I(M))
end

end
