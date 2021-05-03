module Kernels

export Standard, SE, Perioidc

using LinearAlgebra

abstract type Kernel end

nax = [CartesianIndex()]

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


"""
Gaussian only with limited potential
"""
mutable struct GaussianLite <: Kernel
    hyperparameters::Any
    data::Any
    function GaussianLite(;hyperparameters=nothing, data=nothing)
        new(hyperparameters, data)
    end
end

function (ker::GaussianLite)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, noise::Bool)
    @assert noise
    D, M = size(Xm)
    noise_f =  ker.hyperparameters[end]
    K = ker(Xm, Xn)
    return K + (noise_f * I(M))
end

function (ker::GaussianLite)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2};
                             gradients=false, hessian=false)

    D, M = size(Xm) # Xm : DxM
    A = D > 3 ? D ÷ 3 : 1
    N = size(Xn)[2] # Xn : DxN

    ll = ker.hyperparameters[1:end-2]
    σ_f = ker.hyperparameters[end-1]

    Xmn = (Xm[:, :, nax] .- Xn[:, nax, :]) ./ ll       # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-0.5 * dists)                      # M x N

    if gradients || hessian
        # DxMxN * 1xMxN -> MxDN
        # Derivative coefficient D x M x N
        dc_gd = -Xmn ./ ll
        Kdg = -dc_gd .* K[nax, :, :]
        # Kdg = hcat([Kdg[i, :, :] for i in 1:size(Kdg, 1)]...)
        # DxMxN -> MxDxN
        Kdg = permutedims(Kdg, [2, 1, 3])
        # Xnm = permutedims(Xmn, [2, 1, 3])
        # # DxMx1xN  * 1xMxDxN  -> D x M x D x N
        # dc_dd_glob = -Xmn[:, :, nax, :] .* Xnm[nax, :, :, :]
        # # dc_dd_glob = np.einsum('inm, jnm -> injm', dc_gd, -dc_gd)
        # # ∂_mn exp(Xn - Xm)^2
        # dc_dd_diag = I(D)[:, nax, :, nax] ./ ll
        # # DxMxDxN - DxMxDxN
        # Kdd = @. (dc_dd_glob + dc_dd_diag) * K[nax, :, nax, :]
        # # DM x DN
        # Kdd = reshape(permutedims(Kdd, [2, 1, 4, 3]), D * M, D * N)
        if gradients
            return reshape(Kdg, M, D*N)
        end
    end

    if hessian
        # Delta _ dd
        # dnm = [1:D;]
        # MxDxN
        dc_gd = permutedims(-Xmn ./ ll, [2, 1, 3])
        # dc_gd = -Xnm
        # Loop A :=> Mx3x1xN * Mx1x3xN -> Mx3x3AxN
        dc_hg_glob = zeros(Float64, M, 3, D, N)
        for a=1:A
            i, j = 3a-2, 3a
            dc_hg_glob[:, :, i:j, :] = dc_gd[:, i:j, nax, :] .* dc_gd[:, nax, i:j, :]
        end
        # dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
        # DxMxN -> MxDxN -> MxDDxN
        # ∂_mm K(Xm,Xn) MxDxDxN
        # dc_hg_diag[:, dnm, dnm, :] = -1 ./ ll
        # graph = []
        dc_hg_diag = zeros(M, 3, D, N)
        for a in 1:A
            # nidx = graph[a, :]
            nidx = [1, 2, 3]
            xyz = [3a-2, 3a-1, 3a]
            dc_hg_diag[:, nidx, xyz, :] .= -1 / ll[xyz]
        end
        # Mx3xDxN + Mx3xDxN -> Mx3xDxN
        Khg = dc_hg_glob + dc_hg_diag
        Khg = Khg .* K[:, nax, nax, :]
        return reshape(Khg, M, 3D*N)
    end

    return K
end


"""
Gaussian only with limited potential
"""
mutable struct GaussianLite2 <: Kernel
    hyperparameters::Any
    data::Any
    function GaussianLite2(;hyperparameters=nothing, data=nothing)
        new(hyperparameters, data)
    end
end

function (ker::GaussianLite2)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, noise::Bool)
    @assert noise
    D, M = size(Xm)
    M′ = M < 10 ? M : 10
    noise_f =  ker.hyperparameters[end]
    noise_e =  ker.hyperparameters[end-1]
    noise = zeros(M + D*M′, M + D*M′)
    for d=1:M
        noise[d, d] = noise_e
    end
    for d=M+1:M+D*M′
        noise[d, d] = noise_f
    end
    Kext = ker(Xm, Xn)
    return Kext + noise
end

function (ker::GaussianLite2)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2};
                              potentials=false, gradients=false, hessian=false,
                              covariance=false)

    D, M = size(Xm) # Xm : DxM
    A = D > 3 ? D ÷ 3 : 1
    N = size(Xn)[2] # Xn : DxN

    ll = ker.hyperparameters[1:end-2]
    σ_f = ker.hyperparameters[end-1]

    Xmn = (Xm[:, :, nax] .- Xn[:, nax, :]) ./ ll       # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-0.5 * dists)                      # M x N
    if covariance
        noise_e = ker.hyperparameters[end-1]
        K[1:end, 1:end] .+= noise_e
        return K
    end

    M′ = M < 10 ? M : 10
    fidx = (1:M |> collect)[end-M′+1:end]
    Xm′n = Xmn[:, fidx, :]
    K′ = K[fidx, :]

    # Derivative coefficient D x M` x N
    dc_gd = -Xm′n ./ ll
    # DxM′xN x 1xM′xN -> DxM′xN -> DM′xN
    K′gd = reshape(dc_gd .* K′[nax, :, :], D*M′, N)
    if potentials
        return vcat([K, K′gd])
    end

    # if gradients || hessian
    # DxMxN * 1xMxN -> MxDN
    # Derivative coefficient D x M x N
    dc_gd = -Xmn ./ ll
    Kdg = -dc_gd .* K[nax, :, :]
    # DxMxN -> MxDxN
    Kdg = reshape(permutedims(Kdg, [2, 1, 3]), M, D*N)
    # DxMx1xN  * 1xMxDxN  -> D x M x D x N
    Xnm′ = permutedims(Xm′n, [2, 1, 3])
    # DxM′x1xN  * 1xM′xDxN  -> D x M′ x D x N
    dc_dd_glob = -Xm′n[:, :, nax, :] .* Xnm′[nax, :, :, :]
    # # ∂_mn exp(Xn - Xm)^2
    dc_dd_diag = I(D)[:, nax, :, nax] ./ ll
    # DxMxDxN - DxMxDxN
    K′dd = @. (dc_dd_glob + dc_dd_diag) * K′[nax, :, nax, :]
    # DM′ x DN
    # Kdd = reshape(permutedims(Kdd, [2, 1, 4, 3]), D * M, D * N)
    K′dd = reshape(K′dd, D * M′, D * N)
    if gradients
        return vcat([Kdg, K′dd])
    end
    # end

    if hessian
        # Delta _ dd
        # dnm = [1:D;]
        # MxDxN
        dc_gd = permutedims(-Xmn ./ ll, [2, 1, 3])
        # dc_gd = -Xnm
        # Loop A :=> Mx3x1xN * Mx1x3xN -> Mx3x3AxN
        dc_hg_glob = zeros(Float64, M, 3, D, N)
        for a=1:A
            i, j = 3a-2, 3a
            dc_hg_glob[:, :, i:j, :] = dc_gd[:, i:j, nax, :] .* dc_gd[:, nax, i:j, :]
        end
        # dc_hg_glob = np.einsum('inm, jnm -> nijm', -dc_gd, -dc_gd)
        # DxMxN -> MxDxN -> MxDDxN
        # ∂_mm K(Xm,Xn) MxDxDxN
        # dc_hg_diag[:, dnm, dnm, :] = -1 ./ ll
        # graph = []
        dc_hg_diag = zeros(M, 3, D, N)
        for a in 1:A
            # nidx = graph[a, :]
            nidx = [1, 2, 3]
            xyz = [3a-2, 3a-1, 3a]
            dc_hg_diag[:, nidx, xyz, :] .= -1 / ll[xyz]
        end
        # Mx3xDxN + Mx3xDxN -> Mx3xDxN
        Khg = dc_hg_glob + dc_hg_diag
        Khg = reshape(Khg .* K[:, nax, nax, :], M, 3D*N)

        #########################
        K′hg = Khg[fidx, :, :, :]
        dc_gd′ = dc_gd[:, fidx, :]
        # Bacground term: ∂_mmn K(Xm,Xn) DxM′x1x1xN * 1xM′x3xDxN -> DxM′x3xDxN
        dc_hd_back′ = zeros(D, M′, 3, D, N)
        for a in 1:A
            nidx = [1, 2, 3]
            xyz = [3a-2, 3a-1, 3a]
            dc_hd_back′ = dc_gd′[:, fidx, [1, 2, 3], xyz, :] * K′hg[nax, :, xyz, xyz, :]
        end
        # dc_hd_back′ = dc_gd′[:, :, nax, nax, :] * K′hg[nax, :, :, :, :]
        # Diagonal term : DxMxDxDxN * 1xMxDxDxN -> DxMxDxDxN
        dc_hd_diag′ = zeros(D, M′, 3, D, N)
        # Global term :
        dc_hd_glob′ = zeros(D, M′, 3, D, N)
        # dc_hd_glob[dnm, :, dnm, :, :] += Xmn[nax, :, :, :] / ll / ll
        # dc_hd_glob[dnm, :, :, dnm, :] += Xmn[nax, :, :, :] / ll / ll
        llll = ll[nax, :] * ll[nax, :]
        for a in 1:A
            # nidx = [1, 2, 3]
            # xyz = [3a-2, 3a-1, 3a]
            #dc_hd_glob[xyz, :, nidx, :, :] .+= X′nm[nax, :, :, :] ./ ll[xyz] ./ ll[xyz]
            #dc_hd_glob[xyz, :, :, xyz, :] += X′nm[nax, :, xyz, :] ./ ll[xyz] ./ ll[xyz]
            dc_hd_glob[3a-2, :, 1, :, :] += X′nm[:, :, :] ./ llll
            dc_hd_glob[3a-1, :, 2, :, :] += X′nm[:, :, :] ./ llll
            dc_hd_glob[3a  , :, 3, :, :] += X′nm[:, :, :] ./ llll

            dc_hd_glob[3a-2, :, 1, 3a-2, :] += X′nm[:, 3a-2, :] ./ llll[:, 3a-2]
            dc_hd_glob[3a-1, :, 2, 3a-1, :] += X′nm[:, 3a-1, :] ./ llll[:, 3a-1]
            dc_hd_glob[3a  , :, 3, 3a  , :] += X′nm[:, 3a  , :] ./ llll[:, 3a  ]
            # dc_hd_glob[dnm, :, dnm, :, :] += X′mn[nax, :, :, :] / ll / ll
            # dc_hd_back′ = dc_gd′[:, fidx, [1, 2, 3], xyz, :] * K′hg[nax, :, xyz, xyz, :]
        end
        # for i in 1:D
        #     dc_hd_glob[i, :, i, :, :] += X′nm ./ ll / ll
        #     dc_hd_glob[i, :, :, i, :] += X′nm ./ ll / ll
        # end

        # print(dc_hd_glob[0, 0, 0, :, 0])
        # print(dc_hd_glob.reshape(2, N, 2))
        Khd = (dc_hd_glob + dc_hd_diag) + dc_hd_back
        # Mx3xDxN x Mx1x1xN -> Mx3xDxN
        Khg *= K[:, nax, nax, :]
        # DxM′x3xDxN * 1xM′x3x1xN
        Khd *= K[nax, :, nax, nax, :]
        # Mx3DxN -> M x 3DN
        Khg = reshape(Khg, M, 3D * N)
        # DxM′x3DxN -> DM′ x 3DN
        Khd = reshape(Khd, D * M′, 3D * N)
        # print(Khd.shape)

        return vcat([Khg, Khd])
    end

    # M+DM′ x (D+1)N
    return [K K′dg;
            K′gd K′dd]
end


###########################################################################
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
#######################################
mutable struct Atomic2 <: Kernel
    hyperparameters::Array{Float64, 2}
    data
    idx::Int64
    function Atomic2()
        new([1 1 1; 1  1  1], 1)
    end
    function Atomic2(hyperparameters::Array{Float64, 2})
        new(hyperparameters, nothing, 1)
    end
end
"""
Xm : DxM
Xn : DxN
hyperparameters : A x A(A-1) + 2; [a, ll1, ll2, ..., σ, noise_f]
"""
function (ker::Atomic2)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2})

    D, M = size(Xm) # Xm : DxM
    N = size(Xn)[2] # Xn : DxN

    hyperparameters = ker.hyperparameters[ker.idx, :]
    Σ⁻¹ = zeros(eltype(hyperparameters), D, D)
    for i in 1:D
        Σ⁻¹[i, i] = 1. / hyperparameters[i]
    end
    σ_f = hyperparameters[end-1]
    σ_n = hyperparameters[end]

    Xmn = Xm[:, :, nax] .- Xn[:, nax, :]

    dists = zeros(eltype(hyperparameters), M, N)
    for m in 1:M for n in 1:N
        # dists[m, n] = Xmn[:, m, n]' * Σ⁻¹ * Xmn[:, m, n]
        dists[m, n] = Xmn[:, m, n]' * Xmn[:, m, n] / 0.0001
    end end

    K = σ_f .* exp.(-0.5 * dists)                      # M x N
    return K
end

function (ker::Atomic2)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, noise::Bool)
    @assert noise
    D, M = size(Xm)
    noise_f =  ker.hyperparameters[ker.idx, end]
    K = ker(Xm, Xn)
    return K + (noise_f * I(M))
end

##############################################################

mutable struct Atomic3 <: Kernel
    function Atomic3()
        new()
    end
end

"""
Xm : DxM
Xn : DxN
hyperparameters : A x A(A-1) + 2; [a, ll1, ll2, ..., σ, noise_f]
"""
function (ker::Atomic3)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, θ::Array)

    # hyperparameters = ker.hyperparameters[ker.idx, :]
    ll = θ[1:end-2]
    σ_f = θ[end-1]

    Xmn = (Xm[:, :, nax] .- Xn[:, nax, :]) ./ ll        # DxMxN
    dists = dropdims(sum(Xmn.^2, dims=1), dims=1)      # MxN
    K = σ_f .* exp.(-0.5 * dists)                      # M x N

    return K
end

function (ker::Atomic3)(Xm::Array{Float64, 2}, Xn::Array{Float64, 2}, noise::Bool)
    @assert noise
    D, M = size(Xm)
    noise_f =  ker.hyperparameters[ker.idx, end]
    K = ker(Xm, Xn)
    return K + (noise_f * I(M))
end


end
