module Regression

export regression
using LBFGSB
using LinearAlgebra
using ForwardDiff

Database = include("./database.jl")
# include("./models.jl")
using ..Models

nax = [CartesianIndex()]

function regression!(model::Model, MPI, comm)
    # comm = MPI.COMM_WORLD
    nprc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

    data = Database.read_data(model.imgdata_filename, model.data_ids)
    𝐤, 𝐦, 𝐆 = model.kernel, model.mean, model.descriptor
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    # 𝐗, 𝐘 = Models.data2𝐗𝐘(data)
    𝐗, 𝐘 = 𝐆.data2𝐗𝐘(data)

    A, M = size(𝐘)
    D = 3 * A
    N = M

    # Parallel read data_ids
    remainder = A % nprc
    quant = zeros(Int64, nprc)
    partn = zeros(Int64, nprc)
    temp = 0
    for i=1:nprc
        quant[i] = A ÷ nprc
        if remainder > 0
            quant[i] += 1
            remainder -= 1 # scope of variable
        end
        partn[i] = temp
        temp += quant[i]
    end

    𝐤.hyperparameters = 𝐆.kernel_hyperparameters(model.kernel_hyperparameters)
    𝐦.hyperparameters = 𝐆.mean_hyperparameters(model.mean_hyperparameters)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    nprtn = zeros(Int64, nprc + 1)
    nprtn[2:end] = partn[:]

    sendbuf = []

    optimizer = L_BFGS_B(1024, 17)

    bounds = 𝐆.kernel_bounds(model.kernel_bounds)

    function likelihood2(hyperparameters, Xmn, Y⏦m)
        Σ⁻¹ = zeros(eltype(hyperparameters), 3(A-1), 3(A-1))
        for i in 1:A-1
            Σ⁻¹[3i-2, 3i-2] = 1 / hyperparameters[i]
            Σ⁻¹[3i-1, 3i-1] = 1 / hyperparameters[i]
            Σ⁻¹[3i, 3i] = 1 / hyperparameters[i]
        end
        σ_f = hyperparameters[end-1]
        σ_n = hyperparameters[end]

        dists = zeros(eltype(hyperparameters), M, N)
        for m in 1:M for n in 1:N
            dists[m, n] = Xmn[:, m, n]' * Σ⁻¹ * Xmn[:, m, n]
        end end

        K = σ_f .* exp.(-0.5 * dists)                      # M x N
        if isposdef(K)
            detK = reduce(*, diag(cholesky(K).L))
        else
            detK = max(det(K), 1)
        end
        return log(detK) + 0.5 * Y⏦m' * inv(K + σ_n * I(M)) * Y⏦m
    end

    function likelihood(hyperparameters, Xmn, Y⏦m)
        return likelihood2(hyperparameters, Xmn, Y⏦m),
               ForwardDiff.gradient(x -> likelihood2(x, Xmn, Y⏦m), hyperparameters)
    end

    # for i = nprtn[rank+1]:nprtn[rank+2]-1
    for i = 1:A
        𝐤.idx, 𝐦.idx = i, i
        Xm = 𝐗[i, :, :]                         # 3(A-1) x M
        Y⏦m = 𝐘[i, :] .- 𝐦(Xm)                 # M
        Xmn = (Xm[:, :, nax] .- Xm[:, nax, :])  # 3(A-1) x M x N

        x0 = zeros(Float64, A+1)
        x0[end] = 𝐤.hyperparameters[i, end]
        x0[end-1] = 𝐤.hyperparameters[i, end-1]
        for j = 1:A-1
            x0[j] = 𝐤.hyperparameters[i, 3j]
        end
        # x0 = 𝐆.regression_fold(xout)
        fout, xout = optimizer(x -> likelihood(x, Xmn, Y⏦m), x0, bounds[i, :, :],
                               m=10, factr=1e7, pgtol=1e-5,
                               iprint=-1, maxfun=15000, maxiter=15000)

        𝐤.hyperparameters[i, end] = xout[end]
        𝐤.hyperparameters[i, end-1] = xout[end-1]
        for j = 1:A-1
            𝐤.hyperparameters[i, 3(j-1)+1:3j] .= xout[j]
        end
        # 𝐤.hyperparameters[i, :] = 𝐆.hyperparameters_expand(xout)
        # append(sendbuf, hyperparameters)
    end

    # MPI.ScatterV(sakdfh)

    numbers = model.kernel_hyperparameters["numbers"]
    model.kernel_hyperparameters = 𝐆.kernel_hyperparameters(𝐤.hyperparameters, numbers)
    model.optimized = false
end

function regression!(model::GaussianLite, MPI, comm)
    # comm = MPI.COMM_WORLD
    nprc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

    data = Database.read_data(model.imgdata_filename, model.data_ids)
    𝐤, 𝐦, 𝐆 = model.kernel, model.mean, model.descriptor
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    # 𝐗, 𝐘 = Models.data2𝐗𝐘(data)
    𝐗, 𝐘 = 𝐆.data2𝐗𝐘(data)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    M = size(𝐗)[end]
    #D = 3 * A
    #N = M
#
    ## Parallel read data_ids
    #remainder = A % nprc
    #quant = zeros(Int64, nprc)
    #partn = zeros(Int64, nprc)
    #temp = 0
    #for i=1:nprc
    #    quant[i] = A ÷ nprc
    #    if remainder > 0
    #        quant[i] += 1
    #        remainder -= 1 # scope of variable
    #    end
    #    partn[i] = temp
    #    temp += quant[i]
    #end
#
#
    #nprtn = zeros(Int64, nprc + 1)
    #nprtn[2:end] = partn[:]
#
    #sendbuf = []

    optimizer = L_BFGS_B(1024, 17)

    bounds = 𝐆.model2reg_bounds(model.kernel_bounds)

    """
    hyperparameters : DD + 2 shape 1D array
    """
    function likelihood2(θ, Xmn, Y⏦m)
        𝛉 = 𝐆.reg2ker_hyper(θ)
        ll = 𝛉[1:end-2]       # D
        σ_f = 𝛉[end-1]
        σ_n = 𝛉[end]

        dists = dropdims(sum((Xmn ./ ll).^2, dims=1), dims=1)      # MxN
        K = σ_f .* exp.(-0.5 * dists)                            # M x N
        if isposdef(K)
            detK = reduce(*, diag(cholesky(K).L))
        else
            detK = max(det(K), 1)
        end
        return log(detK) + 0.5 * Y⏦m' * inv(K + σ_n * I(M)) * Y⏦m
    end

    function likelihood(θ, Xmn, Y⏦m)
        return likelihood2(θ, Xmn, Y⏦m),
               ForwardDiff.gradient(x -> likelihood2(x, Xmn, Y⏦m), θ)
    end

    Y⏦m = 𝐘 .- 𝐦(𝐗)                      # M
    Xmn = (𝐗[:, :, nax] .- 𝐗[:, nax, :])  # 3A x M x N

    θ0 = 𝐆.model2reg_hyper(model.kernel_hyperparameters)
    fout, θout = optimizer(θ -> likelihood(θ, Xmn, Y⏦m), θ0, bounds,
                           m=10, factr=1e7, pgtol=1e-5,
                           iprint=-1, maxfun=15000, maxiter=15000)

    # MPI.ScatterV(sakdfh)
    𝐤.hyperparameters = 𝐆.reg2ker_hyper(θout)
    model.kernel_hyperparameters = 𝐆.reg2model_hyper(θout)
    model.optimized = false
end

function regression!(model::GaussianLite2, MPI, comm)
    # comm = MPI.COMM_WORLD
    nprc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

    data = Database.read_data(model.imgdata_filename, model.data_ids)
    𝐤, 𝐦, 𝐆 = model.kernel, model.mean, model.descriptor
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    # 𝐗, 𝐘 = Models.data2𝐗𝐘(data)
    𝐗, 𝐘 = 𝐆.data2𝐗𝐘(data)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    M = size(𝐗)[end]
    #D = 3 * A
    #N = M
#
    ## Parallel read data_ids
    #remainder = A % nprc
    #quant = zeros(Int64, nprc)
    #partn = zeros(Int64, nprc)
    #temp = 0
    #for i=1:nprc
    #    quant[i] = A ÷ nprc
    #    if remainder > 0
    #        quant[i] += 1
    #        remainder -= 1 # scope of variable
    #    end
    #    partn[i] = temp
    #    temp += quant[i]
    #end
#
#
    #nprtn = zeros(Int64, nprc + 1)
    #nprtn[2:end] = partn[:]
#
    #sendbuf = []

    optimizer = L_BFGS_B(1024, 17)

    bounds = 𝐆.model2reg_bounds(model.kernel_bounds)

    """
    hyperparameters : DD + 2 shape 1D array
    """
    function likelihood2(θ, Xmn, Y⏦m)
        𝛉 = 𝐆.reg2ker_hyper(θ)
        ll = 𝛉[1:end-2]       # D
        σ_f = 𝛉[end-2]
        σ_ne = 𝛉[end-1]
        σ_nf = 𝛉[end]

        dists = dropdims(sum((Xmn ./ ll).^2, dims=1), dims=1)      # MxN
        K = σ_f .* exp.(-0.5 * dists)                            # M x N
        M′ = M < 10 ? M : 10
        fidx = (1:M |> collect)[end-M′+1:end]
        Xm′n = Xmn[:, fidx, :]
        K′ = K[fidx, :]

        # Derivative coefficient D x M` x N
        dc_gd = -Xm′n ./ ll
        # DxM′xN x 1xM′xN -> DxM′xN -> DM′xN
        K′gd = reshape(dc_gd .* K′[nax, :, :], D*M′, N)
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
        Kext = [K K′dg;
                K′gd K′dd]

        if isposdef(Kext)
            detK = reduce(*, diag(cholesky(Kext).L))
        else
            detK = max(det(Kext), 1)
        end
        KK = [K+σ_ne*I(M) K′dg; K′gd K′dd+σ_nf*I(D*M′)]
        return log(detK) + 0.5 * Y⏦m' * inv(KK) * Y⏦m
    end

    function likelihood(θ, Xmn, Y⏦m)
        return likelihood2(θ, Xmn, Y⏦m),
               ForwardDiff.gradient(x -> likelihood2(x, Xmn, Y⏦m), θ)
    end

    Y⏦m = 𝐘 .- 𝐦(𝐗)                      # M
    Xmn = (𝐗[:, :, nax] .- 𝐗[:, nax, :])  # 3A x M x N

    θ0 = 𝐆.model2reg_hyper(model.kernel_hyperparameters)
    fout, θout = optimizer(θ -> likelihood(θ, Xmn, Y⏦m), θ0, bounds,
                           m=10, factr=1e7, pgtol=1e-5,
                           iprint=-1, maxfun=15000, maxiter=15000)

    # MPI.ScatterV(sakdfh)
    𝐤.hyperparameters = 𝐆.reg2ker_hyper(θout)
    model.kernel_hyperparameters = 𝐆.reg2model_hyper(θout)
    model.optimized = false
end

function regression!(model::DescriptorGaussian, MPI, comm)
    # comm = MPI.COMM_WORLD
    nprc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
    root = 0

    data = Database.read_data(model.imgdata_filename, model.data_ids)
    𝐤, 𝐦, 𝐆 = model.kernel, model.mean, model.descriptor
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    # 𝐗, 𝐘 = Models.data2𝐗𝐘(data)
    𝐗, 𝐘 = 𝐆.data2𝐗𝐘(data)

    A, M = size(𝐘)
    D = 3 * A
    hyperD = 𝐆.hyperD
    N = M

    # Parallel read data_ids
    remainder = A % nprc
    quant = zeros(Int64, nprc)
    partn = zeros(Int64, nprc)
    temp = 0
    for i=1:nprc
        quant[i] = A ÷ nprc
        if remainder > 0
            quant[i] += 1
            remainder -= 1 # scope of variable
        end
        partn[i] = temp
        temp += quant[i]
    end

    𝐤.hyperparameters = 𝐆.kernel_hyperparameters(model.kernel_hyperparameters)
    𝐦.hyperparameters = 𝐆.mean_hyperparameters(model.mean_hyperparameters)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    nprtn = zeros(Int64, nprc + 1)
    nprtn[2:end] = partn[:]

    sendbuf = []

    optimizer = L_BFGS_B(1024, 17)

    bounds = 𝐆.kernel_bounds(model.kernel_bounds)

    """
    hyperparameters : DD + 2 shape 1D array
    """
    function likelihood2(hyperparameters, Xmn, Y⏦m)
        Σ⁻¹ = zeros(eltype(hyperparameters), hyperD, hyperD)
        for i in 1:hyperD
            Σ⁻¹[i, i] = 1. / hyperparameters[i]
        end
        σ_f = hyperparameters[end-1]
        σ_n = hyperparameters[end]

        dists = zeros(eltype(hyperparameters), M, N)
        for m in 1:M for n in 1:N
            # dists[m, n] = Xmn[:, m, n]' * Σ⁻¹ * Xmn[:, m, n]
            dists[m, n] = Xmn[:, m, n]' * Xmn[:, m, n] / 0.0001
        end end

        K = σ_f .* exp.(-0.5 * dists)                      # M x N
        if isposdef(K)
            detK = reduce(*, diag(cholesky(K).L))
        else
            detK = max(det(K), 1)
        end
        return log(detK) + 0.5 * Y⏦m' * inv(K + σ_n * I(M)) * Y⏦m
    end

    function likelihood(hyperparameters, Xmn, Y⏦m)
        return likelihood2(hyperparameters, Xmn, Y⏦m),
               ForwardDiff.gradient(x -> likelihood2(x, Xmn, Y⏦m), hyperparameters)
    end

    # for i = nprtn[rank+1]:nprtn[rank+2]-1
    for a = 1:A
        𝐤.idx, 𝐦.idx = a, a
        Xm = 𝐗[a, :, :]                         # 3(A-1) x M
        Y⏦m = 𝐘[a, :] .- 𝐦(Xm)                 # M
        Xmn = (Xm[:, :, nax] .- Xm[:, nax, :])  # 3(A-1) x M x N

        x0 = 𝐤.hyperparameters[a, :]
        # x0 = 𝐆.regression_fold(xout)
        # fout, xout = optimizer(x -> likelihood(x, Xmn, Y⏦m), x0, bounds[a, :, :],
        #                        m=10, factr=1e7, pgtol=1e-5,
        #                        iprint=-1, maxfun=15000, maxiter=15000)
        xout = x0

        𝐤.hyperparameters[a, :] = xout
        # 𝐤.hyperparameters[i, :] = 𝐆.hyperparameters_expand(xout)
        # append(sendbuf, hyperparameters)
    end

    # MPI.ScatterV(sakdfh)

    numbers = model.numbers
    model.kernel_hyperparameters = 𝐆.kernel_hyperparameters(𝐤.hyperparameters, numbers)
    model.optimized = false
end

end
