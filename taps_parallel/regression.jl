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
    # 𝐗::Array{Float64, 3};   3xAxM -> A x 3(A-1) x M
    # 𝐘::Array{Float64, 2};   A x M
    𝐗, 𝐘 = Models.data2𝐗𝐘(data)

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

    𝐤, 𝐦 = model.kernel, model.mean
    𝐤.hyperparameters = Models._kernel_hyperparameters(model.kernel_hyperparameters)
    𝐦.hyperparameters = Models._mean_hyperparameters(model.mean_hyperparameters)
    𝐤.data, 𝐦.data  = 𝐘, 𝐘

    nprtn = zeros(Int64, nprc + 1)
    nprtn[2:end] = partn[:]

    sendbuf = []

    optimizer = L_BFGS_B(1024, 17)

    bounds = _kernel_bounds(model.kernel_bounds)

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

    function likelihood(hyperparamters, Xmn, Y⏦m)
        return likelihood2(hyperparamters, Xmn, Y⏦m),
               ForwardDiff.gradient(x -> likelihood2(x, Xmn, Y⏦m), hyperparamters)
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
        fout, xout = optimizer(x -> likelihood(x, Xmn, Y⏦m), x0, bounds[i, :, :],
                               m=10, factr=1e7, pgtol=1e-5,
                               iprint=-1, maxfun=15000, maxiter=15000)

        𝐤.hyperparameters[i, end] = xout[end]
        𝐤.hyperparameters[i, end-1] = xout[end-1]
        for j = 1:A-1
            𝐤.hyperparameters[i, 3(j-1)+1:3j] .= xout[j]
        end
        # append(sendbuf, hyperparamters)
    end

    # MPI.ScatterV(sakdfh)

    numbers = model.kernel_hyperparameters["numbers"]
    model.kernel_hyperparameters = Models._kernel_hyperparameters(𝐤.hyperparameters, numbers)
    model.optimized = false
end

"""
kernel_bounds : Dict {"numbers" => [1, 2, 3], "sig_13" => [2, 3]} A x 3(A-1) + 2
numbers : A
return : Dict
"""
function _kernel_bounds(kernel_bounds)
    numbers = kernel_bounds["numbers"]
    A = length(numbers)
    n = (A - 1) + 2
    bounds = zeros(A, 3, n)
    for (i, ni) in enumerate(numbers)
        # represents the type of bounds imposed on the variables:
        #  0->unbounded,
        #  1->only lower bound
        #  2-> both lower and upper bounds
        #  3->only upper bound
        bounds[i, 1, end] = kernel_bounds["noise_$ni"][i][1]  # Type
        bounds[i, 2, end] = kernel_bounds["noise_$ni"][i][2]  # Lower
        bounds[i, 3, end] = kernel_bounds["noise_$ni"][i][3]  # Upper
        bounds[i, 1, end-1] = kernel_bounds["sig_$ni"][i][1]
        bounds[i, 2, end-1] = kernel_bounds["sig_$ni"][i][2]
        bounds[i, 3, end-1] = kernel_bounds["sig_$ni"][i][3]

        masked = numbers[[i!=j for j in 1:A]]
        for (j, nj) in enumerate(masked)
            bounds[i, 1, j] = kernel_bounds["ll_$nj"][i][1]
            bounds[i, 2, j] = kernel_bounds["ll_$nj"][i][2]
            bounds[i, 3, j] = kernel_bounds["ll_$nj"][i][3]
        end
    end
    return bounds
end

"""
hyperparameters = {'numbers': [28, 79, 78],
                       sig_28: [1, 1, 1], sig_79: [1, 1, 1], sig_78: [1, 1, 1],
                       noise_28: [1e-4]*3, noise_79: [1e-4]*3, noise_78: [1e-4]*3,
                       ll_28, 28): l1, (28, 79): l2, (28, 78): l3,
                       (79, 79): l4, (79, 78): l5, (78, 78): l6}
"""
function _kernel_hyperparameters(hyperparameters::Dict)
    numbers = hyperparameters["numbers"]
    A = size(numbers)[1]
    hyper_kernel = zeros(Float64, A, 3(A-1) + 2)
    for (i, ni) in enumerate(numbers)
        hyper_kernel[i, end-1] = hyperparameters["sig_$ni"][i]
        hyper_kernel[i, end] = hyperparameters["noise_$ni"][i]
        masked = numbers[[i!=j for j in 1:A]]
        for (j, nj) in enumerate(masked)
            ll = hyperparameters["ll_$nj"][i]
            hyper_kernel[i, 3(j-1)+1:3(j)] .= ll
        end
    end
    # A x A(A-1) + 2
    return hyper_kernel
end


end
