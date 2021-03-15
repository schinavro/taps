module Kernel


export Standard, Perioidc

struct Standard
    hyperparameters::Dict{String, Float64}
end

function (kernel::Standard)(Xm::Array{Flaot64, 2}, Xn::Array{Flaot64, 2};
    orig::Bool=false,
    noise::Bool=false,
    hyperparameters::Dict{String, Float64}=nothing,
    gradient_only::Bool=false,
    hessian_only::Bool=false,
    potential_only::Bool=false)

    hyperparameters = hyperparameters == nothing ? kernel.hyperparameters :
                                                    hyperparameters
    ll = get(hyperparameters, "ll")
    sigma_f = get(hyperparameters, "sigma_f")
    Xn = Xn == nothing ? copy(Xm) : Xn
    N = size(Xn)[1]
    M = size(Xm)[1]

    Xnm = zeros(N, M, D)
    for i=1:N
        for j=1:M
            Xnm[i, j] = exp(-2 )



end
