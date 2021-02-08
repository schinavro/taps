module Means

struct Mean
    data_ids::Dict{Any, Any}
    mean_type::String
end

function (mean::Mean)(X;
                      data=nothing, hess=true, mean_type=nothing)
    mean_type = mean_type == nothing ? mean.mean_type : mean_type
    data = data == nothing ? model.data : data

    if mean_type == "zero"
        return 0.
    end

    V = data["V"]
    M, D = size(X)
    F = zeros(D, M)
    if type == "average"
        e = zeros(N) .+ average(V)
    elseif type == "min"
        e = zeros(N) .+ min(V)
    else
        e = zeros(N) .+ max(V)
    end
    if !hess
        return e
    end
    ef = hcat(e, F)
    return vec(ef)
end

end
