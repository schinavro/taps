module Mean

struct Mean
    data::Dict
    mean_type::String
    function Mean(data)
        mean_type = "average"
        new(data, mean_type)
    end
end

function (mean::Mean)(X::Array{Float64, 2}; data::Dict{String, Array},
                      hess::Bool=true, mean_type::String=nothing)
    mean_type = mean_type == nothing ? mean.mean_type : mean_type
    data = data == nothing? model.data : data

    if mean_type == "zero"
        return 0.
    end

    V = data["V"]
    N, D = size(X)
    F = zeros(N, D)
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
    ef = hstack([e, F])
    return flatten(ef)
end
