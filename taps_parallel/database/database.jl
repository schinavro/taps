
using SQLite, DataFrames

export read_data, ImgData

struct ImgData<:Database
    filename::String
end
ImgData() = ImgData("imgdata.db")

function Base.getproperty(database::Database, name::Symbol, x)
    if length(String(name)) > 7 && String(name)[end-6:end] == "_kwargs"
        name = Symbol(String(name)[1:end-7])
        field = getfield(database, name)
        x = x == nothing ? Dict([(f, nothing) for f in fieldnames(field)]) : x
        ans = Dict()
        for (key, value) in pairs(x)
            ans[key] = getproperty(field, key, value)
        end
        return ans
    elseif parentmodule(fieldtype(typeof(database), name)) ∉ [Base, Core]
        typename = typeof(getfield(database, name))
        return "$typename"
    else
        return getfield(database, name)
    end
end

function read_data(filename::String, data_ids)
    db = SQLite.DB(filename)
    db = DBInterface.connect(SQLite.DB, filename)
    img_ids = data_ids["image"]
    N = size(img_ids)
    cmdline = "SELECT * FROM image WHERE rowid IN ("
    cmdline *= join(map(id -> string(id), img_ids), ", ") * ")"
    ds = DBInterface.execute(db, cmdline) |> DataFrame
    data = Dict()
    coords = hcat(map(crd -> reinterpret(Float64, crd), ds.coord)...)
    gradients = hcat(map(grd -> reinterpret(Float64, grd), ds.gradients)...)
    if typeof(ds.potentials) != Array{Missing, 1}
    potentials = hcat(map(ptls -> reinterpret(Float64, ptls), ds.potentials)...)
    data["potentials"] = potentials # A x N ;
    end
    data["coords"] = coords        # D x N ; confirmed
    data["gradients"] = gradients  # D x N ; confirmed
    data["potential"] = Array{Float64, 1}(reinterpret(Float64, ds.potential))
    return data
end

include("./manybody.jl")
